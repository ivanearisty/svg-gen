"""Parallel inference on Modal — fans out across multiple GPU containers."""

from __future__ import annotations

import csv
import os
import time

from svg_gen.config import InferenceConfig, TrainingConfig
from svg_gen.modal_app import (
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    VOLUME_MOUNT,
    app,
    gpu_image,
    volume,
)


@app.function(
    image=gpu_image,
    gpu="A100-40GB",
    timeout=2 * 3600,
    volumes={VOLUME_MOUNT: volume},
)
def generate_batch(
    batch: list[tuple[str, str]],
    adapter_path: str,
    inference_config: InferenceConfig,  # noqa: ARG001
    training_config: TrainingConfig,
) -> list[tuple[str, str, bool]]:
    """Generate SVGs for a batch of (id, prompt) pairs on a single GPU.

    Returns list of (id, svg, is_valid) tuples.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from svg_gen.config import SYSTEM_PROMPT
    from svg_gen.data import extract_svg, fallback_svg, is_valid_svg, normalize_viewbox, repair_svg

    # --- Load model ---
    bnb_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        training_config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print(f"Processing {len(batch)} samples...", flush=True)

    # --- Generate one at a time (most efficient for small 4-bit models) ---
    results: list[tuple[str, str, bool]] = []
    t0 = time.time()

    for i, (sample_id, prompt) in enumerate(batch):
        chat_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                repetition_penalty=1.1,
            )

        decoded = str(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        svg = extract_svg(decoded)

        # Try repair before giving up
        if svg and not is_valid_svg(svg, training_config.svg):
            svg = repair_svg(svg)

        valid = is_valid_svg(svg, training_config.svg)
        svg = fallback_svg() if not valid else normalize_viewbox(svg)

        results.append((sample_id, svg, valid))

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            valid_so_far = sum(1 for *_, v in results if v)
            print(f"  [{i + 1}/{len(batch)}] {rate:.1f} samples/s | valid={valid_so_far}/{i + 1}", flush=True)

    elapsed = time.time() - t0
    valid_count = sum(1 for *_, v in results if v)
    print(f"Batch done: {len(batch)} samples in {elapsed:.1f}s, {valid_count} valid", flush=True)
    return results


@app.function(
    image=gpu_image,
    volumes={VOLUME_MOUNT: volume},
)
def merge_results(all_results: list[list[tuple[str, str, bool]]]) -> str:
    """Merge results from all containers into a single submission CSV."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, "submission.csv")

    total = 0
    invalid = 0
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "svg"])
        for batch_results in all_results:
            for sample_id, svg, valid in batch_results:
                writer.writerow([sample_id, svg])
                total += 1
                if not valid:
                    invalid += 1

    print(f"Wrote {total} rows ({invalid} invalid/fallback) to {output_path}")
    volume.commit()
    return output_path


@app.local_entrypoint()
def main(
    adapter_path: str | None = None,
    num_containers: int = 4,
) -> None:
    """Fan out inference across multiple GPU containers.

    Usage: `modal run src/svg_gen/inference.py --num-containers 4`
    """
    import pandas as pd

    if adapter_path is None:
        # Auto-detect latest checkpoint on volume
        import modal as _modal

        vol = _modal.Volume.from_name("svg-gen-vol")
        checkpoints = [
            e.path.split("/")[-1]
            for e in vol.listdir("checkpoints/")
            if e.path.split("/")[-1].startswith("checkpoint-")
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            adapter_path = os.path.join(CHECKPOINTS_DIR, latest)
            print(f"Auto-detected latest checkpoint: {adapter_path}")
        else:
            adapter_path = os.path.join(CHECKPOINTS_DIR, "final-adapter")
            print(f"No checkpoints found, falling back to: {adapter_path}")

    inference_config = InferenceConfig(max_new_tokens=1536)
    training_config = TrainingConfig()

    # Load test data locally and split into chunks
    test_csv = os.path.join("data", "test.csv")
    test_df = pd.read_csv(test_csv)

    pairs: list[tuple[str, str]] = []
    for _, row in test_df.iterrows():
        sample_id = str(row.get("id", ""))
        prompt = str(row.get("prompt", row.get("text", "")))
        pairs.append((sample_id, prompt))

    # Split into chunks for each container
    chunk_size = (len(pairs) + num_containers - 1) // num_containers
    chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    print(f"Distributing {len(pairs)} samples across {len(chunks)} containers ({chunk_size} each)")
    t0 = time.time()

    # Fan out: each container gets a chunk and loads its own model
    all_results: list[list[tuple[str, str, bool]]] = list(
        generate_batch.map(
            chunks,
            kwargs={
                "adapter_path": adapter_path,
                "inference_config": inference_config,
                "training_config": training_config,
            },
        ),
    )

    elapsed = time.time() - t0
    total = sum(len(r) for r in all_results)
    print(f"All containers done: {total} samples in {elapsed:.1f}s ({total / elapsed:.1f} samples/s)")

    # Merge into single CSV
    output = merge_results.remote(all_results)
    print(f"Submission saved to: {output}")
