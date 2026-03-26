"""Decoding strategy sweep — test different generation configs on the same model."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from svg_gen.modal_app import (
    VOLUME_MOUNT,
    app,
    gpu_image,
    volume,
)


@dataclass
class DecodeConfig:
    """A single decoding strategy to test."""

    name: str
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    num_beams: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    max_new_tokens: int = 1536


STRATEGIES: list[DecodeConfig] = [
    DecodeConfig(name="greedy"),
    DecodeConfig(name="greedy-rep1.1", repetition_penalty=1.1),
    DecodeConfig(name="greedy-rep1.1-ngram8", repetition_penalty=1.1, no_repeat_ngram_size=8),
    DecodeConfig(name="temp0.1", temperature=0.1, top_p=0.95, do_sample=True),
    DecodeConfig(name="temp0.3", temperature=0.3, top_p=0.9, do_sample=True),
    DecodeConfig(name="temp0.7", temperature=0.7, top_p=0.9, do_sample=True, repetition_penalty=1.05),
    DecodeConfig(name="beam4", num_beams=4),
    DecodeConfig(name="beam4-ngram8", num_beams=4, no_repeat_ngram_size=8),
]


@app.function(
    image=gpu_image,
    gpu="A100-40GB",
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: volume},
)
def run_decode_sweep(  # noqa: PLR0915
    adapter_path: str,
    num_samples: int = 10,
) -> list[dict[str, object]]:
    """Test all decoding strategies on the same model + prompts."""
    import pandas as pd
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from svg_gen.config import SYSTEM_PROMPT
    from svg_gen.data import extract_svg, is_valid_svg, repair_svg

    # --- Load model ---
    bnb_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Load test prompts
    test_df = pd.read_csv(os.path.join("/vol/data", "test.csv")).head(num_samples)
    prompts = []
    for _, row in test_df.iterrows():
        prompt = str(row.get("prompt", row.get("text", "")))
        prompts.append(prompt)

    print(f"Testing {len(STRATEGIES)} strategies on {len(prompts)} prompts", flush=True)

    results: list[dict[str, object]] = []

    for strategy in STRATEGIES:
        print(f"\n--- {strategy.name} ---", flush=True)
        t0 = time.time()
        valid_count = 0
        svg_lengths: list[int] = []
        sample_svgs: list[str] = []

        for prompt in prompts:
            chat_text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

            gen_kwargs: dict[str, object] = {
                "max_new_tokens": strategy.max_new_tokens,
                "repetition_penalty": strategy.repetition_penalty,
            }
            if strategy.num_beams > 1:
                gen_kwargs["num_beams"] = strategy.num_beams
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = strategy.do_sample
                if strategy.do_sample:
                    gen_kwargs["temperature"] = strategy.temperature
                    gen_kwargs["top_p"] = strategy.top_p

            if strategy.no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = strategy.no_repeat_ngram_size

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            decoded = str(tokenizer.decode(output_ids[0], skip_special_tokens=True))
            svg = extract_svg(decoded)

            if svg and not is_valid_svg(svg):
                svg = repair_svg(svg)

            valid = is_valid_svg(svg)
            if valid:
                valid_count += 1
            svg_lengths.append(len(svg) if svg else 0)
            sample_svgs.append(svg[:200] if svg else "(empty)")

        elapsed = time.time() - t0
        avg_len = sum(svg_lengths) / len(svg_lengths) if svg_lengths else 0

        result = {
            "strategy": strategy.name,
            "valid": valid_count,
            "total": len(prompts),
            "validity_rate": valid_count / len(prompts),
            "avg_svg_length": avg_len,
            "time_s": elapsed,
            "samples": sample_svgs[:3],
        }
        results.append(result)

        print(
            f"  valid={valid_count}/{len(prompts)} ({valid_count / len(prompts):.0%}), "
            f"avg_len={avg_len:.0f}, time={elapsed:.1f}s",
            flush=True,
        )

    # Print comparison table
    print(f"\n{'=' * 70}")
    print(f"{'Strategy':<25} {'Valid':>8} {'Rate':>8} {'AvgLen':>8} {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['strategy']:<25} "
            f"{r['valid']:>4}/{r['total']:<3} "
            f"{r['validity_rate']:>7.0%} "
            f"{r['avg_svg_length']:>8.0f} "
            f"{r['time_s']:>7.1f}s",
        )
    print(f"{'=' * 70}")

    return results


@app.local_entrypoint()
def main(
    adapter_path: str | None = None,
    num_samples: int = 10,
) -> None:
    """Run decoding strategy sweep.

    Usage: `modal run src/svg_gen/decode_sweep.py --num-samples 10`
    """
    if adapter_path is None:
        import modal as _modal

        vol = _modal.Volume.from_name("svg-gen-vol")
        checkpoints = [
            e.path.split("/")[-1]
            for e in vol.listdir("checkpoints/")
            if e.path.split("/")[-1].startswith("checkpoint-")
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            adapter_path = f"/vol/checkpoints/{latest}"
            print(f"Using latest checkpoint: {adapter_path}")
        else:
            adapter_path = "/vol/checkpoints/final-adapter"

    print(f"Running decode sweep on {adapter_path} with {num_samples} samples")
    results = run_decode_sweep.remote(adapter_path=adapter_path, num_samples=num_samples)

    print(f"\n{'=' * 70}")
    print(f"{'Strategy':<25} {'Valid':>8} {'Rate':>8} {'AvgLen':>8} {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['strategy']:<25} "
            f"{r['valid']:>4}/{r['total']:<3} "
            f"{r['validity_rate']:>7.0%} "
            f"{r['avg_svg_length']:>8.0f} "
            f"{r['time_s']:>7.1f}s",
        )
    print(f"{'=' * 70}")
