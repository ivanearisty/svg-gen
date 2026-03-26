"""Generate sample SVGs and save individually for inspection."""

from __future__ import annotations

import json
import os

from svg_gen.modal_app import VOLUME_MOUNT, app, gpu_image, volume


@app.function(image=gpu_image, gpu="A100-40GB", timeout=30 * 60, volumes={VOLUME_MOUNT: volume})
def generate_samples(adapter_path: str, num_samples: int = 20) -> list[dict[str, str]]:
    """Generate SVGs and return them with prompts."""
    import pandas as pd
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from svg_gen.config import SYSTEM_PROMPT
    from svg_gen.data import extract_svg, fallback_svg, is_valid_svg, normalize_viewbox, repair_svg

    bnb_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    test_df = pd.read_csv("/vol/data/test.csv").head(num_samples)
    results: list[dict[str, str]] = []

    for i, (_, row) in enumerate(test_df.iterrows()):
        prompt = str(row.get("prompt", row.get("text", "")))
        sample_id = str(row.get("id", f"sample_{i}"))

        chat_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, repetition_penalty=1.1)

        decoded = str(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        svg = extract_svg(decoded)

        if svg and not is_valid_svg(svg):
            svg = repair_svg(svg)
        valid = is_valid_svg(svg)
        svg = fallback_svg() if not valid else normalize_viewbox(svg)

        results.append({"id": sample_id, "prompt": prompt, "svg": svg, "valid": str(valid)})
        print(f"  [{i + 1}/{num_samples}] {sample_id}: valid={valid}, len={len(svg)}", flush=True)

    return results


@app.local_entrypoint()
def main(adapter_path: str = "/vol/checkpoints/final-adapter", num_samples: int = 20) -> None:
    """Generate samples and save locally."""
    print(f"Generating {num_samples} samples from {adapter_path}")
    results = generate_samples.remote(adapter_path=adapter_path, num_samples=num_samples)

    out_dir = "sample_outputs"
    os.makedirs(out_dir, exist_ok=True)

    for r in results:
        sid = r["id"]
        with open(os.path.join(out_dir, f"{sid}.svg"), "w") as f:
            f.write(r["svg"])
        with open(os.path.join(out_dir, f"{sid}.prompt.txt"), "w") as f:
            f.write(r["prompt"])

    # Also save summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    valid_count = sum(1 for r in results if r["valid"] == "True")
    print(f"\nSaved {len(results)} samples to {out_dir}/")
    print(f"Valid: {valid_count}/{len(results)} ({valid_count / len(results):.0%})")
