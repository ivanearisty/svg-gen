"""Debug: generate a few SVGs and print raw output to diagnose validity issues."""

from __future__ import annotations

import os

from svg_gen.modal_app import VOLUME_MOUNT, app, gpu_image, volume


@app.function(image=gpu_image, gpu="A100-40GB", timeout=10 * 60, volumes={VOLUME_MOUNT: volume})
def debug_generate(adapter_path: str) -> str:
    """Generate 2 SVGs and print full raw output for debugging."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from svg_gen.config import SYSTEM_PROMPT
    from svg_gen.data import extract_svg, is_valid_svg, repair_svg

    import pandas as pd

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

    test_df = pd.read_csv("/vol/data/test.csv").head(2)
    output_lines: list[str] = []

    for _, row in test_df.iterrows():
        prompt = str(row.get("prompt", row.get("text", "")))
        output_lines.append(f"\n{'=' * 60}")
        output_lines.append(f"PROMPT: {prompt[:150]}")
        output_lines.append(f"{'=' * 60}")

        chat_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        raw = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        decoded = str(tokenizer.decode(output_ids[0], skip_special_tokens=True))

        output_lines.append(f"\nRAW (first 800 chars):\n{raw[:800]}")
        output_lines.append(f"\nDECODED (first 800 chars):\n{decoded[:800]}")

        svg = extract_svg(decoded)
        output_lines.append(f"\nEXTRACTED SVG: {'(EMPTY)' if not svg else svg[:500]}")
        output_lines.append(f"SVG length: {len(svg)}")
        output_lines.append(f"Valid: {is_valid_svg(svg)}")

        if svg and not is_valid_svg(svg):
            repaired = repair_svg(svg)
            output_lines.append(f"Repaired valid: {is_valid_svg(repaired)}")
            output_lines.append(f"Repaired (first 500): {repaired[:500]}")

    result = "\n".join(output_lines)
    print(result, flush=True)
    return result


@app.local_entrypoint()
def main(adapter_path: str = "/vol/checkpoints/checkpoint-8000") -> None:
    """Run debug generation."""
    print(f"Debugging generation with adapter: {adapter_path}")
    result = debug_generate.remote(adapter_path=adapter_path)
    print(result)
