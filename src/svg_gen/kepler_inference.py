"""Standalone inference script for kepler (RTX 2000 Ada 16GB).

No Modal dependency. Generates submission CSV locally.

Usage:
    python -m svg_gen.kepler_inference
    python -m svg_gen.kepler_inference --adapter-path outputs/final-adapter --max-new-tokens 1024
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from svg_gen.config import SYSTEM_PROMPT
from svg_gen.data import extract_svg, fallback_svg, is_valid_svg, normalize_viewbox, repair_svg


def main() -> None:  # noqa: PLR0915
    """Run inference on local GPU."""
    parser = argparse.ArgumentParser(description="Generate SVGs on kepler")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="outputs/final-adapter")
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--output", default="submissions/submission.csv")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Model: {args.model}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Max new tokens: {args.max_new_tokens}")

    # --- Load model ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    print("Model loaded.")

    # --- Load test data ---
    test_df = pd.read_csv(args.test_csv)
    total = len(test_df)
    print(f"Generating SVGs for {total} test prompts...")

    # --- Generate ---
    t0 = time.time()
    invalid_count = 0

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "svg"])

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
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                )

            decoded = str(tokenizer.decode(output_ids[0], skip_special_tokens=True))
            svg = extract_svg(decoded)

            if svg and not is_valid_svg(svg):
                svg = repair_svg(svg)

            valid = is_valid_svg(svg)
            if not valid:
                invalid_count += 1
                svg = fallback_svg()
            else:
                svg = normalize_viewbox(svg)

            writer.writerow([sample_id, svg])

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i + 1}/{total}] {rate:.2f} samples/s | "
                    f"eta={eta / 60:.0f}min | invalid={invalid_count}",
                )

    elapsed = time.time() - t0
    print(f"\nDone! {total} samples in {elapsed / 60:.1f}min")
    print(f"Invalid/fallback: {invalid_count}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
