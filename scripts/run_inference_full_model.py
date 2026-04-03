"""Inference with a full (non-adapter) model. Greedy, rep=1.1, 1024 tokens.

Usage:
    uv run python scripts/run_inference_full_model.py --model models/refined-7000
"""

import argparse
import csv
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from svg_gen.data import extract_svg, fallback_svg, format_chat_prompt, is_valid_svg, normalize_viewbox, repair_svg

TEST_CSV = "data/test.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--rep-penalty", type=float, default=1.1)
    args = parser.parse_args()

    output = args.output or f"results/submissions/{os.path.basename(args.model)}_greedy.csv"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model}")
    print(f"Output: {output}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("Model loaded.")

    test_df = pd.read_csv(TEST_CSV)
    total = len(test_df)
    t0 = time.time()
    invalid_count = 0

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "svg"])

        for i, (_, row) in enumerate(test_df.iterrows()):
            prompt = str(row.get("prompt", row.get("text", "")))
            sample_id = str(row.get("id", f"sample_{i}"))
            chat_text = format_chat_prompt(prompt)
            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, repetition_penalty=args.rep_penalty,
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

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i + 1}/{total}] {rate:.2f}/s | eta={eta / 60:.0f}min | invalid={invalid_count}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! {total} samples in {elapsed / 60:.1f}min, {invalid_count} invalid")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
