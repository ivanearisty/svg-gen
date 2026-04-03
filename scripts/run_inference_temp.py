"""Run inference at a specific temperature.

Usage:
    uv run python scripts/run_inference_temp.py --temp 0.05
"""

import argparse
import csv
import os
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from svg_gen.data import extract_svg, fallback_svg, format_chat_prompt, is_valid_svg, normalize_viewbox, repair_svg

BASE_MODEL = "models/merged-1.5b-r16"
ADAPTER = "outputs-componly/final-adapter"
TEST_CSV = "data/test.csv"
# MAX_NEW_TOKENS set via --max-new-tokens arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--rep-penalty", type=float, default=1.1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    temp = args.temp
    output_path = args.output or f"results/submissions/greedy_rep{args.rep_penalty:.2f}_tok{args.max_new_tokens}.csv"
    os.makedirs("results/submissions", exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Temperature: {temp}, Rep penalty: {args.rep_penalty}, Max tokens: {args.max_new_tokens}")
    print(f"Output: {output_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER)
    model.eval()
    print("Model loaded.")

    test_df = pd.read_csv(TEST_CSV)
    total = len(test_df)
    print(f"Generating {total} SVGs...")

    t0 = time.time()
    invalid_count = 0

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "svg"])

        for i, (_, row) in enumerate(test_df.iterrows()):
            prompt = str(row.get("prompt", row.get("text", "")))
            sample_id = str(row.get("id", f"sample_{i}"))
            chat_text = format_chat_prompt(prompt)
            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                if temp == 0.0:
                    output_ids = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                        repetition_penalty=args.rep_penalty,
                    )
                else:
                    output_ids = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=temp, top_p=0.95, repetition_penalty=args.rep_penalty,
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
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
