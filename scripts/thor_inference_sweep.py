"""Temperature sweep inference for Thor (runs in bf16, no quantization).

Usage (inside Docker):
    python3 scripts/thor_inference_sweep.py
"""

import csv
import os
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from svg_gen.data import extract_svg, fallback_svg, format_chat_prompt, is_valid_svg, normalize_viewbox, repair_svg

MODEL = "models/merged-componly-r32"
TEST_CSV = "data/test.csv"
MAX_NEW_TOKENS = 1536
TEMPERATURES = [0.0]  # Just greedy with 1536 tokens


def generate_at_temp(model, tokenizer, test_df, temperature, output_path):
    total = len(test_df)
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
                if temperature == 0.0:
                    output_ids = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False, repetition_penalty=1.1,
                    )
                else:
                    output_ids = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True, temperature=temperature,
                        top_p=0.95, repetition_penalty=1.1,
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
                print(f"  temp={temperature} [{i + 1}/{total}] {rate:.2f}/s | eta={eta / 60:.0f}min | invalid={invalid_count}", flush=True)

    elapsed = time.time() - t0
    print(f"  temp={temperature} DONE: {total} in {elapsed / 60:.1f}min, {invalid_count} invalid", flush=True)


def main():
    os.makedirs("results/submissions", exist_ok=True)

    # Disable FlashAttention on Thor/Blackwell
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading {MODEL} in bf16 (pre-merged, no adapter)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = model.to("cuda:0")
    model.eval()
    print("Model loaded.")

    test_df = pd.read_csv(TEST_CSV)
    print(f"Test prompts: {len(test_df)}")
    print(f"Temperatures: {TEMPERATURES}")

    for temp in TEMPERATURES:
        output_path = f"results/submissions/thor_temp_{temp:.2f}.csv"
        print(f"\n{'='*60}")
        print(f"Temperature={temp} → {output_path}")
        print(f"{'='*60}")
        generate_at_temp(model, tokenizer, test_df, temp, output_path)

    print("\nAll done! Files:")
    for temp in TEMPERATURES:
        print(f"  results/submissions/thor_temp_{temp:.2f}.csv")


if __name__ == "__main__":
    main()
