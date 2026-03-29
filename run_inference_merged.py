"""Inference for the merged+r32 model."""

import csv
import os
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from svg_gen.data import extract_svg, fallback_svg, format_chat_prompt, is_valid_svg, normalize_viewbox, repair_svg

MERGED_MODEL = "models/merged-1.5b-r16"
ADAPTER_PATH = "outputs/final-adapter"
TEST_CSV = "data/test.csv"
OUTPUT = "submissions/merged-r32-expanded.csv"
MAX_NEW_TOKENS = 1024

os.makedirs("submissions", exist_ok=True)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Loading {MERGED_MODEL} + {ADAPTER_PATH}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
print("Model loaded.")

test_df = pd.read_csv(TEST_CSV)
total = len(test_df)
print(f"Generating SVGs for {total} test prompts...")

t0 = time.time()
invalid_count = 0

with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "svg"])

    for i, (_, row) in enumerate(test_df.iterrows()):
        prompt = str(row.get("prompt", row.get("text", "")))
        sample_id = str(row.get("id", f"sample_{i}"))

        chat_text = format_chat_prompt(prompt)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
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
                flush=True,
            )

elapsed = time.time() - t0
print(f"\nDone! {total} samples in {elapsed / 60:.1f}min")
print(f"Invalid/fallback: {invalid_count}")
print(f"Saved to {OUTPUT}")
