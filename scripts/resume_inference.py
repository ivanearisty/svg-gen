"""Resume inference from where the previous run crashed."""

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
EXISTING_CSV = "results/submissions/merged-r32-expanded.csv"
OUTPUT = "results/submissions/merged-r32-expanded-complete.csv"
MAX_NEW_TOKENS = 1024

# Step 1: Recover good rows from crashed CSV
print("Recovering existing results...")
good_rows = []
with open(EXISTING_CSV, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) == 2:
            good_rows.append(row)

done_ids = {row[0] for row in good_rows}
print(f"Recovered {len(good_rows)} valid samples")

# Step 2: Find remaining test prompts
test_df = pd.read_csv(TEST_CSV)
remaining = test_df[~test_df["id"].astype(str).isin(done_ids)]
print(f"Remaining: {len(remaining)} samples to generate")

if len(remaining) == 0:
    print("All samples already generated!")
    # Just write the clean CSV
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "svg"])
        writer.writerows(good_rows)
    print(f"Saved to {OUTPUT}")
    exit()

# Step 3: Load model
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
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

# Step 4: Generate remaining
total = len(remaining)
print(f"Generating {total} remaining SVGs...")

t0 = time.time()
invalid_count = 0
new_rows = []

for i, (_, row) in enumerate(remaining.iterrows()):
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

    new_rows.append([sample_id, svg])

    if (i + 1) % 10 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (total - i - 1) / rate if rate > 0 else 0
        print(
            f"  [{i + 1}/{total}] {rate:.2f} samples/s | "
            f"eta={eta / 60:.0f}min | invalid={invalid_count}",
            flush=True,
        )

# Step 5: Merge and save
print(f"\nMerging {len(good_rows)} recovered + {len(new_rows)} new...")
all_rows = good_rows + new_rows

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "svg"])
    writer.writerows(all_rows)

elapsed = time.time() - t0
print(f"Done! {len(new_rows)} new samples in {elapsed / 60:.1f}min")
print(f"Invalid/fallback: {invalid_count}")
print(f"Total: {len(all_rows)} samples saved to {OUTPUT}")
