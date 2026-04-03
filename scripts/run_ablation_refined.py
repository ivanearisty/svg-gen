"""Ablation for the refined-7000 full fine-tune model.

This model is a full model (not adapter), loaded differently from the
PEFT-based ablation script.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_dir, "src"))
os.chdir(_project_dir)

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from svg_gen.config import SYSTEM_PROMPT
from svg_gen.data import extract_svg, fallback_svg, is_valid_svg, normalize_viewbox, repair_svg
from svg_gen.score import score_sample

VAL_CSV = "data/val_ablation.csv"
RESULTS_DIR = "results/ablations"
MODEL_PATH = "models/refined-7000"


def format_prompt(prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="refined_ft")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    val_df = pd.read_csv(VAL_CSV)
    total = len(val_df)
    print(f"Val set: {total} samples")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading refined model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("Model loaded.")

    output_csv = os.path.join(RESULTS_DIR, f"{args.name}_predictions.csv")
    scores_path = os.path.join(RESULTS_DIR, f"{args.name}_scores.json")

    all_scores: list[dict[str, float]] = []
    invalid_count = 0
    fallback_count = 0
    t0 = time.time()

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "svg_pred", "svg_ref"])

        for i, (_, row) in enumerate(val_df.iterrows()):
            prompt = str(row["prompt"])
            sample_id = str(row["id"])
            ref_svg = str(row["svg"])

            chat_text = format_prompt(prompt)
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

            if svg and not is_valid_svg(svg):
                svg = repair_svg(svg)

            valid = is_valid_svg(svg)
            if not valid:
                invalid_count += 1
                fallback_count += 1
                svg = fallback_svg()
            else:
                svg = normalize_viewbox(svg)

            sample_score = score_sample(svg, ref_svg)
            all_scores.append(sample_score)
            writer.writerow([sample_id, prompt, svg, ref_svg])

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                avg_comp = sum(s["composite"] for s in all_scores) / len(all_scores)
                print(
                    f"  [{i + 1}/{total}] {rate:.2f} s/s | "
                    f"eta={eta / 60:.1f}min | "
                    f"avg_score={100 * avg_comp:.2f} | "
                    f"invalid={invalid_count}",
                    flush=True,
                )

    elapsed = time.time() - t0
    n = len(all_scores)
    result = {
        "ablation_name": args.name,
        "n_samples": n,
        "elapsed_min": round(elapsed / 60, 1),
        "invalid_count": invalid_count,
        "fallback_count": fallback_count,
        "validity_rate": sum(s["valid"] for s in all_scores) / n,
        "avg_visual": sum(s["visual"] for s in all_scores) / n,
        "avg_structural": sum(s["structural"] for s in all_scores) / n,
        "avg_compactness": sum(s["compactness"] for s in all_scores) / n,
        "avg_composite": sum(s["composite"] for s in all_scores) / n,
        "final_score": 100 * sum(s["composite"] for s in all_scores) / n,
        "config": {
            "model": MODEL_PATH,
            "adapter": None,
            "system_prompt": True,
            "repair": True,
            "rep_penalty": 1.1,
            "max_tokens": 1024,
        },
    }

    with open(scores_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ABLATION: {args.name}")
    print(f"FINAL SCORE: {result['final_score']:.2f} / 100")
    print(f"{'=' * 60}")

    done_file = os.path.join(RESULTS_DIR, f"{args.name}.done")
    with open(done_file, "w") as f:
        f.write(f"{result['final_score']:.2f}\n")


if __name__ == "__main__":
    main()
