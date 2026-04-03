"""System prompt sensitivity analysis.

Tests different system prompts with the same fine-tuned model.
Reuses the ablation infrastructure but varies only the system prompt text.

2hr budget = 2 variants (~75 min each).
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from svg_gen.data import extract_svg, fallback_svg, is_valid_svg, normalize_viewbox, repair_svg
from svg_gen.score import score_sample

VAL_CSV = "data/val_ablation.csv"
RESULTS_DIR = "results/ablations"
MERGED_BASE = "models/merged-1.5b-r16"
ADAPTER_PATH = "outputs-componly/final-adapter"

# System prompt variants to test
PROMPTS = {
    "minimal": "Output valid SVG code only.",
    "specific": (
        "You generate SVG icons using path elements in a 200x200 viewBox. "
        "Return only the <svg> element with xmlns, viewBox, width, and height attributes. "
        "Always close with </svg>. Keep paths compact."
    ),
}

# Original for reference (not re-run, use baseline score)
# "original": "You generate compact, valid SVG markup from user requests. "
#              "Return only SVG code with a single root <svg> element. "
#              "Keep the SVG under 16000 characters."


def format_prompt(prompt: str, system_prompt: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def run_variant(name: str, system_prompt: str, model, tokenizer, val_df: pd.DataFrame) -> dict:
    """Run a single system prompt variant."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total = len(val_df)

    output_csv = os.path.join(RESULTS_DIR, f"sysprompt_{name}_predictions.csv")
    scores_path = os.path.join(RESULTS_DIR, f"sysprompt_{name}_scores.json")

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

            chat_text = format_prompt(prompt, system_prompt)
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
        "ablation_name": f"sysprompt_{name}",
        "system_prompt": system_prompt,
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
    }

    with open(scores_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"SYSTEM PROMPT VARIANT: {name}")
    print(f"Prompt: \"{system_prompt[:80]}...\"")
    print(f"FINAL SCORE: {result['final_score']:.2f} / 100")
    print(f"Invalid: {invalid_count}, Time: {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")

    done_file = os.path.join(RESULTS_DIR, f"sysprompt_{name}.done")
    with open(done_file, "w") as f:
        f.write(f"{result['final_score']:.2f}\n")

    return result


def main():
    print(f"=== SYSTEM PROMPT SENSITIVITY ANALYSIS: {time.strftime('%H:%M:%S')} ===")
    print(f"Budget: 2 hours, {len(PROMPTS)} variants")

    val_df = pd.read_csv(VAL_CSV)
    print(f"Val set: {len(val_df)} samples")

    # Load model once, reuse for all variants
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading {MERGED_BASE} + {ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_BASE)
    base_model = AutoModelForCausalLM.from_pretrained(
        MERGED_BASE,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("Model loaded.")

    results = []
    for name, prompt_text in PROMPTS.items():
        done_file = os.path.join(RESULTS_DIR, f"sysprompt_{name}.done")
        if os.path.exists(done_file):
            print(f"[SKIP] sysprompt_{name} already done")
            continue

        print(f"\n--- Variant: {name} ---")
        print(f"Prompt: \"{prompt_text}\"")
        r = run_variant(name, prompt_text, model, tokenizer, val_df)
        results.append(r)

    # Summary
    print(f"\n{'=' * 60}")
    print("SYSTEM PROMPT SENSITIVITY SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Variant':<20} | {'Score':>8} | {'Prompt'}")
    print(f"{'-'*20}-+-{'-'*8}-+-{'-'*40}")
    print(f"{'original (baseline)':<20} | {'53.79':>8} | You generate compact, valid SVG markup...")
    print(f"{'none':<20} | {'18.80':>8} | (no system prompt)")
    for r in results:
        name = r["ablation_name"].replace("sysprompt_", "")
        print(f"{name:<20} | {r['final_score']:>8.2f} | {r['system_prompt'][:40]}...")

    # Signal all done
    with open(os.path.join(RESULTS_DIR, "SYSPROMPT_DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    print(f"\n=== COMPLETE: {time.strftime('%H:%M:%S')} ===")


if __name__ == "__main__":
    main()
