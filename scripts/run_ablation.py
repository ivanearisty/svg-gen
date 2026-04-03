"""Ablation study runner — inference + local scoring on val set.

Runs a single ablation variant: loads model, generates SVGs for val prompts,
scores against ground truth, saves results.

Usage:
    # Full system (baseline)
    python3 scripts/run_ablation.py --name baseline

    # Base model (no fine-tuning)
    python3 scripts/run_ablation.py --name base_model --no-adapter

    # No system prompt
    python3 scripts/run_ablation.py --name no_sys_prompt --no-system-prompt

    # No repetition penalty
    python3 scripts/run_ablation.py --name rep_1.0 --rep-penalty 1.0

    # Short generation
    python3 scripts/run_ablation.py --name max_tok_512 --max-tokens 512

    # No repair pipeline
    python3 scripts/run_ablation.py --name no_repair --no-repair
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

# Add src to path before any local imports
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
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MERGED_BASE = "models/merged-1.5b-r16"
ADAPTER_PATH = "outputs-componly/final-adapter"


def format_prompt(prompt: str, use_system_prompt: bool = True) -> str:
    """Format prompt for inference, optionally without system prompt."""
    if use_system_prompt:
        return (
            "<|im_start|>system\n"
            f"{SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        return (
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )


def load_model(use_adapter: bool = True):
    """Load model — with adapter (default) or base model only."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if use_adapter:
        print(f"Loading {MERGED_BASE} + {ADAPTER_PATH}...")
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(MERGED_BASE)
        base_model = AutoModelForCausalLM.from_pretrained(
            MERGED_BASE,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print(f"Loading base model {BASE_MODEL} (no fine-tuning)...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


def run_ablation(args: argparse.Namespace) -> None:
    """Run a single ablation: inference + scoring."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load val data
    val_df = pd.read_csv(VAL_CSV)
    total = len(val_df)
    print(f"Val set: {total} samples")

    # Load model
    model, tokenizer = load_model(use_adapter=args.use_adapter)
    print("Model loaded.")

    # Generate + score
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

            # Generate
            chat_text = format_prompt(prompt, use_system_prompt=args.use_system_prompt)
            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    repetition_penalty=args.rep_penalty,
                )

            decoded = str(tokenizer.decode(output_ids[0], skip_special_tokens=True))
            svg = extract_svg(decoded)

            # Post-processing pipeline
            if args.use_repair:
                if svg and not is_valid_svg(svg):
                    svg = repair_svg(svg)

                valid = is_valid_svg(svg)
                if not valid:
                    invalid_count += 1
                    fallback_count += 1
                    svg = fallback_svg()
                else:
                    svg = normalize_viewbox(svg)
            else:
                # No repair — just extract and normalize, fallback if empty
                if not svg:
                    invalid_count += 1
                    fallback_count += 1
                    svg = fallback_svg()
                else:
                    svg = normalize_viewbox(svg)

            # Score against ground truth
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

    # Aggregate scores
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
            "use_adapter": args.use_adapter,
            "use_system_prompt": args.use_system_prompt,
            "use_repair": args.use_repair,
            "rep_penalty": args.rep_penalty,
            "max_tokens": args.max_tokens,
        },
    }

    with open(scores_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"ABLATION: {args.name}")
    print(f"{'=' * 60}")
    print(f"Samples:      {n}")
    print(f"Time:         {elapsed / 60:.1f} min")
    print(f"Validity:     {result['validity_rate']:.0%}")
    print(f"Invalid:      {invalid_count} ({fallback_count} fallbacks)")
    print(f"Visual:       {result['avg_visual']:.4f}")
    print(f"Structural:   {result['avg_structural']:.4f}")
    print(f"Compactness:  {result['avg_compactness']:.4f}")
    print(f"Composite:    {result['avg_composite']:.4f}")
    print(f"FINAL SCORE:  {result['final_score']:.2f} / 100")
    print(f"{'=' * 60}")
    print(f"Predictions: {output_csv}")
    print(f"Scores:      {scores_path}")

    # Signal completion
    done_file = os.path.join(RESULTS_DIR, f"{args.name}.done")
    with open(done_file, "w") as f:
        f.write(f"{result['final_score']:.2f}\n")
    print(f"Done signal:  {done_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation study variant")
    parser.add_argument("--name", required=True, help="Ablation name (used for output files)")
    parser.add_argument("--no-adapter", dest="use_adapter", action="store_false", default=True,
                        help="Use base model without fine-tuning")
    parser.add_argument("--no-system-prompt", dest="use_system_prompt", action="store_false", default=True,
                        help="Omit system prompt at inference")
    parser.add_argument("--no-repair", dest="use_repair", action="store_false", default=True,
                        help="Skip repair pipeline")
    parser.add_argument("--rep-penalty", type=float, default=1.1,
                        help="Repetition penalty (default: 1.1)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max new tokens (default: 1024)")
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
