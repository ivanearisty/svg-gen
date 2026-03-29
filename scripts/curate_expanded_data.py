"""Build expanded training dataset from competition data + external sources.

Applies rel+int preprocessing, filters by quality, and merges into a single CSV.

Usage:
    uv run python curate_expanded_data.py
    uv run python curate_expanded_data.py --max-external 30000
"""

import argparse
import csv
import io
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from svg_gen.data import clean_svg

# ─── Preprocessing ───


def round_to_int(svg: str) -> str:
    """Round all decimal numbers to integers."""
    def _round(m: re.Match) -> str:
        return str(round(float(m.group(0))))
    return re.sub(r"\d+\.\d+", _round, svg)


def to_relative_int_simple(svg: str) -> str:
    """Convert path coordinates to relative integers using regex (fast, no svgpathtools).

    Handles the common case: single-path SVGs with M/L/C/Q/Z commands.
    Falls back to integer-only if path parsing fails.
    """
    # First, round all floats to integers
    svg = round_to_int(svg)

    # Try to convert path d= attributes to relative
    def _convert_d(match: re.Match) -> str:
        d = match.group(1)
        try:
            return f'd="{_make_relative(d)}"'
        except Exception:
            return match.group(0)

    return re.sub(r'd="([^"]*)"', _convert_d, svg)


def _make_relative(d: str) -> str:
    """Convert path command string to relative coordinates."""
    # Tokenize: split into commands and numbers
    tokens = re.findall(r'[MLCQSAHVZmlcqsahvz]|[-+]?\d+(?:\.\d+)?', d)
    if not tokens:
        return d

    result = []
    cur_x, cur_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0
    i = 0

    while i < len(tokens):
        cmd = tokens[i]

        if cmd == 'M':
            i += 1
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            result.append(f"M {int(x)} {int(y)}")
            cur_x, cur_y = x, y
            start_x, start_y = x, y
            # Implicit lineto after M
            while i < len(tokens) and tokens[i] not in 'MLCQSAHVZmlcqsahvz':
                x, y = float(tokens[i]), float(tokens[i + 1])
                dx, dy = x - cur_x, y - cur_y
                result.append(f"l {int(round(dx))} {int(round(dy))}")
                cur_x, cur_y = x, y
                i += 2

        elif cmd == 'L':
            i += 1
            while i < len(tokens) and tokens[i] not in 'MLCQSAHVZmlcqsahvz':
                x, y = float(tokens[i]), float(tokens[i + 1])
                dx, dy = x - cur_x, y - cur_y
                result.append(f"l {int(round(dx))} {int(round(dy))}")
                cur_x, cur_y = x, y
                i += 2

        elif cmd == 'C':
            i += 1
            while i + 5 < len(tokens) and tokens[i] not in 'MLCQSAHVZmlcqsahvz':
                x1, y1 = float(tokens[i]) - cur_x, float(tokens[i + 1]) - cur_y
                x2, y2 = float(tokens[i + 2]) - cur_x, float(tokens[i + 3]) - cur_y
                x, y = float(tokens[i + 4]) - cur_x, float(tokens[i + 5]) - cur_y
                result.append(
                    f"c {int(round(x1))} {int(round(y1))} "
                    f"{int(round(x2))} {int(round(y2))} "
                    f"{int(round(x))} {int(round(y))}"
                )
                cur_x += x
                cur_y += y
                i += 6

        elif cmd == 'Q':
            i += 1
            while i + 3 < len(tokens) and tokens[i] not in 'MLCQSAHVZmlcqsahvz':
                x1, y1 = float(tokens[i]) - cur_x, float(tokens[i + 1]) - cur_y
                x, y = float(tokens[i + 2]) - cur_x, float(tokens[i + 3]) - cur_y
                result.append(
                    f"q {int(round(x1))} {int(round(y1))} "
                    f"{int(round(x))} {int(round(y))}"
                )
                cur_x += x
                cur_y += y
                i += 4

        elif cmd == 'H':
            i += 1
            x = float(tokens[i])
            dx = x - cur_x
            result.append(f"h {int(round(dx))}")
            cur_x = x
            i += 1

        elif cmd == 'V':
            i += 1
            y = float(tokens[i])
            dy = y - cur_y
            result.append(f"v {int(round(dy))}")
            cur_y = y
            i += 1

        elif cmd in ('Z', 'z'):
            result.append("z")
            cur_x, cur_y = start_x, start_y
            i += 1

        elif cmd in ('m', 'l', 'c', 'q', 's', 'h', 'v', 'a'):
            # Already relative — pass through
            result.append(cmd)
            i += 1
            while i < len(tokens) and tokens[i] not in 'MLCQSAHVZmlcqsahvz':
                result.append(tokens[i])
                i += 1

        elif cmd in ('S', 'A'):
            # Complex commands — keep absolute with integers
            result.append(cmd)
            i += 1
            while i < len(tokens) and tokens[i] not in 'MLCQSAHVZmlcqsahvz':
                result.append(str(int(round(float(tokens[i])))))
                i += 1

        else:
            i += 1

    return " ".join(result)


def preprocess_svg(svg: str) -> str:
    """Full preprocessing pipeline: clean + rel+int."""
    svg = clean_svg(svg)
    svg = to_relative_int_simple(svg)
    return svg


def is_valid_svg(svg: str) -> bool:
    """Quick validity check."""
    if not svg or len(svg) < 30:
        return False
    try:
        root = ET.fromstring(svg)
        return root.tag.endswith("svg")
    except ET.ParseError:
        return False


# ─── Main ───


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-external", type=int, default=30000,
                        help="Max samples from SVGX-SFT dataset")
    parser.add_argument("--max-mmsvg", type=int, default=30000,
                        help="Max samples from MMSVG-Icon dataset")
    parser.add_argument("--max-tokens", type=int, default=1900,
                        help="Max Qwen tokens for SVG to fit in 2048 context")
    parser.add_argument("--output", default="data/train_expanded.csv")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

    def token_count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))

    # ─── 1. Competition data ───
    print("=== Competition data (train.csv) ===")
    comp_df = pd.read_csv("data/train.csv")
    print(f"  Loaded: {len(comp_df)}")

    comp_df["svg_clean"] = comp_df["svg"].apply(preprocess_svg)
    comp_df["valid"] = comp_df["svg_clean"].apply(is_valid_svg)
    comp_df = comp_df[comp_df["valid"]].copy()
    print(f"  After validity filter: {len(comp_df)}")

    comp_df["tok_count"] = comp_df["svg_clean"].apply(token_count)
    comp_df = comp_df[comp_df["tok_count"] <= args.max_tokens]
    print(f"  After token filter (<={args.max_tokens}): {len(comp_df)}")

    comp_rows = []
    for _, row in comp_df.iterrows():
        comp_rows.append({
            "id": row["id"],
            "prompt": row["prompt"],
            "svg": row["svg_clean"],
            "source": "competition",
        })
    print(f"  Final competition samples: {len(comp_rows)}")

    # ─── 2. External data (SVGX-SFT) ───
    print(f"\n=== External data (SVGX-SFT-1M) ===")
    ext_path = "data/external/svgx_sft_51k_int.csv"
    if not os.path.exists(ext_path):
        print(f"  {ext_path} not found — skipping external data")
        ext_rows = []
    else:
        ext_df = pd.read_csv(ext_path)
        print(f"  Loaded: {len(ext_df)}")

        # Deduplicate by SVG (keep first prompt variant)
        ext_df = ext_df.drop_duplicates(subset=["svg"], keep="first")
        print(f"  After SVG dedup: {len(ext_df)}")

        # Filter out trivially short prompts
        ext_df = ext_df[ext_df["prompt"].str.len() >= 10]
        print(f"  After prompt length filter: {len(ext_df)}")

        # Preprocess
        ext_df["svg_clean"] = ext_df["svg"].apply(preprocess_svg)
        ext_df["valid"] = ext_df["svg_clean"].apply(is_valid_svg)
        ext_df = ext_df[ext_df["valid"]].copy()
        print(f"  After validity filter: {len(ext_df)}")

        ext_df["tok_count"] = ext_df["svg_clean"].apply(token_count)
        ext_df = ext_df[ext_df["tok_count"] <= args.max_tokens]
        print(f"  After token filter (<={args.max_tokens}): {len(ext_df)}")

        # Sample if too many
        if len(ext_df) > args.max_external:
            ext_df = ext_df.sample(n=args.max_external, random_state=42)
            print(f"  Sampled down to: {len(ext_df)}")

        ext_rows = []
        for _, row in ext_df.iterrows():
            ext_rows.append({
                "id": row["id"],
                "prompt": row["prompt"],
                "svg": row["svg_clean"],
                "source": "svgx_sft",
            })
        print(f"  Final external samples: {len(ext_rows)}")

    # ─── 3. MMSVG-Icon data ───
    mmsvg_path = "data/external/mmsvg_icon.csv"
    if os.path.exists(mmsvg_path):
        print(f"\n=== External data (MMSVG-Icon) ===")
        mm_df = pd.read_csv(mmsvg_path)
        print(f"  Loaded: {len(mm_df)}")

        mm_df = mm_df.drop_duplicates(subset=["svg"], keep="first")
        print(f"  After SVG dedup: {len(mm_df)}")

        mm_df = mm_df[mm_df["prompt"].str.len() >= 10]
        print(f"  After prompt length filter: {len(mm_df)}")

        mm_df["svg_clean"] = mm_df["svg"].apply(preprocess_svg)
        mm_df["valid"] = mm_df["svg_clean"].apply(is_valid_svg)
        mm_df = mm_df[mm_df["valid"]].copy()
        print(f"  After validity filter: {len(mm_df)}")

        mm_df["tok_count"] = mm_df["svg_clean"].apply(token_count)
        mm_df = mm_df[mm_df["tok_count"] <= args.max_tokens]
        print(f"  After token filter (<={args.max_tokens}): {len(mm_df)}")

        # MMSVG uses same viewBox as competition (200x200) — high value, but cap to avoid drowning competition data
        if len(mm_df) > args.max_mmsvg:
            mm_df = mm_df.sample(n=args.max_mmsvg, random_state=42)
            print(f"  Sampled down to: {len(mm_df)}")

        mm_rows = []
        for _, row in mm_df.iterrows():
            mm_rows.append({
                "id": row["id"],
                "prompt": row["prompt"],
                "svg": row["svg_clean"],
                "source": "mmsvg_icon",
            })
        print(f"  Final MMSVG samples: {len(mm_rows)}")
    else:
        print(f"\n  {mmsvg_path} not found — skipping MMSVG-Icon")
        mm_rows = []

    # ─── 4. Merge ───
    all_rows = comp_rows + ext_rows + mm_rows
    np.random.seed(42)
    np.random.shuffle(all_rows)

    print(f"\n=== Merged dataset ===")
    print(f"  Competition:  {len(comp_rows)}")
    print(f"  SVGX-SFT:     {len(ext_rows)}")
    print(f"  MMSVG-Icon:   {len(mm_rows)}")
    print(f"  Total:        {len(all_rows)}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "svg", "source"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  Saved to {args.output}")

    # Stats
    svg_lens = [len(r["svg"]) for r in all_rows]
    tok_counts = [token_count(r["svg"]) for r in all_rows]
    print(f"\n  SVG chars: mean={np.mean(svg_lens):.0f}, median={np.median(svg_lens):.0f}, max={max(svg_lens)}")
    print(f"  Qwen tokens: mean={np.mean(tok_counts):.0f}, median={np.median(tok_counts):.0f}, max={max(tok_counts)}")

    source_counts = {}
    for r in all_rows:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
    for src, cnt in sorted(source_counts.items()):
        print(f"  Source {src}: {cnt}")


if __name__ == "__main__":
    main()
