"""Generate before/after renders for preprocessing comparison.

Creates a directory with side-by-side PNGs for visual audit:
  audit_renders/
    001_original.png    001_int.png    001_relint.png
    002_original.png    002_int.png    002_relint.png
    ...

Also prints token counts and SSIM scores.
"""

import csv
import io
import os
import re
import sys

import cairosvg
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity
from svgpathtools import parse_path, Line, CubicBezier, QuadraticBezier, Arc

from svg_gen.data import clean_svg

RENDER_SIZE = 256
NUM_SAMPLES = 100
OUTPUT_DIR = "audit_renders"
SEED = 42


# ─── Preprocessing functions ───


def round_to_int(svg: str) -> str:
    """Round all decimal numbers in SVG to integers."""
    # Match floats like 93.3 or 93.30000305175781, replace with rounded int
    def _round(m: re.Match) -> str:
        return str(round(float(m.group(0))))
    return re.sub(r"\d+\.\d+", _round, svg)


def to_relative_paths(svg: str) -> str:
    """Convert absolute path commands to relative where possible.

    Uses svgpathtools to parse paths and reconstruct with relative commands.
    Falls back to original if parsing fails.
    """
    def _convert_path_d(match: re.Match) -> str:
        d_attr = match.group(1)
        try:
            path = parse_path(d_attr)
        except Exception:
            return match.group(0)  # return unchanged on parse failure

        if len(path) == 0:
            return match.group(0)

        parts = []
        cur = complex(0, 0)

        for seg in path:
            if isinstance(seg, Line):
                if seg.start != cur:
                    # moveto
                    parts.append(f"M {seg.start.real:.1f} {seg.start.imag:.1f}")
                    cur = seg.start
                dx = seg.end.real - cur.real
                dy = seg.end.imag - cur.imag
                parts.append(f"l {dx:.1f} {dy:.1f}")
                cur = seg.end

            elif isinstance(seg, CubicBezier):
                if seg.start != cur:
                    parts.append(f"M {seg.start.real:.1f} {seg.start.imag:.1f}")
                    cur = seg.start
                c1 = seg.control1 - cur
                c2 = seg.control2 - cur
                end = seg.end - cur
                parts.append(
                    f"c {c1.real:.1f} {c1.imag:.1f} "
                    f"{c2.real:.1f} {c2.imag:.1f} "
                    f"{end.real:.1f} {end.imag:.1f}"
                )
                cur = seg.end

            elif isinstance(seg, QuadraticBezier):
                if seg.start != cur:
                    parts.append(f"M {seg.start.real:.1f} {seg.start.imag:.1f}")
                    cur = seg.start
                c = seg.control - cur
                end = seg.end - cur
                parts.append(
                    f"q {c.real:.1f} {c.imag:.1f} "
                    f"{end.real:.1f} {end.imag:.1f}"
                )
                cur = seg.end

            elif isinstance(seg, Arc):
                if seg.start != cur:
                    parts.append(f"M {seg.start.real:.1f} {seg.start.imag:.1f}")
                    cur = seg.start
                # Arcs are complex, keep absolute
                parts.append(
                    f"A {seg.radius.real:.1f} {seg.radius.imag:.1f} "
                    f"{seg.rotation:.1f} {int(seg.large_arc)} {int(seg.sweep)} "
                    f"{seg.end.real:.1f} {seg.end.imag:.1f}"
                )
                cur = seg.end

        # Check if path was closed (last point == first point of a subpath)
        if len(path) > 0 and path[-1].end == path[0].start:
            parts.append("z")

        return f'd="{" ".join(parts)}"'

    return re.sub(r'd="([^"]*)"', _convert_path_d, svg)


def to_relative_int(svg: str) -> str:
    """Convert to relative paths with integer coordinates."""
    def _convert_path_d(match: re.Match) -> str:
        d_attr = match.group(1)
        try:
            path = parse_path(d_attr)
        except Exception:
            return match.group(0)

        if len(path) == 0:
            return match.group(0)

        parts = []
        cur = complex(0, 0)

        for seg in path:
            if isinstance(seg, Line):
                if seg.start != cur:
                    parts.append(f"M {round(seg.start.real)} {round(seg.start.imag)}")
                    cur = seg.start
                dx = round(seg.end.real - cur.real)
                dy = round(seg.end.imag - cur.imag)
                parts.append(f"l {dx} {dy}")
                cur = seg.end

            elif isinstance(seg, CubicBezier):
                if seg.start != cur:
                    parts.append(f"M {round(seg.start.real)} {round(seg.start.imag)}")
                    cur = seg.start
                c1 = seg.control1 - cur
                c2 = seg.control2 - cur
                end = seg.end - cur
                parts.append(
                    f"c {round(c1.real)} {round(c1.imag)} "
                    f"{round(c2.real)} {round(c2.imag)} "
                    f"{round(end.real)} {round(end.imag)}"
                )
                cur = seg.end

            elif isinstance(seg, QuadraticBezier):
                if seg.start != cur:
                    parts.append(f"M {round(seg.start.real)} {round(seg.start.imag)}")
                    cur = seg.start
                c = seg.control - cur
                end = seg.end - cur
                parts.append(
                    f"q {round(c.real)} {round(c.imag)} "
                    f"{round(end.real)} {round(end.imag)}"
                )
                cur = seg.end

            elif isinstance(seg, Arc):
                if seg.start != cur:
                    parts.append(f"M {round(seg.start.real)} {round(seg.start.imag)}")
                    cur = seg.start
                parts.append(
                    f"A {round(seg.radius.real)} {round(seg.radius.imag)} "
                    f"{round(seg.rotation)} {int(seg.large_arc)} {int(seg.sweep)} "
                    f"{round(seg.end.real)} {round(seg.end.imag)}"
                )
                cur = seg.end

        if len(path) > 0 and path[-1].end == path[0].start:
            parts.append("z")

        return f'd="{" ".join(parts)}"'

    result = re.sub(r'd="([^"]*)"', _convert_path_d, svg)
    # Also round any remaining float attrs (cx, cy, r, x, y, width, height, etc.)
    result = re.sub(r'(\d+)\.\d+', r'\1', result)
    return result


# ─── Rendering & scoring ───


def render_svg(svg_text: str) -> np.ndarray | None:
    try:
        png = cairosvg.svg2png(bytestring=svg_text.encode(), output_width=RENDER_SIZE, output_height=RENDER_SIZE)
        img = Image.open(io.BytesIO(png)).convert("L")
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


def save_png(svg_text: str, path: str) -> bool:
    try:
        png = cairosvg.svg2png(bytestring=svg_text.encode(), output_width=RENDER_SIZE, output_height=RENDER_SIZE)
        with open(path, "wb") as f:
            f.write(png)
        return True
    except Exception:
        return False


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, data_range=255))


# ─── Main ───


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv("data/train.csv")
    # Sample diverse SVGs: mix of short, medium, long
    df["_len"] = df["svg"].str.len()
    df = df.sort_values("_len")

    # Pick evenly across length distribution
    indices = np.linspace(0, len(df) - 1, NUM_SAMPLES, dtype=int)
    samples = df.iloc[indices].reset_index(drop=True)

    # Load tokenizer for token counting
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

    results = []
    fail_count = 0

    print(f"Processing {NUM_SAMPLES} samples...")
    print(f"Output: {OUTPUT_DIR}/")
    print()

    for i, (_, row) in enumerate(samples.iterrows()):
        svg_original = clean_svg(row["svg"])
        svg_int = round_to_int(svg_original)

        try:
            svg_relint = to_relative_int(svg_original)
        except Exception as e:
            svg_relint = svg_int  # fallback
            fail_count += 1

        prefix = f"{i + 1:03d}"
        prompt = row["prompt"][:60]

        # Render all variants
        ok_orig = save_png(svg_original, f"{OUTPUT_DIR}/{prefix}_original.png")
        ok_int = save_png(svg_int, f"{OUTPUT_DIR}/{prefix}_int.png")
        ok_relint = save_png(svg_relint, f"{OUTPUT_DIR}/{prefix}_relint.png")

        if not all([ok_orig, ok_int, ok_relint]):
            fail_count += 1
            continue

        # Compute SSIM
        img_orig = render_svg(svg_original)
        img_int = render_svg(svg_int)
        img_relint = render_svg(svg_relint)

        if img_orig is None or img_int is None or img_relint is None:
            fail_count += 1
            continue

        ssim_int = compute_ssim(img_orig, img_int)
        ssim_relint = compute_ssim(img_orig, img_relint)

        # Token counts
        tok_orig = len(tok.encode(svg_original, add_special_tokens=False))
        tok_int = len(tok.encode(svg_int, add_special_tokens=False))
        tok_relint = len(tok.encode(svg_relint, add_special_tokens=False))

        results.append({
            "id": prefix,
            "prompt": prompt,
            "chars_orig": len(svg_original),
            "chars_int": len(svg_int),
            "chars_relint": len(svg_relint),
            "tokens_orig": tok_orig,
            "tokens_int": tok_int,
            "tokens_relint": tok_relint,
            "ssim_int": ssim_int,
            "ssim_relint": ssim_relint,
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{NUM_SAMPLES}] processed")

    # ─── Summary ───
    print(f"\n{'='*70}")
    print(f"PREPROCESSING AUDIT RESULTS ({len(results)} samples, {fail_count} failures)")
    print(f"{'='*70}")

    if not results:
        print("No results!")
        return

    avg = lambda key: sum(r[key] for r in results) / len(results)

    print(f"\nCharacter counts (avg):")
    print(f"  Original:  {avg('chars_orig'):.0f}")
    print(f"  Integer:   {avg('chars_int'):.0f}  ({100*(1 - avg('chars_int')/avg('chars_orig')):+.1f}%)")
    print(f"  Rel+Int:   {avg('chars_relint'):.0f}  ({100*(1 - avg('chars_relint')/avg('chars_orig')):+.1f}%)")

    print(f"\nQwen token counts (avg):")
    print(f"  Original:  {avg('tokens_orig'):.0f}")
    print(f"  Integer:   {avg('tokens_int'):.0f}  ({100*(1 - avg('tokens_int')/avg('tokens_orig')):+.1f}%)")
    print(f"  Rel+Int:   {avg('tokens_relint'):.0f}  ({100*(1 - avg('tokens_relint')/avg('tokens_orig')):+.1f}%)")

    print(f"\nSSIM vs original (avg):")
    print(f"  Integer:   {avg('ssim_int'):.6f}")
    print(f"  Rel+Int:   {avg('ssim_relint'):.6f}")

    ssim_int_min = min(r["ssim_int"] for r in results)
    ssim_relint_min = min(r["ssim_relint"] for r in results)
    print(f"\nSSIM vs original (worst case):")
    print(f"  Integer:   {ssim_int_min:.6f}")
    print(f"  Rel+Int:   {ssim_relint_min:.6f}")

    # Samples that fit in 2048 context (1900 tokens for SVG)
    fit_orig = sum(1 for r in results if r["tokens_orig"] <= 1900)
    fit_int = sum(1 for r in results if r["tokens_int"] <= 1900)
    fit_relint = sum(1 for r in results if r["tokens_relint"] <= 1900)
    print(f"\nFit in 2048 context (<=1900 SVG tokens):")
    print(f"  Original:  {fit_orig}/{len(results)} ({100*fit_orig/len(results):.0f}%)")
    print(f"  Integer:   {fit_int}/{len(results)} ({100*fit_int/len(results):.0f}%)")
    print(f"  Rel+Int:   {fit_relint}/{len(results)} ({100*fit_relint/len(results):.0f}%)")

    # Save CSV summary
    summary_path = f"{OUTPUT_DIR}/summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed results: {summary_path}")
    print(f"Renders: {OUTPUT_DIR}/{{NNN}}_{{original,int,relint}}.png")


if __name__ == "__main__":
    main()
