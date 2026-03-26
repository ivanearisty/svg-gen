"""Local scoring — approximates the competition metric.

Implements:
- Visual Fidelity: 0.7 * SSIM + 0.3 * EdgeF1
- Structural Similarity: exp(-TED / 25)
- Compactness: exp(-|log((len_pred+50)/(len_ref+50))|)
- Composite: V^0.85 * S^0.12 * C^0.03

Usage:
    python -m svg_gen.score --submission submissions/submission.csv --reference data/train.csv --samples 100
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import cairosvg
import numpy as np
from skimage.feature import canny
from skimage.metrics import structural_similarity

if TYPE_CHECKING:
    from numpy.typing import NDArray


def render_svg_to_array(svg_text: str, size: int = 256) -> NDArray[np.uint8] | None:
    """Render SVG to a grayscale numpy array. Returns None on failure."""
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(png_data)).convert("L")
        return np.array(img, dtype=np.uint8)
    except Exception:  # noqa: BLE001
        return None


def compute_ssim(pred: NDArray[np.uint8], ref: NDArray[np.uint8]) -> float:
    """Compute SSIM between two grayscale images."""
    return float(structural_similarity(pred, ref, data_range=255))


def compute_edge_f1(pred: NDArray[np.uint8], ref: NDArray[np.uint8]) -> float:
    """Compute Edge F1 score between two grayscale images."""
    pred_edges = canny(pred.astype(float), sigma=1.0)
    ref_edges = canny(ref.astype(float), sigma=1.0)

    # Dilate edges slightly for tolerance (3px)
    from scipy.ndimage import binary_dilation

    pred_dilated = binary_dilation(pred_edges, iterations=1)
    ref_dilated = binary_dilation(ref_edges, iterations=1)

    pred_count = pred_edges.sum()
    precision = float((pred_edges & ref_dilated).sum()) / pred_count if pred_count > 0 else 0.0

    ref_count = ref_edges.sum()
    recall = float((ref_edges & pred_dilated).sum()) / ref_count if ref_count > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_visual_fidelity(pred_img: NDArray[np.uint8], ref_img: NDArray[np.uint8]) -> float:
    """V = 0.7 * SSIM + 0.3 * EdgeF1."""
    ssim = compute_ssim(pred_img, ref_img)
    edge_f1 = compute_edge_f1(pred_img, ref_img)
    return 0.7 * ssim + 0.3 * edge_f1


def compute_tree_edit_distance(pred_svg: str, ref_svg: str) -> int:
    """Approximate tree edit distance using element-level comparison.

    Full TED is expensive (O(n^3)). This is a simplified version that counts
    element type mismatches at each level — good enough for relative comparison.
    """
    def get_element_sequence(svg_text: str) -> list[str]:
        try:
            root = ET.fromstring(svg_text)  # noqa: S314
        except ET.ParseError:
            return []
        else:
            elements: list[str] = []
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                elements.append(tag)
            return elements

    pred_elems = get_element_sequence(pred_svg)
    ref_elems = get_element_sequence(ref_svg)

    # Simple edit distance on element sequences (not full tree edit distance)
    # This is a reasonable proxy for the actual TED
    max_len = max(len(pred_elems), len(ref_elems))
    if max_len == 0:
        return 0

    matches = sum(1 for p, r in zip(pred_elems, ref_elems, strict=False) if p == r)
    return max_len - matches


def compute_structural_similarity(pred_svg: str, ref_svg: str) -> float:
    """S = exp(-TED / 25)."""
    ted = compute_tree_edit_distance(pred_svg, ref_svg)
    return math.exp(-ted / 25)


def compute_compactness(pred_svg: str, ref_svg: str) -> float:
    """C = exp(-|log((len_pred+50)/(len_ref+50))|)."""
    ratio = (len(pred_svg) + 50) / (len(ref_svg) + 50)
    return math.exp(-abs(math.log(ratio)))


def score_sample(pred_svg: str, ref_svg: str) -> dict[str, float]:
    """Score a single prediction against a reference.

    Returns dict with individual components and composite score.
    """
    # Validity gate
    pred_img = render_svg_to_array(pred_svg)
    ref_img = render_svg_to_array(ref_svg)

    if pred_img is None:
        return {"valid": 0.0, "visual": 0.0, "structural": 0.0, "compactness": 0.0, "composite": 0.0}

    if ref_img is None:
        # Can't score without reference render
        return {"valid": 1.0, "visual": 0.0, "structural": 0.0, "compactness": 0.0, "composite": 0.0}

    # Components
    visual = compute_visual_fidelity(pred_img, ref_img)
    structural = compute_structural_similarity(pred_svg, ref_svg)
    compactness = compute_compactness(pred_svg, ref_svg)

    # Composite (geometric mean with weights)
    composite = (visual ** 0.85) * (structural ** 0.12) * (compactness ** 0.03)

    return {
        "valid": 1.0,
        "visual": visual,
        "structural": structural,
        "compactness": compactness,
        "composite": composite,
    }


def score_submission(
    submission_csv: str,
    reference_csv: str,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Score a full submission CSV against training references.

    Samples random training examples, generates a mapping, and scores.
    Returns aggregate metrics.
    """
    import pandas as pd

    sub_df = pd.read_csv(submission_csv)
    ref_df = pd.read_csv(reference_csv)

    # Match by ID
    merged = sub_df.merge(ref_df, on="id", suffixes=("_pred", "_ref"))
    if max_samples and len(merged) > max_samples:
        merged = merged.sample(n=max_samples, random_state=42)

    print(f"Scoring {len(merged)} samples...")

    scores: list[dict[str, float]] = []
    for i, (_, row) in enumerate(merged.iterrows()):
        pred_svg = str(row.get("svg_pred", row.get("svg", "")))
        ref_svg = str(row.get("svg_ref", row.get("svg", "")))
        s = score_sample(pred_svg, ref_svg)
        scores.append(s)

        if (i + 1) % 10 == 0:
            avg_composite = sum(x["composite"] for x in scores) / len(scores)
            valid_rate = sum(x["valid"] for x in scores) / len(scores)
            print(f"  [{i + 1}/{len(merged)}] avg_composite={avg_composite:.4f}, valid={valid_rate:.0%}")

    # Aggregate
    n = len(scores)
    result = {
        "n_samples": float(n),
        "validity_rate": sum(s["valid"] for s in scores) / n,
        "avg_visual": sum(s["visual"] for s in scores) / n,
        "avg_structural": sum(s["structural"] for s in scores) / n,
        "avg_compactness": sum(s["compactness"] for s in scores) / n,
        "avg_composite": sum(s["composite"] for s in scores) / n,
        "final_score": 100 * sum(s["composite"] for s in scores) / n,
    }

    print(f"\n{'=' * 50}")
    print(f"Samples:      {n}")
    print(f"Validity:     {result['validity_rate']:.0%}")
    print(f"Visual:       {result['avg_visual']:.4f}")
    print(f"Structural:   {result['avg_structural']:.4f}")
    print(f"Compactness:  {result['avg_compactness']:.4f}")
    print(f"Composite:    {result['avg_composite']:.4f}")
    print(f"Final Score:  {result['final_score']:.2f} / 100")
    print(f"{'=' * 50}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:  # noqa: PLR2004
        print("Usage: python -m svg_gen.score --submission <csv> --reference <csv> [--samples N]")
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()

    score_submission(args.submission, args.reference, args.samples)
