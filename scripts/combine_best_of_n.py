"""Best-of-N combiner — pick the best SVG per prompt across multiple runs.

For each prompt, scores all candidate SVGs and picks the best one.
Scoring heuristic (no ground truth needed):
1. Must be valid XML with <svg> root
2. Must have </svg> closing tag
3. Longer complete SVGs score higher (more detail)
4. Penalize fallback/generic SVGs
5. Prefer standard viewBox (0 0 200 200)
6. Penalize truncated paths (unclosed d attributes)

Usage:
    uv run python scripts/combine_best_of_n.py
"""

import re
import sys
import xml.etree.ElementTree as ET

import pandas as pd


def score_svg(svg: str) -> float:
    """Score an SVG heuristically — higher is better."""
    if not svg or not svg.strip():
        return 0.0

    score = 0.0

    # Valid XML?
    try:
        root = ET.fromstring(svg)
        if not root.tag.endswith("svg"):
            return 1.0  # parseable but not SVG
        score += 10.0
    except ET.ParseError:
        return 0.5  # invalid XML

    # Has closing tag?
    if "</svg>" in svg:
        score += 5.0

    # Length (more detail = likely better visual match)
    score += min(len(svg) / 200.0, 10.0)  # cap at 10 points for length

    # Penalize generic fallback SVGs
    if 'cx="128" cy="128" r="64"' in svg:
        score -= 20.0

    # Penalize very short SVGs
    if len(svg) < 200:
        score -= 5.0

    # Standard viewBox preferred
    if "0.0 0.0 200.0 200.0" in svg or "0 0 200 200" in svg:
        score += 2.0

    # Count path elements (more paths = more detail)
    path_count = svg.count("<path")
    score += min(path_count * 0.5, 5.0)

    # Penalize truncated paths (d attribute without closing quote)
    unclosed_d = len(re.findall(r'd="[^"]*$', svg))
    score -= unclosed_d * 3.0

    # Has fill colors (not just black/white)
    color_count = len(re.findall(r'fill="#[0-9a-fA-F]{6}"', svg))
    score += min(color_count * 0.3, 3.0)

    return score


def main():
    submissions = [
        ("results/submissions/componly-r32-clean.csv", "greedy_rep1.1 (16.87)"),
        ("results/submissions/greedy_rep1.05_tok1024.csv", "greedy_rep1.05"),
        ("results/submissions/greedy_rep1.15_tok1024.csv", "greedy_rep1.15"),
    ]

    # Check if 1536 tokens run is complete
    try:
        df_1536 = pd.read_csv("results/submissions/greedy_rep1.10_tok1536.csv")
        if len(df_1536) >= 1000:
            submissions.append(("results/submissions/greedy_rep1.10_tok1536.csv", "greedy_1536tok"))
    except Exception:
        pass

    # Load all submissions
    dfs = {}
    for path, name in submissions:
        try:
            df = pd.read_csv(path)
            if len(df) >= 1000:
                dfs[name] = df.set_index("id")["svg"].to_dict()
                print(f"Loaded {name}: {len(df)} samples")
            else:
                print(f"Skipping {name}: only {len(df)} samples")
        except Exception as e:
            print(f"Skipping {name}: {e}")

    if not dfs:
        print("No submissions loaded!")
        sys.exit(1)

    # Get all prompt IDs from the best submission
    best_name = list(dfs.keys())[0]
    all_ids = list(dfs[best_name].keys())

    # For each prompt, pick the best SVG
    results = []
    improvements = 0
    best_source_counts: dict[str, int] = {}

    for sample_id in all_ids:
        candidates = {}
        for name, svgs in dfs.items():
            if sample_id in svgs:
                candidates[name] = svgs[sample_id]

        # Score all candidates
        scored = [(name, svg, score_svg(svg)) for name, svg in candidates.items()]
        scored.sort(key=lambda x: -x[2])

        best_name_for_id, best_svg, best_score = scored[0]
        original_svg = candidates.get("greedy_rep1.1 (16.87)", "")
        original_score = score_svg(original_svg) if original_svg else 0

        if best_score > original_score and best_name_for_id != "greedy_rep1.1 (16.87)":
            improvements += 1

        best_source_counts[best_name_for_id] = best_source_counts.get(best_name_for_id, 0) + 1
        results.append({"id": sample_id, "svg": best_svg})

    # Save combined submission
    output_path = "results/submissions/best_of_n_combined.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Combined {len(dfs)} submissions → {output_path}")
    print(f"Total samples: {len(results)}")
    print(f"Improvements over baseline: {improvements}/{len(results)}")
    print(f"\nSource distribution:")
    for name, count in sorted(best_source_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} samples ({100*count/len(results):.1f}%)")

    # Show specific improvements for short SVGs
    print(f"\nShort SVG improvements:")
    for r in results:
        original = dfs.get("greedy_rep1.1 (16.87)", {}).get(r["id"], "")
        if len(original) < 200 and len(r["svg"]) > len(original):
            print(f"  {r['id'][:12]}: {len(original)} → {len(r['svg'])} chars")


if __name__ == "__main__":
    main()
