"""Local SVG rendering and comparison utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cairosvg


def render_svg_to_png(svg_text: str, output_path: str, size: int = 256) -> bool:
    """Render an SVG string to a PNG file. Returns True on success."""
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
        Path(output_path).write_bytes(png_data)
    except Exception as e:  # noqa: BLE001
        print(f"  Render failed: {e}")
        return False
    else:
        return True


def render_experiment_results(results_json: str, output_dir: str = "renders") -> None:
    """Render all SVGs from an experiment results JSON file."""
    with open(results_json) as f:
        data = json.load(f)

    exp_name = data["name"]
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Rendering {len(data['samples'])} samples from experiment '{exp_name}'")
    print(f"  Train loss: {data['train_loss']:.4f}")
    print(f"  Validity:   {data['num_valid']}/{data['num_generated']} ({data['validity_rate']:.0%})")

    for sample in data["samples"]:
        sample_id = sample["id"]
        svg = sample["svg"]
        valid = sample["valid"]
        png_path = os.path.join(exp_dir, f"{sample_id}.png")

        success = render_svg_to_png(svg, png_path)
        status = "ok" if success else "FAIL"
        print(f"  {sample_id}: {status} (valid={valid}, {len(svg)} chars)")

    print(f"Renders saved to {exp_dir}/")


def compare_experiments(results_dir: str = "experiment_results") -> None:
    """Print a comparison table of all experiment results."""
    results = []
    for filename in sorted(Path(results_dir).glob("*.json")):
        with open(filename) as f:
            data = json.load(f)
        results.append(data)

    if not results:
        print(f"No results found in {results_dir}/")
        return

    # Header
    print(f"{'Name':<25} {'Loss':>8} {'Valid':>8} {'Rate':>8} {'AvgLen':>8} {'TrainT':>8} {'GenT':>8}")
    print("-" * 85)

    for r in results:
        print(
            f"{r['name']:<25} "
            f"{r['train_loss']:>8.4f} "
            f"{r['num_valid']:>4}/{r['num_generated']:<3} "
            f"{r['validity_rate']:>7.0%} "
            f"{r['avg_svg_length']:>8.0f} "
            f"{r['train_time_s']:>7.0f}s "
            f"{r['generation_time_s']:>7.0f}s",
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        dir_arg = sys.argv[2] if len(sys.argv) > 2 else "experiment_results"  # noqa: PLR2004
        compare_experiments(dir_arg)
    elif len(sys.argv) > 1:
        render_experiment_results(sys.argv[1])
    else:
        print("Usage:")
        print("  python -m svg_gen.render <results.json>    — render SVGs to PNGs")
        print("  python -m svg_gen.render compare [dir]     — compare experiments")
