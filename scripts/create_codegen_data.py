"""Convert (prompt, SVG) pairs into (prompt, Python code) training data.

Parses each SVG and generates Python code using the SVGBuilder API.
For basic shapes (rect, circle, ellipse, line, polygon) → clean API calls.
For complex paths → svg.path(d="...") with the raw path data.

Usage:
    uv run python scripts/create_codegen_data.py
"""

import csv
import re
import sys
import xml.etree.ElementTree as ET

import pandas as pd

from svg_gen.data import clean_svg


def svg_to_python(svg_text: str) -> str | None:
    """Convert an SVG string to Python code using SVGBuilder API."""
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return None

    if not root.tag.endswith("svg"):
        return None

    # Get viewBox dimensions
    vb = root.get("viewBox", "0 0 200 200")
    parts = vb.replace(",", " ").split()
    try:
        vb_w = int(float(parts[2])) if len(parts) > 2 else 200
        vb_h = int(float(parts[3])) if len(parts) > 3 else 200
    except (ValueError, IndexError):
        vb_w, vb_h = 200, 200

    lines = [f"svg = create_svg({vb_w}, {vb_h})"]

    # Check for background rect (first rect covering full canvas)
    first_child = list(root)
    if first_child:
        fc = first_child[0]
        tag = fc.tag.split("}")[-1] if "}" in fc.tag else fc.tag
        if tag == "rect":
            w = fc.get("width", "0")
            h = fc.get("height", "0")
            try:
                if float(w) >= vb_w * 0.9 and float(h) >= vb_h * 0.9:
                    fill = fc.get("fill", "white")
                    lines.append(f'svg.background("{fill}")')
                    first_child = first_child[1:]
            except ValueError:
                pass

    def process_element(elem, depth=0):
        """Convert a single SVG element to a Python line."""
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag == "rect":
            x = elem.get("x", "0")
            y = elem.get("y", "0")
            w = elem.get("width", "0")
            h = elem.get("height", "0")
            fill = elem.get("fill", "black")
            stroke = elem.get("stroke", "none")
            rx = elem.get("rx", "0")
            call = f'svg.rect({x}, {y}, {w}, {h}, fill="{fill}"'
            if stroke != "none":
                sw = elem.get("stroke-width", "1")
                call += f', stroke="{stroke}", stroke_width={sw}'
            if rx != "0":
                call += f", rx={rx}"
            call += ")"
            lines.append(call)

        elif tag == "circle":
            cx = elem.get("cx", "0")
            cy = elem.get("cy", "0")
            r = elem.get("r", "0")
            fill = elem.get("fill", "black")
            stroke = elem.get("stroke", "none")
            call = f'svg.circle({cx}, {cy}, {r}, fill="{fill}"'
            if stroke != "none":
                sw = elem.get("stroke-width", "1")
                call += f', stroke="{stroke}", stroke_width={sw}'
            call += ")"
            lines.append(call)

        elif tag == "ellipse":
            cx = elem.get("cx", "0")
            cy = elem.get("cy", "0")
            rx = elem.get("rx", "0")
            ry = elem.get("ry", "0")
            fill = elem.get("fill", "black")
            call = f'svg.ellipse({cx}, {cy}, {rx}, {ry}, fill="{fill}")'
            lines.append(call)

        elif tag == "line":
            x1 = elem.get("x1", "0")
            y1 = elem.get("y1", "0")
            x2 = elem.get("x2", "0")
            y2 = elem.get("y2", "0")
            stroke = elem.get("stroke", "black")
            sw = elem.get("stroke-width", "2")
            lines.append(f'svg.line({x1}, {y1}, {x2}, {y2}, stroke="{stroke}", stroke_width={sw})')

        elif tag == "polygon":
            pts_str = elem.get("points", "")
            fill = elem.get("fill", "black")
            # Parse points: "x1,y1 x2,y2 ..." or "x1 y1 x2 y2 ..."
            nums = re.findall(r"[\d.]+", pts_str)
            if len(nums) >= 4:
                pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
                pts_code = ", ".join(f"({x}, {y})" for x, y in pairs)
                lines.append(f'svg.polygon([{pts_code}], fill="{fill}")')

        elif tag == "polyline":
            pts_str = elem.get("points", "")
            stroke = elem.get("stroke", "black")
            fill = elem.get("fill", "none")
            nums = re.findall(r"[\d.]+", pts_str)
            if len(nums) >= 4:
                pairs = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
                pts_code = ", ".join(f"({x}, {y})" for x, y in pairs)
                lines.append(f'svg.polyline([{pts_code}], fill="{fill}", stroke="{stroke}")')

        elif tag == "path":
            d = elem.get("d", "")
            fill = elem.get("fill", "black")
            stroke = elem.get("stroke", "none")
            if d:
                # Truncate very long paths to keep training data manageable
                if len(d) > 1500:
                    return  # Skip very complex paths
                call = f'svg.path("{d}", fill="{fill}"'
                if stroke != "none":
                    sw = elem.get("stroke-width", "1")
                    call += f', stroke="{stroke}", stroke_width={sw}'
                call += ")"
                lines.append(call)

        elif tag == "text":
            x = elem.get("x", "0")
            y = elem.get("y", "0")
            content = elem.text or ""
            fill = elem.get("fill", "black")
            fs = elem.get("font-size", "16")
            if content.strip():
                lines.append(f'svg.text({x}, {y}, "{content.strip()}", font_size={fs}, fill="{fill}")')

        elif tag == "g":
            transform = elem.get("transform", "")
            for child in elem:
                process_element(child, depth + 1)

        # Process children for non-group elements
        if tag != "g":
            for child in elem:
                process_element(child, depth + 1)

    for elem in (first_child if first_child != list(root) else root):
        if isinstance(elem, str):
            continue
        process_element(elem)

    if len(lines) <= 1:
        return None  # Only create_svg, no actual content

    return "\n".join(lines)


def main():
    df = pd.read_csv("data/train.csv")
    print(f"Input: {len(df)} samples")

    results = []
    skipped = 0

    for _, row in df.iterrows():
        svg = clean_svg(str(row["svg"]))
        prompt = str(row["prompt"])
        code = svg_to_python(svg)

        if code is None:
            skipped += 1
            continue

        # Skip if the code is too short (trivial) or too long (won't fit in context)
        if len(code) < 30 or len(code) > 3000:
            skipped += 1
            continue

        results.append({"prompt": prompt, "code": code, "id": row["id"]})

    output_path = "data/train_codegen.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"Output: {len(results)} samples ({skipped} skipped)")
    print(f"Saved to {output_path}")

    # Show a few examples
    print("\n=== EXAMPLES ===")
    for r in results[:3]:
        print(f"\nPrompt: {r['prompt'][:80]}")
        print(f"Code ({len(r['code'])} chars):")
        print(r["code"][:300])
        print("---")


if __name__ == "__main__":
    main()
