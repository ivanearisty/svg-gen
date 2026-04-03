"""Postprocess a submission CSV — validate, repair, and improve fallbacks.

Reads a submission CSV, re-validates every SVG, attempts deeper repairs on
invalid ones, and replaces hopeless cases with prompt-aware fallbacks instead
of generic black circles.

Usage:
    uv run python scripts/postprocess.py results/submissions/merged-r32-expanded-complete.csv
    uv run python scripts/postprocess.py input.csv --output cleaned.csv
"""

import argparse
import csv
import re
import xml.etree.ElementTree as ET

import pandas as pd

from svg_gen.data import SVGConstraints, is_valid_svg, normalize_viewbox


# ─── Deeper repair than data.repair_svg ───


def deep_repair(svg: str) -> str:
    """Aggressive SVG repair — try multiple strategies."""
    if is_valid_svg(svg):
        return svg

    # 1. Strip everything after </svg> (model appended garbage)
    if "</svg>" in svg:
        svg = svg[: svg.index("</svg>") + len("</svg>")]
        if is_valid_svg(svg):
            return svg

    # 2. Close unclosed quotes
    if svg.count('"') % 2 == 1:
        svg += '"'

    # 3. Close unclosed tags — find all open tags without matching close
    last_open = svg.rfind("<")
    last_close = svg.rfind(">")
    if last_open > last_close:
        # Check if it looks like a closing tag attempt
        if svg[last_open:].startswith("</"):
            svg += ">"
        else:
            svg += "/>"

    # 4. Ensure </svg>
    if "</svg>" not in svg:
        # Close any open path/g/etc elements
        open_tags = re.findall(r"<(path|g|rect|circle|ellipse|line|polyline|polygon|defs|text)\b", svg)
        close_tags = re.findall(r"</(path|g|rect|circle|ellipse|line|polyline|polygon|defs|text)>", svg)

        open_counts: dict[str, int] = {}
        for tag in open_tags:
            open_counts[tag] = open_counts.get(tag, 0) + 1
        for tag in close_tags:
            open_counts[tag] = open_counts.get(tag, 0) - 1

        # Close any unclosed non-self-closing tags
        for tag, count in reversed(list(open_counts.items())):
            if count > 0 and tag in ("g", "defs", "text"):
                for _ in range(count):
                    svg += f"</{tag}>"

        svg += "</svg>"

    if is_valid_svg(svg):
        return svg

    # 5. Nuclear option: extract just the path data and rebuild
    paths = re.findall(r'<path\s[^>]*/?>', svg)
    rects = re.findall(r'<rect\s[^>]*/?>', svg)
    circles = re.findall(r'<circle\s[^>]*/?>', svg)
    ellipses = re.findall(r'<ellipse\s[^>]*/?>', svg)

    elements = paths + rects + circles + ellipses
    if elements:
        # Ensure all elements are self-closing
        clean_elements = []
        for elem in elements:
            if not elem.endswith("/>"):
                elem = elem.rstrip(">").rstrip("/") + "/>"
            clean_elements.append(elem)

        rebuilt = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" '
            'width="256" height="256">'
            + "".join(clean_elements)
            + "</svg>"
        )
        if is_valid_svg(rebuilt):
            return rebuilt

    return ""


# ─── Prompt-aware fallback ───

# Common color keywords → hex
COLOR_MAP = {
    "red": "#E74C3C", "blue": "#3498DB", "green": "#2ECC71", "yellow": "#F1C40F",
    "orange": "#E67E22", "purple": "#9B59B6", "pink": "#FF69B4", "black": "#2C3E50",
    "white": "#ECF0F1", "gray": "#95A5A6", "grey": "#95A5A6", "brown": "#8B4513",
    "gold": "#FFD700", "navy": "#1F3A93", "teal": "#008080", "cyan": "#00BCD4",
}

# Shape keywords → SVG elements
SHAPE_MAP = {
    "circle": '<circle cx="100" cy="100" r="70" fill="{color}"/>',
    "square": '<rect x="30" y="30" width="140" height="140" fill="{color}"/>',
    "rectangle": '<rect x="20" y="50" width="160" height="100" fill="{color}"/>',
    "triangle": '<polygon points="100,20 180,180 20,180" fill="{color}"/>',
    "star": '<polygon points="100,10 120,75 190,75 135,120 155,185 100,145 45,185 65,120 10,75 80,75" fill="{color}"/>',
    "heart": '<path d="M100,180 C60,140 10,100 10,60 C10,30 40,10 70,10 C85,10 95,20 100,30 C105,20 115,10 130,10 C160,10 190,30 190,60 C190,100 140,140 100,180Z" fill="{color}"/>',
    "diamond": '<polygon points="100,10 180,100 100,190 20,100" fill="{color}"/>',
    "arrow": '<polygon points="100,20 180,100 130,100 130,180 70,180 70,100 20,100" fill="{color}"/>',
}


def prompt_aware_fallback(prompt: str) -> str:
    """Generate a better fallback SVG based on prompt keywords."""
    prompt_lower = prompt.lower()

    # Detect color
    color = "#555555"  # default gray
    for keyword, hex_val in COLOR_MAP.items():
        if keyword in prompt_lower:
            color = hex_val
            break

    # Detect background color
    bg_color = "#FFFFFF"
    if "dark background" in prompt_lower or "black background" in prompt_lower:
        bg_color = "#2C3E50"
    elif "blue background" in prompt_lower:
        bg_color = "#3498DB"

    # Detect shape
    shape_svg = ""
    for keyword, template in SHAPE_MAP.items():
        if keyword in prompt_lower:
            shape_svg = template.format(color=color)
            break

    if not shape_svg:
        # Default: colored circle (better than nothing)
        shape_svg = f'<circle cx="100" cy="100" r="60" fill="{color}"/>'

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" '
        f'width="256" height="256">'
        f'<rect x="0" y="0" width="200" height="200" fill="{bg_color}"/>'
        f'{shape_svg}'
        f'</svg>'
    )


# ─── Strip disallowed elements ───


def strip_disallowed(svg: str, constraints: SVGConstraints | None = None) -> str:
    """Remove disallowed elements/attributes while keeping the rest."""
    if constraints is None:
        constraints = SVGConstraints()

    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return svg

    # Remove disallowed elements
    def clean_element(elem: ET.Element) -> None:
        to_remove = []
        for child in elem:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag not in constraints.allowed_elements:
                to_remove.append(child)
            else:
                clean_element(child)
        for child in to_remove:
            elem.remove(child)

    clean_element(root)

    # Remove disallowed attributes
    for elem in root.iter():
        attrs_to_remove = []
        for attr in elem.attrib:
            attr_lower = attr.lower()
            if any(pat in attr_lower for pat in constraints.disallowed_patterns):
                attrs_to_remove.append(attr)
        for attr in attrs_to_remove:
            del elem.attrib[attr]

    return ET.tostring(root, encoding="unicode")


# ─── Main ───


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess submission CSV")
    parser.add_argument("input", help="Input submission CSV")
    parser.add_argument("--output", help="Output CSV (default: input with -clean suffix)")
    parser.add_argument("--prompts", default="data/test.csv", help="Test prompts CSV for fallbacks")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace(".csv", "-clean.csv")

    # Load submission
    sub_df = pd.read_csv(args.input)
    print(f"Loaded {len(sub_df)} submissions from {args.input}")

    # Load prompts for fallback generation
    prompts_df = pd.read_csv(args.prompts)
    prompt_map = dict(zip(prompts_df["id"].astype(str), prompts_df["prompt"].astype(str)))

    constraints = SVGConstraints()
    stats = {"valid_already": 0, "repaired": 0, "stripped": 0, "fallback": 0, "normalized": 0}

    rows = []
    for _, row in sub_df.iterrows():
        sample_id = str(row["id"])
        svg = str(row["svg"])

        # Step 1: Check if already valid
        if is_valid_svg(svg, constraints):
            svg = normalize_viewbox(svg)
            stats["valid_already"] += 1
            stats["normalized"] += 1
            rows.append({"id": sample_id, "svg": svg})
            continue

        # Step 2: Try deep repair
        repaired = deep_repair(svg)
        if repaired and is_valid_svg(repaired, constraints):
            repaired = normalize_viewbox(repaired)
            stats["repaired"] += 1
            rows.append({"id": sample_id, "svg": repaired})
            continue

        # Step 3: Try stripping disallowed elements
        stripped = strip_disallowed(svg, constraints)
        if is_valid_svg(stripped, constraints):
            stripped = normalize_viewbox(stripped)
            stats["stripped"] += 1
            rows.append({"id": sample_id, "svg": stripped})
            continue

        # Step 4: Prompt-aware fallback
        prompt = prompt_map.get(sample_id, "")
        fallback = prompt_aware_fallback(prompt)
        stats["fallback"] += 1
        rows.append({"id": sample_id, "svg": fallback})

    # Save
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "svg"])
        writer.writeheader()
        writer.writerows(rows)

    # Report
    print(f"\n{'='*50}")
    print(f"POSTPROCESSING RESULTS")
    print(f"{'='*50}")
    print(f"  Total samples:    {len(rows)}")
    print(f"  Already valid:    {stats['valid_already']}")
    print(f"  Repaired:         {stats['repaired']}")
    print(f"  Stripped:         {stats['stripped']}")
    print(f"  Fallback:         {stats['fallback']}")
    print(f"  Saved to:         {args.output}")

    # Verify output
    verify_df = pd.read_csv(args.output)
    valid_count = sum(1 for _, r in verify_df.iterrows() if is_valid_svg(str(r["svg"])))
    print(f"\n  Verification: {valid_count}/{len(verify_df)} valid SVGs")

    svg_lens = [len(str(r["svg"])) for _, r in verify_df.iterrows()]
    over_limit = sum(1 for l in svg_lens if l > 16000)
    print(f"  Over 16k chars:   {over_limit}")
    print(f"  Avg SVG length:   {sum(svg_lens)/len(svg_lens):.0f} chars")


if __name__ == "__main__":
    main()
