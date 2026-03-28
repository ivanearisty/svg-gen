"""Data loading, preprocessing, and SVG cleaning utilities."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pandas as pd

from svg_gen.config import SYSTEM_PROMPT, SVGConstraints

# --- P1: SVG Preprocessing Regexes ---

# Round excessive decimal precision: 93.30000305175781 -> 93.3
# This saves ~53% of total SVG characters across the training set.
_DECIMAL_RE = re.compile(r"(\d+\.\d)\d+")

# Strip default SVG attributes that waste tokens
_DEFAULT_ATTRS = re.compile(
    r'\s(?:fill-opacity|stroke-opacity|opacity)=["\']1(?:\.0)?["\']'
    r'|\s(?:fill-rule)=["\']nonzero["\']'
    r'|\s(?:stroke)=["\']none["\']',
)

# Strip XML comments
_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)

# Collapse multiple whitespace (but preserve single spaces/newlines for readability)
_MULTI_SPACE_RE = re.compile(r"  +")
_BLANK_LINES_RE = re.compile(r"\n\s*\n")


def round_svg_numbers(svg: str) -> str:
    """Round all floats in SVG to 1 decimal place, saving tokens.

    Training data has 6.5M floats with 2+ decimals — this saves ~66MB of text.
    """
    return _DECIMAL_RE.sub(r"\1", svg)


def strip_default_attrs(svg: str) -> str:
    """Remove attributes that are already the SVG default.

    Found ~178k default attributes across 50k training samples.
    """
    return _DEFAULT_ATTRS.sub("", svg)


def strip_comments(svg: str) -> str:
    """Remove XML comments from SVG."""
    return _COMMENT_RE.sub("", svg)


def collapse_whitespace(svg: str) -> str:
    """Collapse redundant whitespace without breaking structure."""
    result = _MULTI_SPACE_RE.sub(" ", svg)
    return _BLANK_LINES_RE.sub("\n", result)


def normalize_viewbox(svg: str, target_w: int = 256, target_h: int = 256) -> str:
    """Set SVG width/height for competition rendering without altering viewBox.

    The viewBox defines the internal coordinate space (82% of training data
    uses 0 0 200 200). Changing it would shrink/misalign content. Instead,
    we only set width/height so the renderer scales the SVG to fill the
    target canvas (256x256).
    """
    result = re.sub(r'width=["\'][^"\']*["\']', f'width="{target_w}"', svg, count=1)
    return re.sub(r'height=["\'][^"\']*["\']', f'height="{target_h}"', result, count=1)


def clean_svg(svg: str) -> str:
    """Apply all preprocessing steps to an SVG string.

    Order matters: round numbers first (biggest savings), then strip attrs,
    then comments, then whitespace.
    """
    result = svg.strip()
    result = round_svg_numbers(result)
    result = strip_default_attrs(result)
    result = strip_comments(result)
    return collapse_whitespace(result).strip()


# --- P4: Post-processing & Validation ---


def is_valid_svg(svg_text: str, constraints: SVGConstraints | None = None) -> bool:  # noqa: PLR0911
    """Check if SVG text is valid XML with an <svg> root and passes constraints."""
    if not svg_text or not svg_text.strip():
        return False

    if constraints is None:
        constraints = SVGConstraints()

    if len(svg_text) > constraints.max_svg_length:
        return False

    lower = svg_text.lower()
    if any(pat in lower for pat in constraints.disallowed_patterns):
        return False

    try:
        root = ET.fromstring(svg_text)  # noqa: S314
    except ET.ParseError:
        return False

    if not root.tag.endswith("svg"):
        return False

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag not in constraints.allowed_elements:
            return False

    path_count = sum(1 for elem in root.iter() if elem.tag.endswith("path"))
    return path_count <= constraints.max_path_count


def repair_svg(svg_text: str) -> str:
    """Attempt to repair common model output issues.

    - Close unclosed attributes/tags
    - Ensure </svg> ending
    - Strip content after </svg>
    """
    # If we have </svg>, just truncate after it
    if "</svg>" in svg_text:
        end_idx = svg_text.index("</svg>") + len("</svg>")
        return svg_text[:end_idx]

    # Model was cut off mid-generation. Try to salvage:
    # 1. Close any unclosed quotes (for truncated attribute values)
    # Count quotes — if odd, add one
    if svg_text.count('"') % 2 == 1:
        svg_text += '"'

    # 2. Close any unclosed tag (find last < without >)
    last_open = svg_text.rfind("<")
    last_close = svg_text.rfind(">")
    if last_open > last_close:
        # We're inside an unclosed tag — close it as self-closing
        svg_text += "/>"

    # 3. Add </svg>
    svg_text += "</svg>"
    return svg_text


def extract_svg(text: str) -> str:
    """Extract the first proper <svg ...>...</svg> block from model output.

    Matches <svg followed by a space or attribute (not bare <svg> in prose).
    Also strips any text before the assistant's SVG output.
    """
    # First, try to isolate the assistant's response (after the last "assistant" marker)
    assistant_marker = text.rfind("assistant")
    if assistant_marker >= 0:
        text = text[assistant_marker:]

    # Match <svg with attributes (space after svg), not bare <svg> in prose
    match: re.Match[str] | None = re.search(r"<svg\s[\s\S]*?</svg>", text, flags=re.IGNORECASE)
    if match:
        return match.group(0).strip()
    # Try to salvage: model started an SVG but didn't close it
    start = re.search(r"<svg\s", text, flags=re.IGNORECASE)
    if start:
        return repair_svg(text[start.start() :])
    return ""


def fallback_svg() -> str:
    """Return a minimal valid SVG for failed generations."""
    return (
        "<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256' "
        "viewBox='0 0 256 256'>"
        "<rect x='0' y='0' width='256' height='256' fill='white'/>"
        "<circle cx='128' cy='128' r='64' fill='black'/>"
        "</svg>"
    )


# --- Data Loading & Formatting ---


def load_train_data(csv_path: str) -> pd.DataFrame:
    """Load and validate competition training CSV."""
    df: pd.DataFrame = pd.read_csv(csv_path)
    expected_cols = {"id", "prompt", "svg"}
    if not expected_cols.issubset(set(df.columns)):
        cols = list(df.columns)
        msg = f"Expected columns {expected_cols}, got {cols}"
        raise ValueError(msg)
    return df


def load_test_data(csv_path: str) -> pd.DataFrame:
    """Load competition test CSV (id + prompt only)."""
    return pd.read_csv(csv_path)


def curate_training_data(
    df: pd.DataFrame,
    max_svg_tokens: int = 1900,
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
) -> pd.DataFrame:
    """P2: Filter and clean the training dataset.

    - Remove exact SVG duplicates
    - Clean all SVGs (P1 preprocessing)
    - Remove SVGs that fail XML parsing
    - Filter out SVGs that exceed max_svg_tokens (so model always sees complete SVGs)
    - Filter out empty/trivial results
    """
    original_len = len(df)

    # Drop exact SVG duplicates
    df = df.drop_duplicates(subset=["svg"], keep="first")

    # Clean SVGs
    df = df.copy()
    df["svg"] = df["svg"].apply(clean_svg)

    # Filter out empty or trivially short SVGs after cleaning
    min_svg_len = 50
    df = df[df["svg"].str.len() >= min_svg_len]

    # Validate XML parsing
    def parses_ok(svg: str) -> bool:
        try:
            root = ET.fromstring(svg)  # noqa: S314
            return root.tag.endswith("svg")
        except ET.ParseError:
            return False

    df = df[df["svg"].apply(parses_ok)]
    after_parse = len(df)

    # Filter by token count — ensure every SVG fits completely in context
    # This is critical: the model must see </svg> + <|im_end|> during training
    # to learn when to stop generating
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def token_count(svg: str) -> int:
        return len(tokenizer.encode(svg, add_special_tokens=False))

    df["_token_count"] = df["svg"].apply(token_count)
    too_long = (df["_token_count"] > max_svg_tokens).sum()
    df = df[df["_token_count"] <= max_svg_tokens]
    df = df.drop(columns=["_token_count"])

    print(
        f"Curated: {original_len} -> {len(df)} samples "
        f"({original_len - after_parse} parse failures, "
        f"{too_long} too long, "
        f"{original_len - len(df)} total removed)",
    )
    return df.reset_index(drop=True)


def format_chat_prompt(prompt: str, svg: str | None = None) -> str:
    """Format a prompt (and optional SVG) into the Qwen chat template.

    For training: include the SVG as assistant response.
    For inference: leave assistant turn open.
    """
    text = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if svg is not None:
        text += f"{svg}<|im_end|>"
    return text
