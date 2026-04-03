"""Code-generation SVG inference — model writes Python, we execute it.

Usage:
    uv run python scripts/code_gen_inference.py
"""

import argparse
import csv
import os
import re
import time
import traceback

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from svg_gen.data import fallback_svg, normalize_viewbox

# Import the builder so exec() can find it
from scripts.svg_builder import SVGBuilder, create_svg  # noqa: F401

API_DOCS = '''You write Python code to create SVG images. Use ONLY this API:

svg = create_svg(200, 200)
svg.background("white")
svg.rect(x, y, width, height, fill="blue", stroke="none", rx=0)
svg.circle(cx, cy, r, fill="red", stroke="none")
svg.ellipse(cx, cy, rx, ry, fill="green")
svg.line(x1, y1, x2, y2, stroke="black", stroke_width=2)
svg.polygon([(x1,y1), (x2,y2), (x3,y3)], fill="yellow")
svg.text(x, y, "text", font_size=16, fill="black")
svg.path("M 10 10 L 90 90 Z", fill="none", stroke="red")

Canvas is 200x200. Coordinates range from 0 to 200. Center is (100, 100).
Return ONLY Python code. No markdown. No explanations. No function definitions.
Start with: svg = create_svg(200, 200)
End with the last svg method call. Do NOT call svg.render() or print().'''

MODEL = "models/codegen-1.5b"
TEST_CSV = "data/test.csv"
MAX_NEW_TOKENS = 512


def extract_python_code(text: str) -> str:
    """Extract Python code from model output, handling markdown blocks."""
    # Remove markdown code blocks
    code = re.sub(r"```python\s*", "", text)
    code = re.sub(r"```\s*", "", code)

    # Find the code starting with svg = create_svg or import
    lines = code.strip().split("\n")
    start = 0
    for i, line in enumerate(lines):
        if "create_svg" in line or "svg." in line:
            start = i
            break

    # Filter lines: keep only svg builder calls and variable assignments
    filtered = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Stop at non-code lines
        if stripped.startswith("def ") or stripped.startswith("class "):
            continue
        if stripped.startswith("print(") or stripped.startswith("return "):
            continue
        if "render()" in stripped:
            continue
        filtered.append(line)

    return "\n".join(filtered)


def execute_svg_code(code: str) -> str | None:
    """Safely execute Python code that builds an SVG."""
    # Provide the builder in the execution namespace
    namespace = {
        "create_svg": create_svg,
        "SVGBuilder": SVGBuilder,
    }

    try:
        exec(code, namespace)  # noqa: S102
    except Exception:
        return None

    # Find the svg object in namespace
    for val in namespace.values():
        if isinstance(val, SVGBuilder):
            return val.render()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/submissions/code_gen.csv")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print(f"Mode: CODE GENERATION (Python → SVG)")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("Model loaded.")

    test_df = pd.read_csv(TEST_CSV)
    total = len(test_df)
    print(f"Generating {total} SVGs via code generation...")

    t0 = time.time()
    invalid_count = 0
    exec_fail_count = 0

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "svg"])

        for i, (_, row) in enumerate(test_df.iterrows()):
            prompt = str(row.get("prompt", row.get("text", "")))
            sample_id = str(row.get("id", f"sample_{i}"))

            chat = (
                f"<|im_start|>system\n{API_DOCS}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\nsvg = create_svg(200, 200)\n"
            )

            inputs = tokenizer(chat, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    do_sample=False, repetition_penalty=1.1,
                )

            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            marker = decoded.rfind("assistant")
            response = decoded[marker + len("assistant"):].strip() if marker >= 0 else decoded

            # Ensure it starts with create_svg
            code = extract_python_code(response)
            if not code.startswith("svg"):
                code = "svg = create_svg(200, 200)\n" + code

            # Execute the code
            svg = execute_svg_code(code)

            if svg is None:
                exec_fail_count += 1
                # Try with a simpler extraction
                svg = execute_svg_code("svg = create_svg(200, 200)\nsvg.background('white')\n" + code)

            if svg is None:
                invalid_count += 1
                svg = fallback_svg()
            else:
                svg = normalize_viewbox(svg)

            writer.writerow([sample_id, svg])

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i + 1}/{total}] {rate:.2f}/s | eta={eta / 60:.0f}min | "
                    f"invalid={invalid_count} exec_fail={exec_fail_count}",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"\nDone! {total} samples in {elapsed / 60:.1f}min")
    print(f"Invalid/fallback: {invalid_count}, Exec failures: {exec_fail_count}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
