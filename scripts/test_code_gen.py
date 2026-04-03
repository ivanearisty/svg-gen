"""Quick test: can Qwen2.5-Coder generate Python that builds SVGs?

Tests both the fine-tuned model and the base model with a code-generation prompt.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

API_DOCS = '''You write Python code to create SVG images using this API:

svg = create_svg(200, 200)  # creates 200x200 canvas
svg.background("white")  # fill background
svg.rect(x, y, w, h, fill="blue")  # rectangle
svg.circle(cx, cy, r, fill="red")  # circle
svg.ellipse(cx, cy, rx, ry, fill="green")  # ellipse
svg.line(x1, y1, x2, y2, stroke="black", stroke_width=2)  # line
svg.polygon([(x1,y1), (x2,y2), (x3,y3)], fill="yellow")  # polygon
svg.text(x, y, "hello", font_size=16, fill="black")  # text
svg.path("M 10 10 L 90 90", fill="none", stroke="red")  # SVG path

Return ONLY the Python code. No markdown. No explanation. Start with svg = create_svg()'''

PROMPTS = [
    "a red circle on a white background",
    "a blue triangle pointing up",
    "a simple smiley face",
    "a green rectangle with a yellow circle inside",
    "the letter A in black on white",
]


def test_model(model_name, is_base=False):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} ({'base' if is_base else 'fine-tuned'})")
    print(f"{'='*60}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()

    for prompt in PROMPTS:
        chat = (
            f"<|im_start|>system\n{API_DOCS}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = tokenizer(chat, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False, repetition_penalty=1.1)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)

        # Extract assistant response
        marker = decoded.rfind("assistant")
        response = decoded[marker + len("assistant"):].strip() if marker >= 0 else decoded

        print(f"\n--- Prompt: {prompt} ---")
        print(response[:500])
        print()


if __name__ == "__main__":
    # Test base model first
    test_model("Qwen/Qwen2.5-Coder-1.5B-Instruct", is_base=True)
