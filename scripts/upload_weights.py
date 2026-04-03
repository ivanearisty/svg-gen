"""Upload all model weights to HuggingFace."""

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
os.chdir(_project_dir)

from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN", "")
REPO = "kaleidoscopicwhether/svg-gen-weights"

api = HfApi(token=TOKEN)

# Models to upload with their descriptions
MODELS = [
    {
        "local_path": "models/r16-3epoch",
        "hf_folder": "r16-3epoch",
        "desc": "LoRA r=16 adapter, 3 epochs, 46k comp data (Kaggle: 15.47)",
    },
    {
        "local_path": "models/merged-1.5b-r16",
        "hf_folder": "merged-1.5b-r16",
        "desc": "Qwen2.5-Coder-1.5B with r16 adapter merged into weights",
    },
    {
        "local_path": "outputs-componly/final-adapter",
        "hf_folder": "componly-r32-adapter",
        "desc": "BEST MODEL - LoRA r=32 adapter, comp-only data (Kaggle: 16.87, 6th place). Load on top of merged-1.5b-r16.",
    },
    {
        "local_path": "outputs/final-adapter",
        "hf_folder": "mixed-r32-adapter",
        "desc": "LoRA r=32 adapter, 76k mixed data (Kaggle: 14.64). Load on top of merged-1.5b-r16.",
    },
    {
        "local_path": "models/refined-7000",
        "hf_folder": "refined-7000",
        "desc": "Full fine-tune checkpoint 7000, loss 0.308 (Kaggle: 16.26). Standalone model.",
    },
    {
        "local_path": "models/codegen-1.5b",
        "hf_folder": "codegen-1.5b",
        "desc": "Code-gen experiment model (Kaggle: 12.26). Standalone model.",
    },
]

# Create a README for the repo
readme = """# SVG Generation Model Weights

Model weights for the DL Spring 2026 Kaggle Competition — Text-to-SVG Generation.

**Team:** Ivan Aristy (NYU Tandon)
**Final Score:** 16.87/100 (6th place / 58 teams)
**Base Model:** Qwen/Qwen2.5-Coder-1.5B-Instruct

## Models

| Model | Type | Kaggle Score | Description |
|---|---|---|---|
| `componly-r32-adapter` | LoRA adapter | **16.87** | **Best model.** Load on `merged-1.5b-r16`. |
| `refined-7000` | Full model | 16.26 | Full fine-tune, loss 0.308 |
| `r16-3epoch` | LoRA adapter | 15.47 | First adapter, load on Qwen2.5-Coder-1.5B |
| `mixed-r32-adapter` | LoRA adapter | 14.64 | Mixed data experiment |
| `codegen-1.5b` | Full model | 12.26 | Code generation experiment |
| `merged-1.5b-r16` | Full model | — | Base model with r16 knowledge baked in |

## Usage (Best Model)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)

# Load merged base
tokenizer = AutoTokenizer.from_pretrained("kaleidoscopicwhether/svg-gen-weights", subfolder="merged-1.5b-r16")
model = AutoModelForCausalLM.from_pretrained(
    "kaleidoscopicwhether/svg-gen-weights", subfolder="merged-1.5b-r16",
    quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16,
)

# Load best adapter
model = PeftModel.from_pretrained(model, "kaleidoscopicwhether/svg-gen-weights", subfolder="componly-r32-adapter")
model.eval()

# Generate
prompt = "<|im_start|>system\\nOutput valid SVG code only.<|im_end|>\\n<|im_start|>user\\nA red circle<|im_end|>\\n<|im_start|>assistant\\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024, do_sample=False, repetition_penalty=1.1)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training Details

See [GitHub repo](https://github.com/ivanearisty/svg-gen) and paper in `report/main.pdf`.
"""

# Upload README first
with open("/tmp/hf_readme.md", "w") as f:
    f.write(readme)
api.upload_file(path_or_fileobj="/tmp/hf_readme.md", path_in_repo="README.md", repo_id=REPO)
print("Uploaded README.md")

# Upload each model
for m in MODELS:
    local = m["local_path"]
    hf_folder = m["hf_folder"]

    if not os.path.isdir(local):
        print(f"SKIP {local} — not found")
        continue

    files = os.listdir(local)
    print(f"\nUploading {hf_folder}/ ({len(files)} files from {local})...")

    for fname in files:
        fpath = os.path.join(local, fname)
        if not os.path.isfile(fpath):
            continue
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {fname} ({size_mb:.1f} MB)...", end="", flush=True)
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"{hf_folder}/{fname}",
            repo_id=REPO,
        )
        print(" done")

    print(f"  {hf_folder}/ complete")

print("\n=== ALL UPLOADS COMPLETE ===")
print(f"Repo: https://huggingface.co/{REPO}")
