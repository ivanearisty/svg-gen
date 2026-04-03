"""Fine-tune for code-generation SVG output on Thor.

Teaches the model to output Python code that builds SVGs using the SVGBuilder API.

Usage (inside Docker):
    python3 -m svg_gen.thor_train_codegen
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

SEED = 42

SYSTEM_PROMPT = '''You write Python code to create SVG images. Use this API:

svg = create_svg(200, 200)
svg.background("white")
svg.rect(x, y, width, height, fill="blue")
svg.circle(cx, cy, r, fill="red")
svg.ellipse(cx, cy, rx, ry, fill="green")
svg.line(x1, y1, x2, y2, stroke="black", stroke_width=2)
svg.polygon([(x1,y1), (x2,y2), ...], fill="yellow")
svg.text(x, y, "text", font_size=16, fill="black")
svg.path("M 10 10 L 90 90 Z", fill="none", stroke="red")
svg.arrow(x, y, width, height, fill="orange", direction="right")
svg.star(cx, cy, r, points=5, fill="gold")

Canvas is 200x200. Center is (100,100). Return ONLY Python code.
Start with: svg = create_svg(200, 200)'''


def format_chat(prompt: str, code: str) -> str:
    """Format a (prompt, code) pair into chat template."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{code}<|im_end|>"
    )


def main() -> None:  # noqa: PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs-thor-refined/checkpoints/checkpoint-7000")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--train-csv", default="data/train_codegen.csv")
    parser.add_argument("--output-dir", default="outputs-thor-codegen")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)  # noqa: NPY002
    torch.manual_seed(SEED)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mode: CODE-GEN FINE-TUNE")
    print(f"Training: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}x{args.grad_accum}")

    # Load data
    df = pd.read_csv(args.train_csv)
    print(f"Loaded {len(df)} code-gen training samples")

    # Filter by token length
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = args.max_seq_length

    def format_example(row):
        return {"text": format_chat(str(row["prompt"]), str(row["code"]))}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # Filter by length
    def fits_in_context(example):
        return len(tokenizer.encode(example["text"], add_special_tokens=False)) <= args.max_seq_length - 50

    before = len(dataset)
    dataset = dataset.filter(fits_in_context)
    print(f"After context filter: {len(dataset)}/{before} samples")

    splits = dataset.train_test_split(test_size=0.02, seed=SEED)
    train_ds = splits["train"]
    eval_ds = splits["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Load model
    print(f"\nLoading {args.model} in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda:0")
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} (100% trainable)")

    # Training
    output_dir = os.path.join(args.output_dir, "checkpoints")
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    save_steps = max(100, steps_per_epoch // 5)

    print(f"Steps/epoch: {steps_per_epoch}, save every {save_steps}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=int(0.1 * steps_per_epoch * args.epochs),
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        label_smoothing_factor=0.1,
        logging_steps=10,
        eval_strategy="no",
        save_steps=save_steps,
        save_total_limit=5,
        gradient_checkpointing=True,
        report_to="tensorboard",
        seed=SEED,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    print(f"\nStarting CODE-GEN FINE-TUNE at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    t0 = time.time()

    last_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(output_dir, max(checkpoints, key=lambda x: int(x.split("-")[1])))
            print(f"Resuming from {last_checkpoint}")

    result = trainer.train(resume_from_checkpoint=last_checkpoint)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 3600:.1f}h")
    print(f"Final loss: {result.training_loss:.4f}")

    model_path = os.path.join(args.output_dir, "final-model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    import json
    summary = {
        "model": args.model,
        "mode": "codegen_full_fine_tune",
        "lr": args.lr,
        "epochs": args.epochs,
        "train_samples": len(train_ds),
        "train_loss": result.training_loss,
        "training_hours": elapsed / 3600,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
