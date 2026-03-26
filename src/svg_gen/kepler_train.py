"""Standalone training script for kepler (RTX 2000 Ada 16GB).

No Modal dependency. Uses Unsloth for memory-efficient training.

Usage:
    python -m svg_gen.kepler_train
    python -m svg_gen.kepler_train --model Qwen/Qwen2.5-Coder-3B-Instruct --lora-r 32
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
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from svg_gen.config import SYSTEM_PROMPT
from svg_gen.data import curate_training_data

SEED = 42


def main() -> None:  # noqa: PLR0915
    """Run training on local GPU."""
    parser = argparse.ArgumentParser(description="Train SVG generation model on kepler")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-svg-tokens", type=int, default=1900)
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    # --- Reproducibility ---
    random.seed(SEED)
    np.random.seed(SEED)  # noqa: NPY002
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Model: {args.model}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Training: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}x{args.grad_accum}")

    # --- Load & curate data ---
    print(f"\nLoading {args.train_csv}...")
    df = pd.read_csv(args.train_csv)
    df = curate_training_data(df, max_svg_tokens=args.max_svg_tokens, model_name=args.model)

    # --- Format as chat ---
    def format_example(row: dict[str, str]) -> dict[str, str]:
        svg = str(row.get("svg", ""))
        prompt = str(row.get("prompt", ""))
        return {
            "text": (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{svg}<|im_end|>"
            ),
        }

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    splits = dataset.train_test_split(test_size=0.02, seed=SEED)
    train_ds = splits["train"]
    eval_ds = splits["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # --- Model + LoRA via Unsloth ---
    print(f"\nLoading {args.model} with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # --- Training ---
    output_dir = os.path.join(args.output_dir, "checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        save_total_limit=2,
        gradient_checkpointing=True,
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    print(f"\nStarting training at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    t0 = time.time()

    # Resume from checkpoint if available
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

    # --- Save ---
    adapter_path = os.path.join(args.output_dir, "final-adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
