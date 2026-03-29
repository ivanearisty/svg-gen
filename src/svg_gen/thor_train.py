"""Full fine-tuning script for Thor (DGX Spark, 128GB unified memory).

Unlike kepler's QLoRA approach, this updates ALL model weights — no low-rank
bottleneck. This should break through the 0.34 loss ceiling if it's caused by
LoRA's limited capacity to reshape coordinate prediction behavior.

Usage:
    # Must run with video group for GPU access:
    sg video -c 'python -m svg_gen.thor_train'
    sg video -c 'python -m svg_gen.thor_train --model Qwen/Qwen2.5-Coder-3B-Instruct --epochs 2'
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

from svg_gen.data import curate_training_data, format_chat_prompt

SEED = 42


def main() -> None:  # noqa: PLR0915
    """Run full fine-tuning on Thor GPU."""
    parser = argparse.ArgumentParser(description="Full fine-tune SVG model on Thor")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-svg-tokens", type=int, default=1900)
    parser.add_argument("--train-csv", default="data/train.csv")
    parser.add_argument("--output-dir", default="outputs-thor")
    args = parser.parse_args()

    # --- Reproducibility ---
    random.seed(SEED)
    np.random.seed(SEED)  # noqa: NPY002
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {mem_gb:.1f} GB")
    print(f"Model: {args.model}")
    print(f"Mode: FULL FINE-TUNE (all parameters)")
    print(f"Training: {args.epochs} epochs, lr={args.lr}, warmup={args.warmup_ratio}, batch={args.batch_size}x{args.grad_accum}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    # --- Load & curate data ---
    print(f"\nLoading {args.train_csv}...")
    df = pd.read_csv(args.train_csv)
    df = curate_training_data(df, max_svg_tokens=args.max_svg_tokens, model_name=args.model)

    # --- Format as chat ---
    def format_example(row: dict[str, str]) -> dict[str, str]:
        return {"text": format_chat_prompt(str(row.get("prompt", "")), str(row.get("svg", "")))}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    splits = dataset.train_test_split(test_size=0.02, seed=SEED)
    train_ds = splits["train"]
    eval_ds = splits["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # --- Load model in bf16 (NO quantization — full precision weights) ---
    print(f"\nLoading {args.model} in bf16 (full precision)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = args.max_seq_length

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable:,} ({100 * trainable / total_params:.1f}%)")

    # Estimate memory usage
    param_mem = total_params * 2 / 1024**3  # bf16 = 2 bytes
    optim_mem = total_params * 8 / 1024**3  # AdamW = 8 bytes per param (2 momentum + variance)
    grad_mem = total_params * 2 / 1024**3   # bf16 gradients
    print(f"Estimated memory: params={param_mem:.1f}GB + optim={optim_mem:.1f}GB + grads={grad_mem:.1f}GB = {param_mem + optim_mem + grad_mem:.1f}GB")

    # --- Training ---
    output_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")

    # Save frequently — Thor crashes are common, don't lose progress
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    save_steps = 500
    eval_steps = 500
    print(f"Steps per epoch: {steps_per_epoch}, save/eval every {save_steps} steps")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=int(args.warmup_ratio * steps_per_epoch * args.epochs),
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="adamw_torch",  # Full AdamW — no paged/8bit needed with 128GB
        bf16=True,
        # logging_dir=log_dir,  # deprecated in trl 0.29+
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
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

    print(f"\nStarting FULL FINE-TUNE at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
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
    print(f"Final train loss: {result.training_loss:.4f}")

    # --- Final eval ---
    eval_result = trainer.evaluate()
    print(f"Final eval loss: {eval_result['eval_loss']:.4f}")

    # --- Memory report ---
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {peak_mem:.1f}GB")

    # --- Save full model ---
    model_path = os.path.join(args.output_dir, "final-model")
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Full model saved to {model_path}")

    # --- Save training summary ---
    summary = {
        "model": args.model,
        "mode": "full_fine_tune",
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds),
        "train_loss": result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "training_hours": elapsed / 3600,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
    }
    import json
    summary_path = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
