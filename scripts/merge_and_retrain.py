"""Merge existing LoRA adapter into base model, then apply a new higher-rank LoRA and train.

This 'stacks' training: the r=16 knowledge gets baked into the weights permanently,
then a new r=32/r=64 adapter adds fresh capacity on top.

Usage:
    uv run python merge_and_retrain.py
    uv run python merge_and_retrain.py --new-r 64 --epochs 3
    uv run python merge_and_retrain.py --new-r 32 --train-csv data/train_expanded.csv
"""

import argparse
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel

from svg_gen.data import curate_training_data, format_chat_prompt


SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Merge r=16 adapter and retrain with higher rank")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-path", default="filtered-r16-3epoch")
    parser.add_argument("--merged-path", default="models/merged-1.5b-r16")
    parser.add_argument("--new-r", type=int, default=32)
    parser.add_argument("--new-alpha", type=int, default=64)
    parser.add_argument("--train-csv", default="data/train_expanded.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step if already done")
    args = parser.parse_args()

    # ─── Step 1: Merge existing adapter into base model ───
    if not args.skip_merge or not os.path.exists(args.merged_path):
        print("=" * 60)
        print("Step 1: Merging r=16 adapter into base model...")
        print("=" * 60)

        print(f"Loading {args.base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        # Load in fp16 for merging (need full precision, not 4-bit)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="cpu",  # merge on CPU to save GPU memory
        )

        print(f"Loading adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)

        print("Merging adapter into base weights...")
        model = model.merge_and_unload()

        print(f"Saving merged model to {args.merged_path}...")
        os.makedirs(args.merged_path, exist_ok=True)
        model.save_pretrained(args.merged_path)
        tokenizer.save_pretrained(args.merged_path)
        print(f"Merged model saved ({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

        # Free memory
        del model, base_model
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    else:
        print(f"Skipping merge — using existing merged model at {args.merged_path}")

    # ─── Step 2: Load merged model with Unsloth + new LoRA ───
    print()
    print("=" * 60)
    print(f"Step 2: Loading merged model + applying new LoRA (r={args.new_r})...")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.merged_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.new_r,
        lora_alpha=args.new_alpha,
        lora_dropout=0,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"New LoRA: r={args.new_r}, alpha={args.new_alpha}")
    print(f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.0f}M ({100 * trainable / total:.1f}%)")

    # ─── Step 3: Prepare data ───
    print()
    print("=" * 60)
    print("Step 3: Preparing training data...")
    print("=" * 60)

    import numpy as np
    import pandas as pd
    from datasets import Dataset

    df = pd.read_csv(args.train_csv)
    print(f"Loaded {len(df)} samples from {args.train_csv}")

    # The expanded CSV already has preprocessed SVGs — just filter by token count
    df = curate_training_data(df, max_svg_tokens=1900, model_name=args.base_model)

    def format_example(row: dict[str, str]) -> dict[str, str]:
        return {"text": format_chat_prompt(str(row.get("prompt", "")), str(row.get("svg", "")))}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    splits = dataset.train_test_split(test_size=0.02, seed=SEED)
    train_ds = splits["train"]
    eval_ds = splits["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # ─── Step 4: Train ───
    print()
    print("=" * 60)
    print("Step 4: Training...")
    print("=" * 60)

    from trl import SFTTrainer
    from transformers import TrainingArguments

    output_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")

    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    save_steps = max(100, steps_per_epoch // 5)
    eval_steps = save_steps
    print(f"Steps per epoch: {steps_per_epoch}, save/eval every {save_steps} steps")

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
        logging_dir=log_dir,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="tensorboard",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        max_seq_length=args.max_seq_length,
        args=training_args,
    )

    print(f"\nStarting training at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    t0 = time.time()

    # Resume from checkpoint if available
    last_checkpoint = None
    if os.path.isdir(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(output_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
            print(f"Resuming from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 3600:.1f}h")

    # ─── Step 5: Save ───
    adapter_path = os.path.join(args.output_dir, "final-adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"New adapter saved to {adapter_path}")
    print(f"Base model for inference: {args.merged_path}")


if __name__ == "__main__":
    main()
