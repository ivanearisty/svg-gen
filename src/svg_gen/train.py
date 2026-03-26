"""LoRA fine-tuning on Modal GPU."""

from __future__ import annotations

import os

from svg_gen.config import TrainingConfig
from svg_gen.modal_app import (
    DATA_DIR,
    VOLUME_MOUNT,
    app,
    gpu_image,
    volume,
)


@app.function(
    image=gpu_image,
    gpu="A100-40GB",
    timeout=10 * 3600,
    volumes={VOLUME_MOUNT: volume},
)
def train(config: TrainingConfig | None = None) -> str:  # noqa: PLR0915
    """Run LoRA fine-tuning on the competition dataset.

    Returns the path to the saved adapter.
    """
    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    if config is None:
        config = TrainingConfig()

    # --- Reproducibility ---
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)  # noqa: NPY002

    # --- Load & curate data ---
    train_csv = os.path.join(DATA_DIR, "train.csv")
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} training samples")

    from svg_gen.data import curate_training_data
    df = curate_training_data(df)

    if config.max_train_samples is not None:
        df = df.sample(n=min(config.max_train_samples, len(df)), random_state=config.seed)
        print(f"Subsampled to {len(df)} samples")

    # --- Format data ---
    from svg_gen.config import SYSTEM_PROMPT

    def format_example(row: dict[str, str]) -> dict[str, str]:
        svg = str(row.get("svg", ""))  # Already cleaned by curate_training_data
        prompt = str(row.get("prompt", ""))
        text = (
            "<|im_start|>system\n"
            f"{SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{svg}<|im_end|>"
        )
        return {"text": text}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    splits = dataset.train_test_split(test_size=config.eval_size, seed=config.seed)
    train_ds = splits["train"]
    eval_ds = splits["test"]
    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # --- Model + QLoRA ---
    bnb_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.model_max_length = config.max_seq_length
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)  # type: ignore[assignment]

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=list(config.lora.target_modules),
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        bf16=True,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        gradient_checkpointing=True,
        report_to="none",
        seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    # Resume from checkpoint if one exists
    last_checkpoint = None
    if os.path.isdir(config.output_dir):
        checkpoints = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(config.output_dir, max(checkpoints, key=lambda x: int(x.split("-")[1])))
            print(f"Resuming from checkpoint: {last_checkpoint}")

    result = trainer.train(resume_from_checkpoint=last_checkpoint)
    print(f"Training complete. Loss: {result.training_loss:.4f}")

    # --- Save ---
    adapter_path = os.path.join(config.output_dir, "final-adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    volume.commit()

    print(f"Adapter saved to {adapter_path}")
    return adapter_path


@app.local_entrypoint()
def main(
    max_samples: int | None = None,
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    grad_norm: float = 0.3,
    output_dir: str = "/vol/checkpoints",
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
) -> None:
    """CLI entrypoint: `modal run src/svg_gen/train.py [--max-samples 1000]`."""
    from svg_gen.config import LoRAConfig

    effective_alpha = lora_alpha if lora_alpha is not None else lora_r * 2
    config = TrainingConfig(
        model_name=model_name,
        max_train_samples=max_samples,
        num_train_epochs=epochs,
        learning_rate=lr,
        max_grad_norm=grad_norm,
        output_dir=output_dir,
        lora=LoRAConfig(r=lora_r, lora_alpha=effective_alpha, lora_dropout=lora_dropout),
    )
    print(f"Starting training: {config.model_name}, epochs={config.num_train_epochs}, lr={config.learning_rate}")
    adapter_path = train.remote(config)
    print(f"Done! Adapter at: {adapter_path}")
