"""Quick experiment runner — train on small subset, generate samples, score them."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

from svg_gen.config import SVGConstraints
from svg_gen.modal_app import (
    DATA_DIR,
    VOLUME_MOUNT,
    app,
    gpu_image,
    volume,
)

EXPERIMENT_DIR = "/vol/experiments"


@dataclass
class ExperimentConfig:
    """Configuration for a quick experiment run."""

    name: str = "default"

    # Training
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    max_train_samples: int = 2000
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    max_seq_length: int = 2048

    # Data preprocessing
    round_to_int: bool = False  # Round coords to integers instead of 1 decimal

    # System prompt override (None = use default)
    system_prompt: str | None = None

    # Inference
    num_test_samples: int = 20
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = True
    num_beams: int = 1
    no_repeat_ngram_size: int = 0

    seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a quick experiment."""

    name: str
    config: dict[str, object]

    # Training metrics
    train_loss: float
    train_time_s: float
    train_steps: int

    # Generation metrics
    num_generated: int
    num_valid: int
    validity_rate: float
    avg_svg_length: float
    generation_time_s: float

    # Sample outputs (id, prompt, svg)
    samples: list[dict[str, str]] = field(default_factory=list)


@app.function(
    image=gpu_image,
    gpu="A100-40GB",
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: volume},
)
def run_experiment(exp: ExperimentConfig) -> ExperimentResult:  # noqa: PLR0915
    """Run a full experiment: train on subset, generate test SVGs, return metrics."""
    import re

    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer

    from svg_gen.config import SYSTEM_PROMPT
    from svg_gen.data import (
        curate_training_data,
        extract_svg,
        fallback_svg,
        is_valid_svg,
        normalize_viewbox,
        repair_svg,
    )

    system_prompt = exp.system_prompt or SYSTEM_PROMPT
    print(f"=== Experiment: {exp.name} ===", flush=True)
    print(f"Model: {exp.model_name}, r={exp.lora_r}, lr={exp.learning_rate}", flush=True)

    # --- Reproducibility ---
    torch.manual_seed(exp.seed)
    np.random.seed(exp.seed)  # noqa: NPY002

    # --- Load & preprocess data ---
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    df = curate_training_data(df)
    df = df.sample(n=min(exp.max_train_samples, len(df)), random_state=exp.seed)
    print(f"Training on {len(df)} samples", flush=True)

    # Optional: round to integers
    if exp.round_to_int:
        int_re = re.compile(r"(\d+)\.\d+")
        df["svg"] = df["svg"].apply(lambda s: int_re.sub(r"\1", s))

    def format_example(row: dict[str, str]) -> dict[str, str]:
        svg = str(row.get("svg", ""))
        prompt = str(row.get("prompt", ""))
        return {
            "text": (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{svg}<|im_end|>"
            ),
        }

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # --- Model + QLoRA ---
    bnb_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(exp.model_name)
    tokenizer.model_max_length = exp.max_seq_length
    model = AutoModelForCausalLM.from_pretrained(
        exp.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)  # type: ignore[assignment]

    lora_config = PeftLoraConfig(
        r=exp.lora_r,
        lora_alpha=exp.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Train ---
    exp_output_dir = os.path.join(EXPERIMENT_DIR, exp.name)
    os.makedirs(exp_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=exp_output_dir,
        num_train_epochs=exp.num_train_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=exp.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=True,
        report_to="none",
        seed=exp.seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    train_t0 = time.time()
    result = trainer.train()
    train_time = time.time() - train_t0
    train_loss = result.training_loss
    train_steps = result.global_step
    print(f"Training done: loss={train_loss:.4f}, steps={train_steps}, time={train_time:.0f}s", flush=True)

    # --- Generate test SVGs ---
    model.eval()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_df = test_df.head(exp.num_test_samples)

    constraints = SVGConstraints()
    samples: list[dict[str, str]] = []
    valid_count = 0
    svg_lengths: list[int] = []

    gen_t0 = time.time()
    for _, row in test_df.iterrows():
        prompt = str(row.get("prompt", row.get("text", "")))
        sample_id = str(row.get("id", ""))

        chat_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        gen_kwargs: dict[str, object] = {
            "max_new_tokens": 1536,
            "repetition_penalty": exp.repetition_penalty,
        }
        if exp.num_beams > 1:
            gen_kwargs["num_beams"] = exp.num_beams
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = exp.do_sample
            gen_kwargs["temperature"] = exp.temperature
            gen_kwargs["top_p"] = exp.top_p

        if exp.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = exp.no_repeat_ngram_size

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        decoded = str(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        svg = extract_svg(decoded)

        if svg and not is_valid_svg(svg, constraints):
            svg = repair_svg(svg)

        valid = is_valid_svg(svg, constraints)
        if valid:
            valid_count += 1
            svg = normalize_viewbox(svg)
        else:
            svg = fallback_svg()

        svg_lengths.append(len(svg))
        samples.append({"id": sample_id, "prompt": prompt, "svg": svg, "valid": str(valid)})

    gen_time = time.time() - gen_t0
    validity_rate = valid_count / len(test_df) if len(test_df) > 0 else 0
    avg_len = sum(svg_lengths) / len(svg_lengths) if svg_lengths else 0

    print(f"Generation: {valid_count}/{len(test_df)} valid ({validity_rate:.0%}), avg_len={avg_len:.0f}", flush=True)

    # Save results to volume
    result_data = ExperimentResult(
        name=exp.name,
        config=asdict(exp),
        train_loss=train_loss,
        train_time_s=train_time,
        train_steps=train_steps,
        num_generated=len(test_df),
        num_valid=valid_count,
        validity_rate=validity_rate,
        avg_svg_length=avg_len,
        generation_time_s=gen_time,
        samples=samples,
    )

    results_path = os.path.join(exp_output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(asdict(result_data), f, indent=2)
    volume.commit()

    return result_data


@app.local_entrypoint()
def main(
    name: str = "default",
    lora_r: int = 16,
    lr: float = 2e-4,
    round_int: bool = False,
    temperature: float = 0.7,
    num_beams: int = 1,
    no_repeat_ngram: int = 0,
    samples: int = 2000,
    test_samples: int = 20,
) -> None:
    """Run a quick experiment.

    Usage: `modal run src/svg_gen/experiment.py --name exp1-r32 --lora-r 32`
    """
    exp = ExperimentConfig(
        name=name,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        learning_rate=lr,
        round_to_int=round_int,
        temperature=temperature,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram,
        max_train_samples=samples,
        num_test_samples=test_samples,
        do_sample=num_beams <= 1,
    )

    print(f"Launching experiment: {name}")
    result = run_experiment.remote(exp)

    print(f"\n{'=' * 60}")
    print(f"Experiment: {result.name}")
    print(f"{'=' * 60}")
    print(f"Train loss:     {result.train_loss:.4f}")
    print(f"Train time:     {result.train_time_s:.0f}s ({result.train_steps} steps)")
    print(f"Validity rate:  {result.num_valid}/{result.num_generated} ({result.validity_rate:.0%})")
    print(f"Avg SVG length: {result.avg_svg_length:.0f} chars")
    print(f"Gen time:       {result.generation_time_s:.0f}s")
    print()

    # Show first 3 samples
    for s in result.samples[:3]:
        print(f"--- {s['id']} (valid={s['valid']}) ---")
        print(f"Prompt: {s['prompt'][:80]}")
        print(f"SVG:    {s['svg'][:120]}...")
        print()
