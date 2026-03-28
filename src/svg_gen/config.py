"""Typed configuration for training, inference, and SVG constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SVGConstraints:
    """Competition-mandated SVG constraints."""

    canvas_width: int = 256
    canvas_height: int = 256
    max_svg_length: int = 16_000
    max_path_count: int = 256
    allowed_elements: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "svg", "g", "path", "rect", "circle", "ellipse",
            "line", "polyline", "polygon", "defs", "use",
            "symbol", "clipPath", "mask", "linearGradient",
            "radialGradient", "stop", "text", "tspan", "title",
            "desc", "style", "pattern", "marker", "filter",
        }),
    )
    disallowed_patterns: tuple[str, ...] = (
        "script", "onclick", "onload", "onmouseover",
        "animate", "foreignObject", "xlink:href",
    )


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA adapter hyperparameters."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    bias: Literal["none", "all", "lora_only"] = "none"


@dataclass(frozen=True)
class TrainingConfig:
    """Full training configuration."""

    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    max_seq_length: int = 2048

    # LoRA
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(r=32, lora_alpha=64, lora_dropout=0))

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"

    # Logging & saving
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 3
    eval_steps: int = 100

    # Data
    eval_size: float = 0.02
    max_train_samples: int | None = None

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "/vol/checkpoints"

    # SVG constraints
    svg: SVGConstraints = field(default_factory=SVGConstraints)


@dataclass(frozen=True)
class InferenceConfig:
    """Inference / generation configuration."""

    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    do_sample: bool = False


SYSTEM_PROMPT = (
    "You generate compact, valid SVG markup from user requests. "
    "Return only SVG code with a single root <svg> element. "
    "Keep the SVG under 16000 characters."
)
