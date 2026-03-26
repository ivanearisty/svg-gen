"""Modal app definition — shared image and volume configuration."""

from __future__ import annotations

import modal

app = modal.App("svg-gen")

# Persistent volume for checkpoints, data, and outputs
volume = modal.Volume.from_name("svg-gen-vol", create_if_missing=True)

# GPU image with all ML dependencies + local svg_gen package
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0",
        "transformers>=4.48.0",
        "peft>=0.14.0",
        "trl>=0.14.0",
        "accelerate>=1.3.0",
        "bitsandbytes>=0.45.0",
        "datasets>=3.2.0",
        "pandas>=2.2.0",
        "cairosvg>=2.7.0",
        "lxml>=5.0.0",
        "scipy>=1.14.0",
    )
    .apt_install("libcairo2-dev", "libffi-dev", "pkg-config")
    .add_local_python_source("svg_gen")
)

# Minimal image for non-GPU tasks (data upload, etc.)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas>=2.2.0")
    .add_local_python_source("svg_gen")
)

VOLUME_MOUNT = "/vol"
DATA_DIR = f"{VOLUME_MOUNT}/data"
CHECKPOINTS_DIR = f"{VOLUME_MOUNT}/checkpoints"
OUTPUTS_DIR = f"{VOLUME_MOUNT}/outputs"
