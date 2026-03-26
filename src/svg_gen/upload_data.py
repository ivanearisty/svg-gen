"""Upload local CSV data to Modal volume."""

from __future__ import annotations

import os
from pathlib import Path

import modal

from svg_gen.modal_app import DATA_DIR, VOLUME_MOUNT, app, base_image, volume


@app.function(image=base_image, volumes={VOLUME_MOUNT: volume})
def _create_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


@app.local_entrypoint()
def main(data_dir: str = "data") -> None:
    """Upload CSVs from local `data/` to Modal volume.

    Usage: `modal run src/svg_gen/upload_data.py --data-dir ./data`
    """
    _create_dirs.remote()

    vol = modal.Volume.from_name("svg-gen-vol")
    data_path = Path(data_dir)

    with vol.batch_upload() as batch:
        for csv_file in data_path.glob("*.csv"):
            remote_path = f"data/{csv_file.name}"
            print(f"Uploading {csv_file} -> {remote_path}")
            batch.put_file(csv_file, remote_path)

    print("Upload complete!")
