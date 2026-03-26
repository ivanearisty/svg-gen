"""Download submission CSV from Modal volume to local machine."""

from __future__ import annotations

import os

import modal


def main(output_dir: str = "submissions") -> None:
    """Download submission.csv from Modal volume.

    Usage: `python -m svg_gen.download_results`
    """
    os.makedirs(output_dir, exist_ok=True)

    vol = modal.Volume.from_name("svg-gen-vol")
    remote_path = "outputs/submission.csv"
    local_path = os.path.join(output_dir, "submission.csv")

    print(f"Downloading {remote_path} -> {local_path}")
    data = b""
    for chunk in vol.read_file(remote_path):
        data += chunk

    with open(local_path, "wb") as f:
        f.write(data)

    print(f"Saved to {local_path} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
