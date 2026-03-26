"""Submit CSV to Kaggle competition."""

from __future__ import annotations

import subprocess
import sys

COMPETITION_SLUG = "dl-spring-2026-svg-generation"


def main(csv_path: str = "submissions/submission.csv", message: str = "auto-submit") -> None:
    """Submit a CSV to the Kaggle competition.

    Usage: `python -m svg_gen.submit [--csv-path path] [--message msg]`
    """
    print(f"Submitting {csv_path} to {COMPETITION_SLUG}...")

    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "kaggle",
            "competitions",
            "submit",
            "-c", COMPETITION_SLUG,
            "-f", csv_path,
            "-m", message,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(result.stdout)

    # Check submissions
    result2 = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "kaggle",
            "competitions",
            "submissions",
            "-c", COMPETITION_SLUG,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(result2.stdout)


if __name__ == "__main__":
    main()
