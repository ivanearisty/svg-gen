"""Orchestrator: runs all ablation experiments sequentially.

Usage:
    nohup python3 scripts/orchestrate_ablations.py > logs/ablations.log 2>&1 &
"""

import os
import subprocess
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
os.chdir(_project_dir)

DONE_DIR = os.path.join(_project_dir, "results", "ablations")
LOG_DIR = os.path.join(_project_dir, "logs")

os.makedirs(DONE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Define all ablation experiments
ABLATIONS = [
    {
        "name": "baseline",
        "args": ["--name", "baseline"],
        "script": "scripts/run_ablation.py",
        "desc": "Full system (adapter + sys prompt + repair + rep=1.1 + 1024 tok)",
    },
    {
        "name": "base_model",
        "args": ["--name", "base_model", "--no-adapter"],
        "script": "scripts/run_ablation.py",
        "desc": "Base model zero-shot (no fine-tuning)",
    },
    {
        "name": "no_sys_prompt",
        "args": ["--name", "no_sys_prompt", "--no-system-prompt"],
        "script": "scripts/run_ablation.py",
        "desc": "No system prompt at inference",
    },
    {
        "name": "rep_1.0",
        "args": ["--name", "rep_1.0", "--rep-penalty", "1.0"],
        "script": "scripts/run_ablation.py",
        "desc": "No repetition penalty (rep=1.0)",
    },
    {
        "name": "max_tok_512",
        "args": ["--name", "max_tok_512", "--max-tokens", "512"],
        "script": "scripts/run_ablation.py",
        "desc": "Short generation (max_tokens=512)",
    },
    {
        "name": "max_tok_1536",
        "args": ["--name", "max_tok_1536", "--max-tokens", "1536"],
        "script": "scripts/run_ablation.py",
        "desc": "Long generation (max_tokens=1536)",
    },
    {
        "name": "no_repair",
        "args": ["--name", "no_repair", "--no-repair"],
        "script": "scripts/run_ablation.py",
        "desc": "No repair pipeline (raw extraction + fallback)",
    },
    {
        "name": "refined_ft",
        "args": ["--name", "refined_ft"],
        "script": "scripts/run_ablation_refined.py",
        "desc": "Refined full fine-tune model (checkpoint-7000)",
    },
]


def is_done(name: str) -> bool:
    return os.path.exists(os.path.join(DONE_DIR, f"{name}.done"))


def run_ablation(ablation: dict) -> None:
    name = ablation["name"]
    if is_done(name):
        print(f"[SKIP] {name} already done")
        return

    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] Starting: {name}")
    print(f"  {ablation['desc']}")
    print(f"{'='*60}\n", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(_project_dir, "src")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, ablation["script"]] + ablation["args"],
        env=env,
        cwd=_project_dir,
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n[{time.strftime('%H:%M:%S')}] Completed: {name} ({elapsed/60:.1f} min)")
    else:
        print(f"\n[{time.strftime('%H:%M:%S')}] FAILED: {name} (exit code {result.returncode})")


def print_summary() -> None:
    print(f"\n{'='*60}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Ablation':<24} | {'Score':>8} | {'Description'}")
    print(f"{'-'*24}-+-{'-'*8}-+-{'-'*40}")

    for abl in ABLATIONS:
        done_file = os.path.join(DONE_DIR, f"{abl['name']}.done")
        if os.path.exists(done_file):
            with open(done_file) as f:
                score = f.read().strip()
            print(f"{abl['name']:<24} | {score:>8} | {abl['desc']}")
        else:
            print(f"{abl['name']:<24} | {'MISSING':>8} | {abl['desc']}")


def main():
    print(f"=== ABLATION STUDY START: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Val set: data/val_ablation.csv (200 samples)")
    print(f"Results: {DONE_DIR}/")
    print(f"Python: {sys.executable}")
    print(f"Experiments: {len(ABLATIONS)}")

    for ablation in ABLATIONS:
        try:
            run_ablation(ablation)
        except Exception as e:
            print(f"ERROR running {ablation['name']}: {e}")

    print_summary()

    # Signal all done
    with open(os.path.join(DONE_DIR, "ALL_DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\n=== ALL ABLATIONS COMPLETE: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()
