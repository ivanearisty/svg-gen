#!/bin/bash
# Run all ablation experiments sequentially.
# Each writes a .done file when finished so cron can detect completion.
#
# Usage: nohup bash scripts/run_all_ablations.sh > logs/ablations.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
DONE_DIR="results/ablations"
LOG="logs/ablations.log"

mkdir -p results/ablations logs

echo "=== ABLATION STUDY START: $(date) ==="
echo "Val set: data/val_ablation.csv (200 samples)"
echo "Results: results/ablations/"
echo ""

# 1. Baseline (full system — adapter + system prompt + repair + rep=1.1 + 1024 tokens)
if [ ! -f "$DONE_DIR/baseline.done" ]; then
    echo "[$(date +%H:%M)] Starting: baseline (full system)"
    python3 scripts/run_ablation.py --name baseline
    echo "[$(date +%H:%M)] Completed: baseline"
else
    echo "[SKIP] baseline already done"
fi

# 2. Base model — no fine-tuning
if [ ! -f "$DONE_DIR/base_model.done" ]; then
    echo "[$(date +%H:%M)] Starting: base_model (no fine-tuning)"
    python3 scripts/run_ablation.py --name base_model --no-adapter
    echo "[$(date +%H:%M)] Completed: base_model"
else
    echo "[SKIP] base_model already done"
fi

# 3. No system prompt
if [ ! -f "$DONE_DIR/no_sys_prompt.done" ]; then
    echo "[$(date +%H:%M)] Starting: no_sys_prompt"
    python3 scripts/run_ablation.py --name no_sys_prompt --no-system-prompt
    echo "[$(date +%H:%M)] Completed: no_sys_prompt"
else
    echo "[SKIP] no_sys_prompt already done"
fi

# 4. No repetition penalty (rep=1.0)
if [ ! -f "$DONE_DIR/rep_1.0.done" ]; then
    echo "[$(date +%H:%M)] Starting: rep_1.0 (no repetition penalty)"
    python3 scripts/run_ablation.py --name rep_1.0 --rep-penalty 1.0
    echo "[$(date +%H:%M)] Completed: rep_1.0"
else
    echo "[SKIP] rep_1.0 already done"
fi

# 5. Max tokens = 512
if [ ! -f "$DONE_DIR/max_tok_512.done" ]; then
    echo "[$(date +%H:%M)] Starting: max_tok_512 (short generation)"
    python3 scripts/run_ablation.py --name max_tok_512 --max-tokens 512
    echo "[$(date +%H:%M)] Completed: max_tok_512"
else
    echo "[SKIP] max_tok_512 already done"
fi

# 6. Max tokens = 1536
if [ ! -f "$DONE_DIR/max_tok_1536.done" ]; then
    echo "[$(date +%H:%M)] Starting: max_tok_1536 (long generation)"
    python3 scripts/run_ablation.py --name max_tok_1536 --max-tokens 1536
    echo "[$(date +%H:%M)] Completed: max_tok_1536"
else
    echo "[SKIP] max_tok_1536 already done"
fi

# 7. No repair pipeline
if [ ! -f "$DONE_DIR/no_repair.done" ]; then
    echo "[$(date +%H:%M)] Starting: no_repair (skip repair pipeline)"
    python3 scripts/run_ablation.py --name no_repair --no-repair
    echo "[$(date +%H:%M)] Completed: no_repair"
else
    echo "[SKIP] no_repair already done"
fi

# 8. Refined model (full fine-tune checkpoint) — load differently
if [ ! -f "$DONE_DIR/refined_ft.done" ]; then
    echo "[$(date +%H:%M)] Starting: refined_ft (full fine-tune model)"
    python3 scripts/run_ablation_refined.py --name refined_ft
    echo "[$(date +%H:%M)] Completed: refined_ft"
else
    echo "[SKIP] refined_ft already done"
fi

echo ""
echo "=== ALL ABLATIONS COMPLETE: $(date) ==="
echo ""

# Print summary table
echo "=== RESULTS SUMMARY ==="
echo "Ablation                | Score"
echo "------------------------|--------"
for f in "$DONE_DIR"/*.done; do
    name=$(basename "$f" .done)
    score=$(cat "$f")
    printf "%-24s| %s\n" "$name" "$score"
done

# Signal all done
touch "$DONE_DIR/ALL_DONE"
echo ""
echo "All results in: $DONE_DIR/"
