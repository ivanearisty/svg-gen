#!/bin/bash
# Overnight inference queue — runs sequentially on kepler
# Each run ~5.5h, three runs = ~16.5h (finishes by ~noon tomorrow)

cd /home/kepler/WorkDir/ivan-training/svg-gen

echo "=== Run 1: rep_penalty=1.05 (already running, skip if CSV exists) ==="
if [ -f "results/submissions/greedy_rep1.05_tok1024.csv" ]; then
    lines=$(wc -l < "results/submissions/greedy_rep1.05_tok1024.csv")
    if [ "$lines" -gt 999 ]; then
        echo "Run 1 already complete ($lines lines), skipping"
    else
        echo "Run 1 incomplete ($lines lines), running..."
        uv run python scripts/run_inference_temp.py --temp 0.0 --rep-penalty 1.05
    fi
else
    echo "Run 1 not found, running..."
    uv run python scripts/run_inference_temp.py --temp 0.0 --rep-penalty 1.05
fi

echo ""
echo "=== Run 2: rep_penalty=1.15 ==="
uv run python scripts/run_inference_temp.py --temp 0.0 --rep-penalty 1.15

echo ""
echo "=== Run 3: max_tokens=1536 (default rep=1.1) ==="
uv run python scripts/run_inference_temp.py --temp 0.0 --rep-penalty 1.1 --max-new-tokens 1536

echo ""
echo "=== All runs complete ==="
echo "Results:"
ls -la results/submissions/greedy_*.csv
