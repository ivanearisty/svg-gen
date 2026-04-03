"""Create a fixed validation split for ablation studies.

Samples 200 prompts from train.csv, stratified by SVG token length
(proxy for complexity). Saves both the prompts (for inference) and
full rows (for scoring against ground truth).
"""

import pandas as pd
import numpy as np

SEED = 42
N_SAMPLES = 200
INPUT = "data/train.csv"
OUTPUT = "data/val_ablation.csv"

df = pd.read_csv(INPUT)
print(f"Loaded {len(df)} samples from {INPUT}")

# Stratify by SVG length (proxy for complexity)
df["_svg_len"] = df["svg"].str.len()
df["_complexity_bin"] = pd.qcut(df["_svg_len"], q=5, labels=["xs", "s", "m", "l", "xl"])

# Sample proportionally from each bin
val = df.groupby("_complexity_bin", group_keys=False).apply(
    lambda x: x.sample(n=max(1, int(N_SAMPLES * len(x) / len(df))), random_state=SEED),
    include_groups=True,
)

# Ensure exactly N_SAMPLES
if len(val) > N_SAMPLES:
    val = val.sample(n=N_SAMPLES, random_state=SEED)
elif len(val) < N_SAMPLES:
    remaining = df[~df.index.isin(val.index)].sample(n=N_SAMPLES - len(val), random_state=SEED)
    val = pd.concat([val, remaining])

val = val.drop(columns=["_svg_len", "_complexity_bin"]).reset_index(drop=True)
val.to_csv(OUTPUT, index=False)

# Print stats
svg_lens = val["svg"].str.len()
print(f"Saved {len(val)} samples to {OUTPUT}")
print(f"SVG length: min={svg_lens.min()}, median={svg_lens.median():.0f}, max={svg_lens.max()}")
print(f"Prompt examples:")
for i in range(min(5, len(val))):
    print(f"  {val.iloc[i]['prompt'][:80]}...")
