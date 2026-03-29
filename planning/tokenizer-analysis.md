# Tokenizer & Rounding Analysis — SVG Generation Competition

**Date:** March 27, 2026
**Goal:** Find the optimal model + preprocessing combination for SVG coordinate efficiency

---

## The Problem

SVG paths are sequences of commands and coordinates:
```
M 156.3 42.8 C 60.5 30.2 80.7 40.3 100.0 50.5 ...
```

LLM tokenizers split these differently. Some treat multi-digit numbers as single tokens, others split every digit. Since SVG paths are 80%+ coordinate numbers by token count, this determines how much visual information fits in the model's context window.

---

## Experiment: Tokenize the Same SVG Path With 8 Models

### Test input
A representative SVG path with 30 coordinate values (typical of a medium-complexity icon):
```
M 50.3 20.1 C 60.5 30.2 80.7 40.3 100.0 50.5 C 120.3 60.7 140.5 80.2
160.3 100.0 L 180.5 120.3 C 170.2 140.5 150.3 160.7 130.5 180.2
L 100.0 190.5 C 80.3 180.2 60.5 160.3 40.7 140.5 C 20.2 120.3
10.5 100.7 20.3 80.5 C 30.2 60.3 40.5 40.7 50.3 20.1 Z
```

Four variants tested:
- **Float (1-dec):** Current preprocessing — `156.3`
- **Integer:** Round to nearest int — `156`
- **Relative:** Delta from last point — `c 10.2 10.1 30.4 20.2 ...`
- **Rel+Int:** Both combined — `c 10 10 30 20 ...`

### Results

| Model | Year | Params | Quality | Path(float) | Path(int) | Path(rel) | Rel+Int | `156.3` split |
|---|---|---|---|---|---|---|---|---|
| GPT-2-XL | 2019 | 1.5B | 3/10 | **136** | **52** | — | **71** | `[156][.][3]` (3t) |
| Pythia-1.4B | 2023 | 1.4B | 4/10 | **136** | **52** | — | **71** | `[156][.][3]` (3t) |
| **Pythia-2.8B** | 2023 | 2.8B | 5/10 | **136** | **52** | — | **71** | `[156][.][3]` (3t) |
| Qwen2.5-Coder-0.5B | 2024 | 0.5B | 3/10 | 240 | 156 | 219 | 136 | `[1][5][6][.][3]` (5t) |
| Qwen2.5-Coder-1.5B | 2024 | 1.5B | 6/10 | 240 | 156 | 219 | 136 | `[1][5][6][.][3]` (5t) |
| **Qwen2.5-Coder-3B** | 2024 | 3B | 8/10 | 240 | 156 | 219 | 136 | `[1][5][6][.][3]` (5t) |
| StarCoder2-3B | 2024 | 3B | 6/10 | 240 | 156 | 219 | 136 | `[1][5][6][.][3]` (5t) |
| Phi-3.5-mini | 2024 | 3.8B | 9/10 | 240 | 156 | 219 | 136 | `[][1][5][6][.][3]` (6t) |
| Mistral-7B | 2024 | 7B | — | 240 | 156 | 219 | 136 | `[][1][5][6][.][3]` (6t) |

Quality rating = general model capability pre-fine-tuning (instruction following, code, reasoning).

Training time estimates on RTX 2000 Ada (16GB), batch_size=1, per epoch of 46k samples:
- 0.5B: ~3h | 1.5B: ~8h | 2.8B: ~14h | 3B: ~18h | 3.8B: ~22h

### Key Findings

**1. Two tokenizer families exist:**
- **GPT-2/Pythia (GPT-NeoX tokenizer):** Multi-digit number tokens. `156` = 1 token, `20` = 1 token.
- **Everything else (Qwen, Mistral, Phi, StarCoder):** Per-digit splitting. `156` = 3 tokens, `20` = 2 tokens.

**2. GPT-2/Pythia is 1.8x more efficient on float SVG paths** (136 vs 240 tokens). On integer paths, the gap widens to **3x** (52 vs 156 tokens).

**3. Qwen3.5-2B does not exist.** The starter notebook hallucinated it. All Qwen3/Qwen3.5 model IDs return 404 on HuggingFace.

**4. Every modern code model (Qwen, StarCoder2, Phi, Mistral) uses per-digit number tokenization.** This is not a Qwen-specific bug — it's a design choice in recent BPE tokenizers. Only the older GPT-2/GPT-NeoX vocabulary retained multi-digit number tokens.

**5. Rel+Int preprocessing almost closes the gap:**
- Qwen with rel+int: 136 tokens
- Pythia with raw floats: 136 tokens
- They converge because relative deltas are small numbers (1-2 digits), where per-digit splitting hurts less.

---

## Rounding Analysis: Does 156 vs 156.3 Actually Matter?

### The Math

The training data uses `viewBox="0 0 200 200"`. The competition renders at 256x256 pixels.

**Scale factor: 1 viewBox unit = 256/200 = 1.28 pixels**

| Precision | Example | Max error | Pixel error | Values per axis |
|---|---|---|---|---|
| 1 decimal | 156.3 | ±0.05 units | ±0.064 px | 2,001 |
| **Integer** | **156** | **±0.5 units** | **±0.64 px** | **201** |
| Half-grid (0-99) | 78 | ±1.0 units | ±1.28 px | 100 |

### Why ±0.64 Pixels Is Invisible to the Scoring Metric

**SSIM (70% of visual fidelity score):**
- Uses an 11x11 pixel Gaussian-weighted window
- A sub-pixel shift gets averaged out within the window
- SSIM difference between 0.0px and 0.64px error: effectively 0

**EdgeF1 (30% of visual fidelity score):**
- Uses Canny edge detection with sigma=1.0
- Then dilates edges by 1 pixel before comparing (from `score.py`: `binary_dilation(iterations=1)`)
- A 0.64px shift is well within the 1px dilation tolerance
- Edge correspondence will be identical

**Tree Edit Distance (12% of total score):**
- Compares SVG element structure, not coordinate values
- Integer vs float coordinates make zero difference

**Compactness (3% of total score):**
- `exp(-|log((len_pred+50)/(len_ref+50))|)`
- Integer coordinates are SHORTER, so compactness improves

### Why the Earlier Integer Experiment Showed Higher CE Loss

The Day 1 experiment trained with integer coordinates and measured higher cross-entropy loss (0.5423 vs 0.5105 baseline). This is expected:

1. **Ambiguous rounding targets:** `156.3` could round to `156` or `157`. During training, the model sees `156` but could equally justify `157`. This spreads probability mass across multiple valid roundings, raising CE loss.

2. **CE loss penalizes wrong digits, not wrong shapes:** Predicting `157` instead of `156` is a 1-unit error (0.64 pixels) that's visually invisible, but CE loss counts it as 3 wrong digit predictions.

3. **The experiment measured loss, not competition score.** Higher CE loss does not mean worse visual quality. The right experiment is: train with integers, generate SVGs, render them, and measure SSIM/EdgeF1.

### Token Savings From Integer Rounding

| | Characters | Qwen tokens | Pythia tokens |
|---|---|---|---|
| `156.3` | 5 chars | 5 tokens | 3 tokens |
| `156` | 3 chars | 3 tokens | 1 token |
| **Savings** | **40%** | **40%** | **67%** |

Over a full SVG with 200 coordinate values, integer rounding saves ~400 Qwen tokens or ~400 Pythia tokens. That's the difference between a complex shape fitting in context or being truncated.

### Verdict: Integer Rounding Is Almost Certainly a Win

- Visual quality loss: negligible (sub-pixel at render resolution)
- Token savings: 35-40% on coordinate data
- More SVGs fit in context: fewer truncations during training
- Shorter sequences at inference: faster generation, less chance of truncation
- **Needs validation:** Run a quick experiment — train both ways, score with SSIM, confirm no regression

---

## Preprocessing Impact: Cumulative Token Savings

Starting from our current approach (1-decimal absolute coordinates), each optimization stacks:

### On Qwen2.5-Coder-3B

| Preprocessing | Path tokens | Cumulative savings |
|---|---|---|
| Current (1-dec, absolute) | 240 | — |
| + Integer rounding | 156 | -35% |
| + Relative commands | 136 | -43% |
| Both (rel+int) | **136** | **-43%** |

Note: relative alone saves only 9% because Qwen's per-digit splitting still hurts even on small deltas. Integer rounding is the bigger win for Qwen because it eliminates the decimal point token AND the decimal digit token.

### On Pythia-2.8B (GPT-NeoX tokenizer)

| Preprocessing | Path tokens | Cumulative savings |
|---|---|---|
| Current (1-dec, absolute) | 136 | — |
| + Integer rounding | 52 | -62% |
| + Relative commands | ~95 | ~-30% |
| Both (rel+int) | **71** | **-48%** |

Pythia benefits massively from integers because each int coordinate becomes a single token. But relative commands add negative signs and more varied numbers, which partially offsets the delta-size savings.

### What This Means for Context Budget

With `max_seq_length=2048` and ~100 tokens for system prompt + chat template:

| Config | Tokens for SVG | Approx path coords that fit |
|---|---|---|
| Qwen, current (1-dec abs) | ~1900 | ~380 coords |
| Qwen, rel+int | ~1900 | **~665 coords** |
| Pythia, float abs | ~1900 | ~665 coords |
| Pythia, rel+int | ~1900 | **~1010 coords** |

The median training SVG has ~200 coordinate values. Complex icons have 400-800. With rel+int on Qwen, virtually all training SVGs fit in context.

---

## Model Decision: Qwen2.5-Coder-3B vs Pythia-2.8B

### The Case for Pythia-2.8B
- 3x more token-efficient on integer coordinates (52 vs 156 tokens for same path)
- Even with rel+int, still ~2x more efficient (71 vs 136)
- More SVGs fit in context → better training signal for complex shapes
- Under 4B parameter limit
- Supported by Unsloth

### The Case for Qwen2.5-Coder-3B
- Much stronger base model (8/10 vs 5/10)
- Instruction-tuned — already understands "generate SVG code"
- Trained on code — knows XML syntax, structured output, tag matching
- With rel+int preprocessing, token efficiency gap narrows to 1.9x (136 vs 71)
- Larger community, more tooling support, better documented
- The model is only bad at NUMBER tokenization — everything else (XML tags, attributes, colors) it handles well

### The Case for Both
- Train Qwen2.5-Coder-3B with rel+int preprocessing (primary)
- Train Pythia-2.8B with integer preprocessing as a comparison (secondary, if time)
- Compare competition scores
- Use whichever scores better

### Recommendation

**Start with Qwen2.5-Coder-3B + rel+int preprocessing.** The stronger base model + instruction tuning is likely worth more than the tokenizer advantage, especially now that we have 1M+ external training data. The preprocessing almost closes the tokenizer gap.

Run Pythia as a second experiment only if time allows.

---

## Experiments to Run

### Experiment A: Integer Rounding Visual Quality (30 min)
- Take 100 training SVGs
- Render original (1-decimal) and integer-rounded versions to PNG
- Compute SSIM between them
- If avg SSIM > 0.99, integer rounding is visually lossless → proceed

### Experiment B: Relative Path Conversion (1-2 hours)
- Implement absolute→relative path conversion using `svgpathtools`
- Measure character and token savings on full training set
- Verify round-trip correctness (convert, render, compare)

### Experiment C: Combined Preprocessing Quick Train (overnight)
- Apply rel+int preprocessing to training data
- Train Qwen2.5-Coder-3B with QLoRA, 1 epoch
- Compare training loss against 1-decimal baseline
- Generate 100 samples, score with SSIM

### Experiment D: Model Comparison (weekend)
- Train Pythia-2.8B with same data + integer preprocessing
- Compare competition scores vs Qwen2.5-Coder-3B
- Only do this if Experiment C results are disappointing
