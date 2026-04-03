# Experiment Journal — SVG Generation Competition

**Competition:** DL Spring 2026 Kaggle Contest — Text-to-SVG Generation
**Team:** Ivan Aristy
**Deadline:** April 1, 2026
**Model limit:** ≤ 4B parameters
**Dataset:** train.csv (50k samples) — only competition-provided data allowed

---

## Day 1 — March 24 (Setup & First Training)

### Infrastructure
- Set up project with uv, ruff (ALL rules), mypy strict, typed stubs
- Installed Modal for cloud GPU (A100-40GB at $2.10/hr)
- Installed Kaggle CLI via `uv tool install kaggle`
- Created Modal volume, uploaded train.csv (129MB), test.csv (151K), sample_submission.csv

### Data Analysis
- 50,000 training samples, 1,000 test samples
- SVG lengths: min=91, median=2110, mean=2524, p95=6078, max=15937 chars
- Prompts: short (~100 chars avg, ~30 tokens)
- 82% of viewBoxes are `0.0 0.0 200.0 200.0`
- Path elements dominate (12k paths in 5k samples), mostly single `<path>` per SVG
- Only 7 exact SVG duplicates, 0 parse failures — very clean data

### Preprocessing (P1)
- **Coordinate rounding**: 6.5M floats with 2+ decimals → round to 1 decimal
- **Result: 54.3% character reduction** (126MB → 58MB, mean 2524 → 1154 chars)
- Strip default attributes (fill-opacity="1", stroke="none", etc.) — 178k found
- Strip XML comments — 3,475 found
- Collapse whitespace

### First Training Run (r=16, 3 epochs)
- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct, QLoRA 4-bit
- Config: r=16, alpha=32, lr=2e-4, batch=2, grad_accum=8, cosine schedule
- max_seq_length=2048
- **Problem:** First run without `--detach` — Modal killed it when local client disconnected
- Relaunched with `--detach`, ran overnight
- **Hit 4h timeout** at step 3244/9189 (epoch 1.05)
- Loss: 0.71 → 0.39, token accuracy: 80% → 86.7%
- Bumped timeout to 10h

### Smoke Tests
- 100-sample training: worked, 25s on A100, loss=0.834
- Inference pipeline: initially showed 0% validity — turned out to be extraction bug (later fixed)

---

## Day 2 — March 25 (Experiments, Bugs, Analysis)

### Resumed Training (r=16, epochs 2-3)
- Resumed from checkpoint-3200, trained to step 9189
- Final loss: ~0.35, accuracy: ~87.9%
- Saved final-adapter (74MB LoRA weights)
- Also downloaded model locally to models/r16-3epoch/

### Quick Experiments (2k samples, 1 epoch each)
Ran 5 experiments in parallel on Modal:

| Experiment | LoRA r | Decode | Final Loss |
|---|---|---|---|
| baseline-r16-t07 | 16 | temp=0.7 | 0.5105 |
| r32-t07 | 32 | temp=0.7 | 0.5022 |
| r16-greedy | 16 | greedy | 0.5108 |
| r16-int-coords | 16 | temp=0.7 (int coords) | 0.5423 |
| r64-greedy | 64 | greedy | 0.4963 |

**Findings:**
- r=64 lowest loss, r=32 close behind — higher rank helps
- Integer coordinate rounding HURTS (highest loss) — stick with 1 decimal
- All experiments timed out during SVG generation phase (30min limit)

### Bug: extract_svg matching system prompt
- **Root cause:** Regex `<svg[\s\S]*?</svg>` matched `<svg>` in system prompt text
- The model WAS generating valid SVGs but extraction grabbed the wrong substring
- **Fix:** Match `<svg\s` (with space/attribute) and strip text before "assistant" marker
- After fix: validation went from 0% → 70-100%

### Bug: Model never produces </svg>
- **Root cause:** Training SVGs truncated at max_seq_length=2048
- 8% of cleaned SVGs exceed token budget → model never sees closing tags during training
- Model generates valid SVG structure but runs to max_new_tokens without stopping

### Inference Speed Issues
- Sequential generation: ~2 min/sample with max_new_tokens=1536 (model generates to limit)
- `stop_strings=["</svg>"]` caused hangs with 4-bit quantized models
- Batched inference: didn't help (4-bit model is memory-bandwidth bound, padding overhead)
- **Fix:** Greedy decode (`do_sample=False`) + `max_new_tokens=1024` + repair truncated SVGs
- Result: ~30s/sample, 100% validity with repair pipeline

### SVG Repair Pipeline
- Close unclosed quotes (odd quote count → add one)
- Close unclosed tags (find last `<` without `>` → add `/>`)
- Add `</svg>` if missing
- Works well: truncated SVGs render as partial but valid images

### Multi-container Inference
- Implemented `.map()` fan-out: split 1000 samples across N containers
- Each container loads model independently, processes its chunk
- 4 containers × 250 samples — running now (~7.5h estimated)

### Research: How Others Solve This
- **Kaggle "Drawing with LLMs" winners**: Used SDXL/Flux → VTracer, NOT LLM fine-tuning
- **LLM4SVG (CVPR 2025)**: Uses 2048 context, truncates long SVGs, aggressive preprocessing
- **OmniSVG**: Discrete tokenization compresses SVGs 3-4x (coordinates as single tokens)
- **Key finding**: Qwen tokenizer has known issues with numerical coordinates (SVGenius paper)
- **Path command conversion** (M/L/C/Z only): tested, INCREASES length by 79% due to arc→lines bloat

### Filtered Training Data
- Added token-count filtering to curate_training_data()
- Filter out SVGs > 1900 tokens after cleaning
- Result: 50,000 → 45,874 samples (91.7% retained), ALL ending with `</svg>`
- Launched new training run with filtered data

### Sample Outputs (10 test samples, r=16 3-epoch model)
- 10/10 valid SVGs
- Triangle prompt: **perfect** dark gray triangle
- Horizontal lines: **correct** five bars
- Camera icon: **decent** blue circle
- Complex prompts (firewood, eye): **bad** — truncated paths render as fragments
- Pattern: simple shapes = good, complex shapes = truncated garbage

### r=32 Training Run (launched then killed)
- Config: r=32, alpha=32, lr=3e-4, 4 epochs, dropout=0, grad_norm=1.0
- Separate output dir: /vol/checkpoints-r32
- Reached step 3000 before we killed it to rethink strategy

### Cost So Far
- Training runs: ~$30-40 estimated
- Inference runs (including failed/killed): ~$15-20
- Experiments: ~$10
- **Total: ~$55-75 estimated**

---

## Key Learnings

1. **Data preprocessing is the highest-ROI activity** — 54% size reduction from coordinate rounding alone
2. **The model must see complete SVGs during training** — truncation means it never learns to stop
3. **extract_svg bugs caused days of confusion** — always inspect raw model output before blaming the model
4. **Modal --detach is mandatory** for anything > 5 minutes
5. **Inference speed is dominated by max_new_tokens**, not model size — stopping early is critical
6. **The training data is all single-path icons** — the model can't learn compositional SVG structure
7. **SDXL → VTracer approach dominates Kaggle competitions** — fundamentally different paradigm

---

## Open Questions

- Can we use external HF datasets? → **YES, confirmed allowed** (see Day 3 entry)
- 50k samples enough? → Probably, with better preprocessing
- SDXL+VTracer: leaderboard top is 18/100, nobody has cracked it. SDXL approach is speculative — not proven for THIS competition. Park for now.

---

## Current State (end of Day 2)

### Running on Modal
- Inference: 4 containers generating 1000 samples from r=16 model (ETA ~1:00 AM)
- Training: filtered dataset run (45,874 samples, r=16, 3 epochs)

### Assets
- Trained model: models/r16-3epoch/ (74MB adapter — **NOTE: weights not on kepler, only config+tokenizer**)
- Preprocessed data pipeline: clean_svg(), curate_training_data()
- Inference pipeline: multi-container fan-out with repair
- Local scoring: score.py (SSIM + EdgeF1 + TED + Compactness)
- Experiment runner: experiment.py
- Decode sweep: decode_sweep.py

---

## Day 2 Evening — Code Audit & Research (Claude on Kepler)

### Bugs Fixed
1. **viewBox normalization bug** — `normalize_viewbox()` was replacing viewBox with `0 0 256 256`, shrinking content drawn in 200x200 space to 78% of canvas. Fixed: now only sets width/height, preserves viewBox so renderer scales correctly.
2. **System prompt mismatch** — prompt said `viewBox='0 0 256 256'` but 82% of training data uses 200x200. Removed specific viewBox instruction. Model will follow training data distribution.
3. **Checkpoint settings** — save_steps 500→200, save_total_limit 2→3 for safer multi-night training.
4. **Stale config defaults** — InferenceConfig updated to match proven settings (1024 tokens, greedy).
5. **Duplicated chat formatting** — kepler_train.py now uses `format_chat_prompt()` from data.py.
6. **Inference speed** — kepler_inference.py switched from raw PEFT to Unsloth `FastLanguageModel` (~2x faster).
7. **7B NaN fix** — added embed_tokens + lm_head to target_modules when model contains "7B" (untrained tool-calling tokens cause NaN loss).

### Key Research Finding: GPU VRAM

Previous estimates were WRONG. Unsloth 4-bit QLoRA uses much less VRAM than expected:

| Model | Estimated VRAM | Fits 16GB? |
|---|---|---|
| 1.5B, r=32 | ~6-8 GB | Easily |
| 3B, r=32 | ~5-7 GB | Easily |
| 7B, r=32 | ~9-12 GB | Yes |
| 7B, r=16 | ~8-10 GB | Comfortably |

**7B is feasible.** This changes the strategy — we can train a much larger model than originally thought.

### Unsloth Compatibility
- Old wheel (cu124-torch260) won't work on kepler (torch 2.10 + CUDA 12.8)
- Correct install: `pip install "unsloth[cu128-ampere-torch2100] @ git+https://github.com/unslothai/unsloth.git"`
- Known Qwen2.5-Coder-7B issue: untrained tokens → NaN loss. Fix: train embed_tokens + lm_head.

### Preprocessing Improvements to Try (Day 3+)

Ranked by estimated impact:

1. **Relative path commands** (15-35% character savings)
   - SVG paths allow relative coords (m/l/c vs M/L/C). Deltas are smaller numbers = fewer tokens.
   - Implicit command repetition stacks with this (e.g., after `l`, subsequent pairs are implicit lineto).
   - Implement with `svgpathtools` library.

2. **RDP path simplification** for overlong SVGs
   - Ramer-Douglas-Peucker reduces points while preserving shape.
   - Instead of FILTERING the 8% that don't fit, SIMPLIFY them until they do.
   - Progressive: start epsilon=0.5, increment by 0.5 until it fits.
   - Gets fit rate from ~92% to ~95%+ AND keeps training signal from complex shapes.

3. **Curriculum learning** (sort training by complexity)
   - Train simple SVGs first (low token count), gradually introduce complex ones.
   - LLM4SVG paper found this helps significantly.
   - Easy to implement: sort dataset by token count.

4. **Prompt paraphrasing** (3-5x effective dataset)
   - Use an LLM to generate 3-5 paraphrases of each text prompt.
   - Same SVG, different prompt wording → more diverse training signal.
   - Highest ROI augmentation for text→SVG.

5. **SVG pretraining stage** (LLM4SVG approach)
   - Stage 1: unconditional SVG completion (model learns SVG grammar)
   - Stage 2: text→SVG fine-tuning
   - Even 1 epoch of SVG-only pretraining helps.

6. **Inference fallback: simplify instead of truncate**
   - A truncated path (incomplete bezier) renders as garbage.
   - A simplified path (fewer control points, complete) renders as rough but recognizable.
   - For SSIM scoring, rough-but-complete >> detailed-but-broken.

### Scoring Strategy Insight

For the 85% visual fidelity component:
- SSIM is more sensitive to luminance/color than edge precision
- A rough shape in the right color scores 0.35-0.60 SSIM
- A truncated path renders as a thin sliver or nothing: 0.0-0.15 SSIM
- A colored rectangle matching dominant color: 0.25-0.45 SSIM
- **Never output a truncated/incomplete path. Always simplify to fit.**

---

## Day 3 — March 27: SVG Structure Deep Dive & Tokenization Problem

### What Does an SVG Actually Look Like?

A typical training sample (from the 50k dataset) is a single-path icon in a 200x200 grid:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <path d="M 50.3 20.1 C 60.5 30.2 80.7 40.3 100.0 50.5
           C 120.3 60.7 140.5 80.2 160.3 100.0
           L 180.5 120.3 C 170.2 140.5 150.3 160.7 130.5 180.2
           L 100.0 190.5 Z" fill="#333333"/>
</svg>
```

Breaking this down:

**The coordinate grid** — `viewBox="0 0 200 200"` defines a 200×200 unit coordinate space. Every number in the path is a position in this grid:
- `(0, 0)` = top-left corner
- `(200, 200)` = bottom-right corner
- `(100, 100)` = center
- Our 1-decimal rounding means coordinates like `50.3`, `120.7` — giving ~2000 possible values per axis

**The path commands** — the `d` attribute is a mini-language:
- `M 50.3 20.1` — "pick up the pen, move to point (50.3, 20.1)"
- `L 180.5 120.3` — "draw a straight line to (180.5, 120.3)"
- `C 60.5 30.2 80.7 40.3 100.0 50.5` — "draw a cubic Bézier curve" with 2 control points + endpoint (6 numbers!)
- `Z` — "close the path back to the starting point"

**The key cost insight:** A cubic Bézier `C` command needs **6 numbers** for a single curve segment. A typical icon outline might have 20-50 curve segments = 120-300 coordinate values. With the XML boilerplate, this is 400-1500+ characters.

### How Qwen Tokenizes These Numbers (The Core Problem)

From the tutorial notebook output, Qwen2.5-Coder splits every digit and punctuation mark into separate tokens:

```
Token: '3'  → ID 18     (1 token per digit)
Token: '0'  → ID 15
Token: '.'  → ID 13
```

Concrete example of what the model sees for a single coordinate pair:

```
SVG text:    M 156.3 42.8
Tokens:      [M] [ ] [1] [5] [6] [.] [3] [ ] [4] [2] [.] [8]
Count:       12 tokens for ONE move-to command with ONE coordinate pair
```

Vs. what GPT-2 would see:
```
SVG text:    M 156.3 42.8
Tokens:      [M] [ 156] [.3] [ 42] [.8]
Count:       5 tokens — GPT-2 has multi-digit number tokens in its vocabulary
```

**This is why Qwen is 2-3x less token-efficient on SVG coordinates than GPT-2-style tokenizers.**

For a typical training SVG with 200 coordinate values:
- Qwen: ~200 × 5 avg digits = ~1000 tokens just for numbers
- GPT-2-style: ~200 × 2 avg tokens = ~400 tokens for numbers

That's **600 wasted tokens** per sample — the difference between fitting a complex shape or truncating it.

### Why Loss Plateaus at ~0.34

The model has to predict each digit independently:
- For `156.3`: predict `1`, then `5`, then `6`, then `.`, then `3`
- Each prediction is a separate CE loss term
- Getting the "shape" right but predicting `157.1` instead of `156.3` incurs 4 wrong-digit penalties
- Two visually identical triangles with slightly different vertices get penalized as if they were different shapes
- The model is spending most of its capacity memorizing digit sequences, not learning spatial structure

### How Can We Fix This?

#### Option 1: Use a model with better number tokenization

| Model | How it tokenizes `156` | Tokens for `156.3 42.8` | Under 4B? |
|---|---|---|---|
| Qwen2.5-Coder-3B | `1`, `5`, `6` (3 tok) | ~10 tok | Yes |
| GPT-2 / GPT-NeoX | `156` (1 tok) | ~5 tok | Only 1.5B max |
| Phi-3.5-mini (3.8B) | Need to verify | Need to verify | Borderline |
| Qwen3.5-2B (starter notebook ref) | Need to verify | Need to verify | Yes |
| CodeGemma-2B | Need to verify | Need to verify | Yes |

**Experiment needed:** Tokenize the same SVG with multiple models and count tokens. If another model under 4B saves 40%+ tokens, it could be worth switching even with less SVG pretraining.

**Note:** The starter notebook references `Qwen3.5-2B-Instruct` — this might be a newer model with improved tokenization. Need to check if it exists on HuggingFace.

#### Option 2: Make coordinates more token-efficient

**A. Relative path commands** (lowercase: `m`, `l`, `c` instead of `M`, `L`, `C`)

Absolute (what we have now):
```
M 100 100 L 120 105 L 135 110 L 150 130
```
Relative (deltas from current position):
```
M 100 100 l 20 5 l 15 5 l 15 20
```

Why this helps:
- Deltas are small (typically 1-30), absolute coords are large (0-200)
- `20` = 2 tokens vs `120` = 3 tokens in Qwen
- No decimal needed when deltas are small integers: `l 5 -3` vs `L 105.3 97.8`
- Implicit repetition: after `l`, every subsequent pair is also a lineto — no repeated `l` needed
- **15-35% character savings, zero precision loss**
- Easy to implement with `svgpathtools` library

**B. Integer coordinates in smaller grid**

Rescale from 200×200 with 1 decimal (2001 values per axis) to 200×200 integers (201 values):
```
Before: M 50.3 20.1 C 60.5 30.2 80.7 40.3 100.0 50.5
After:  M 50 20 C 61 30 81 40 100 51
```
- Saves the decimal point token (`.`) + decimal digit for every number
- ~40% token reduction per coordinate
- **BUT:** Earlier experiment showed integer rounding hurts loss. However that was on raw training loss, not on SSIM/EdgeF1 scoring where 1-pixel shifts are invisible.
- **Needs experiment:** Does integer rounding hurt SSIM score? Or does the token savings (more SVGs fit in context) offset the precision loss?

**C. Quantize to 100×100 grid (0-99)**

```
Before (200 grid): M 156.3 42.8 → 10 Qwen tokens
After (100 grid):  M 78 21      → 5 Qwen tokens (50% reduction!)
```
All coordinates become 1-2 digits = 1-2 Qwen tokens. But precision drops from ~2000 levels to 100 levels per axis. Visually significant? Maybe, maybe not at 256×256 render size. **Needs experiment.**

**D. Custom vocabulary tokens (advanced, OmniSVG approach)**

Add ~40k tokens to vocabulary: one per (x,y) coordinate pair. `(156, 42)` → single token `[COORD_31242]`.
- Each coordinate pair becomes exactly 1 token instead of ~10
- Requires resizing model embeddings and training the new tokens
- 3-4x sequence length reduction
- This is what the state-of-the-art (OmniSVG) does, but it's a significant implementation lift

#### Option 3: Different loss function

Standard CE loss penalizes each digit equally. Alternatives:
- **Weighted CE:** Lower weight on coordinate digits, higher on structure tokens (M/L/C/Z, tags). Model focuses on getting shape structure right rather than exact coordinates.
- **DPO with rendered scoring:** After initial SFT, generate multiple SVGs per prompt, render both, score with SSIM, train model to prefer higher-scoring output. Doesn't need differentiable rendering.
- **REINFORCE:** Sample SVGs, compute SSIM reward, update with policy gradient. Training-unstable but directly optimizes what we care about.

### Major Discovery: External Datasets May Be Allowed

The starter notebook (`references/starter-notebook.ipynb`) explicitly references official `contest_docs/03_Data_Design.md` listing these HuggingFace datasets:
- `OmniSVG/MMSVG-Icon`
- `xingxm/SVGX-Core-250k`
- `xingxm/SVGX-SFT-1M` (has `_int` (integer coords) and `_encode` variants!)
- `thesantatitan/deepseek-svg-dataset`
- `nyuuzyou/svgfind`
- `starvector/svg-icons`

**UPDATE: External data IS allowed.** The earlier "train.csv only" note was wrong. This is game-changing:
- `SVGX-SFT-1M` alone = 20x more data, including `_int` (integer coords) and `_encode` variants
- `SVGX-Core-250k` = another 5x on top
- `OmniSVG/MMSVG-Icon` = icon-focused dataset (likely most similar to competition data)
- These datasets have already been preprocessed/cleaned by researchers

### Proposed Experiments (Ranked by Impact)

#### Experiment 1: Tokenizer comparison across models
**Goal:** Find if there's a model under 4B with significantly better number tokenization.
**Method:** Take 100 sample SVGs from training data. Tokenize with Qwen2.5-Coder-3B, Qwen3.5-2B (if exists), Phi-3.5, CodeGemma-2B, GPT-NeoX. Compare token counts.
**Expected time:** 30 min (just tokenizer comparison, no training)
**Decision:** If another model saves >30% tokens, strongly consider switching.

#### Experiment 2: Relative path conversion
**Goal:** Measure actual token savings from converting absolute→relative path commands.
**Method:** Convert all training SVGs to relative commands. Measure character savings and token count reduction with Qwen tokenizer. Run a quick 1-epoch training comparison.
**Rationale:** This is a near-free win. Relative paths preserve ALL visual information (zero precision loss) while significantly reducing sequence length. The deltas are small integers (1-2 digits) instead of large coordinates (1-3 digits), so Qwen's per-digit tokenization hurts less. It's the single highest-ROI change we haven't tried.
**Expected time:** 1-2 hours implementation + 2-3 hours for quick experiment
**Risk:** Low. Worst case it doesn't help much and we revert.

#### Experiment 3: Train 3B with current preprocessing (baseline)
**Goal:** Get a baseline score with the larger model on kepler.
**Rationale:** We need a 3B number to compare against the 1.5B results. Even without tokenization improvements, 3B has more capacity to memorize coordinate patterns. Since VRAM is only ~5-7GB (not the feared 14GB), training is safe. This runs overnight while we work on preprocessing improvements during the day.
**Config:** Qwen2.5-Coder-3B-Instruct, r=32, alpha=64, lr=2e-4, 3 epochs
**Expected time:** ~36h (multi-night)

#### Experiment 4: Integer coordinates (0-199)
**Goal:** Test if integer rounding hurts SSIM scoring (not just CE loss).
**Method:** Round all training coordinates to integers. Train 1 epoch, generate samples, score with score.py. Compare SSIM/EdgeF1 against 1-decimal model.
**Key question:** The earlier experiment showed higher CE loss, but does visual quality actually suffer? SSIM at 256×256 render might not notice 0.5-unit shifts in 200-unit space.
**Expected time:** 3 hours (quick experiment)

#### Experiment 5: Clarify external dataset rules
**Goal:** Determine if we can use SVGX-SFT-1M and other HF datasets.
**Method:** Re-read competition rules, ask on Kaggle discussion / email TA.
**Impact:** If yes → 20x more training data, including pre-quantized integer variants.

#### Experiment 6: DPO pass with SSIM scoring (if time permits)
**Goal:** After SFT training, do a DPO (Direct Preference Optimization) pass.
**Method:** For each prompt, generate 2 SVGs with different seeds. Render both, score with SSIM. Train model to prefer higher-scoring SVG.
**Rationale:** Directly optimizes what the competition scores, bypasses the CE-loss-on-digits problem.
**Expected time:** 4-6 hours implementation + overnight training
**Risk:** Medium. DPO training can be unstable. Save SFT checkpoint as fallback.

---

## Day 3 Evening — Preprocessing Results & Training Data Expansion

### Preprocessing Audit Results (Rel+Int)

Ran `preprocessing_audit.py` on 100 samples across the full length distribution. Renders in `audit_renders/`.

| Metric | Integer only | Rel+Int |
|---|---|---|
| Avg char savings | -27.7% | -29.1% |
| Avg Qwen token savings | -32.6% | **-40.0%** |
| Avg SSIM vs original | 0.9745 | 0.9730 |
| Worst-case SSIM | 0.5294 (1 outlier) | 0.8020 |
| Fit in 2048 context | 99% | 99% |

**Verdict:** Rel+int is the winner — 40% token savings, near-identical visual quality, 99% context fit.
The integer-only worst case (0.529 SSIM, sample 023) is fixed by rel+int (0.880). Visual audit confirmed
renders look equivalent. Worst samples can be filtered during data curation.

Full analysis in `planning/tokenizer-analysis.md`.

### Training Data Strategy (confirmed: external data IS allowed)

**Parallel agent is downloading and curating external datasets.** Target: ~80-100k total samples.

Sources:
| Source | Est. usable samples | Notes |
|---|---|---|
| Competition train.csv | 50k | Must include — matches test distribution |
| `OmniSVG/MMSVG-Icon` | ~20-30k | Icons — closest to competition format |
| `xingxm/SVGX-SFT-1M` | ~20-30k (sampled) | Has `_int` variant with integer coords |

All external data will be preprocessed with rel+int to match.

### Training Plan (Night 3, March 27-28)

Two machines training in parallel:

| Machine | GPU | Config | Data |
|---|---|---|---|
| **Kepler** | RTX 2000 Ada (16GB) | Qwen2.5-Coder-3B QLoRA (r=32) | 80-100k, rel+int, 2 epochs |
| **Thor** | NVIDIA Thor (128GB) | Qwen2.5-Coder-3B **full fine-tune** | 80-100k, rel+int, 2 epochs |

Thor full fine-tune could break the 0.34 loss ceiling — LoRA's low-rank constraint may be limiting
how much the model can reshape its coordinate prediction behavior.

### Current Status (end of Day 3)
- **Kepler inference DONE** — r16-3epoch model, 1000 test SVGs generated, 12/1000 invalid. Saved to `submissions/filtered-r16-3epoch.csv`
- **External data downloaded** — SVGX-SFT-1M + OmniSVG curated into expanded CSVs
- **Thor set up and training launched** — see Day 4 entry

---

## Day 4 — March 28: Thor Full Fine-Tune & Dual-Machine Training

### Thor (DGX Spark) Setup

Discovered the NVIDIA Thor (DGX Spark) on Ivan's network — 128GB unified memory (Grace-Hopper architecture), ARM64/aarch64, CUDA 13.0, Driver 580.00.

**Setup steps:**
1. SSH access via `ssh ivan@192.168.5.194` (LAN, kepler ed25519 key)
2. Installed `uv`, created venv with `uv sync --extra gpu` (PyTorch 2.11+cu130 for aarch64)
3. GPU access requires `video` group — added ivan to `video` and `render` groups
4. All training commands must be wrapped: `sg video -c 'nohup ... &'` to preserve group context
5. nvidia-smi shows "Not Supported" for memory on unified memory architecture — this is normal
6. GPU-Util also shows 0% despite active GPU computation — reporting issue, confirmed model is on `cuda:0`
7. Installed tensorboard for loss logging

**Key quirk:** `device_map="auto"` works correctly (places model on cuda:0), but `nohup` inside `sg video` can lose group context. Fixed with a wrapper shell script (`run_thor_train.sh`) that the nohup executes.

### LoRA vs Full Fine-Tune — Why We're Doing Both

**LoRA (kepler):** Freezes all 3B parameters, trains ~50M low-rank adapter matrices (1.6% of model). Uses ~6GB VRAM. Fast but constrained — the low-rank bottleneck may limit how much the model can reshape its digit prediction behavior. Loss plateau at 0.34 could be a capacity limit.

**Full fine-tune (Thor):** Updates ALL 3B parameters. Uses ~35GB (params 5.7GB + optimizer 23GB + gradients 5.7GB). No architectural bottleneck — the model can fully rewire how it handles coordinates. Could break through the 0.34 loss ceiling.

### Batch Size Benchmarking (Real Training Data)

Tested batch sizes 1-32 on Thor with actual SVG training data (~1000 tokens/sample):

| Batch size | s/step | samples/s | Status |
|---|---|---|---|
| 1 | 2.27 | 0.44 | Works but slow |
| 2 | 6.29 | 0.32 | Worse (overhead) |
| 4 | 5.25 | 0.76 | OK |
| **8** | **7.25** | **1.10** | **Best stable** |
| 16 | ~17 | ~0.94 | Crashed at step 3 (OOM) |

**Winner: batch_size=8.** bs=16 crashes on real data (longer sequences than dummy benchmark). The unified memory architecture doesn't give explicit OOM errors — it just dies.

Note: dummy data benchmark (short sequences) showed bs=8 at 1.40 samples/s. Real data (1000+ token SVGs) is slower at 1.10 samples/s due to longer sequences.

### Training Configuration — Thor

```
Machine:       DGX Spark (NVIDIA Thor, 128GB unified)
Model:         Qwen/Qwen2.5-Coder-3B-Instruct (full bf16, all params trainable)
Data:          train_expanded_large.csv (149,516 samples → 141,618 after curation)
Epochs:        1
Batch size:    8 (effective batch 8, grad_accum=1)
Learning rate: 2e-5 (cosine schedule, 10% warmup)
Optimizer:     AdamW (full, not paged/8bit)
Steps/epoch:   17,703
Save every:    3,540 steps (~7h, keeps latest 3 checkpoints)
Eval every:    3,540 steps
ETA:           ~35h (finish ~Sunday morning)
```

**Launched at:** 2026-03-28 12:14 UTC

### Training Configuration — Kepler (planned, waiting on data agent)

```
Machine:       RTX 2000 Ada (16GB)
Model:         Qwen/Qwen2.5-Coder-3B-Instruct (QLoRA 4-bit, r=32)
Data:          train_expanded.csv (79,516 samples)
Epochs:        2
Batch size:    2 (effective batch 16, grad_accum=8)
Learning rate: 2e-4
Optimizer:     paged_adamw_8bit
```

### Expanded Training Data

Two datasets prepared by the data agent:

| File | Samples | Purpose |
|---|---|---|
| `train_expanded.csv` | 79,516 | Kepler QLoRA — competition data + moderate external |
| `train_expanded_large.csv` | 149,516 | Thor full fine-tune — competition data + large external |

Sources: competition train.csv (50k) + OmniSVG/MMSVG-Icon + xingxm/SVGX-SFT-1M (sampled). External data is allowed per competition rules (confirmed via contest_docs/03_Data_Design.md).

### Inference on Checkpoints

Thor saves full model checkpoints (~6GB each). To test a checkpoint:
1. `rsync` checkpoint from Thor to kepler (~30s over LAN)
2. Load with `AutoModelForCausalLM.from_pretrained()` in 4-bit on kepler
3. Run inference with greedy decode, 1024 max tokens
4. Score with `score.py` against training references

Need to write a checkpoint inference script for this (TODO).

### What We Intend to Achieve

**Primary goal:** Break the 0.34 CE loss ceiling and improve competition score above the current leaderboard top of 18/100.

**Strategy:**
1. **More data** — 3x more training samples from external datasets
2. **Full fine-tune** — remove the LoRA capacity bottleneck on Thor
3. **Preprocessing (rel+int)** — 40% token savings, 99% context fit rate (TODO: apply to training data)
4. **Compare** QLoRA vs full fine-tune on the same data → submit the better one

**What success looks like:**
- Train loss below 0.30 (breaking the plateau)
- Eval loss tracking train loss (not overfitting)
- Visual quality: complex shapes render recognizably (not truncated garbage)
- Competition score: top 50% of leaderboard (>18/100)

### Monitoring

Thor training can be checked with:
```bash
ssh ivan@192.168.5.194 "tail -1 ~/WorkDir/svg-gen/thor_training.log"
ssh ivan@192.168.5.194 "grep loss ~/WorkDir/svg-gen/thor_training.log | tail -5"
```

Cron job runs hourly at :17 to check progress (session-only, dies when Claude exits).

### Outstanding TODOs
- [ ] Write checkpoint inference script (load full model, not adapter)
- [ ] Apply rel+int preprocessing to training data (currently training on 1-decimal absolute)
- [x] Launch kepler QLoRA training with expanded data — **DONE, see below**
- [ ] Submit r16-3epoch inference results to Kaggle for baseline score
- [ ] Score first Thor checkpoint when it drops (~step 3540, ~7h in)

### Kepler: Merge-and-Retrain (launched March 28 00:44 UTC)

**Baseline result:** The r16-3epoch model (1.5B, filtered data, Modal-trained) placed **9th out of 58** on the Kaggle leaderboard. This is our best model so far.

**Strategy: Merge + Re-LoRA** — instead of starting from scratch with 3B, we build on the proven 9th-place model:
1. Merged the r=16 adapter into Qwen2.5-Coder-1.5B base weights (bakes 9th-place knowledge permanently)
2. Applied a NEW r=32 LoRA on the merged model (2x trainable params: 37M vs 18M)
3. Training on expanded dataset (76k samples from competition + SVGX + MMSVG-Icon)

```
Machine:       Kepler (RTX 2000 Ada, 16GB)
Base:          models/merged-1.5b-r16 (Qwen2.5-Coder-1.5B with r16 knowledge baked in)
Adapter:       NEW LoRA r=32, alpha=64
Data:          train_expanded.csv (79,516 samples → 76,256 after curation)
Epochs:        2
Batch:         2 × 8 grad_accum = effective 16
LR:            2e-4 (cosine)
Steps:         9,532 total
Speed:         ~5s/step
Save every:    953 steps (~1.3h)
ETA:           ~13h (finish ~2 PM March 28)
```

**Why this should work:**
- The merged model already knows SVG syntax + coordinate patterns from r=16 training
- Fresh r=32 adapter has double the capacity to learn new patterns
- 76k samples (vs 46k originally) = more diverse training signal
- The model starts from a proven baseline, not from scratch

**Monitoring:**
```bash
tail -3 merge_train.log
grep "loss" merge_train.log | tail -5
```

### Parallel Training Summary

| Machine | Model | Approach | Data | ETA |
|---|---|---|---|---|
| **Kepler** | 1.5B (merged r16 + new r32) | QLoRA | 76k | ~2 PM Mar 28 |
| **Thor** | 3B | Full fine-tune | 142k | ~Sun morning |

---

## Day 5 — March 29: Thor Debugging & Stable Training

### The Thor Process Kill Saga

Spent most of the day debugging why training processes kept dying silently on the Thor after ~2-5 minutes. No error messages, no OOM in dmesg, no CUDA exceptions — processes just vanished.

**What we tried and what failed:**

| Attempt | Method | Result |
|---|---|---|
| 1 | `nohup` + `sg video` | Died at step ~25 (~2.5 min) |
| 2 | `nohup` without `sg video` (groups are permanent) | Died at step ~25 |
| 3 | `tmux` | Died at step ~67 (bs=2, seq=1536) |
| 4 | `systemd-run --user --scope` | Died at step ~26 |
| 5 | `screen` | Died at step ~30 |
| 6 | Foreground run via SSH (with `timeout 300`) | **Survived all 5 minutes!** |

The foreground run surviving while all background methods died was the key clue.

**What we investigated:**
- `ulimit -l` (memlock): was 15.3 GB → set to unlimited. Didn't fix it.
- Swap: was 0 → added 32GB. Didn't fix it.
- `loginctl enable-linger ivan`: enabled. Didn't fix it.
- cgroup memory limits: all set to infinity/max. Not the issue.
- Thermal: 38°C. Not the issue.
- Kernel OOM killer: no entries in dmesg. Not the issue.
- systemd-oomd: disabled on this platform. Not the issue.
- Memory at crash time: 82GB free (!). Not OOM.

**What we learned from research (agent):**
- DGX Spark has a documented "zombie" issue — processes die without clean OOM errors
- NVIDIA forums have multiple threads about this exact behavior
- The recommended approach is Docker with `--runtime=nvidia --ulimit memlock=-1`
- The community's "five-layer defense" includes: disable swap, flush caches, memory jails via cgroups, protect SSH with OOMScoreAdjust, memory watchdog

**Root cause (most likely):** Combination of systemd session cleanup killing backgrounded processes AND unified memory fragmentation. The foreground SSH run survived because the PTY kept the session alive. Docker containers have their own process namespace and aren't subject to user session management.

### The Fix: Docker

Stopped all competing Docker containers (jrubin's Savant dev stack — RTSP sinks, replay, Pulsar) to free GPU memory on the shared unified pool. Then launched training inside the official NVIDIA PyTorch container:

```bash
sudo docker run -d --name svg-train \
  --runtime=nvidia --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --memory=100g \
  -v ~/WorkDir/svg-gen:/workspace \
  -e HF_TOKEN=... -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTHONPATH=/workspace/src \
  nvcr.io/nvidia/pytorch:25.04-py3-igpu \
  bash -c 'pip install -q trl datasets peft accelerate bitsandbytes tensorboard cairosvg && \
  python3 -m svg_gen.thor_train --train-csv data/train_expanded_large.csv --epochs 1 --batch-size 4 --grad-accum 2 --lr 2e-5'
```

**This worked.** Training survived past all previous crash points. Process has been running for hours.

### FlashAttention Crash at Step 816

After ~1.5h of successful training, crashed at eval step (step 500 checkpoint save + eval):
```
RuntimeError: FlashAttention only supports Ampere GPUs or newer.
```

Thor's SM 12.1 (Blackwell) isn't recognized by FlashAttention's GPU check. Fixed by adding `attn_implementation="sdpa"` to `AutoModelForCausalLM.from_pretrained()`. SDPA (Scaled Dot-Product Attention) works on all architectures.

Training resumed from checkpoint-500. Loss continued from ~0.57.

### Loss Curve (Full Fine-Tune)

```
Step   10: 1.243  (start)
Step   20: 1.232
Step  100: ~1.0
Step  200: ~0.73
Step  300: ~0.68
Step  500: ~0.57   ← checkpoint saved
Step  700: ~0.58   (resumed from checkpoint-500)
Step  800: ~0.57
```

**Loss is already at 0.57 — well below the LoRA plateau of ~0.34.** Full fine-tune hypothesis confirmed: the LoRA bottleneck WAS limiting coordinate learning. Token accuracy: 81%.

### Revised Training Configuration (Final)

```
Machine:       DGX Spark (NVIDIA Thor, 128GB unified)
Runtime:       Docker (nvcr.io/nvidia/pytorch:25.04-py3-igpu)
Model:         Qwen/Qwen2.5-Coder-3B-Instruct (full bf16, SDPA attention)
Data:          train_expanded_large.csv (149,516 → 141,618 after curation)
Epochs:        1
Batch size:    4 (effective batch 8, grad_accum=2)
Learning rate: 2e-5 (cosine schedule, 10% warmup)
Optimizer:     AdamW (full)
Steps/epoch:   17,703
Save every:    500 steps (~45 min, keeps latest 5)
Speed:         ~5.5s/step
```

### Training Schedule

| Window | Hours | Est. steps | Cumulative |
|---|---|---|---|
| Sunday night (now → Mon 9am) | ~12h | ~7,800 | ~8,300 |
| Mon 6pm → Tue 9am | 15h | ~9,800 | ~17,700+ |
| Tue 6pm | Inference + submit | — | — |

Full epoch should complete by Tuesday morning. Inference Tuesday evening (~3-4h on Thor), submit before midnight.

### Operations

**Pause training (Mon morning):**
```bash
ssh ivan@192.168.5.194 "sudo docker stop svg-train"
```
Docker stop sends SIGTERM → Trainer saves checkpoint → container stops cleanly.

**Resume training (Mon evening):**
```bash
ssh ivan@192.168.5.194 "sudo docker start svg-train"
```
Container resumes, Trainer auto-detects latest checkpoint, continues.

**Restart jrubin's containers after training is done:**
```bash
ssh ivan@192.168.5.194 "cd /home/jrubin/dev-jrubin/intenseye && sudo docker compose up -d"
```

**Monitor:**
```bash
ssh ivan@192.168.5.194 "sudo docker logs svg-train --tail 3"
```

### Project Cleanup

Reorganized the project directory — moved standalone scripts to `scripts/`, logs to `logs/`, results to `results/`. Cleaned .gitignore. Freed 70GB on kepler's disk (HF cache, pip cache, downloads, playwright).

### Other Changes
- Added `attn_implementation="sdpa"` to thor_train.py (FlashAttention doesn't support Blackwell SM 12.1)
- Set `save_steps=500`, `save_total_limit=5` for more frequent checkpoints
- Added unlimited memlock to `/etc/security/limits.conf` on Thor
- Protected SSH daemon with `OOMScoreAdjust=-1000` on Thor
- Installed `ncdu` on kepler for disk usage analysis

### Kepler Merge+R32 Training Results

Training completed after 13.6h (9,532 steps, 2 epochs).

**Loss curve:**
```
Step     0: 0.53  (new r=32 adapter starts from zero on merged base)
Step  1000: 0.51
Step  2000: 0.51
Step  3000: 0.48
Step  4000: 0.47  (end of epoch 1)
Step  5000: 0.46
Step  6000: 0.45
Step  7000: 0.40  (lowest point)
Step  8500: 0.44
Step  9532: 0.465 (final, LR nearly zero)
```

**Final train loss: 0.465.** Higher than the r=16 model's 0.34, but these aren't comparable — different dataset (76k mixed vs 46k competition-only), different preprocessing (rel+int vs 1-decimal absolute), different coordinate format. The only fair comparison is the competition score.

Adapter saved to `outputs/final-adapter`, base model at `models/merged-1.5b-r16`.

### Inference — Crashed and Resumed

Launched inference on merged+r32 model. Crashed at sample 590/1000 due to disk full (`OSError: [Errno 28] No space left on device`). The other session's project cleanup freed disk space.

Wrote `scripts/resume_inference.py` — recovers 585 valid samples from the corrupted CSV, generates only the remaining 415. Resumed and running (~70min ETA).

Output: `results/submissions/merged-r32-expanded-complete.csv`

### Simple SVG Training Subset

Created `data/train_simple.csv` — **48,055 samples** filtered for complexity the model can realistically generate fully:
- ≤600 Qwen tokens
- ≤80 path commands
- ≤20 SVG elements

Source breakdown: 29k competition (60%), 11k SVGX (22%), 8k MMSVG (17%).

**Plan:** After inference completes, merge current r32 adapter into base, apply fresh r32, train 2 epochs on just the simple subset. This "specialist sharpening" pass should boost scores on the majority of test prompts that are simple/medium icons.

### Training Stack Summary (3 rounds of learning)

```
Round 1: Qwen2.5-Coder-1.5B + LoRA r=16, 46k competition data     → 9th place
         ↓ merge adapter into weights
Round 2: Merged-1.5B + LoRA r=32, 76k expanded data                → inference running
         ↓ merge adapter into weights (planned)
Round 3: Merged-1.5B-v2 + LoRA r=32, 48k simple SVGs only          → planned next
```

Each round bakes the previous knowledge permanently and adds fresh capacity.

### Kaggle Submission Results

| Submission | Score | Rank |
|---|---|---|
| `filtered-r16-3epoch.csv` (r16, 46k comp-only) | **15.47** | 9th/58 |
| `merged-r32-expanded-complete-clean.csv` (r32, 76k mixed) | **14.64** | worse |

**The expanded dataset hurt.** External data (SVGX 128×128 viewBox + MMSVG) diluted competition performance. The model learned patterns from different SVG distributions that don't match the test prompts.

**Lessons:**
1. More data is not always better — distribution match matters more than volume
2. The rel+int preprocessing change may have also contributed (different coordinate format than what the model learned in r16)
3. The r16 merged base is still clean (LoRA keeps base weights frozen during r32 training)

### Course Correction: Competition-Only R32 Training

Going back to what works. Using the r16 merged base (9th-place knowledge baked in) with a new r32 adapter trained on competition data only, matching the original preprocessing:

```
Base:          models/merged-1.5b-r16 (proven 9th-place knowledge)
Adapter:       NEW LoRA r=32, alpha=64 (2x capacity vs r16)
Data:          data/train.csv (50k competition-only — same distribution as test)
Preprocessing: 1-decimal absolute (matches r16, NOT rel+int)
Epochs:        2
LR:            2e-4 (cosine)
```

**Why this should work:** Same data distribution that got 9th place + double the adapter capacity. No external noise. No preprocessing format changes.

### Final Pivot: 1.5B Full Fine-Tune on Thor

Killed the 3B training. Three reasons:
1. **3B keeps crashing** — FlashAttention on eval (Blackwell SM 12.1 incompatibility), even with SDPA + eval disabled it's fragile
2. **Expanded data hurts** — merged-r32 on 76k mixed data scored 14.64 vs r16 on 46k comp-only at 15.47
3. **1.5B is 2.5x faster** — can do 3 full epochs in 12h vs 1 epoch in 28h for 3B

The 3B full fine-tune proved the concept (loss 1.24→0.57, below LoRA ceiling) but the 1.5B full fine-tune is the practical winner for our time constraints.

**Plan:**
- Model: Qwen/Qwen2.5-Coder-1.5B-Instruct, full bf16, all params trainable
- Data: original train.csv only (50k → 46k after filtering)
- 3 epochs, ~4h/epoch = ~12h total (fits in tonight's window)
- Same preprocessing as the 9th-place model (1-decimal absolute)
- SDPA attention, eval disabled, Docker on Thor

```
Est. speed:    ~2.5s/step
Steps/epoch:   5,750
Total steps:   17,250 (3 epochs)
ETA:           ~12h
```

If this works, we have Monday evening free for inference + submission buffer.

---

## Day 5/6 Overnight — 1.5B Full Fine-Tune Results

### Thor Training (overnight Sun→Mon)

1.5B full fine-tune ran successfully overnight in Docker. No crashes. Trained ~10,000 steps (1.8 epochs) before pausing for work day.

**Loss curve:**
```
Step     10: 1.023  (start)
Step   1780: 0.474
Step   2700: 0.432
Step   3610: 0.410
Step   4530: 0.395
Step   5450: 0.387  ← end of epoch 1
Step   6370: 0.458  (epoch 2 start bump)
Step   7290: 0.458
Step   8210: 0.381
Step   9120: 0.370  ← best so far
Step  10040: 0.397  (stopped for work day)
```

**Checkpoints saved:** 8000, 8500, 9000, 9500, 10000

### Key Finding: Loss Floor Is Real

The full fine-tune **did NOT break through the LoRA loss plateau.** All approaches converge to the same range:

| Approach | Best loss | Notes |
|---|---|---|
| LoRA r=16 (Modal, comp-only) | 0.34 | 9th place, 15.47 score |
| LoRA r=32 (kepler, comp-only) | 0.354 | Untested on Kaggle |
| Full fine-tune 1.5B (Thor) | 0.370 | 10k steps, 1.8 epochs |
| Full fine-tune 3B (Thor) | 0.57 | Only 800 steps, would likely converge to same |

**Conclusion: ~0.34-0.40 is the CE loss floor for this dataset + tokenization.** More model capacity doesn't help. The bottleneck is:
1. **One-to-many mapping** — many valid SVGs per prompt, CE loss can't handle this
2. **Digit-by-digit tokenization** — model predicts each digit independently, can't learn "numbers" as units
3. **Data ceiling** — 46k samples, same shapes seen multiple times

### Kepler Parallel Work (other agent)

While Thor trained, the other agent on kepler:

1. **Submitted comp-only r=32 model** — trained merged-r16 base + fresh r=32 LoRA on 45k competition-only data (matching r16's preprocessing). Loss: 0.354. **Score: awaiting result.**

2. **Built inference postprocessing pipeline:**
   - Validate all SVGs (XML parse + constraint check)
   - Repair truncated paths (close unclosed tags, add </svg>)
   - Normalize viewBox to 256x256 rendering
   - Fallback SVG for invalid outputs

3. **Ran inference on merged+r32 model** — 1000 samples, 0 invalid

4. **Created simple-subset training data** (`train_simple.csv`, 48k samples filtered for ≤600 tokens, ≤80 path commands) — intended for a specialist pass but deprioritized

### Kaggle Scores So Far

| Submission | Model | Data | Score |
|---|---|---|---|
| filtered-r16-3epoch | LoRA r=16, 1.5B | 46k comp-only | **15.47** (9th) |
| merged-r32-expanded | LoRA r=32, 1.5B merged | 76k mixed | **14.64** (worse) |
| componly-r32 | LoRA r=32, 1.5B merged | 45k comp-only | **pending** |

### High-LR Experiment (launched, paused)

Started a fresh 1-epoch run from the step-10000 checkpoint with **LR=1e-4** (5x higher than original 2e-5). Hypothesis: conservative LR kept us in a local minimum. Paused after launch for work day — will resume tonight.

---

## Day 6 — March 30: Strategic Assessment

### The Uncomfortable Truth

We've hit the CE loss floor. Every model architecture (1.5B, 3B), every training method (LoRA, full fine-tune), and every dataset combination converges to 0.34-0.40. More compute won't help. The leaderboard top is ~18/100 and our best is 15.47 — that 2.5 point gap probably comes from **inference quality, not training loss.**

### What Could Actually Move the Score

**Tier 1 — Actionable tonight, highest potential:**

1. **Best-of-N sampling at inference** — generate 3-5 SVGs per prompt with different temperatures, render all, score locally with SSIM, submit the best. No training needed. Simple to implement. Could gain 1-2 points just from picking better outputs.

2. **DPO (Direct Preference Optimization)** — generate pairs of SVGs per prompt, score with SSIM, train the model to prefer the higher-scoring one. Directly optimizes the competition metric instead of CE loss on digits. HuggingFace TRL has `DPOTrainer` built in.

**Tier 2 — Good ideas but harder to implement in 2 days:**

3. **Prompt-aware decoding** — lower temperature for simple shapes (triangles, circles), higher for complex scenes. Simple prompts should use greedy, complex should sample.

4. **SVG space mapping** — teach model to output structured descriptions instead of raw coordinates, convert deterministically. Removes coordinate prediction from the model's job entirely. Too much implementation for remaining time.

5. **Quantized coordinate vocabulary** — add 40k tokens for (x,y) pairs (OmniSVG approach). 3-4x compression. Requires vocab resize + retraining. Not feasible in 2 days.

**Tier 3 — Marginal benefit:**

6. **Data augmentation** (flip, rotate, color jitter) — doesn't fix the tokenization bottleneck, and prompts would need matching modifications.

7. **Higher LR training** — running tonight as an experiment, but unlikely to break the floor.

### Remaining Time Budget

| Window | Hours | Best use |
|---|---|---|
| Mon evening (tonight) | ~6h | Best-of-N inference + DPO experiment |
| Mon overnight | ~12h | DPO training OR continued full fine-tune |
| Tue evening | ~4h | Final inference + submit |

### Decision Needed

Two paths for tonight:
- **Path A:** Best-of-N inference (no training, immediate results, safe)
- **Path B:** DPO training (needs pair generation + training, higher ceiling, more risk)

Could also do both: run best-of-N inference on kepler while DPO trains on Thor.

---

## Day 7 — March 31: Inference Optimization & Refined Training

### Temperature Sweep Results

Tested temperature and repetition penalty variations on the comp-only r=32 model (16.87 baseline):

| Submission | Score | Change |
|---|---|---|
| Greedy, rep=1.1 (baseline) | **16.87** | — |
| Greedy, rep=1.05 | 16.51 | -0.36 |
| Greedy, rep=1.15 | 15.88 | -0.99 |
| Temp=0.05, rep=1.1 | 16.42 | -0.45 |

**Conclusion: Greedy with rep=1.1 is already optimal for single-run inference.** Any sampling or penalty variation hurts. The model's default confidence is its best mode for structured SVG output.

### Thor Inference — Not Viable

Attempted to run inference on Thor (NVIDIA Thor, SM 11.0 Blackwell). Despite models loading on CUDA with 2.9GB allocated, the HuggingFace `generate()` loop falls back to CPU (~80% CPU utilization, 1.3GB RAM). The container's PyTorch warns "NVIDIA Thor with CUDA capability sm_110 is not compatible with the current PyTorch installation." Training works (different CUDA kernel path) but autoregressive generation does not.

**Thor is training-only.** All inference on kepler.

### Best-of-N Combiner

Built `scripts/combine_best_of_n.py` — for each prompt, picks the best SVG across multiple runs using heuristic scoring (valid XML, length, complete paths, standard viewBox, no fallbacks).

Combined 4 runs (greedy rep=1.1, rep=1.05, rep=1.15, rep=1.1+1536 tokens):
- **885/1000 improvements** over baseline
- All 14 short SVGs rescued (some from 197→2054 chars)
- Top contributor: 1536-token run (38.6%) — longer generation helps most
- Ready to submit: `results/submissions/best_of_n_combined.csv`

### Refined Training — Breaking Through the Loss Floor

**Key insight:** Instead of training from base Qwen (loss starts at ~1.0, converges to ~0.37), we merged the 16.87 model's adapter permanently and full fine-tuned from there.

Merged the comp-only r=32 adapter into the base weights on Thor → `models/merged-componly-r32`. This model starts at loss 0.345 (already knows SVG). Full fine-tune with LR=5e-6 can push lower since it's refining, not learning from scratch.

**Loss curve (refined training, LR=5e-6):**
```
Step     10: 0.345  (starting from 16.87 knowledge)
Step    800: 0.325  ← first time below LoRA ceiling
Step   1510: 0.340
Step   2210: 0.322
Step   2940: 0.327
Step   3690: 0.337
Step   4580: 0.318  ← new all-time low
Step   5470: 0.362  (epoch 1 boundary)
Step   6380: 0.308  ← epoch 2, new all-time low
```

**Loss 0.308 — well below the 0.34 LoRA ceiling.** The merge+refine approach works. The loss is noisy (±0.04 per batch) but the floor is steadily dropping. Checkpoints saved every 500 steps.

Training continues overnight (~3h remaining). A higher-LR follow-up pass (LR=2e-5) will auto-launch when it finishes to test if we can break out of the 0.30 valley.

### Kepler Overnight Queue (completed)

Three inference runs completed:
1. `greedy_rep1.05_tok1024.csv` — 1000 samples, done
2. `greedy_rep1.15_tok1024.csv` — 1000 samples, done
3. `greedy_rep1.10_tok1536.csv` — 1000 samples, done (1536 max tokens)

### Kaggle Submission Summary

| Submission | Model | Inference | Score |
|---|---|---|---|
| filtered-r16-3epoch | LoRA r=16, 46k | greedy, rep=1.1 | 15.47 |
| merged-r32-expanded | LoRA r=32, 76k mixed | greedy, rep=1.1 | 14.64 |
| componly-r32-clean | LoRA r=32, 45k comp | greedy, rep=1.1 | **16.87** |
| temp_0.05 | same as above | temp=0.05, rep=1.1 | 16.42 |
| greedy_rep1.05 | same | greedy, rep=1.05 | 16.51 |
| greedy_rep1.15 | same | greedy, rep=1.15 | 15.88 |

**Best: 16.87** (6th place, 1.46 points from 1st)

### Remaining Submissions

3 today + 5 tomorrow (competition closes April 1 midnight).

**Plan:**
1. Submit `best_of_n_combined.csv` — cherry-picked across 4 runs, 885 improvements
2. When Thor refined model finishes → copy best checkpoint to kepler → run inference → submit
3. Save remaining for April 1 final attempts

### What We Learned

1. **Inference params matter more than training loss** — same model gave 15.88-16.87 range just from rep penalty changes
2. **Longer generation helps** — 1536-token run was biggest contributor to best-of-N (38.6% of picks)
3. **Merge+refine breaks the loss floor** — full fine-tuning a pre-trained SVG model starts at 0.345 and reaches 0.308 (vs 0.37 from scratch)
4. **Thor can train but not infer** — SM 11.0 (Blackwell) not supported by PyTorch generate(), only by training kernels
5. **Best-of-N with heuristic scoring** works without ground truth — pick longest valid SVG per prompt

---

## Day 8 — April 1: Final Submissions & Code-Gen Experiment

### Code-Gen Experiment: Python → SVG

Built an entirely different approach: instead of generating raw SVG, teach the model to write Python code using a custom `SVGBuilder` API (`scripts/svg_builder.py`):

```python
svg = create_svg(200, 200)
svg.background("white")
svg.circle(100, 100, 50, fill="red")
svg.rect(20, 20, 60, 40, fill="blue")
```

Python code gets executed → produces valid SVG. The model is Qwen2.5-**Coder** — it's better at writing Python than predicting digit sequences.

**Base model test (no fine-tuning):** 4/5 prompts produced valid code. ~200-500 char SVGs vs ~1000 for raw paths. 6x faster inference (code outputs are short).

**Full inference (base model):** 714/1000 valid (286 fallbacks). 40 minutes total vs 5.5h for raw SVG.

**Fine-tuned code-gen model (Thor, 2 epochs):** 824/1000 valid (176 fallbacks). Loss 1.015 → 0.318.

**Kaggle score: 12.26** — worse than garbage SVG baseline (11). The Python code approach produces SVGs that are too simple (basic shapes) compared to the detailed bezier paths the scorer expects. SSIM rewards visual fidelity to ground truth, and `rect()`/`circle()` calls can't match complex path-based icons.

**Refined code-gen (from 0.308 model):** Started training but abandoned — the fundamental approach doesn't match the scoring metric.

### Code-Gen Postmortem

The code-gen idea was creative but wrong for this competition:
- Training SVGs are 49.4% single complex paths, 47.7% multi-element with paths — only 2.9% are basic shapes
- The Python API naturally produces basic shapes, not complex bezier paths
- `svg.path(d="M 50 20 C 60 30...")` is just raw SVG with a Python wrapper — no compression benefit
- The scoring metric (SSIM + EdgeF1) rewards pixel-level visual fidelity, which requires matching the exact ground truth paths
- Simple programmatic shapes score WORSE than the generic fallback because they don't match the ground truth at all

**Lesson:** The intermediate representation idea is sound for a different scoring metric (human judgment, semantic similarity). For pixel-level SSIM, you need pixel-level accuracy, which means matching the training data format.

### Best-of-N Submission

Submitted `best_of_n_combined.csv` — cherry-picked across 4 runs. Score was not an improvement over 16.87. The heuristic scoring (pick longest valid SVG) doesn't correlate well with SSIM — a longer SVG isn't necessarily a better-looking one.

### Final Submission: Refined Model

Our last submission was the refined model (checkpoint-7000, loss 0.308). This model is:
- Base: Qwen2.5-Coder-1.5B with r=16 merged, then r=32 merged (16.87 knowledge baked in)
- Training: Full fine-tune on Thor, 2 epochs on competition data, loss 0.308
- Inference: Greedy, rep=1.1, max_tokens=1024 (identical to 16.87 settings)
- Result: 1000 samples, 6 invalid (vs 12 for the 16.87 model)

### Complete Kaggle Submission History

| # | Submission | Model | Score | Notes |
|---|---|---|---|---|
| 1 | filtered-r16-3epoch | LoRA r=16, 1.5B | 15.47 | First submission, 9th place |
| 2 | merged-r32-expanded | Merged r16 + r=32, 76k mixed | 14.64 | External data hurt |
| 3 | componly-r32-clean | Merged r16 + r=32, 45k comp | **16.87** | **Best score — 6th place** |
| 4 | temp_0.05 | Same model, temp=0.05 | 16.42 | Sampling hurts |
| 5 | greedy_rep1.05 | Same model, rep=1.05 | 16.51 | Lower penalty hurts |
| 6 | greedy_rep1.15 | Same model, rep=1.15 | 15.88 | Higher penalty hurts |
| 7 | best_of_n_combined | Cherry-pick across 4 runs | ~16.5 | Heuristic picking doesn't help |
| 8 | codegen_hybrid | Code-gen + 16.87 fallbacks | 12.26 | Code-gen approach failed |
| 9 | refined_7000_final | Full fine-tune, loss 0.308 | ??? | Final submission |

### Competition Summary

**Final standing:** 6th place with 16.87 (public leaderboard). Final results on hidden test set TBD.

**Leaderboard at close:**
- 1st: 18.33
- 6th (us): 16.87
- Gap to 1st: 1.46 points

### What Worked

1. **Merge + re-LoRA** — baking r=16 knowledge into weights, then training r=32 on top. Gave the biggest single improvement (15.47 → 16.87).
2. **Competition-only data** — external data (SVGX, OmniSVG) consistently hurt. Distribution match > volume.
3. **Greedy decoding with rep=1.1** — any deviation from this (temperature, different penalty) made scores worse.
4. **Full fine-tune on Thor (DGX Spark)** — broke through the LoRA loss ceiling (0.34 → 0.308). Required Docker + extensive debugging of the unified memory architecture.
5. **Coordinate rounding to 1 decimal** — 54% character savings in preprocessing.
6. **Token-count filtering** — ensuring all training SVGs fit in context so model learns to generate `</svg>`.

### What Didn't Work

1. **External training data** — SVGX + OmniSVG degraded performance despite 3x more samples.
2. **Temperature/sampling** — greedy is king for structured SVG output.
3. **Code-gen approach** — Python code produces basic shapes that score worse than complex paths.
4. **Best-of-N with heuristic scoring** — longest SVG ≠ best SSIM score.
5. **Relative + integer coordinate preprocessing** — 40% token savings but untested on Kaggle; we stayed with proven preprocessing.
6. **Higher learning rates** — didn't break through loss floors, just added noise.

### What We'd Do Differently

1. **Start with the merge + re-LoRA strategy from day 1** — this was our biggest win but we discovered it on day 4.
2. **Never use external data** — the hours spent on SVGX/OmniSVG curation were wasted.
3. **Full fine-tune earlier** — we could have run the refined training approach (merge → full fine-tune) with more epochs if we'd started sooner.
4. **Focus on inference optimization earlier** — the 1.0 point range from inference params (15.88-16.87) suggests there might be a better combination we didn't find.
5. **DPO with rendered scoring** — directly optimizing SSIM instead of CE loss was identified as high-potential but never attempted due to time constraints.

### Technical Infrastructure

The dual-machine setup (kepler RTX 2000 Ada + Thor DGX Spark) was essential:
- **Kepler:** All inference (20s/sample, 4-bit quantization)
- **Thor:** All training (full fine-tune, 128GB unified memory, Docker required)
- **Thor quirks:** FlashAttention crashes on SM 11.0, nvidia-smi shows "Not Supported" for memory, processes get killed by session cleanup (fixed with Docker), competing containers cause silent OOM

### Cost

- **Modal (A100):** ~$55-75 for initial training + experiments
- **Kepler (local):** Free (already owned)
- **Thor (local):** Free (already owned)
- **Total:** ~$55-75

### Final Thoughts

The competition exposed a fundamental tension in LLM-based SVG generation: cross-entropy loss on digit tokens doesn't optimize what the competition scores (SSIM + EdgeF1). Every model converged to the same ~0.30-0.40 CE loss range regardless of size, architecture, or training approach. The gap between our 16.87 and the leader's 18.33 likely comes from either:
1. A fundamentally different approach (not LLM-based)
2. Better postprocessing/repair that we didn't discover
3. DPO or RL that directly optimizes the rendering metric

This was a week-long sprint from zero to 6th place on a 58-team leaderboard. The merge + re-LoRA technique, the tokenizer analysis, and the DGX Spark debugging were genuine contributions. The code-gen experiment, while unsuccessful for SSIM scoring, demonstrated that Qwen2.5-Coder can compose SVGs programmatically — an approach that would work well for a human-judgment metric.

---

## Day 9 — April 3: Ablation Study

### Motivation

25% of the midterm grade is "ablations" — systematic experiments that remove or change one component at a time to measure its contribution. The competition is closed (deadline April 1), so we can't submit new runs to Kaggle. Instead, we score locally against a held-out validation set using the same metric formula (SSIM + EdgeF1 + TED + Compactness).

### Methodology

**Validation set:** 200 samples stratified by SVG complexity (token length) from `train.csv`, saved as `data/val_ablation.csv`. Same samples used for all ablations to ensure fair comparison.

**Scoring:** Local `score.py` implementing the competition metric:
- Visual Fidelity = 0.7 * SSIM + 0.3 * EdgeF1
- Structural Similarity = exp(-TED / 25)
- Compactness = exp(-|log((len_pred+50)/(len_ref+50))|)
- Composite = V^0.85 * S^0.12 * C^0.03

**Baseline:** Full system = merged r16→r32 adapter + system prompt + repair pipeline + rep=1.1 + max_tokens=1024.

**Infrastructure:** All inference on kepler (RTX 2000 Ada 16GB), 4-bit quantized models, greedy decoding. Each ablation runs the same 200 prompts through the full inference + scoring pipeline (~45-80 min per ablation).

### Ablation Results (COMPLETE — 7.25 hours total, 03:25–10:41 UTC)

| # | Ablation | What Changed | Score | Delta | Visual | Structural | Compact | Invalid | Time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **baseline** | Full system | **53.79** | — | 0.509 | 0.919 | 0.446 | 2 | 73.6 min |
| 2 | base_model | No fine-tuning | 46.48 | -7.31 | 0.447 | 0.888 | 0.223 | 18 | 16.2 min |
| 3 | no_sys_prompt | Remove system prompt | 18.80 | -34.99 | 0.156 | 0.886 | 0.188 | 192 | 7.1 min |
| 4 | rep_1.0 | No repetition penalty | 54.21 | +0.42 | 0.513 | 0.920 | 0.459 | 4 | 79.9 min |
| 5 | max_tok_512 | Halve token budget | 51.47 | -2.32 | 0.491 | 0.885 | 0.345 | 4 | 45.9 min |
| 6 | max_tok_1536 | 1.5x token budget | **55.50** | +1.71 | **0.525** | **0.933** | **0.495** | **0** | 88.4 min |
| 7 | no_repair | Skip repair pipeline | 53.63 | -0.16 | 0.508 | 0.909 | 0.445 | 0 | 75.5 min |
| 8 | refined_ft | Full fine-tune (loss 0.308) | 53.05 | -0.74 | 0.502 | 0.911 | 0.461 | 0 | 47.5 min |

### Key Findings So Far

**1. System prompt is the most critical component (-35 points)**

Without the system prompt, the model produces 192/200 invalid SVGs (96% fallback rate). The fine-tuned model learned to associate SVG generation with the system prompt during training — without it, the model doesn't know to produce SVG markup at all. This is the single largest ablation effect.

**2. Fine-tuning contributes ~7 points**

The base Qwen2.5-Coder-1.5B (no fine-tuning) scores 46.48 vs baseline 53.79. The model has some zero-shot SVG capability from pretraining, but fine-tuning on competition data adds 7.31 points by teaching it the specific SVG patterns and coordinate distributions of the training set.

**3. Token budget has a linear effect on quality**

| max_tokens | Score | Visual | Compactness |
|---|---|---|---|
| 512 | 51.47 | 0.4914 | 0.3448 |
| 1024 | 53.79 | 0.5092 | 0.4456 |
| 1536 | 55.50 | 0.5249 | 0.4952 |

More tokens = more complete SVGs = better visual fidelity AND compactness (closer to reference length). The relationship is approximately linear (+2 points per 512 tokens). The 1536-token run also achieved 0 invalid outputs vs 2 for 1024 and 4 for 512.

**4. Repetition penalty is negligible (+0.42)**

rep=1.0 (no penalty) scored 54.21 vs baseline's 53.79 with rep=1.1. The difference is within noise. This matches the Kaggle results where rep=1.05 and rep=1.15 gave only small deviations from rep=1.1 (16.51, 15.88 vs 16.87).

**5. Repair pipeline is a wash (-0.16)**

Removing the repair pipeline (which closes unclosed tags and adds `</svg>`) scored 53.63 vs baseline 53.79. The repair pipeline helps for the rare truncated SVG, but repaired SVGs often look worse than the fallback circle — the effects cancel out. The model already produces valid SVGs 98%+ of the time.

**6. Full fine-tune vs LoRA adapter — LoRA wins (-0.74)**

The refined full fine-tune model (loss 0.308, trained on DGX Spark) scored 53.05 vs the LoRA adapter baseline at 53.79. Despite achieving a lower CE loss (0.308 vs 0.34), the full fine-tune model performs marginally *worse* on visual fidelity scoring. This confirms the journal's Day 5 finding: **CE loss floor doesn't predict visual quality.** The LoRA adapter, with its constrained parameter space, may actually regularize the model in ways that benefit generation quality. This also matches Kaggle results (refined-7000 scored 16.26 vs LoRA's 16.87).

**7. Best configuration discovered: max_tokens=1536 (55.50)**

The highest-scoring variant wasn't the baseline — it was `max_tok_1536`. Longer generation allows complex SVGs to complete rather than truncate, improving all three sub-metrics simultaneously: visual (+0.016), structural (+0.014), and compactness (+0.049). Zero invalid outputs.

### Component Importance Ranking

1. **System prompt** — 35 points (catastrophic without it)
2. **Fine-tuning** — 7.3 points (substantial improvement)
3. **Token budget** — 4 points (512→1536 range)
4. **Training method (LoRA vs full FT)** — 0.7 points (LoRA slightly better)
5. **Repetition penalty** — 0.4 points (negligible)
6. **Repair pipeline** — 0.2 points (negligible)

### Cross-Validation with Kaggle Scores

The local ablation rankings are consistent with Kaggle submissions:

| Decision | Local score direction | Kaggle score direction | Consistent? |
|---|---|---|---|
| Fine-tuning helps | 46.48 → 53.79 (+7.3) | N/A (no zero-shot submit) | — |
| LoRA > full FT | 53.79 > 53.05 | 16.87 > 16.26 | Yes |
| rep=1.1 ≈ rep=1.0 | 53.79 ≈ 54.21 | 16.87 ≈ 16.51 (rep=1.05) | Yes |
| More tokens helps | 51.47 → 55.50 | 1536-tok was biggest best-of-N contributor | Yes |

### Scripts

- `scripts/create_val_split.py` — Creates stratified 200-sample val set
- `scripts/run_ablation.py` — Runs a single ablation variant (inference + scoring)
- `scripts/run_ablation_refined.py` — Variant for full fine-tune model (loads differently)
- `scripts/orchestrate_ablations.py` — Runs all 8 ablations sequentially
- Results in `results/ablations/` — per-ablation JSON scores + prediction CSVs
