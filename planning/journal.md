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

- Should we pivot to SDXL + VTracer approach? (Better visual quality, competition rules allow it)
- Can we use external HF datasets? → **NO, TA confirmed train.csv only**
- 50k samples enough? → Probably, with better preprocessing
- Is the scoring metric (SSIM + EdgeF1) achievable with LLM approach, or does it require raster-first?

---

## Current State (end of Day 2)

### Running on Modal
- Inference: 4 containers generating 1000 samples from r=16 model (ETA ~1:00 AM)
- Training: filtered dataset run (45,874 samples, r=16, 3 epochs)

### Assets
- Trained model: models/r16-3epoch/ (74MB adapter)
- Preprocessed data pipeline: clean_svg(), curate_training_data()
- Inference pipeline: multi-container fan-out with repair
- Local scoring: score.py (SSIM + EdgeF1 + TED + Compactness)
- Experiment runner: experiment.py
- Decode sweep: decode_sweep.py
