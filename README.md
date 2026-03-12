# MatrixArena 🏟️

> **A decentralised, peer-review evaluation framework where top-tier LLMs dynamically generate tasks, solve them, and judge each other.**

Static benchmarks like MMLU are easily gamed. MatrixArena replaces them with a continuous, rotating evaluation loop: one model generates a coding problem, **every model** independently solves it, and **all non-generator models** cross-judge every solution. The result is an Elo-based leaderboard that reflects multi-dimensional coding ability and is hard to overfit.

---

## Architecture

```
MatrixArena/
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── .env.example             # API key template
│
├── config/
│   ├── settings.py          # Config loader (YAML + env vars, auto-disables missing keys)
│   └── models.yaml          # Pool of participating models
│
├── prompts/                 # Prompt templates
│   ├── generator.txt        # Novel problem generation + Elo incentive rules
│   ├── solver.txt           # Solution output + originality warning + Elo incentive
│   └── judge.txt            # Scoring rubric + execution result + Judge Elo incentive
│
├── core/
│   ├── gateway.py           # litellm async wrapper (retries, backoff, empty/truncation recovery)
│   ├── orchestrator.py      # Full Gen → Solve × N → Execute × N → Judge × N² loop
│   └── elo_rating.py        # Elo for Solvers (K=32) + Generator calibration (K=16)
│
├── sandbox/
│   ├── Dockerfile           # Isolated execution image (non-root, no network)
│   └── executor.py          # SubprocessExecutor: syntax check + per-test harness
│
├── data/                    # Runtime outputs (git-ignored)
│   ├── battles.jsonl        # Append-only compact battle log
│   ├── leaderboard.json     # Latest Elo snapshot
│   └── cycles/              # Full per-cycle artifacts
│       └── <YYYYMMDD_HHMMSS>_c<N>/
│           ├── summary.json
│           ├── generator_problem.json
│           ├── solver_<model_slug>.json       # one per solver
│           ├── execution_<model_slug>.json    # one per solver
│           └── judgments_<model_slug>.json    # one per solver (all judges)
│
└── dashboard/
    └── app.py               # Streamlit UI (leaderboard, Elo history, cycle detail)
```

---

## Evaluation Cycle

Each cycle runs **5 sequential phases**:

```
### Phase 1 ── GENERATION ──────────────────────────────────────────
  **Modes:**
  - **Random:** A model is randomly chosen as the Generator.
  - **Sequential (NEW):** If the total number of cycles specified (`--cycles`) is a multiple of the number of available models, the system enter **Sequential Mode**. Each model takes turns being the Generator exactly once (or twice, if `cycles = 2 * N`, etc.). This ensures 100% fairness in problem generation opportunities.
  
  The Generator creates a novel coding problem including:
  (JSON: title, description, test_cases, evaluation_criteria)

### Phase 2 ── SOLVING (all N models, concurrent) ──────────────────
  Every model in the pool — including the Generator — independently
  writes a Python solution to the same problem.
  **Resilience:** If a solver times out (default 240s) or fails, it receives a score based on its actual performance (usually 0).

### Phase 3 ── SANDBOX EXECUTION ───────────────────────────────────
  Each of the N solutions is run locally in a subprocess sandbox.
  Syntax check → test harness injection → per-test pass/fail.
  **Hardened Sandbox:** Prevents infinite loops or hung processes using `stdin=null` and strict timeouts.

### Phase 4 ── JUDGING (concurrent) ────────────────────────────────
  All N models (including the Generator) act as Judges.
  Each Judge scores every solution except its own.
  → N × (N-1) total scoring calls, all concurrent.
  **Fairness & Reliability:** 
  - A model never judges its own solution.
  - **Score Aggregation:** If a judge fails to provide a score (timeout or API error), its "vote" is discarded. It no longer defaults to 5.0, ensuring that unreliable judges don't skew the leaderboard.

### Phase 5 ── ELO UPDATE ──────────────────────────────────────────
  Each Solver's Elo is updated from its aggregate judge score (K=32).
  The Generator's Elo is updated by a calibration formula:
    outcome = 1 − |normalized_score − 0.5| × 2
  Peaks at score=5/10 (good difficulty), penalises trivial or
  impossible problems.
---
## Key Features & Resilience

- **Sequential Generator Mode:** Automatically triggered when `cycles` is a multiple of the model count, ensuring every model serves as a generator equally.
- **Enhanced Timeouts:** Default 240s timeout for API calls to accommodate slow-reasoning models (e.g., Qwen, O1/O3).
- **Raw Output Logging:** Every raw LLM response is saved in `data/cycles/<timestamp>/raw/` for deep-dive analysis and debugging.
- **Hard Abort on Generation Failure:** If a generator fails to produce a valid problem, the cycle is aborted immediately (preventing "Two Sum" fallbacks) to save execution time and credits.
- **Robust JSON Parsing:** Advanced multi-stage parsing (Regex + Brace Balance) allows the system to extract JSON from models that provide conversational "fluff" or Markdown wrappers.
```

**Fairness rule:** a model never judges its own solution.

### Role matrix for N=10 models

| Role | Count | Who |
|---|---|---|
| Generator | 1 | Randomly chosen each cycle |
| Solvers | 10 | All models (inc. Generator) |
| Judges per solution | 9 | All models except the Solver being judged |
| Total judge API calls | ~90 | Fully concurrent |

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/Wes1eyyy/MatrixArena.git
cd MatrixArena
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Set OPENROUTER_API_KEY (required)
# Set ARK_API_KEY if you want Doubao Seed 2.0 Code (optional)
```

### 3. Run evaluation cycles

```bash
# Run 3 cycles (default from .env NUM_CYCLES or hardcoded default)
python main.py

# Run a specific number of cycles
python main.py --cycles 10
```

Progress is printed in real time as each phase completes:

```
--- Cycle 1/3 ---
  Generator  : openrouter/anthropic/claude-opus-4.6
  Solvers    : all 10 models
  Judges pool: 10 models (all, each skips own solution)
  [1/5] Generating problem...
        Problem : "Chrono-Gated Courier Network"
  [2/5] Solving (10 solvers in parallel)...
        [claude-opus-4.6] → 42 line(s) of code
        [gemini-3-pro-preview] → 38 line(s) of code
        ...
  [3/5] Running sandbox execution for all solutions...
        [claude-opus-4.6] 5/5 tests passed
        ...
  [4/5] Judging (90 calls: 10 solutions × ~9 judges each)...
        [grok-4.1-fast] judged [gemini-3-pro-preview] → 8.5/10
        ...
  [5/5] Updating Elo ratings...
  Overall avg : 7.42/10
  Best solver : claude-opus-4.6 (8.90/10)
```

### 4. View the dashboard

```bash
streamlit run dashboard/app.py
```

Opens a Streamlit web app with:
- **Leaderboard** — Elo bar chart + ranked table
- **Elo History** — per-cycle line chart with model filter
- **Battle Log** — summary table of all cycles
- **Cycle Detail** — per-solver code, execution results, judge radar chart, feedback

---

## CLI Reference

| Command | Description |
|---|---|
| `python main.py` | Run cycles (default or `NUM_CYCLES` from `.env`) |
| `python main.py --cycles N` | Run exactly N cycles |
| `python main.py --health-check` | Ping all models and exit |
| `python main.py --no-health-check` | Skip health check and run immediately |
| `python main.py --reset-last` | Remove the most recent cycle and revert Elo |
| `python main.py --reset-all` | Wipe all battles, leaderboard, and cycle artifacts |

### Health check behaviour

- **`[OK ]`** — model responded within timeout
- **`[429]`** — model is temporarily rate-limited upstream; auto-excluded from this run
- **`[FAIL]`** — hard error (auth, not found, etc.); run aborts unless `--no-health-check`

---

## Gateway Resilience

`core/gateway.py` handles production edge cases automatically:

| Situation | Behaviour |
|---|---|
| Rate limit (429) | Waits `retry_after_seconds` from the error, then retries |
| Transient error | Exponential backoff: 1 s → 2 s → 4 s … |
| Empty response | Detected as failure, retried with backoff |
| Truncated response (`finish_reason=length`) | Retries with continuation prompt asking for complete JSON |
| Health check 429 | Fast-fails immediately (no wait); model skipped for this run |

---

## Data Storage

### `data/battles.jsonl` — compact append-only log

One JSON line per cycle. Fields:

```jsonc
{
  "cycle_num": 1,
  "timestamp": 1741696367,
  "generator": "openrouter/anthropic/claude-opus-4.6",
  "solvers": ["openrouter/anthropic/claude-opus-4.6", ...],   // all N models
  "judges_pool": ["openrouter/anthropic/claude-opus-4.6", ...],    // all N
  "problem_title": "Chrono-Gated Courier Network",
  "average_scores": {"openrouter/anthropic/claude-opus-4.6": 8.9, ...},
  "overall_average": 7.42,
  "elo_after": {"openrouter/anthropic/claude-opus-4.6": 1218.4, ...}
}
```

### `data/cycles/<YYYYMMDD_HHMMSS>_c<N>/` — full per-cycle artifacts

| File | Contents |
|---|---|
| `summary.json` | Roles, per-solver scores, overall average, Elo snapshot |
| `generator_problem.json` | Complete Generator output: title, description, test cases, criteria |
| `solver_<slug>.json` | Runnable Python code + explanation (one file per solver) |
| `execution_<slug>.json` | Sandbox result: per-test status, pass/fail count, stderr |
| `judgments_<slug>.json` | All judge scores + feedback for this solver (one file per solver) |

**Example layout after 1 cycle with 10 models:**

```
data/cycles/20260311_143022_c1/
├── summary.json
├── generator_problem.json
├── solver_openrouter_anthropic_claude-opus-4_6.json
├── solver_openrouter_google_gemini-3-pro-preview.json
├── ... (×10 solvers)
├── execution_openrouter_anthropic_claude-opus-4_6.json
├── ... (×10)
├── judgments_openrouter_anthropic_claude-opus-4_6.json
└── ... (×10)
```

> All files under `data/` are git-ignored and never committed.

---

## Configuration

Edit `config/models.yaml` to change the model pool. Any model with an `api_key_env` pointing to an unset environment variable is automatically disabled at startup.

```yaml
models:
  - id: openrouter/anthropic/claude-opus-4.6
    provider: openrouter
    display_name: Claude Opus 4.6

  # Direct endpoint (non-OpenRouter)
  - id: openai/doubao-seed-2.0-code
    provider: ark
    display_name: Doubao Seed 2.0 Code
    api_base: https://ark.cn-beijing.volces.com/api/coding/v3
    api_key_env: ARK_API_KEY   # disabled if ARK_API_KEY is not set

initial_elo: 1200
```

Minimum pool size: **3 models** (1 generator + 1 solver + 1 judge).

---

## Roadmap

- [x] Real sandbox execution via subprocess (syntax check + per-test harness)
- [x] Generator Elo incentive (problem calibration quality scoring)
- [x] Full per-cycle artifact storage (`data/cycles/`)
- [x] OpenRouter multi-provider support (10 models)
- [x] ARK (ByteDance) direct endpoint support for Doubao
- [x] All-vs-all interaction: every model solves, all non-generator models judge
- [x] Streamlit dashboard (leaderboard, Elo history, cycle detail, per-solver view)
- [x] Health check with rate-limit awareness (429 → skip, hard fail → abort)
- [x] Gateway resilience (empty response retry, truncation recovery, exponential backoff)
- [x] `--reset-last` / `--reset-all` data management commands
- [ ] Docker sandbox upgrade (replace subprocess with `docker run`)
- [ ] Task difficulty classification and weighted scoring
- [ ] Multi-language support (JavaScript, Go, …)
