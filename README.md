# MatrixArena 🏟️

> **A decentralised, peer-review evaluation framework where top-tier LLMs dynamically generate tasks, solve them, and judge each other.**

Static benchmarks like MMLU are easily gamed. MatrixArena replaces them with a continuous, rotating evaluation loop: models act as Generators, Solvers, and Judges in turn, producing an Elo-based leaderboard that is hard to overfit.

---

## Architecture

```
MatrixArena/
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── .env.example             # API key template
│
├── config/
│   ├── settings.py          # Config loader (YAML + env vars)
│   └── models.yaml          # Pool of participating models
│
├── prompts/                 # Prompt templates
│   ├── generator.txt        # Novel problem generation + Elo incentive rules
│   ├── solver.txt           # Solution output + originality warning
│   └── judge.txt            # Scoring rubric + execution result integration
│
├── core/
│   ├── gateway.py           # litellm wrapper (async, retries, api_base support)
│   ├── orchestrator.py      # Gen → Solve → Judge loop
│   └── elo_rating.py        # Elo for Solver + Generator calibration score
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
│           ├── solver_solution.json
│           ├── execution_result.json
│           └── judge_<model_slug>.json
│
└── dashboard/
    └── app.py               # Streamlit UI (stub)
```

---

## Evaluation Cycle

```
┌─────────────┐     coding problem      ┌─────────────┐
│  Generator  │ ──────────────────────▶ │   Solver    │
│  (Model A)  │                         │  (Model B)  │
└─────────────┘                         └──────┬──────┘
                                               │ solution
                                               ▼
                                     ┌──────────────────┐
                                     │   Judge Pool     │
                                     │ (all except B)   │
                                     └────────┬─────────┘
                                              │ scores (JSON)
                                              ▼
                                       Elo Rating Update
```

**Fairness Rule:** A model **cannot judge its own answer**. The Solver is explicitly excluded from the Judge pool every cycle.

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
# Edit .env and add your OpenAI, Anthropic, and Google API keys
```

### 3. Run an evaluation cycle

```bash
# Run 3 evaluation cycles (default)
python main.py

# Run a specific number of cycles
python main.py --cycles 5
```

The leaderboard is printed at the end and persisted to `data/leaderboard.json`. Each cycle is logged to `data/battles.jsonl`.

---

## Data Storage

Every evaluation cycle writes outputs to two locations:

### `data/battles.jsonl` — compact append-only log

One JSON line per cycle, containing role assignments, aggregate scores, per-judge scores, execution summary, and the post-cycle Elo snapshot. **Full solution code is omitted** to keep the file scannable.

Useful for: Elo trend analysis, win-rate statistics, quick `grep` / `jq` queries.

### `data/cycles/<YYYYMMDD_HHMMSS>_c<N>/` — full per-cycle artifacts

One directory per cycle, named by UTC timestamp and cycle number.

| File | Contents |
|---|---|
| `summary.json` | Roles, scores, Elo snapshot, execution summary (no code) |
| `generator_problem.json` | Complete Generator output: title, description, test cases, evaluation criteria |
| `solver_solution.json` | Complete Solver output: runnable Python code + explanation with complexity |
| `execution_result.json` | Sandbox per-test results: status, pass/fail per test case, stderr |
| `judge_<slug>.json` | Per-judge output: 5-dimension scores, overall score, feedback text |

**Example directory layout after 2 cycles:**

```
data/
├── battles.jsonl
├── leaderboard.json
└── cycles/
    ├── 20260311_143022_c1/
    │   ├── summary.json
    │   ├── generator_problem.json
    │   ├── solver_solution.json
    │   ├── execution_result.json
    │   ├── judge_openrouter_anthropic_claude-opus-4_6.json
    │   └── judge_openrouter_google_gemini-3-pro-preview.json
    └── 20260311_143301_c2/
        └── ...
```

> All files under `data/` are git-ignored and never committed to the repository.

---

## Configuration

Edit `config/models.yaml` to change the model pool:

```yaml
models:
  - id: gpt-4o
    provider: openai
    display_name: GPT-4o
  - id: claude-3-5-sonnet-20240620
    provider: anthropic
    display_name: Claude 3.5 Sonnet
  - id: gemini/gemini-1.5-pro
    provider: google
    display_name: Gemini 1.5 Pro

initial_elo: 1200
```

At least **3 models** are required (one Generator, one Solver, one or more Judges).

---

## Roadmap

- [x] Real sandbox execution via subprocess (syntax check + per-test harness)
- [x] Generator Elo incentive (problem calibration quality scoring)
- [x] Full per-cycle artifact storage (`data/cycles/`)
- [x] OpenRouter multi-provider support (10 models)
- [x] ARK (ByteDance) direct endpoint support for Doubao
- [ ] Streamlit leaderboard dashboard (`dashboard/app.py`)
- [ ] Docker sandbox upgrade (replace subprocess with `docker run`)
- [ ] Persistent Elo history with time-series plots
- [ ] Task difficulty classification and weighted scoring
