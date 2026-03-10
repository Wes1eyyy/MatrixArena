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
├── prompts/                 # Prompt Engineering Library
│   ├── generator.txt        # How to generate coding tasks
│   ├── solver.txt           # How to output code solutions
│   └── judge.txt            # Scoring dimensions + JSON schema
│
├── core/
│   ├── gateway.py           # litellm wrapper (async, retries)
│   ├── orchestrator.py      # Gen → Solve → Judge loop
│   └── elo_rating.py        # Elo rating updates
│
├── sandbox/
│   ├── Dockerfile           # Isolated execution env (placeholder)
│   └── executor.py          # Mock executor (returns "Pass" in MVP)
│
├── data/
│   ├── battles.jsonl        # Append-only battle log
│   └── leaderboard.json     # Latest Elo snapshot
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

- [ ] Real sandbox execution via Docker (replace `MockExecutor`)
- [ ] Streamlit leaderboard dashboard (`dashboard/app.py`)
- [ ] Persistent Elo history with time-series plots
- [ ] Support for more model providers (Cohere, Mistral, etc.)
- [ ] Task difficulty classification and weighted scoring
