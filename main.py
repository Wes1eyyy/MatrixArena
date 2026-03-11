"""
MatrixArena - CLI entry point.

Run an evaluation cycle where LLMs generate tasks, solve them, and judge each other.

Usage:
    python main.py [--cycles N] [--no-health-check]
"""

import argparse
import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from config.settings import Settings
from core.gateway import call_model
from core.orchestrator import Orchestrator

_HEALTH_PROMPT = "Reply with exactly one word: OK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MatrixArena: LLM peer-review evaluation framework"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Number of evaluation cycles to run (overrides NUM_CYCLES in .env)",
    )
    parser.add_argument(
        "--no-health-check",
        action="store_true",
        help="Skip the model health check before running evaluation cycles",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run the model health check and exit without starting evaluation cycles",
    )
    return parser.parse_args()


_RATE_LIMIT_MARKERS = ("ratelimiterror", "429", "rate_limit", "rate-limit", "retry_after")


def _is_rate_limit(err: str) -> bool:
    low = err.lower()
    return any(m in low for m in _RATE_LIMIT_MARKERS)


async def health_check(settings: Settings) -> tuple[list[str], list[str]]:
    """
    Ping every enabled model concurrently with a minimal prompt.

    Returns
    -------
    hard_failed : list[str]
        Model IDs with hard errors (auth, not found, etc.) — caller should abort.
    rate_limited : list[str]
        Model IDs that returned a 429 rate-limit — caller should skip for this run.
    """
    models = settings.models
    print(f"Health check ({len(models)} model(s))...")

    async def _ping(cfg) -> tuple[str, bool, float, str]:
        start = time.monotonic()
        try:
            await call_model(
                cfg.id,
                _HEALTH_PROMPT,
                temperature=0.0,
                max_tokens=16,
                retries=1,
                backoff_on_rate_limit=False,   # health check must be fast
                **settings.model_extra(cfg.id),
            )
            elapsed = time.monotonic() - start
            return cfg.display_name, True, elapsed, ""
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            return cfg.display_name, False, elapsed, str(exc)

    results = await asyncio.gather(*[_ping(m) for m in models])

    hard_failed: list[str] = []
    rate_limited: list[str] = []
    for m, (display, ok, elapsed, err) in zip(models, results):
        timing = f"[{elapsed:.2f}s]"
        if ok:
            print(f"  [OK ]  {display:<40} {timing}")
        elif _is_rate_limit(err):
            print(f"  [429 ] {display:<40} {timing}  rate-limited (will skip this run)")
            rate_limited.append(m.id)
        else:
            short_err = err.splitlines()[0][:80] if err else "unknown error"
            print(f"  [FAIL] {display:<40} {timing}  {short_err}")
            hard_failed.append(m.id)

    total_failed = len(hard_failed) + len(rate_limited)
    passed = len(models) - total_failed
    print(f"Health check complete: {passed}/{len(models)} passed.\n")
    return hard_failed, rate_limited



async def run(cycles: int, skip_health_check: bool = False) -> None:
    settings = Settings()
    orchestrator = Orchestrator(settings)

    print("=" * 60)
    print("  MatrixArena — LLM Peer-Review Evaluation")
    print("=" * 60)
    print(f"Enabled models ({len(settings.model_names)}):")
    for name in settings.model_names:
        print(f"  + {name}")
    print()

    if not skip_health_check:
        hard_failed, rate_limited = await health_check(settings)

        # Rate-limited: temporarily remove from this run's model pool
        if rate_limited:
            print(f"WARNING: {len(rate_limited)} model(s) are rate-limited and will be excluded from this run:")
            for mid in rate_limited:
                print(f"  - {mid}")
            settings.models = [m for m in settings.models if m.id not in rate_limited]
            if len(settings.models) < 3:
                print(f"ABORT: Only {len(settings.models)} model(s) remain after excluding rate-limited ones.")
                print("Need at least 3 models (1 generator + 1 solver + 1 judge). Re-run later or add more models.")
                raise SystemExit(1)
            print(f"  Continuing with {len(settings.models)} model(s).\n")

        # Hard failures: abort
        if hard_failed:
            print(f"ABORT: {len(hard_failed)} model(s) failed health check:")
            for f in hard_failed:
                print(f"  - {f}")
            print("Fix the failing models or run with --no-health-check to skip.")
            raise SystemExit(1)
    else:
        print("(Health check skipped)\n")

    print(f"Running {cycles} evaluation cycle(s)...\n")

    for cycle_num in range(1, cycles + 1):
        print(f"--- Cycle {cycle_num}/{cycles} ---")
        result = await orchestrator.run_cycle(cycle_num=cycle_num)

        print(f"  Generator : {result['generator']}")
        print(f"  Solver    : {result['solver']}")
        print(f"  Judges    : {', '.join(result['judges'])}")
        print(f"  Exec      : {result['execution_result']['summary']}")
        print(f"  Avg Score : {result['average_score']:.2f}/10")

        # Persist logs
        _append_battle_log(result)
        _save_cycle_artifacts(result)

    # Print final leaderboard
    leaderboard = orchestrator.elo.get_leaderboard()
    _save_leaderboard(leaderboard)

    print("\n" + "=" * 60)
    print("  Final Leaderboard (Elo Ratings)")
    print("=" * 60)
    for rank, (model, rating) in enumerate(leaderboard, start=1):
        print(f"  {rank}. {model:<40} {rating:.1f}")
    print()


def _append_battle_log(result: dict) -> None:
    """
    Append a compact summary line to data/battles.jsonl.

    Full solution code and per-test execution details are intentionally
    omitted here to keep the JSONL scannable.  Full artifacts are written
    by _save_cycle_artifacts().
    """
    execution = result.get("execution_result", {})
    compact = {
        "cycle_num":    result.get("cycle_num"),
        "timestamp":    result["timestamp"],
        "generator":    result["generator"],
        "solver":       result["solver"],
        "judges":       result["judges"],
        "problem_title": result["problem"].get("title", "unknown"),
        "exec_status":  execution.get("status"),
        "exec_summary": execution.get("summary"),
        "exec_passed":  execution.get("passed"),
        "exec_failed":  execution.get("failed"),
        "judge_scores": [
            {
                "judge":         jr.get("judge"),
                "overall_score": jr.get("overall_score"),
                "scores":        jr.get("scores"),
            }
            for jr in result.get("judge_scores", [])
        ],
        "average_score": result["average_score"],
        "elo_after":     result["elo_after"],
    }
    log_path = os.path.join(os.path.dirname(__file__), "data", "battles.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(compact, ensure_ascii=False) + "\n")


def _save_cycle_artifacts(result: dict) -> None:
    """
    Write full per-cycle artifacts to data/cycles/<dir>/.

    Directory name: <YYYYMMDD_HHMMSS>_c<cycle_num>

    Files written
    -------------
    summary.json          — roles, scores, elo snapshot (no code)
    generator_problem.json — complete Generator output
    solver_solution.json  — complete Solver output (code + explanation)
    execution_result.json — sandbox per-test results
    judge_<slug>.json     — one file per Judge (scores + feedback)
    """
    ts = datetime.fromtimestamp(result["timestamp"], tz=timezone.utc)
    dir_name = f"{ts.strftime('%Y%m%d_%H%M%S')}_c{result.get('cycle_num', 0)}"
    cycle_dir = os.path.join(os.path.dirname(__file__), "data", "cycles", dir_name)
    os.makedirs(cycle_dir, exist_ok=True)

    def _write(filename: str, data: object) -> None:
        with open(os.path.join(cycle_dir, filename), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # summary.json
    _write("summary.json", {
        "cycle_num":     result.get("cycle_num"),
        "timestamp":     result["timestamp"],
        "timestamp_utc": ts.isoformat(),
        "generator":     result["generator"],
        "solver":        result["solver"],
        "judges":        result["judges"],
        "problem_title": result["problem"].get("title", "unknown"),
        "average_score": result["average_score"],
        "exec_status":   result["execution_result"].get("status"),
        "exec_summary":  result["execution_result"].get("summary"),
        "exec_passed":   result["execution_result"].get("passed"),
        "exec_failed":   result["execution_result"].get("failed"),
        "elo_after":     result["elo_after"],
    })

    # generator_problem.json
    _write("generator_problem.json", {
        "model": result["generator"],
        **result["problem"],
    })

    # solver_solution.json
    _write("solver_solution.json", {
        "model": result["solver"],
        **result["solution"],
    })

    # execution_result.json
    _write("execution_result.json", result["execution_result"])

    # judge_<slug>.json  (one file per judge)
    for jr in result.get("judge_scores", []):
        judge_id = jr.get("judge", "unknown")
        # Derive a safe filename from the model ID
        slug = re.sub(r"[^\w-]", "_", judge_id)[:60]
        _write(f"judge_{slug}.json", {"model": judge_id, **jr})


def _save_leaderboard(leaderboard: list[tuple[str, float]]) -> None:
    lb_path = os.path.join(os.path.dirname(__file__), "data", "leaderboard.json")
    os.makedirs(os.path.dirname(lb_path), exist_ok=True)
    data = [{"rank": i + 1, "model": m, "elo": round(r, 2)} for i, (m, r) in enumerate(leaderboard)]
    with open(lb_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.health_check:
        settings = Settings()
        print("=" * 60)
        print("  MatrixArena — Model Health Check")
        print("=" * 60)
        hard_failed, rate_limited = asyncio.run(health_check(settings))
        raise SystemExit(1 if (hard_failed or rate_limited) else 0)

    cycles = args.cycles
    if cycles is None:
        cycles = int(os.getenv("NUM_CYCLES", "3"))

    asyncio.run(run(cycles, skip_health_check=args.no_health_check))


if __name__ == "__main__":
    main()
