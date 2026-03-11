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
    Ping every enabled model concurrently.
    Each result is printed the moment it arrives — no waiting for stragglers.

    Returns
    -------
    hard_failed : list[str]
    rate_limited : list[str]
    """
    models = settings.models
    n = len(models)
    print(f"Health check ({n} model(s))...")

    hard_failed: list[str] = []
    rate_limited: list[str] = []
    # Use a queue so each task can push its result and the loop prints immediately.
    queue: asyncio.Queue = asyncio.Queue()

    async def _ping(cfg) -> None:
        start = time.monotonic()
        err = ""
        ok = False
        try:
            await call_model(
                cfg.id,
                _HEALTH_PROMPT,
                temperature=0.0,
                max_tokens=16,
                retries=1,
                backoff_on_rate_limit=False,
                **settings.model_extra(cfg.id),
            )
            ok = True
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        elapsed = time.monotonic() - start
        await queue.put((cfg, ok, elapsed, err))

    # Fire all pings concurrently
    tasks = [asyncio.create_task(_ping(m)) for m in models]

    # Consume results as they arrive
    for i in range(n):
        cfg, ok, elapsed, err = await queue.get()
        timing = f"[{elapsed:.2f}s]"
        if ok:
            print(f"  [OK ]  {cfg.display_name:<40} {timing}", flush=True)
        elif _is_rate_limit(err):
            print(f"  [429 ] {cfg.display_name:<40} {timing}  rate-limited (will skip this run)", flush=True)
            rate_limited.append(cfg.id)
        else:
            short_err = err.splitlines()[0][:80] if err else "unknown error"
            print(f"  [FAIL] {cfg.display_name:<40} {timing}  {short_err}", flush=True)
            hard_failed.append(cfg.id)

    await asyncio.gather(*tasks)  # ensure all tasks are fully cleaned up

    total_failed = len(hard_failed) + len(rate_limited)
    passed = n - total_failed
    print(f"Health check complete: {passed}/{n} passed.\n")
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

        def _progress(msg: str) -> None:
            print(f"  {msg}", flush=True)

        result = await orchestrator.run_cycle(cycle_num=cycle_num, on_progress=_progress)

        print(f"  Overall avg : {result['overall_average']:.2f}/10")
        top = max(result['average_scores'].items(), key=lambda x: x[1])
        print(f"  Best solver : {top[0].split('/')[-1]} ({top[1]:.2f}/10)")

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
    Full artifacts are written by _save_cycle_artifacts().
    """
    compact = {
        "cycle_num":      result.get("cycle_num"),
        "timestamp":      result["timestamp"],
        "generator":      result["generator"],
        "solvers":        result["solvers"],
        "judges_pool":    result["judges_pool"],
        "problem_title":  result["problem"].get("title", "unknown"),
        "average_scores": result["average_scores"],
        "overall_average": result["overall_average"],
        "elo_after":      result["elo_after"],
    }
    log_path = os.path.join(os.path.dirname(__file__), "data", "battles.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(compact, ensure_ascii=False) + "\n")


def _save_cycle_artifacts(result: dict) -> None:
    """
    Write full per-cycle artifacts to data/cycles/<dir>/.

    Directory: <YYYYMMDD_HHMMSS>_c<cycle_num>

    Files written
    -------------
    summary.json                  — roles, scores, elo snapshot
    generator_problem.json        — complete Generator output
    solver_<slug>.json            — solution per solver (N files)
    execution_<slug>.json         — sandbox result per solver (N files)
    judgments_<slug>.json         — all judge results for each solver (N files)
    """
    ts = datetime.fromtimestamp(result["timestamp"], tz=timezone.utc)
    dir_name = f"{ts.strftime('%Y%m%d_%H%M%S')}_c{result.get('cycle_num', 0)}"
    cycle_dir = os.path.join(os.path.dirname(__file__), "data", "cycles", dir_name)
    os.makedirs(cycle_dir, exist_ok=True)

    def _write(filename: str, data: object) -> None:
        with open(os.path.join(cycle_dir, filename), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def _slug(model_id: str) -> str:
        return re.sub(r"[^\w-]", "_", model_id)[:60]

    # summary.json
    _write("summary.json", {
        "cycle_num":       result.get("cycle_num"),
        "timestamp":       result["timestamp"],
        "timestamp_utc":   ts.isoformat(),
        "generator":       result["generator"],
        "solvers":         result["solvers"],
        "judges_pool":     result["judges_pool"],
        "problem_title":   result["problem"].get("title", "unknown"),
        "average_scores":  result["average_scores"],
        "overall_average": result["overall_average"],
        "elo_after":       result["elo_after"],
    })

    # generator_problem.json
    _write("generator_problem.json", {
        "model": result["generator"],
        **result["problem"],
    })

    # per-solver: solution, execution, judgments
    for solver_id, solution in result["solutions"].items():
        slug = _slug(solver_id)
        _write(f"solver_{slug}.json", {"model": solver_id, **solution})
        _write(f"execution_{slug}.json", result["execution_results"][solver_id])
        _write(f"judgments_{slug}.json", {
            "solver": solver_id,
            "average_score": result["average_scores"].get(solver_id),
            "judge_results": result["judge_scores"].get(solver_id, []),
        })


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
