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
import time

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


async def health_check(settings: Settings) -> list[str]:
    """
    Ping every enabled model concurrently with a minimal prompt.

    Prints a live status line per model and returns the list of model IDs
    that failed, so the caller can decide whether to abort.
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
                **settings.model_extra(cfg.id),
            )
            elapsed = time.monotonic() - start
            return cfg.display_name, True, elapsed, ""
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            return cfg.display_name, False, elapsed, str(exc)

    results = await asyncio.gather(*[_ping(m) for m in models])

    failed: list[str] = []
    for m, (display, ok, elapsed, err) in zip(models, results):
        icon = "OK " if ok else "FAIL"
        timing = f"[{elapsed:.2f}s]"
        if ok:
            print(f"  [{icon}]  {display:<40} {timing}")
        else:
            short_err = err.splitlines()[0][:80] if err else "unknown error"
            print(f"  [{icon}] {display:<40} {timing}  {short_err}")
            failed.append(m.id)

    passed = len(models) - len(failed)
    print(f"Health check complete: {passed}/{len(models)} passed.\n")
    return failed



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
        failed = await health_check(settings)
        if failed:
            print(f"ABORT: {len(failed)} model(s) failed health check:")
            for f in failed:
                print(f"  - {f}")
            print("Fix the failing models or run with --no-health-check to skip.")
            raise SystemExit(1)
    else:
        print("(Health check skipped)\n")

    print(f"Running {cycles} evaluation cycle(s)...\n")

    for cycle_num in range(1, cycles + 1):
        print(f"--- Cycle {cycle_num}/{cycles} ---")
        result = await orchestrator.run_cycle()

        print(f"  Generator : {result['generator']}")
        print(f"  Solver    : {result['solver']}")
        print(f"  Judges    : {', '.join(result['judges'])}")
        print(f"  Avg Score : {result['average_score']:.2f}/10")

        # Persist battle log
        _append_battle_log(result)

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
    log_path = os.path.join(os.path.dirname(__file__), "data", "battles.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(result) + "\n")


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
        failed = asyncio.run(health_check(settings))
        raise SystemExit(1 if failed else 0)

    cycles = args.cycles
    if cycles is None:
        cycles = int(os.getenv("NUM_CYCLES", "3"))

    asyncio.run(run(cycles, skip_health_check=args.no_health_check))


if __name__ == "__main__":
    main()
