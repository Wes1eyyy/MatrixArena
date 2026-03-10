"""
MatrixArena - CLI entry point.

Run an evaluation cycle where LLMs generate tasks, solve them, and judge each other.

Usage:
    python main.py [--cycles N]
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from config.settings import Settings
from core.orchestrator import Orchestrator


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
    return parser.parse_args()


async def run(cycles: int) -> None:
    settings = Settings()
    orchestrator = Orchestrator(settings)

    print("=" * 60)
    print("  MatrixArena — LLM Peer-Review Evaluation")
    print("=" * 60)
    print(f"Models in pool: {', '.join(settings.model_names)}")
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

    cycles = args.cycles
    if cycles is None:
        cycles = int(os.getenv("NUM_CYCLES", "3"))

    asyncio.run(run(cycles))


if __name__ == "__main__":
    main()
