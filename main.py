"""
MatrixArena - CLI entry point.

Run an evaluation cycle where LLMs generate tasks, solve them, and judge each other.

Usage:
    python main.py --health-check   # just run the health check and exit
    python main.py [--cycles N] [--no-health-check]
    python main.py --reset-last     # undo the most recent cycle
    python main.py --reset-all      # wipe all run data and start fresh
"""

import argparse
import asyncio
import json
import os
import re
import shutil
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
    parser.add_argument(
        "--reset-last",
        action="store_true",
        help="Remove the most recent cycle's artifacts and revert Elo to the previous state",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Wipe ALL run data (battles.jsonl, leaderboard.json, data/cycles/) and exit",
    )
    return parser.parse_args()


_DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
_BATTLES_PATH  = os.path.join(_DATA_DIR, "battles.jsonl")
_LB_PATH       = os.path.join(_DATA_DIR, "leaderboard.json")
_CYCLES_DIR    = os.path.join(_DATA_DIR, "cycles")


def _read_battles() -> list[dict]:
    if not os.path.exists(_BATTLES_PATH):
        return []
    lines = []
    with open(_BATTLES_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return lines


def _write_battles(battles: list[dict]) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_BATTLES_PATH, "w", encoding="utf-8") as fh:
        for b in battles:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")


def _elo_snap_to_leaderboard(elo_after: dict) -> list[dict]:
    ranked = sorted(elo_after.items(), key=lambda x: -x[1])
    return [{"rank": i + 1, "model": m, "elo": round(r, 2)} for i, (m, r) in enumerate(ranked)]


def cmd_reset_last() -> None:
    """Remove the most recent cycle and roll Elo back to the previous snapshot."""
    battles = _read_battles()
    if not battles:
        print("Nothing to reset — battles.jsonl is empty.")
        return

    last = battles[-1]
    cycle_num = last.get("cycle_num", "?")
    ts        = last.get("timestamp", 0)
    title     = last.get("problem_title", "?")
    print(f"Removing cycle {cycle_num}  [{title}]  ts={ts}")

    # 1. Remove last line from battles.jsonl
    _write_battles(battles[:-1])

    # 2. Restore leaderboard from the previous cycle's elo_after snapshot
    if len(battles) >= 2:
        prev_elo = battles[-2].get("elo_after", {})
        lb = _elo_snap_to_leaderboard(prev_elo)
        print(f"  Elo restored to snapshot from cycle {battles[-2].get('cycle_num', '?')}")
    else:
        lb = []   # no previous cycle — reset to blank
        print("  Elo reset to initial (no previous cycle).")
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_LB_PATH, "w", encoding="utf-8") as fh:
        json.dump(lb, fh, indent=2)

    # 3. Delete the matching cycle directory (match by timestamp and cycle_num)
    removed_dirs = []
    if os.path.isdir(_CYCLES_DIR):
        ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d_%H%M%S") if ts else ""
        pattern = re.compile(rf"{ts_str}_c{cycle_num}$") if ts_str else None
        for d in os.listdir(_CYCLES_DIR):
            full = os.path.join(_CYCLES_DIR, d)
            if os.path.isdir(full) and (pattern and pattern.match(d)):
                shutil.rmtree(full)
                removed_dirs.append(d)
        if not removed_dirs:
            # Fallback: delete the lexicographically last directory
            dirs = sorted(os.listdir(_CYCLES_DIR))
            if dirs:
                full = os.path.join(_CYCLES_DIR, dirs[-1])
                shutil.rmtree(full)
                removed_dirs.append(dirs[-1])
    if removed_dirs:
        print(f"  Deleted cycle dir(s): {', '.join(removed_dirs)}")
    else:
        print("  No matching cycle directory found.")

    print("Done. Last cycle removed.")


def cmd_reset_all() -> None:
    """Wipe every piece of run data."""
    confirm = input(
        "This will delete ALL battles, leaderboard data, and cycle artifacts.\n"
        "Type 'yes' to confirm: "
    ).strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    removed = []
    for path in (_BATTLES_PATH, _LB_PATH):
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    if os.path.isdir(_CYCLES_DIR):
        shutil.rmtree(_CYCLES_DIR)
        removed.append(_CYCLES_DIR)

    if removed:
        for r in removed:
            print(f"  Deleted: {r}")
    else:
        print("  Nothing to delete — data directory was already clean.")
    print("Done. All run data cleared.")





def _is_rate_limit(err: str) -> bool:
    _RATE_LIMIT_MARKERS = ("ratelimiterror", "429", "rate_limit", "rate-limit", "retry_after")
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

    hard_failed: list[tuple[str, str]] = []   # (model_id, error_msg)
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
                retries=2,
                backoff_on_rate_limit=False,
                request_timeout=40,
                **settings.model_extra(cfg.id),
            )
            ok = True
        except Exception as exc:  # noqa: BLE001
            err = " ".join(str(exc).split())
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
            short_err = err.splitlines()[0][:120] if err else "unknown error"
            print(f"  [FAIL] {cfg.display_name:<40} {timing}  {short_err}", flush=True)
            hard_failed.append((cfg.id, short_err))

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

        # Collect all models to exclude (hard failures + rate-limited)
        exclude_ids: set[str] = {mid for mid, _ in hard_failed} | set(rate_limited)

        if hard_failed:
            print(f"WARNING: {len(hard_failed)} model(s) failed health check and will be excluded from this run:")
            for mid, reason in hard_failed:
                print(f"  - {mid}")
                print(f"      {reason}")

        if rate_limited:
            print(f"WARNING: {len(rate_limited)} model(s) are rate-limited and will be excluded from this run:")
            for mid in rate_limited:
                print(f"  - {mid}")

        if exclude_ids:
            settings.models = [m for m in settings.models if m.id not in exclude_ids]
            if len(settings.models) < 3:
                print(f"ABORT: Only {len(settings.models)} model(s) remain after exclusions.")
                print("Need at least 3 models (1 generator + 1 solver + 1 judge). Re-run later or add more models.")
                raise SystemExit(1)
            print(f"  Continuing with {len(settings.models)} model(s).\n")
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
    log_path = _BATTLES_PATH
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
    raw/generator.txt             — raw LLM output for the generator
    raw/solver_<slug>.txt         — raw LLM output per solver (N files)
    raw/judge_<j>_for_<s>.txt     — raw LLM output per judge×solver pair
    """
    ts = datetime.fromtimestamp(result["timestamp"], tz=timezone.utc)
    dir_name = f"{ts.strftime('%Y%m%d_%H%M%S')}_c{result.get('cycle_num', 0)}"
    cycle_dir = os.path.join(_CYCLES_DIR, dir_name)
    os.makedirs(cycle_dir, exist_ok=True)

    def _strip_raw(obj: object) -> object:
        """Recursively remove '_raw_output' keys before writing JSON."""
        if isinstance(obj, dict):
            return {k: _strip_raw(v) for k, v in obj.items() if k != "_raw_output"}
        if isinstance(obj, list):
            return [_strip_raw(x) for x in obj]
        return obj

    def _write(filename: str, data: object) -> None:
        with open(os.path.join(cycle_dir, filename), "w", encoding="utf-8") as fh:
            json.dump(_strip_raw(data), fh, indent=2, ensure_ascii=False)

    def _write_raw(filename: str, text: str) -> None:
        raw_dir = os.path.join(cycle_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        with open(os.path.join(raw_dir, filename), "w", encoding="utf-8") as fh:
            fh.write(text)

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

    # Raw output: generator
    gen_raw = result["problem"].get("_raw_output")
    if gen_raw:
        _write_raw("generator.txt", gen_raw)

    # per-solver: solution, execution, judgments, raw outputs
    for solver_id, solution in result["solutions"].items():
        slug = _slug(solver_id)
        _write(f"solver_{slug}.json", {"model": solver_id, **solution})
        _write(f"execution_{slug}.json", result["execution_results"][solver_id])
        _write(f"judgments_{slug}.json", {
            "solver": solver_id,
            "average_score": result["average_scores"].get(solver_id),
            "judge_results": result["judge_scores"].get(solver_id, []),
        })
        # Raw output: solver
        solver_raw = solution.get("_raw_output")
        if solver_raw:
            _write_raw(f"solver_{slug}.txt", solver_raw)

    # Raw output: judges (one file per judge×solver pair)
    for solver_id, judge_results in result["judge_scores"].items():
        s_slug = _slug(solver_id)
        for jr in judge_results:
            judge_raw = jr.get("_raw_output")
            if judge_raw:
                j_slug = _slug(jr.get("judge", "unknown"))
                _write_raw(f"judge_{j_slug}_for_{s_slug}.txt", judge_raw)


def _save_leaderboard(leaderboard: list[tuple[str, float]]) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    data = [{"rank": i + 1, "model": m, "elo": round(r, 2)} for i, (m, r) in enumerate(leaderboard)]
    with open(_LB_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.reset_all:
        cmd_reset_all()
        return

    if args.reset_last:
        cmd_reset_last()
        return

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
