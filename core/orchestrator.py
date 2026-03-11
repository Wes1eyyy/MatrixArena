"""
Orchestrator — the core evaluation loop for MatrixArena.

Each call to ``run_cycle()`` executes one full Gen → Solve → Judge round:

1. **Generator** (randomly chosen) creates a coding problem.
2. **Solver** (a different model) writes a solution.
3. **Judges** (all remaining models, *excluding* the Solver) score the answer
   concurrently.
4. Elo ratings are updated based on the aggregate score.

Fairness Rule
-------------
The Solver is **never** included in the Judge pool — a model cannot evaluate
its own answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any

from config.settings import Settings
from core.elo_rating import EloRating
from core.gateway import GatewayError, call_model, parse_json_response
from sandbox.executor import SubprocessExecutor

logger = logging.getLogger(__name__)


class Orchestrator:
    """Drives the MatrixArena evaluation loop."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.elo = EloRating(
            initial_ratings={m: settings.initial_elo for m in settings.model_names}
        )
        self.executor = SubprocessExecutor()

        # Pre-load prompt templates
        self._generator_prompt = settings.load_prompt("generator")
        self._solver_prompt_template = settings.load_prompt("solver")
        self._judge_prompt_template = settings.load_prompt("judge")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_cycle(
        self,
        cycle_num: int = 0,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """
        Execute one full evaluation cycle.

        New interaction model
        --------------------
        * **Generator** (1 model, randomly chosen) creates the problem.
        * **All N models** (including Generator) independently solve it in parallel.
        * **All N models** judge every solution except their own in parallel.
          Fairness rule: a judge never scores its own solution.
        """
        def _emit(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        models = list(self.settings.model_names)
        n = len(models)

        # ── 1. Role Assignment ──────────────────────────────────────────
        generator = random.choice(models)
        solvers   = models                                     # all N models solve
        judges_pool = models                              # all N models judge

        _emit(f"Generator  : {generator}")
        _emit(f"Solvers    : all {n} models")
        _emit(f"Judges pool: {len(judges_pool)} models (all, each skips own solution)")

        # ── 2. Generation Phase ─────────────────────────────────────────
        _emit("[1/5] Generating problem...")
        problem = await self._generate_problem(generator)
        _emit(f"      Problem : \"{problem.get('title', 'unknown')}\"")

        # ── 3. Solving Phase (all models in parallel) ───────────────────
        _emit(f"[2/5] Solving ({n} solvers in parallel)...")
        solve_queue: asyncio.Queue = asyncio.Queue()

        async def _solve_and_queue(model: str) -> None:
            sol = await self._solve_problem(model, problem)
            await solve_queue.put((model, sol))

        solve_tasks = [asyncio.create_task(_solve_and_queue(m)) for m in solvers]
        solutions: dict[str, Any] = {}
        for _ in solvers:
            model, sol = await solve_queue.get()
            solutions[model] = sol
            lines = len(sol.get("solution_code", "").splitlines())
            _emit(f"      [{model.split('/')[-1]}] → {lines} line(s) of code")
        await asyncio.gather(*solve_tasks)

        # ── 4. Sandbox Execution (sequential, local CPU — fast) ─────────
        _emit("[3/5] Running sandbox execution for all solutions...")
        execution_results: dict[str, Any] = {}
        for model in solvers:
            exec_r = self.executor.run(
                solutions[model].get("solution_code", ""),
                test_cases=problem.get("test_cases", []),
            )
            execution_results[model] = exec_r
            _emit(f"      [{model.split('/')[-1]}] {exec_r['summary']}")

        # ── 5. Judging Phase (all pairs concurrent) ─────────────────────
        # For each solver S: judges = judges_pool \ {S}  (can't judge yourself)
        judge_pairs = [
            (s, j) for s in solvers for j in judges_pool if j != s
        ]
        _emit(f"[4/5] Judging ({len(judge_pairs)} calls: {n} solutions × ~{len(judges_pool)-1} judges each)...")

        judge_queue: asyncio.Queue = asyncio.Queue()

        async def _judge_and_queue(solver_model: str, judge_model: str) -> None:
            jr = await self._call_single_judge(
                judge_model, problem,
                solutions[solver_model], execution_results[solver_model],
            )
            await judge_queue.put((solver_model, jr))

        judge_tasks = [
            asyncio.create_task(_judge_and_queue(s, j)) for s, j in judge_pairs
        ]
        judge_scores: dict[str, list] = {s: [] for s in solvers}
        for _ in judge_pairs:
            solver_id, jr = await judge_queue.get()
            judge_scores[solver_id].append(jr)
            score     = jr.get("overall_score", "?")
            j_name    = jr.get("judge", "?").split("/")[-1]
            s_name    = solver_id.split("/")[-1]
            status    = "error" if jr.get("error") else f"{score:.1f}/10"
            _emit(f"      [{j_name}] judged [{s_name}] → {status}")
        await asyncio.gather(*judge_tasks)

        # ── 6. Aggregate scores per solver ──────────────────────────────
        average_scores: dict[str, float] = {}
        for solver_model, results in judge_scores.items():
            scores = [r.get("overall_score", 5.0) for r in results]
            average_scores[solver_model] = sum(scores) / len(scores) if scores else 5.0
        overall_average = (
            sum(average_scores.values()) / len(average_scores)
            if average_scores else 5.0
        )

        # ── 7. Elo Update ───────────────────────────────────────────────
        _emit("[5/5] Updating Elo ratings...")
        for solver_model, avg_score in average_scores.items():
            effective_judges = [j for j in judges_pool if j != solver_model]
            self.elo.update(solver=solver_model, judges=effective_judges, score=avg_score)
        self.elo.update_generator(
            generator=generator, judges=judges_pool, score=overall_average
        )

        return {
            "cycle_num":      cycle_num,
            "timestamp":      int(time.time()),
            "generator":      generator,
            "solvers":        solvers,
            "judges_pool":    judges_pool,
            "problem":        problem,
            "problem_title":  problem.get("title", "unknown"),
            "solutions":      solutions,
            "execution_results": execution_results,
            "judge_scores":   judge_scores,    # {solver_id: [judge_result, ...]}
            "average_scores": average_scores,  # {solver_id: float}
            "overall_average": overall_average,
            "elo_after":      self.elo.ratings_snapshot(),
        }

    # ------------------------------------------------------------------
    # Role assignment (legacy — inlined into run_cycle)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _extra(self, model: str) -> dict[str, str]:
        """Return extra litellm kwargs (api_base, api_key) for *model*."""
        return self.settings.model_extra(model)

    async def _generate_problem(self, generator: str) -> dict[str, Any]:
        """Ask the Generator model to produce a coding problem."""
        try:
            raw = await call_model(
                generator,
                self._generator_prompt,
                temperature=0.9,
                max_tokens=16384,
                **self._extra(generator),
            )
            return parse_json_response(raw)
        except GatewayError as exc:
            logger.error("Generation failed (gateway): %s", exc)
            return self._fallback_problem()
        except ValueError as exc:
            logger.error("Generation failed (JSON parse): %s", exc)
            return self._fallback_problem()

    async def _solve_problem(self, solver: str, problem: dict[str, Any]) -> dict[str, Any]:
        """Ask the Solver model to produce a solution to *problem*."""
        problem_text = json.dumps(problem, indent=2)
        prompt = self._solver_prompt_template.replace("{problem}", problem_text)
        try:
            raw = await call_model(
                solver,
                prompt,
                temperature=0.4,
                max_tokens=16384,
                **self._extra(solver),
            )
            return parse_json_response(raw)
        except (GatewayError, ValueError) as exc:
            logger.error("Solving failed: %s", exc)
            return {"solution_code": "", "explanation": "Solver failed to produce a response."}

    async def _call_single_judge(
        self,
        judge: str,
        problem: dict[str, Any],
        solution: dict[str, Any],
        execution_result: dict,
    ) -> dict[str, Any]:
        """Call one judge model and return its structured score dict."""
        problem_text   = json.dumps(problem, indent=2)
        solution_text  = json.dumps(solution, indent=2)
        criteria       = problem.get("evaluation_criteria", "Correctness, efficiency, and clarity.")
        execution_text = (
            f"Status : {execution_result['status']}\n"
            f"Summary: {execution_result['summary']}\n"
        )
        if execution_result.get("tests"):
            for t in execution_result["tests"]:
                execution_text += (
                    f"  Test {t['test']}: [{t['status'].upper()}] "
                    f"input={t['input']!r} "
                    f"expected={t['expected']!r} "
                    f"actual={t['actual']!r}"
                    + (f" error={t['error']!r}" if t.get("error") else "") + "\n"
                )
        if execution_result.get("stderr"):
            execution_text += f"Stderr : {execution_result['stderr']}\n"

        prompt = (
            self._judge_prompt_template
            .replace("{problem}", problem_text)
            .replace("{evaluation_criteria}", criteria)
            .replace("{solution}", solution_text)
            .replace("{execution_result}", execution_text)
        )
        try:
            raw = await call_model(judge, prompt, temperature=0.2, **self._extra(judge))
            result = parse_json_response(raw)
            result["judge"] = judge
            return result
        except (GatewayError, ValueError) as exc:
            logger.error("Judge %s failed: %s", judge, exc)
            return {"judge": judge, "overall_score": 5.0, "error": str(exc)}

    # ------------------------------------------------------------------
    # Score aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_scores(judge_results: list[dict[str, Any]]) -> float:
        """Return the mean overall_score across all judge results."""
        scores = [r.get("overall_score", 5.0) for r in judge_results]
        return sum(scores) / len(scores) if scores else 5.0

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_problem() -> dict[str, Any]:
        """Return a minimal hard-coded problem used when generation fails."""
        return {
            "title": "Two Sum",
            "description": (
                "Given an array of integers and a target integer, return the indices "
                "of the two numbers that add up to the target.  Assume exactly one "
                "solution exists.  Input: nums=[2,7,11,15], target=9.  "
                "Output: [0,1]."
            ),
            "test_cases": [
                {"input": "nums=[2,7,11,15], target=9", "expected_output": "[0,1]"},
                {"input": "nums=[3,2,4], target=6", "expected_output": "[1,2]"},
                {"input": "nums=[3,3], target=6", "expected_output": "[0,1]"},
            ],
            "evaluation_criteria": "Correctness, O(n) time complexity, clean code.",
        }
