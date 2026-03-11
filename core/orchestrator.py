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

    async def run_cycle(self) -> dict[str, Any]:
        """
        Execute one full evaluation cycle.

        Returns
        -------
        dict
            A summary record suitable for appending to battles.jsonl.
        """
        models = list(self.settings.model_names)

        # ── 1. Role Assignment ──────────────────────────────────────────
        generator, solver, judges = self._assign_roles(models)
        logger.info("Roles — Generator: %s | Solver: %s | Judges: %s", generator, solver, judges)

        # ── 2. Generation Phase ─────────────────────────────────────────
        problem = await self._generate_problem(generator)

        # ── 3. Solving Phase ────────────────────────────────────────────
        solution = await self._solve_problem(solver, problem)

        # ── 4. Sandbox Execution ───────────────────────────────────────────
        test_cases = problem.get("test_cases", [])
        execution_result = self.executor.run(
            solution.get("solution_code", ""),
            test_cases=test_cases,
        )
        logger.info(
            "Execution: %s — %s",
            execution_result["status"],
            execution_result["summary"],
        )

        # ── 5. Judging Phase (concurrent) ───────────────────────────────
        judge_results = await self._judge_solution(judges, problem, solution, execution_result)

        # ── 6. Aggregate Scores ─────────────────────────────────────────
        average_score = self._aggregate_scores(judge_results)

        # ── 7. Elo Update ───────────────────────────────────────────────
        # Solver: rewarded for code quality (direct score)
        self.elo.update(solver=solver, judges=judges, score=average_score)
        # Generator: rewarded for problem calibration (score near midpoint = good problem)
        self.elo.update_generator(generator=generator, judges=judges, score=average_score)

        return {
            "timestamp": int(time.time()),
            "generator": generator,
            "solver": solver,
            "judges": judges,
            "problem_title": problem.get("title", "unknown"),
            "execution_result": execution_result,
            "judge_scores": judge_results,
            "average_score": average_score,
            "elo_after": self.elo.ratings_snapshot(),
        }

    # ------------------------------------------------------------------
    # Role assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_roles(models: list[str]) -> tuple[str, str, list[str]]:
        """
        Randomly assign Generator, Solver, and Judges from *models*.

        Guarantees:
        * Generator ≠ Solver
        * Solver ∉ Judges  (fairness rule)
        """
        shuffled = random.sample(models, len(models))
        generator = shuffled[0]
        solver = shuffled[1]
        # All remaining models become judges (Solver is excluded by construction)
        judges = shuffled[2:]
        return generator, solver, judges

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
            raw = await call_model(generator, self._generator_prompt, temperature=0.9, **self._extra(generator))
            return parse_json_response(raw)
        except (GatewayError, ValueError) as exc:
            logger.error("Generation failed: %s", exc)
            return self._fallback_problem()

    async def _solve_problem(self, solver: str, problem: dict[str, Any]) -> dict[str, Any]:
        """Ask the Solver model to produce a solution to *problem*."""
        problem_text = json.dumps(problem, indent=2)
        prompt = self._solver_prompt_template.replace("{problem}", problem_text)
        try:
            raw = await call_model(solver, prompt, temperature=0.4, **self._extra(solver))
            return parse_json_response(raw)
        except (GatewayError, ValueError) as exc:
            logger.error("Solving failed: %s", exc)
            return {"solution_code": "", "explanation": "Solver failed to produce a response."}

    async def _judge_solution(
        self,
        judges: list[str],
        problem: dict[str, Any],
        solution: dict[str, Any],
        execution_result: dict,
    ) -> list[dict[str, Any]]:
        """Call all judges concurrently and collect their scores."""
        problem_text = json.dumps(problem, indent=2)
        solution_text = json.dumps(solution, indent=2)
        criteria = problem.get("evaluation_criteria", "Correctness, efficiency, and clarity.")
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

        async def _judge_one(judge: str) -> dict[str, Any]:
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

        tasks = [_judge_one(j) for j in judges]
        return await asyncio.gather(*tasks)

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
