"""
Elo rating module for MatrixArena.

Implements the standard Elo update formula used in chess.  Each evaluation
cycle results in a set of 1-vs-1 match outcomes between the Solver and each
Judge's implicit reference rating derived from the problem score.

Usage
-----
    elo = EloRating(initial_ratings={"gpt-4o": 1200, ...})
    elo.update(solver="gpt-4o", judges=["claude-3-5-sonnet-20240620"], score=7.5)
    leaderboard = elo.get_leaderboard()
"""

from __future__ import annotations

import math
from typing import Iterable


class EloRating:
    """Maintains and updates Elo ratings for all models in the arena."""

    # Standard Elo K-factor — controls how much a single result shifts ratings.
    K: float = 32.0

    def __init__(self, initial_ratings: dict[str, float] | None = None) -> None:
        self._ratings: dict[str, float] = dict(initial_ratings or {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_model(self, model: str, initial: float = 1200.0) -> None:
        """Register *model* with *initial* rating if not already tracked."""
        self._ratings.setdefault(model, initial)

    def get_rating(self, model: str) -> float:
        """Return the current rating of *model* (1200 if unknown)."""
        return self._ratings.get(model, 1200.0)

    def update(
        self,
        solver: str,
        judges: Iterable[str],
        score: float,
        max_score: float = 10.0,
    ) -> None:
        """
        Update Elo ratings after one evaluation cycle.

        The Solver's performance (``score / max_score``) is treated as the
        win probability in a 1-vs-1 match against each Judge.  A score above
        the midpoint means the Solver "won" that judge's round; below means
        the Solver "lost".

        Parameters
        ----------
        solver:
            Model ID of the Solver.
        judges:
            Model IDs of the judges in this cycle.
        score:
            Aggregate score awarded by the judges (0–max_score).
        max_score:
            Maximum possible score (default 10).
        """
        # Normalise score to [0, 1]; treat it as the Solver's actual outcome.
        actual_outcome = max(0.0, min(1.0, score / max_score))

        for judge in judges:
            self.ensure_model(solver)
            self.ensure_model(judge)

            solver_rating = self._ratings[solver]
            judge_rating = self._ratings[judge]

            expected_solver = self._expected(solver_rating, judge_rating)
            expected_judge = 1.0 - expected_solver

            # Judge's "outcome" is the complement of the solver's
            judge_outcome = 1.0 - actual_outcome

            self._ratings[solver] = solver_rating + self.K * (actual_outcome - expected_solver)
            self._ratings[judge] = judge_rating + self.K * (judge_outcome - expected_judge)

    def get_leaderboard(self) -> list[tuple[str, float]]:
        """Return models sorted by rating descending as ``(model_id, rating)`` tuples."""
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        """Standard Elo expected score for player A against player B."""
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))
