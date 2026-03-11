"""
Elo rating module for MatrixArena.

Implements the standard Elo update formula used in chess.  Each evaluation
cycle results in a set of 1-vs-1 match outcomes between the Solver and each
Judge's implicit reference rating derived from the problem score.

Generator Elo
-------------
The Generator is rewarded for producing *well-calibrated* problems:
- If the Solver's normalised score is close to 0.5 (neither trivial nor
  impossible), the Generator is treated as having "won" against the judges.
- If the problem is too easy (score → 1) or too hard / unclear (score → 0),
  the Generator loses.  Formally the Generator's outcome is:

      generator_outcome = 1 − |normalised_score − 0.5| × 2

  which peaks at 1.0 when score = 5.0/10 and reaches 0 at the extremes.

Usage
-----
    elo = EloRating(initial_ratings={"gpt-4o": 1200, ...})
    elo.update(solver="gpt-4o", judges=["claude-3-5-sonnet"], score=7.5)
    elo.update_generator(generator="gemini-pro", judges=["claude-3-5-sonnet"], score=7.5)
    leaderboard = elo.get_leaderboard()
"""

from __future__ import annotations

import math
from typing import Iterable


class EloRating:
    """Maintains and updates Elo ratings for all models in the arena."""

    # Standard Elo K-factor — controls how much a single result shifts ratings.
    K: float = 32.0
    # Reduced K for Generator updates (softer signal, harder to measure directly)
    K_GENERATOR: float = 16.0

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
        Update Elo ratings for the Solver after one evaluation cycle.

        The Solver's performance (``score / max_score``) is treated as the
        win probability in a 1-vs-1 match against each Judge.
        """
        actual_outcome = max(0.0, min(1.0, score / max_score))

        for judge in judges:
            self.ensure_model(solver)
            self.ensure_model(judge)

            solver_rating = self._ratings[solver]
            judge_rating = self._ratings[judge]

            expected_solver = self._expected(solver_rating, judge_rating)
            judge_outcome = 1.0 - actual_outcome

            self._ratings[solver] = solver_rating + self.K * (actual_outcome - expected_solver)
            self._ratings[judge] = judge_rating + self.K * (judge_outcome - (1.0 - expected_solver))

    def update_generator(
        self,
        generator: str,
        judges: Iterable[str],
        score: float,
        max_score: float = 10.0,
    ) -> None:
        """
        Update the Generator's Elo based on problem calibration quality.

        A well-calibrated problem produces a solver score near the midpoint
        (``max_score / 2``).  The generator outcome is:

            generator_outcome = 1 − |normalised_score − 0.5| × 2

        This peaks at 1.0 when the problem is perfectly balanced and drops
        to 0.0 for trivially easy or completely unsolvable problems.
        The update uses a smaller K (``K_GENERATOR``) because problem
        quality is a softer signal than coding correctness.
        """
        normalised = max(0.0, min(1.0, score / max_score))
        # Calibration score: 1.0 at midpoint, 0.0 at extremes
        generator_outcome = 1.0 - abs(normalised - 0.5) * 2.0

        for judge in judges:
            self.ensure_model(generator)
            self.ensure_model(judge)

            gen_rating = self._ratings[generator]
            judge_rating = self._ratings[judge]

            expected_gen = self._expected(gen_rating, judge_rating)
            judge_outcome = 1.0 - generator_outcome

            self._ratings[generator] = gen_rating + self.K_GENERATOR * (generator_outcome - expected_gen)
            self._ratings[judge] = judge_rating + self.K_GENERATOR * (judge_outcome - (1.0 - expected_gen))

    def get_leaderboard(self) -> list[tuple[str, float]]:
        """Return models sorted by rating descending as ``(model_id, rating)`` tuples."""
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    def ratings_snapshot(self) -> dict[str, float]:
        """
        Return a stable ``{model_id: rating}`` snapshot sorted alphabetically by model ID.

        Using alphabetical order (rather than by rating) ensures the dict
        has a deterministic key order across cycles, making time-series
        comparisons in battles.jsonl straightforward.
        """
        return {m: round(r, 2) for m, r in sorted(self._ratings.items())}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        """Standard Elo expected score for player A against player B."""
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))
