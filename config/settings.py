"""
Global configuration loader for MatrixArena.

Reads config/models.yaml and environment variables, then exposes a
single Settings object consumed by the rest of the application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_CONFIG_DIR = Path(__file__).parent
_MODELS_YAML = _CONFIG_DIR / "models.yaml"


@dataclass
class ModelConfig:
    id: str
    provider: str
    display_name: str


@dataclass
class Settings:
    """Aggregated runtime settings."""

    models: list[ModelConfig] = field(default_factory=list)
    initial_elo: float = 1200.0
    prompts_dir: Path = _CONFIG_DIR.parent / "prompts"

    def __post_init__(self) -> None:
        self._load_models()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def model_names(self) -> list[str]:
        """Return the list of model IDs (as understood by litellm)."""
        return [m.id for m in self.models]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        if not _MODELS_YAML.exists():
            raise FileNotFoundError(f"models.yaml not found at {_MODELS_YAML}")

        with _MODELS_YAML.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        self.initial_elo = float(raw.get("initial_elo", self.initial_elo))
        self.models = [
            ModelConfig(
                id=entry["id"],
                provider=entry.get("provider", "unknown"),
                display_name=entry.get("display_name", entry["id"]),
            )
            for entry in raw.get("models", [])
        ]

        if len(self.models) < 3:
            raise ValueError(
                "At least 3 models are required (one Generator, one Solver, one+ Judge)."
            )

    def load_prompt(self, name: str) -> str:
        """Load a prompt template from the prompts/ directory by filename stem."""
        path = self.prompts_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
