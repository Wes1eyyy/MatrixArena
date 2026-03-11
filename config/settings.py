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
    api_base: str | None = None   # custom base URL for OpenAI-compatible endpoints
    api_key_env: str | None = None  # env var name that holds the API key for this model


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

    @property
    def model_map(self) -> dict[str, ModelConfig]:
        """Return a dict mapping model ID → ModelConfig for O(1) lookup."""
        return {m.id: m for m in self.models}

    def model_extra(self, model_id: str) -> dict[str, str]:
        """
        Return extra kwargs to pass to litellm for *model_id*.

        Currently supports:
        - ``api_base``: forwarded when set
        - ``api_key``:  resolved from the env var named in ``api_key_env``
        """
        cfg = self.model_map.get(model_id)
        if cfg is None:
            return {}
        extra: dict[str, str] = {}
        if cfg.api_base:
            extra["api_base"] = cfg.api_base
        if cfg.api_key_env:
            key_value = os.getenv(cfg.api_key_env, "")
            if key_value:
                extra["api_key"] = key_value
        return extra

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        if not _MODELS_YAML.exists():
            raise FileNotFoundError(f"models.yaml not found at {_MODELS_YAML}")

        with _MODELS_YAML.open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)

        self.initial_elo = float(raw.get("initial_elo", self.initial_elo))

        all_models: list[ModelConfig] = [
            ModelConfig(
                id=entry["id"],
                provider=entry.get("provider", "unknown"),
                display_name=entry.get("display_name", entry["id"]),
                api_base=entry.get("api_base"),
                api_key_env=entry.get("api_key_env"),
            )
            for entry in raw.get("models", [])
        ]

        # Filter out models whose required API key env var is not set
        enabled: list[ModelConfig] = []
        for m in all_models:
            if m.api_key_env and not os.getenv(m.api_key_env, "").strip():
                print(
                    f"  [disabled] {m.display_name} — "
                    f"{m.api_key_env} not set in environment, skipping."
                )
            else:
                enabled.append(m)
        self.models = enabled

        if len(self.models) < 3:
            raise ValueError(
                f"At least 3 models are required (one Generator, one Solver, one+ Judge), "
                f"but only {len(self.models)} model(s) are enabled. "
                f"Check that the required API keys are set in your .env file."
            )

    def load_prompt(self, name: str) -> str:
        """Load a prompt template from the prompts/ directory by filename stem."""
        path = self.prompts_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
