"""
Gateway module — litellm wrapper for unified API routing.

Provides async helpers for calling any model in the pool with
consistent error handling, retries, and response normalisation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


class GatewayError(Exception):
    """Raised when a model call fails after all retries."""


async def call_model(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    retries: int = 2,
    api_base: str | None = None,
    api_key: str | None = None,
) -> str:
    """
    Send a prompt to *model* and return the response text.

    Parameters
    ----------
    model:
        litellm model identifier, e.g. ``"gpt-4o"`` or
        ``"openai/doubao-seed-2.0-code"``.
    prompt:
        The full prompt to send as the user message.
    temperature:
        Sampling temperature (0 = deterministic).
    max_tokens:
        Maximum number of tokens to generate.
    retries:
        How many additional attempts to make on transient errors.
    api_base:
        Optional custom base URL (for OpenAI-compatible third-party endpoints).
    api_key:
        Optional API key override (resolved from env by Settings.model_extra).

    Returns
    -------
    str
        The model's response text (stripped of leading/trailing whitespace).

    Raises
    ------
    GatewayError
        If all attempts fail.
    """
    messages = [{"role": "user", "content": prompt}]
    last_exc: Exception | None = None

    # Build optional extra kwargs for non-standard endpoints
    extra: dict[str, str] = {}
    if api_base:
        extra["api_base"] = api_base
    if api_key:
        extra["api_key"] = api_key

    for attempt in range(1, retries + 2):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
            content: str = response.choices[0].message.content or ""
            return content.strip()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Model %s failed (attempt %d/%d): %s",
                model,
                attempt,
                retries + 1,
                exc,
            )

    raise GatewayError(
        f"All {retries + 1} attempts to call '{model}' failed."
    ) from last_exc


def parse_json_response(raw: str) -> dict[str, Any]:
    """
    Parse a JSON response from a model, stripping common artefacts such as
    markdown code fences that some models add despite instructions.

    Parameters
    ----------
    raw:
        Raw string returned by the model.

    Returns
    -------
    dict
        Parsed JSON object.

    Raises
    ------
    ValueError
        If the string cannot be parsed as JSON after clean-up.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove the opening fence line (e.g. ```json)
        lines = lines[1:]
        # Remove the closing fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse model response as JSON: {exc}\nRaw:\n{raw}") from exc
