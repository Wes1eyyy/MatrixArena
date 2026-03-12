"""
Gateway module — litellm wrapper for unified API routing.

Provides async helpers for calling any model in the pool with
consistent error handling, retries, and response normalisation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


class GatewayError(Exception):
    """Raised when a model call fails after all retries."""


_RATE_LIMIT_PATTERNS = ("ratelimiterror", "429", "rate_limit", "rate-limit", "retry_after")
_RETRY_AFTER_RE = re.compile(r'retry_after_seconds["\s:]+([0-9]+)')


def _clean_err(exc: Exception) -> str:
    """Collapse whitespace in exception messages to avoid blank-line spam."""
    return " ".join(str(exc).split()[:80])  # max 80 tokens, single line


def _is_rate_limit(exc: Exception) -> bool:
    low = str(exc).lower()
    return any(p in low for p in _RATE_LIMIT_PATTERNS)


def _retry_after(exc: Exception) -> int:
    """Extract retry_after_seconds from the error string, default to 60."""
    m = _RETRY_AFTER_RE.search(str(exc))
    return int(m.group(1)) if m else 60


async def call_model(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    retries: int = 2,
    backoff_on_rate_limit: bool = True,
    request_timeout: int = 600,
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
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **extra,
                ),
                timeout=request_timeout,
            )
            content: str = response.choices[0].message.content or ""
            finish_reason: str = response.choices[0].finish_reason or ""
            if not content.strip():
                # Empty response — treat as a retryable failure
                last_exc = GatewayError(f"Model {model!r} returned an empty response.")
                logger.warning(
                    "Model %s returned empty content (attempt %d/%d).",
                    model, attempt, retries + 1,
                )
                if attempt <= retries:
                    await asyncio.sleep(2 ** (attempt - 1))
                continue
            if finish_reason == "length":
                # Response was hard-truncated — retry with an explicit continuation prompt
                logger.warning(
                    "Model %s hit max_tokens=%d (finish_reason=length) — "
                    "response likely truncated, retrying.",
                    model, max_tokens,
                )
                if attempt <= retries:
                    # Second attempt: add assistant's partial reply and ask to finish cleanly
                    messages = [
                        {"role": "user",    "content": prompt},
                        {"role": "assistant", "content": content},
                        {"role": "user",    "content":
                            "Your previous response was cut off. "
                            "Please output ONLY the complete JSON object from the beginning, "
                            "with no extra text before or after it."},
                    ]
                    last_exc = GatewayError(f"Model {model!r} truncated (finish_reason=length).")
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue
            return content.strip()
        except (KeyboardInterrupt, SystemExit):
            raise
        except asyncio.TimeoutError:
            logger.warning(
                "Model %s timed out after %ds (attempt %d/%d) — skipping.",
                model, request_timeout, attempt, retries + 1,
            )
            raise GatewayError(
                f"Model '{model}' timed out after {request_timeout}s."
            )
        except asyncio.CancelledError:
            # aiohttp can raise CancelledError during cleanup after a timeout cancel
            logger.warning(
                "Model %s cancelled (attempt %d/%d) — treating as timeout.",
                model, attempt, retries + 1,
            )
            raise GatewayError(
                f"Model '{model}' request was cancelled (possible timeout)."
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            is_rl = _is_rate_limit(exc)
            logger.warning(
                "Model %s failed (attempt %d/%d): %s",
                model,
                attempt,
                retries + 1,
                _clean_err(exc),
            )
            if attempt <= retries:  # don't sleep after the last attempt
                if is_rl and backoff_on_rate_limit:
                    wait = _retry_after(exc)
                    logger.info("Rate-limited — waiting %ds before retry...", wait)
                    await asyncio.sleep(wait)
                else:
                    # Exponential backoff for transient errors: 1s, 2s, 4s…
                    await asyncio.sleep(2 ** (attempt - 1))

    raise GatewayError(
        f"All {retries + 1} attempts to call '{model}' failed."
        + (f" Last error: {_clean_err(last_exc)}" if last_exc else "")
    ) from last_exc


def _extract_json_object(text: str) -> str | None:
    """
    Find the first complete JSON object ``{...}`` in *text* using brace counting.

    This handles two common failure modes:
    - Model prefixes the JSON with prose (``Sure! Here is your JSON: {...}``)
    - Response is wrapped in markdown fences that weren't caught by the fence stripper

    Returns the extracted substring, or None if no complete object is found.
    """
    depth = 0
    start = None
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]
    return None


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
    cleaned = raw.strip()
    
    # 1. Strip markdown code fences more robustly (handles prose before/after)
    start_idx = cleaned.find("```json")
    if start_idx != -1:
        start_content = start_idx + 7
    else:
        start_idx = cleaned.find("```")
        start_content = start_idx + 3 if start_idx != -1 else 0

    if start_content > 0:
        end_idx = cleaned.rfind("```")
        if end_idx > start_content:
            text_to_parse = cleaned[start_content:end_idx].strip()
        else:
            text_to_parse = cleaned[start_content:].strip()
    else:
        text_to_parse = cleaned

    # 2. Fast path: direct parse
    try:
        return json.loads(text_to_parse)
    except json.JSONDecodeError:
        pass

    # 3. Fallback: brace-matching extraction on the isolated text
    extracted = _extract_json_object(text_to_parse)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    # 4. Last resort: brace-matching on the original raw input
    extracted_orig = _extract_json_object(cleaned)
    if extracted_orig:
        try:
            return json.loads(extracted_orig)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse model response as JSON.\nRaw (first 500 chars):\n{raw[:500]}"
    )
