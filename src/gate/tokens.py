"""Prompt token estimation for the gate.

Pure, stdlib-only, deterministic. Estimates the KV-context token size of an
incoming OpenAI request body so the gate can decide whether the vLLM KV cache
has room.

The estimate is a deliberate heuristic, not a tokenizer:

- Prompt characters are counted as ``len(text)`` (Unicode code points) and
  divided by ``cfg.chars_per_token`` with ceiling division.
- A headroom for the completion is added: the request's
  ``max_completion_tokens`` (preferred) or ``max_tokens`` when it is a
  positive int, else ``cfg.default_max_tokens``.

It is intentionally biased to **over-estimate** so the gate is conservative:
a slightly too-large estimate at worst rejects a request that would have fit,
never admits one that would not. No tokenizer dependency in v1 (it would need
model access and a heavy dependency); see AGENTS.md known gotcha #7.

On any input the gate cannot understand (invalid UTF-8, invalid JSON —
including pathologically nested JSON that the parser rejects — a non-object
body, an unrecognized prompt shape) the function returns ``None``.
``None`` means "cannot estimate" and the caller MUST fail open: forward the
request and let vLLM produce the real error.
"""

from __future__ import annotations

import json
from typing import Any, TypeGuard

from gate.config import GateConfig


def _is_positive_int(value: Any) -> TypeGuard[int]:
    """True for real positive ints (bool is excluded on purpose)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _content_chars(content: Any) -> int:
    """Count prompt characters in one message ``content`` value.

    ``str`` counts its code points; a ``list`` (multimodal parts) counts the
    ``text`` of parts whose ``type`` is ``"text"`` and ignores everything else
    (e.g. ``image_url``); any other shape (dict/number/None) contributes 0.
    """
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    total += len(text)
        return total
    return 0


def _prompt_chars(body: dict[str, Any]) -> int | None:
    """Extract total prompt character count from a parsed request body.

    Returns ``None`` when the prompt cannot be understood (caller fails open).
    """
    if "messages" in body:
        messages = body["messages"]
        if not isinstance(messages, list):
            return None
        total = 0
        for item in messages:
            if not isinstance(item, dict):
                continue  # skip non-dict items
            total += _content_chars(item.get("content"))
        return total
    if "prompt" in body:
        prompt = body["prompt"]
        if isinstance(prompt, str):
            return len(prompt)
        if isinstance(prompt, list):
            total = 0
            for piece in prompt:
                if not isinstance(piece, str):
                    return None
                total += len(piece)
            return total
        return None
    # Neither key present: not a generation request the gate understands.
    return None


def _headroom(body: dict[str, Any], cfg: GateConfig) -> int:
    """Completion headroom: max_completion_tokens, else max_tokens, else default."""
    for key in ("max_completion_tokens", "max_tokens"):
        value = body.get(key)
        if _is_positive_int(value):
            return value
    return cfg.default_max_tokens


def estimate_context_tokens(body: bytes, cfg: GateConfig) -> int | None:
    """Estimate the prompt's KV-context token size for an OpenAI request body.

    Returns ``ceil(prompt_chars / cfg.chars_per_token) + headroom``, or
    ``None`` when the body cannot be estimated (caller fails open and
    forwards the request).

    ``body`` is the raw request body as received. The estimate counts prompt
    characters (Unicode code points) plus a completion headroom, so it is a
    conservative over-estimate by design.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        # RecursionError: the C JSON parser rejects pathologically nested
        # input with it; that is an unparseable body, so fail open.
        return None
    if not isinstance(data, dict):
        return None
    prompt_chars = _prompt_chars(data)
    if prompt_chars is None:
        return None
    # Exact integer ceiling (prompt_chars >= 0): no float rounding, no
    # OverflowError for astronomically large values.
    return -(-prompt_chars // cfg.chars_per_token) + _headroom(data, cfg)


def extract_request_model(body: bytes) -> str:
    """Extract the request's ``model`` field (the routing key and stats label).

    Pure and defensive: never raises. Returns the request's ``model`` string
    (the original, unstripped value) when the body is valid UTF-8 JSON with a
    non-empty string ``model`` field; otherwise returns ``"unknown"``.

    The result selects which backend's admission state drives the
    forward/reject decision (multi-backend routing; an unowned or
    ``"unknown"`` model falls to the default backend) and labels the stats.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return "unknown"
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        # RecursionError: the C JSON parser rejects pathologically nested
        # input with it; that is an unparseable body.
        return "unknown"
    if not isinstance(data, dict):
        return "unknown"
    model = data.get("model")
    if isinstance(model, str) and model.strip() != "":
        return model
    return "unknown"
