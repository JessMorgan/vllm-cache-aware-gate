"""Tests for gate.tokens: prompt token estimation (pure, fail-open)."""

from __future__ import annotations

import json
import math

from gate.config import GateConfig
from gate.tokens import estimate_context_tokens


def make_cfg(chars_per_token: int = 4, default_max_tokens: int = 256) -> GateConfig:
    """Small helper to build a GateConfig fixture with the two fields we care about."""
    return GateConfig(chars_per_token=chars_per_token, default_max_tokens=default_max_tokens)


def body_bytes(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


# --- chat: messages ----------------------------------------------------------


def test_chat_single_string_message():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {"messages": [{"role": "user", "content": "hello world"}]}  # 11 chars
    assert estimate_context_tokens(body_bytes(payload), cfg) == 3 + 256  # ceil(11/4)=3


def test_chat_multiple_messages_summed():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {
        "messages": [
            {"role": "system", "content": "be brief"},  # 9
            {"role": "user", "content": "hi there"},  # 8
            {"role": "assistant", "content": "hello!"},  # 6
        ]
    }
    assert estimate_context_tokens(body_bytes(payload), cfg) == math.ceil(23 / 4) + 256


def test_chat_content_as_list_of_parts():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},  # 15
                    {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
                    {"type": "text", "text": "in detail"},  # 9
                    "not-a-dict-part",
                    {"type": "text"},  # missing text key
                    {"type": "text", "text": 42},  # non-str text
                ],
            }
        ]
    }
    assert estimate_context_tokens(body_bytes(payload), cfg) == math.ceil(24 / 4) + 256


def test_chat_non_dict_message_items_skipped():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {"messages": ["str-item", 42, None, {"role": "user", "content": "abcd"}]}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 1 + 256  # ceil(4/4)=1


def test_chat_other_content_types_ignored():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {
        "messages": [
            {"role": "user", "content": None},
            {"role": "user", "content": 42},
            {"role": "user", "content": {"weird": "shape"}},
            {"role": "user", "content": "abcd"},
        ]
    }
    assert estimate_context_tokens(body_bytes(payload), cfg) == 1 + 256


# --- completions: prompt -----------------------------------------------------


def test_completions_prompt_string():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {"prompt": "a" * 40}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 10 + 256


def test_completions_prompt_list_of_strings():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {"prompt": ["abcd", "efgh", "ijkl"]}  # 12 chars
    assert estimate_context_tokens(body_bytes(payload), cfg) == 3 + 256


# --- headroom ----------------------------------------------------------------


def test_max_tokens_added():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {"messages": [{"role": "user", "content": "abcd"}], "max_tokens": 100}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 1 + 100


def test_max_completion_tokens_takes_precedence():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {
        "messages": [{"role": "user", "content": "abcd"}],
        "max_tokens": 100,
        "max_completion_tokens": 32,
    }
    assert estimate_context_tokens(body_bytes(payload), cfg) == 1 + 32


def test_missing_both_uses_default():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=64)
    payload = {"messages": [{"role": "user", "content": "abcd"}]}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 1 + 64


def test_max_tokens_zero_falls_back_to_default():
    """0 is not positive, so the default headroom is used."""
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {"messages": [{"role": "user", "content": "abcd"}], "max_tokens": 0}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 1 + 256


def test_max_tokens_bool_falls_back_to_default():
    """bool is excluded from 'positive int', so the default headroom is used."""
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {"messages": [{"role": "user", "content": "abcd"}], "max_tokens": True}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 1 + 256


def test_max_tokens_float_and_negative_fall_back_to_default():
    """Only a positive int counts; float and negative values use the default."""
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    for bad in (10.5, -5):
        payload = {"messages": [{"role": "user", "content": "abcd"}], "max_tokens": bad}
        assert estimate_context_tokens(body_bytes(payload), cfg) == 1 + 256


def test_max_completion_tokens_zero_falls_back_to_max_tokens():
    """A non-positive max_completion_tokens is ignored; max_tokens still wins."""
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {
        "messages": [{"role": "user", "content": "abcd"}],
        "max_tokens": 100,
        "max_completion_tokens": 0,
    }
    assert estimate_context_tokens(body_bytes(payload), cfg) == 1 + 100


# --- rounding ----------------------------------------------------------------


def test_rounding_ceil():
    """5 chars at 4 chars/token -> ceil = 2 prompt tokens."""
    cfg = make_cfg(chars_per_token=4, default_max_tokens=0)
    payload = {"messages": [{"role": "user", "content": "abcde"}]}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 2


def test_known_end_to_end_example():
    """40 chars, cpt 4, default headroom 256 -> 10 + 256 = 266."""
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {"messages": [{"role": "user", "content": "x" * 40}]}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 266


# --- fail-open -> None ---------------------------------------------------------


def test_invalid_utf8_returns_none():
    cfg = make_cfg()
    assert estimate_context_tokens(b"\xff\xfe\x00invalid", cfg) is None


def test_invalid_json_returns_none():
    cfg = make_cfg()
    assert estimate_context_tokens(b'{"messages": [oops', cfg) is None


def test_json_list_returns_none():
    cfg = make_cfg()
    assert estimate_context_tokens(body_bytes([1, 2, 3]), cfg) is None


def test_messages_not_a_list_returns_none():
    cfg = make_cfg()
    assert estimate_context_tokens(body_bytes({"messages": "hi"}), cfg) is None


def test_prompt_wrong_shape_returns_none():
    cfg = make_cfg()
    assert estimate_context_tokens(body_bytes({"prompt": 42}), cfg) is None


def test_prompt_list_with_non_string_returns_none():
    cfg = make_cfg()
    assert estimate_context_tokens(body_bytes({"prompt": ["ok", 7]}), cfg) is None


def test_neither_messages_nor_prompt_returns_none():
    cfg = make_cfg()
    assert estimate_context_tokens(body_bytes({"model": "x", "temperature": 0.5}), cfg) is None


def test_deeply_nested_json_returns_none():
    """Pathologically nested JSON makes the parser raise RecursionError;
    that is an unparseable body, so the gate fails open (None), not 500."""
    cfg = make_cfg()
    assert estimate_context_tokens(b"[" * 10000 + b"]" * 10000, cfg) is None


def test_messages_wins_when_both_keys_present():
    """When both 'messages' and 'prompt' are present, 'messages' is used."""
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {"messages": [{"role": "user", "content": "abcd"}], "prompt": "x" * 400}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 1 + 256


def test_empty_messages_not_none():
    """Empty messages list is understood: headroom only, ceil(0/4)=0."""
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    assert estimate_context_tokens(body_bytes({"messages": []}), cfg) == 256


# --- unicode ------------------------------------------------------------------


def test_unicode_counted_by_code_points_not_bytes():
    """Emoji/multibyte content counted by len (code points), not bytes."""
    cfg = make_cfg(chars_per_token=4, default_max_tokens=0)
    # 4 code points; 16 bytes in UTF-8. ceil(4/4)=1, not ceil(16/4)=4.
    content = "\U0001f600\U0001f601\U0001f602\U0001f603"
    assert len(content) == 4 and len(content.encode("utf-8")) == 16
    payload = {"messages": [{"role": "user", "content": content}]}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 1


# --- boundary -----------------------------------------------------------------


def test_empty_string_message_headroom_only():
    cfg = make_cfg(chars_per_token=4, default_max_tokens=256)
    payload = {"messages": [{"role": "user", "content": ""}]}
    assert estimate_context_tokens(body_bytes(payload), cfg) == 256
