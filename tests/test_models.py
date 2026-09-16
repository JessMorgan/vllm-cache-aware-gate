"""Tests for gate.models: parse_v1_models and ModelRegistry."""

from __future__ import annotations

import json
import logging

import pytest

from gate.models import ModelRegistry, parse_v1_models

# --- parse_v1_models ---------------------------------------------------------


def test_parse_valid():
    text = json.dumps({"data": [{"id": "a"}, {"id": "b"}, {"id": "c"}]})
    assert parse_v1_models(text) == ["a", "b", "c"]


def test_parse_preserves_order_and_extras():
    text = json.dumps({"object": "list", "data": [{"id": "m1", "object": "model"}, {"id": "m2"}]})
    assert parse_v1_models(text) == ["m1", "m2"]


def test_parse_empty_data():
    assert parse_v1_models(json.dumps({"data": []})) == []


def test_parse_missing_data():
    assert parse_v1_models(json.dumps({"models": []})) is None


def test_parse_data_not_a_list():
    assert parse_v1_models(json.dumps({"data": {"id": "a"}})) is None


def test_parse_non_json():
    assert parse_v1_models("not json at all") is None
    assert parse_v1_models("") is None


def test_parse_json_non_object():
    assert parse_v1_models(json.dumps([{"id": "a"}])) is None


def test_parse_duplicate_ids_preserved():
    text = json.dumps({"data": [{"id": "a"}, {"id": "a"}]})
    assert parse_v1_models(text) == ["a", "a"]


def test_parse_non_dict_entry():
    text = json.dumps({"data": [{"id": "a"}, 42]})
    assert parse_v1_models(text) is None


def test_parse_missing_id():
    text = json.dumps({"data": [{"object": "model"}]})
    assert parse_v1_models(text) is None


def test_parse_empty_id():
    text = json.dumps({"data": [{"id": ""}]})
    assert parse_v1_models(text) is None


def test_parse_non_string_id():
    text = json.dumps({"data": [{"id": 7}]})
    assert parse_v1_models(text) is None


# --- ModelRegistry -----------------------------------------------------------


def test_register_and_resolve():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.sync("a", {"m1", "m2"})
    assert reg.resolve("m1") == "a"
    assert reg.resolve("m2") == "a"
    assert reg.resolve("unknown") is None


def test_sync_unregistered_backend_raises():
    """sync for a never-registered backend raises instead of silently no-opping.

    Ownership for an unregistered name would be invisible to resolve (absent
    from the registration order), so a caller typo must fail loudly.
    """
    reg = ModelRegistry()
    reg.register_backend("a", is_default=True)
    with pytest.raises(ValueError, match="not registered"):
        reg.sync("nope", {"m"})


def test_sync_add_and_remove():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.sync("a", {"m1", "m2"})
    reg.sync("a", {"m2", "m3"})
    assert reg.resolve("m1") is None
    assert reg.resolve("m2") == "a"
    assert reg.resolve("m3") == "a"


def test_sync_second_claim_keeps_first_registered_owner():
    """A model claimed by a later backend while owned by an earlier one stays
    with the first-registered owner (no default involved)."""
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=False)
    reg.sync("a", {"shared"})
    reg.sync("b", {"shared", "b-only"})
    assert reg.resolve("shared") == "a"
    assert reg.resolve("b-only") == "b"


def test_sync_default_backend_claims_model_from_previous_owner():
    """A model claimed by the default backend is resolved to it."""
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=True)
    reg.sync("a", {"shared"})
    reg.sync("b", {"shared", "b-only"})
    assert reg.resolve("shared") == "b"
    assert reg.resolve("b-only") == "b"


def test_collision_default_wins():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=True)
    reg.sync("a", {"m"})
    reg.sync("b", {"m"})
    assert reg.resolve("m") == "b"


def test_collision_first_registered_wins_when_no_default():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=False)
    reg.sync("a", {"m"})
    reg.sync("b", {"m"})
    assert reg.resolve("m") == "a"


def test_collision_default_wins_regardless_of_registration_order():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=True)
    reg.register_backend("b", is_default=False)
    reg.sync("b", {"m"})
    reg.sync("a", {"m"})
    assert reg.resolve("m") == "a"


def test_removed_model_re_resolves_to_other_owner():
    """A model dropped by the winner re-resolves to the remaining owner."""
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=True)
    reg.sync("a", {"m"})
    reg.sync("b", {"m"})
    assert reg.resolve("m") == "b"
    reg.sync("b", set())  # default backend drops the model
    assert reg.resolve("m") == "a"


def test_removed_model_dropped_when_no_owner_remains():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=True)
    reg.sync("a", {"m"})
    reg.sync("b", {"m"})
    reg.sync("b", set())
    reg.sync("a", set())
    assert reg.resolve("m") is None


def test_repeated_sync_of_same_set_does_not_relog(caplog):
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=False)
    with caplog.at_level(logging.WARNING, logger="gate.models"):
        reg.sync("a", {"m"})
        reg.sync("b", {"m"})  # first conflict: one log per (model, backend) pair
        first = len(caplog.records)
        assert first == 2  # (m, a) and (m, b)
        reg.sync("b", {"m"})  # repeated sync: must not re-log
        reg.sync("a", {"m"})  # repeated sync: must not re-log
    assert len(caplog.records) == first
    assert all("m" in r.message for r in caplog.records)


def test_conflict_logged_once_per_pair(caplog):
    """Two distinct colliding models each log once per pair; repeats stay silent."""
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=False)
    with caplog.at_level(logging.WARNING, logger="gate.models"):
        reg.sync("a", {"m1", "m2"})
        reg.sync("b", {"m1", "m2"})  # 2 models x 2 backends = 4 first-time logs
        first = len(caplog.records)
        assert first == 4
        reg.sync("b", {"m1", "m2"})  # repeated sync: must not re-log
    assert len(caplog.records) == first


def test_no_conflict_log_for_disjoint_sets(caplog):
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=False)
    with caplog.at_level(logging.WARNING, logger="gate.models"):
        reg.sync("a", {"m1"})
        reg.sync("b", {"m2"})
    assert caplog.records == []


def test_resolve_unowned_after_sync_returns_none():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=True)
    reg.sync("a", {"m"})
    reg.sync("a", set())
    assert reg.resolve("m") is None


# --- ModelRegistry.items (enumeration) ----------------------------------------


def test_items_empty_registry():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=True)
    assert reg.items() == []


def test_items_lists_owned_models_with_owner():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=True)
    reg.register_backend("b", is_default=False)
    reg.sync("a", {"m1", "m2"})
    reg.sync("b", {"m3"})
    assert reg.items() == [("m1", "a"), ("m2", "a"), ("m3", "b")]


def test_items_applies_collision_winner():
    """A model owned by two backends appears once, under the resolved owner."""
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=True)
    reg.sync("a", {"m", "a-only"})
    reg.sync("b", {"m"})
    # b is the default -> wins the collision; a keeps its own model.
    assert reg.items() == [("a-only", "a"), ("m", "b")]


def test_items_drops_unowned_models():
    """Models no backend owns do not appear."""
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.sync("a", {"m1", "m2"})
    reg.sync("a", {"m1"})
    assert reg.items() == [("m1", "a")]


# --- ModelRegistry.resolve_candidates (v2 routing lookup) ---------------------


def test_resolve_candidates_single_owner_is_one_tuple():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=True)
    reg.sync("a", {"m1"})
    reg.sync("b", {"m2"})
    assert reg.resolve_candidates("m1") == ("a",)
    assert reg.resolve_candidates("m2") == ("b",)


def test_resolve_candidates_multi_owner_in_registration_order():
    """A multi-owner model returns ALL owners in registration (config) order,
    NOT sync order."""
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=False)
    reg.register_backend("c", is_default=True)
    # b syncs first, a second: sync order is (b, a), registration is (a, b).
    reg.sync("b", {"m"})
    reg.sync("a", {"m"})
    assert reg.resolve_candidates("m") == ("a", "b")
    # c does not own m: it must not appear.
    reg.sync("c", {"c-only"})
    assert reg.resolve_candidates("m") == ("a", "b")


def test_resolve_candidates_unowned_model_is_empty_tuple():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.sync("a", {"m1"})
    assert reg.resolve_candidates("unknown") == ()


def test_resolve_candidates_removed_from_all_owners_is_empty():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=True)
    reg.sync("a", {"m"})
    reg.sync("b", {"m"})
    assert reg.resolve_candidates("m") == ("a", "b")
    reg.sync("b", set())
    assert reg.resolve_candidates("m") == ("a",)
    reg.sync("a", set())
    assert reg.resolve_candidates("m") == ()


def test_resolve_candidates_grows_when_second_owner_added_on_later_sync():
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=False)
    reg.sync("a", {"m"})
    assert reg.resolve_candidates("m") == ("a",)
    reg.sync("b", {"m"})
    assert reg.resolve_candidates("m") == ("a", "b")


def test_resolve_and_resolve_candidates_coexist_for_multi_owner():
    """For a multi-owner model, resolve returns the collision winner
    (default-wins) while resolve_candidates returns all owners."""
    reg = ModelRegistry()
    reg.register_backend("a", is_default=False)
    reg.register_backend("b", is_default=True)
    reg.sync("a", {"m", "a-only"})
    reg.sync("b", {"m"})
    assert reg.resolve("m") == "b"  # default wins the collision
    assert reg.resolve_candidates("m") == ("a", "b")  # all owners, config order
    assert reg.resolve("a-only") == "a"
    assert reg.resolve_candidates("a-only") == ("a",)
