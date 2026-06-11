"""Tests for the SynapticConfig parameter census (bead bio_inspired_nanochat-8j9.6).

Locks the census in two ways:
1. The generator's classification is correct (known dead fields are dead, known
   live/ambiguous fields are live, counts are internally consistent).
2. The committed ``docs/parameter_census.json`` is not stale (CI drift guard) —
   so refactors that add/remove/rewire a config field force a census refresh.
"""

from __future__ import annotations

import dataclasses

import pytest

from bio_inspired_nanochat.synaptic import SynapticConfig
from scripts.param_census import build_census, cmaes_phase1_params

pytestmark = pytest.mark.unit

# Verified by hand against the source (see CLAIMS_AUDIT.md / 8j9.5).
KNOWN_DEAD = {
    "enabled",
    "camkii_down",
    "router_sim_threshold",
    "native_presyn",
    "native_metrics",
    "native_plasticity",
}
# Fields whose status is non-obvious (shared names / indirect handles) and must
# resolve LIVE — these are exactly the cases the disambiguation rule exists for.
KNOWN_LIVE = {
    "router_contrastive_lr",
    "router_contrastive_push",
    "native_genetics",
    "init_amp",
    "structural_age_bias",
    "rec_rate",
}


@pytest.fixture(scope="module")
def census() -> dict:
    return build_census()


def test_census_covers_every_field(census: dict) -> None:
    names = {f.name for f in dataclasses.fields(SynapticConfig)}
    census_names = {r["name"] for r in census["fields"]}
    assert census_names == names
    assert census["field_count"] == len(names)


def test_counts_are_consistent(census: dict) -> None:
    assert census["live_count"] + census["dead_count"] == census["field_count"]
    live = sum(1 for r in census["fields"] if r["status"] == "LIVE")
    dead = sum(1 for r in census["fields"] if r["status"] == "DEAD")
    assert (live, dead) == (census["live_count"], census["dead_count"])


def test_known_dead_fields_are_dead(census: dict) -> None:
    status = {r["name"]: r["status"] for r in census["fields"]}
    for name in KNOWN_DEAD:
        assert status[name] == "DEAD", f"{name} should be DEAD"


def test_known_live_fields_are_live(census: dict) -> None:
    status = {r["name"]: r["status"] for r in census["fields"]}
    for name in KNOWN_LIVE:
        assert status[name] == "LIVE", f"{name} should be LIVE"


def test_enabled_collision_is_disambiguated(census: dict) -> None:
    # `enabled` is the one field name shared with other config dataclasses; it
    # must be flagged as a collision yet still resolve DEAD (no syn_cfg reader).
    assert "enabled" in census["collision_fields"]
    rec = next(r for r in census["fields"] if r["name"] == "enabled")
    assert rec["status"] == "DEAD"
    assert rec["runtime_read_sites"] == []


def test_tuned_set_matches_cmaes_specs(census: dict) -> None:
    tuned = {r["name"] for r in census["fields"] if r["tuned_phase1"]}
    assert tuned == set(cmaes_phase1_params())
    assert census["tuned_phase1_count"] == len(tuned)
    # All tuned params must themselves be live (you can't search a dead knob).
    status = {r["name"]: r["status"] for r in census["fields"]}
    assert all(status[name] == "LIVE" for name in tuned)


def test_committed_json_is_not_stale() -> None:
    import json
    from pathlib import Path

    json_path = Path(__file__).resolve().parent.parent / "docs" / "parameter_census.json"
    assert json_path.exists(), "run `uv run python -m scripts.param_census`"
    committed = json.loads(json_path.read_text(encoding="utf-8"))
    fresh = build_census()
    # Compare the stable, semantic fields (ignore ordering of evidence lists).
    assert committed["field_count"] == fresh["field_count"]
    assert committed["live_count"] == fresh["live_count"]
    assert committed["dead_count"] == fresh["dead_count"]
    committed_status = {r["name"]: r["status"] for r in committed["fields"]}
    fresh_status = {r["name"]: r["status"] for r in fresh["fields"]}
    assert committed_status == fresh_status, (
        "docs/parameter_census.json is stale; re-run "
        "`uv run python -m scripts.param_census`."
    )
