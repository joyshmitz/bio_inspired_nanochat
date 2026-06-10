"""
Presyn parameter census — bead or4t.

After the presyn unification (canonical Hill release everywhere), the legacy sigmoid release-prob
params became dead. These tests lock that they are removed, the canonical live params exist, and
the tune-bio-params search space references only real (live) config fields (no orphaned knobs).

Run:  pytest tests/test_presyn_param_census.py -v
"""

from __future__ import annotations

import dataclasses

import pytest

from bio_inspired_nanochat.synaptic import SynapticConfig

_FIELDS = {f.name for f in dataclasses.fields(SynapticConfig)}

# The legacy sigmoid release path (deleted in qcj7) was the only reader of these.
_DEAD = ("alpha_c", "syt1_slope", "syt7_slope", "cpx_thresh", "amp_load", "amp_leak")
# What the canonical (release_canonical / flex / kernels) actually use for the release prob.
_CANONICAL = ("alpha_ca", "syt_fast_kd", "syt_slow_kd", "complexin_bias", "doc2_gain", "tau_c", "q_beta", "qmax")


@pytest.mark.unit
def test_dead_legacy_presyn_params_are_removed():
    present = [d for d in _DEAD if d in _FIELDS]
    assert present == [], f"dead legacy presyn params must be removed (or4t): {present}"


@pytest.mark.unit
def test_canonical_presyn_params_exist():
    missing = [c for c in _CANONICAL if c not in _FIELDS]
    assert missing == [], f"canonical presyn params must exist: {missing}"


@pytest.mark.unit
def test_tune_search_space_references_only_real_fields():
    from scripts.tune_bio_params import TOP10_PARAM_SPECS, _validate_param_specs

    _validate_param_specs(TOP10_PARAM_SPECS)  # raises on duplicate / lower>=upper
    orphans = [s.name for s in TOP10_PARAM_SPECS if s.name not in _FIELDS]
    assert orphans == [], f"tune params not in SynapticConfig (would be no-ops): {orphans}"
    # and none of them is a removed dead param
    dead_in_tune = [s.name for s in TOP10_PARAM_SPECS if s.name in _DEAD]
    assert dead_in_tune == [], f"tune must not optimize dead params: {dead_in_tune}"
