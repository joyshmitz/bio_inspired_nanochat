"""
FlexAttention presyn path migrated to the canonical formulation — bead s3w9.

flex_synaptic.py was a 3rd divergent presyn impl (sigmoid mix + raw AMP). It now uses the SAME
faithful formulation as the live standard path's release_canonical: Hill Syt + Doc2 + complexin/
SNARE fuse for the per-key readiness, and an energy-gated AMPA amplitude. These tests lock the
parity (and that AMP is no longer read).

Run:  pytest tests/test_flex_canonical.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.flex_synaptic import SynapticFlexAttention
from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
)

from _bio_testkit import set_seed

DEV = torch.device("cpu")
DT = torch.float32


@pytest.mark.unit
def test_flex_precompute_matches_canonical_release_prob():
    set_seed(0)
    cfg = SynapticConfig(enable_presyn=True)
    pre = SynapticPresyn(16, cfg)
    flex = SynapticFlexAttention(cfg)
    state = build_presyn_state(1, 4, 2, DEV, DT, cfg)
    with torch.no_grad():
        state["C"].copy_(torch.rand_like(state["C"]) * 2.0)  # vary calcium so the Hill is exercised

    kf, qamp = flex.precompute_bio_factors(state, cfg)

    c, pr, cl = state["C"], state["PR"], state["CL"]
    # canonical fuse_base == _faithful_release_prob with the bilinear driven to 1 (large drive)
    p_fuse = pre._faithful_release_prob(c, pr, cl, torch.full_like(c, 50.0))
    fuse_flex = kf / state["RRP"]
    assert torch.allclose(fuse_flex, p_fuse, atol=1e-4), "flex fuse_base must match the canonical Hill/fuse"

    qamp_expected = torch.sigmoid(cfg.q_beta * (state["E"] - 0.5)) * cfg.qmax
    assert torch.allclose(qamp, qamp_expected, atol=1e-6), "flex qamp must be the energy-gated amplitude"


@pytest.mark.unit
def test_flex_no_longer_reads_amp():
    # AMP is superseded by energy->qamp; precompute must not depend on state["AMP"].
    set_seed(0)
    cfg = SynapticConfig(enable_presyn=True)
    flex = SynapticFlexAttention(cfg)
    state = build_presyn_state(1, 4, 2, DEV, DT, cfg)
    kf0, q0 = flex.precompute_bio_factors(state, cfg)
    with torch.no_grad():
        state["AMP"].fill_(99.0)  # corrupt AMP
    kf1, q1 = flex.precompute_bio_factors(state, cfg)
    assert torch.equal(kf0, kf1) and torch.equal(q0, q1), "flex must not read AMP anymore"


@pytest.mark.unit
def test_flex_forward_runs_finite_if_available():
    # Best-effort end-to-end smoke; flex_attention may require compilation/hardware not present.
    cfg = SynapticConfig(enable_presyn=True)
    flex = SynapticFlexAttention(cfg)
    B, H, T, D = 1, 2, 8, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    state = build_presyn_state(B, T, H, DEV, DT, cfg)
    try:
        out = flex(q, k, v, state, block_mask=None)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"flex_attention unavailable in this environment: {exc}")
    assert out.shape == (B, H, T, D)
    assert torch.isfinite(out).all(), "flex attention output must be finite"
