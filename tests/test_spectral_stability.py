"""
Spectral-radius / contraction monitor for the learned recurrence (bead yw9.7).

The calcium↔buffer subsystem cannot blow up iff the spectral radius of its 2×2 linear transition
M(β) is < 1 for all β=(1−BUF)∈[0,1]. `cb_spectral_radius` is the differentiable closed form;
`LearnableKinetics.spectral_radius` is the worst-case-over-β monitor. See
docs/stable_recurrence_theory.md.

These tests lock: the closed form matches `torch.linalg.eigvals`; the realistic/init kinetics are
comfortably contractive; the yw9.3 parameterization keeps ρ(M) < 1 for ANY finite parameter (the
stability guarantee); and the monitor is differentiable (usable as a telemetry margin / penalty).

Run:  pytest tests/test_spectral_stability.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    LearnableKinetics,
    cb_spectral_radius,
)


def _ref_spectral_radius(rc, rb, aon, aoff, beta):
    M = torch.tensor(
        [[rc - aon * beta, aoff], [aon * beta, rb - aoff]], dtype=torch.float64
    )
    return torch.linalg.eigvals(M).abs().max().item()


# --------------------------------------------------------------------------- #
# 1. Closed form matches torch.linalg.eigvals (real and complex eigenvalue regimes)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_closed_form_matches_eigvals():
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(1000):
        rc, rb = torch.rand(2, generator=g, dtype=torch.float64).tolist()
        aon, aoff = (torch.rand(2, generator=g, dtype=torch.float64) * 0.5).tolist()
        beta = torch.rand((), generator=g, dtype=torch.float64).item()
        got = cb_spectral_radius(
            torch.tensor(rc), torch.tensor(rb), torch.tensor(aon),
            torch.tensor(aoff), torch.tensor(beta),
        ).item()
        worst = max(worst, abs(got - _ref_spectral_radius(rc, rb, aon, aoff, beta)))
    assert worst < 1e-5, f"closed-form spectral radius off by {worst:.2e}"


# --------------------------------------------------------------------------- #
# 2. The realistic (init) kinetics are comfortably contractive
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_init_kinetics_are_contractive():
    kin = LearnableKinetics(SynapticConfig()).double()
    sr = kin.spectral_radius().item()
    assert sr < 1.0
    # dominated by the calcium decay rho_c = exp(-1/tau_c) ≈ 0.846, a wide stability margin
    assert sr == pytest.approx(kin.values()["rho_c"], abs=1e-3)


# --------------------------------------------------------------------------- #
# 3. STRICT contraction for ANY finite parameter (the stability guarantee)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_strictly_contractive_for_arbitrary_finite_params():
    cfg = SynapticConfig()
    g = torch.Generator().manual_seed(1)
    for _ in range(200):
        kin = LearnableKinetics(cfg).double()
        with torch.no_grad():
            for name in ("theta_rho_c", "theta_rho_b", "theta_alpha_ca",
                         "theta_alpha_buf_on", "theta_alpha_buf_off"):
                getattr(kin, name).fill_(float(torch.randn((), generator=g) * 6.0))
        assert kin.spectral_radius().item() < 1.0, "finite params must stay strictly contractive"


# --------------------------------------------------------------------------- #
# 4. Differentiable monitor (can be logged or used as a soft stability penalty)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_spectral_radius_is_differentiable():
    kin = LearnableKinetics(SynapticConfig())
    sr = kin.spectral_radius()
    sr.backward()
    grads = [p.grad for p in kin.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "no gradient from spectral radius"


# --------------------------------------------------------------------------- #
# 5. Marginal limit: as the calcium decay → 1 the spectral radius → 1 (but stays < 1)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_marginal_limit_as_decay_approaches_one():
    kin = LearnableKinetics(SynapticConfig()).double()
    with torch.no_grad():
        kin.theta_rho_c.fill_(12.0)  # rho_c -> ~1 (pure integrator, marginally stable)
    sr = kin.spectral_radius().item()
    assert 0.99 < sr < 1.0, f"near-integrator must approach but not reach 1; got {sr}"
