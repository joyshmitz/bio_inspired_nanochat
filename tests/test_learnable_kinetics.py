"""
SGD-learnable, stability-preserving presynaptic kinetics (bead yw9.3).

The calcium/buffer kinetics (τc, τbuf, αca, αbuf_on/off) were fixed hyperparameters. yw9.3
promotes them to learned Parameters (`LearnableKinetics`), reached through the differentiable
recurrence (yw9.2). They are stability-preserving by construction — decays via ``sigmoid∈(0,1)``,
gains via ``softplus``, buffer-coupling via a bounded ``_ABUF_MAX·sigmoid`` — so SGD can never push
them outside the contractive regime. They are initialized (via the inverse maps) to reproduce the
cfg constants exactly, so turning the flag on is a no-op until training moves them.

These tests lock: (1) init reproduces the hand-tuned forward, (2) the constrained values respect
their bounds for ANY raw parameter value, (3) the kinetics receive gradient through the
differentiable recurrence, and (4) a short optimization moves them and reduces a loss without
divergence, staying constrained-valid throughout.

Run:  pytest tests/test_learnable_kinetics.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    LearnableKinetics,
    build_presyn_state,
    _ABUF_MAX,
)

B, H, T_KEY, K, T = 1, 2, 6, 3, 4


def _presyn(learnable):
    cfg = SynapticConfig(enable_presyn=True, learnable_kinetics=learnable)
    return SynapticPresyn(d_head=8, cfg=cfg).double(), cfg


def _inputs():
    g = torch.Generator().manual_seed(1)
    drive = torch.randn(B, H, T, K, generator=g, dtype=torch.float64) * 0.4 + 0.5
    idx = torch.randint(0, T_KEY, (B, H, T, K), generator=g)
    return drive, idx


# --------------------------------------------------------------------------- #
# 1. INIT reproduces the hand-tuned (cfg) kinetics and forward
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_init_matches_cfg_constants():
    cfg = SynapticConfig()
    kin = LearnableKinetics(cfg).double()
    v = kin.values()
    assert v["rho_c"] == pytest.approx(math.exp(-1.0 / cfg.tau_c), abs=1e-6)
    assert v["rho_b"] == pytest.approx(math.exp(-1.0 / cfg.tau_buf), abs=1e-6)
    assert v["alpha_ca"] == pytest.approx(cfg.alpha_ca, abs=1e-6)
    assert v["alpha_buf_on"] == pytest.approx(cfg.alpha_buf_on, abs=1e-6)
    assert v["alpha_buf_off"] == pytest.approx(cfg.alpha_buf_off, abs=1e-6)


@pytest.mark.unit
def test_learnable_init_forward_matches_fixed():
    drive, idx = _inputs()

    def run(learnable):
        p, cfg = _presyn(learnable)
        st = build_presyn_state(B, T_KEY, H, "cpu", torch.float64, cfg)
        p.ema_e.fill_(1.0)
        e = p.release_canonical(st, drive, idx, train=False)
        return e.detach(), st["C"].detach().clone()

    e_fixed, c_fixed = run(False)
    e_learn, c_learn = run(True)
    assert torch.allclose(e_fixed, e_learn, atol=1e-6), "learnable init must reproduce the cfg forward"
    assert torch.allclose(c_fixed, c_learn, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2. CONSTRAINTS hold for ANY raw parameter value (stability-preserving)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_constraints_hold_for_arbitrary_raw_params():
    cfg = SynapticConfig()
    kin = LearnableKinetics(cfg).double()
    names = ("theta_rho_c", "theta_rho_b", "theta_alpha_ca",
             "theta_alpha_buf_on", "theta_alpha_buf_off")
    with torch.no_grad():
        # shove the raw params to large values in both directions (±20 stays within float
        # resolution; sigmoid(50) rounds to exactly 1.0, which is a numerical, not a model, edge)
        for name in names:
            getattr(kin, name).fill_(20.0)
        v_hi = kin.values()
        for name in names:
            getattr(kin, name).fill_(-20.0)
        v_lo = kin.values()
    for v in (v_hi, v_lo):
        assert 0.0 < v["rho_c"] < 1.0 and 0.0 < v["rho_b"] < 1.0, "decays must stay in (0,1)"
        assert v["alpha_ca"] >= 0.0, "influx gain must stay non-negative"
        assert 0.0 <= v["alpha_buf_on"] <= _ABUF_MAX, "buffer coupling must respect the cap"
        assert 0.0 <= v["alpha_buf_off"] <= _ABUF_MAX


# --------------------------------------------------------------------------- #
# 3. GRADIENT reaches the kinetics through the differentiable recurrence
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_kinetics_receive_gradient():
    # A multi-step differentiable chain is required: the decays rho_c/rho_b act on the PRIOR
    # calcium/buffer, which start at zero, so they only acquire gradient once the state has
    # accumulated over a few steps (correct physics — a decay on nothing has no effect).
    p, cfg = _presyn(True)
    drive, idx = _inputs()
    st = build_presyn_state(B, T_KEY, H, "cpu", torch.float64, cfg)
    p.ema_e.fill_(1.0)
    out = torch.zeros((), dtype=torch.float64)
    for _ in range(4):
        e = p.release_canonical(st, drive, idx, train=False, differentiable=True)
        out = out + e.sum()
    (out + st["C"].sum() + st["BUF"].sum()).backward()
    for name, param in p.kinetics.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), f"{name} got no/NaN grad"
        assert param.grad.abs().sum() > 0, f"{name} gradient is identically zero"


# --------------------------------------------------------------------------- #
# 4. A SHORT TRAINING RUN moves the kinetics and improves the loss, no divergence
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_training_moves_kinetics_and_improves_loss():
    torch.manual_seed(0)
    p, cfg = _presyn(True)
    opt = torch.optim.Adam(p.kinetics.parameters(), lr=0.05)
    g = torch.Generator().manual_seed(3)
    drive = torch.randn(B, H, T, K, generator=g, dtype=torch.float64) * 0.4 + 0.5
    idx = torch.randint(0, T_KEY, (B, H, T, K), generator=g)
    target = torch.rand(B, H, T, K, dtype=torch.float64) * 0.5

    v0 = p.kinetics.values()
    first = last = None
    for step in range(40):
        st = build_presyn_state(B, T_KEY, H, "cpu", torch.float64, cfg)
        p.ema_e.fill_(1.0)
        e = p.release_canonical(st, drive, idx, train=False, differentiable=True)
        loss = ((e - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss), "training must not diverge"
        first = loss.item() if first is None else first
        last = loss.item()

    v1 = p.kinetics.values()
    assert last < first, f"loss must improve: {first:.5f} -> {last:.5f}"
    assert max(abs(v1[k] - v0[k]) for k in v0) > 1e-3, "the kinetics must actually move"
    # still constrained-valid after training
    assert 0.0 < v1["rho_c"] < 1.0 and 0.0 < v1["rho_b"] < 1.0
    assert v1["alpha_ca"] >= 0.0 and 0.0 <= v1["alpha_buf_on"] <= _ABUF_MAX
