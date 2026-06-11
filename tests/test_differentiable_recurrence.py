"""
Differentiable presynaptic state recurrence — release_canonical(differentiable=True) (bead yw9.2).

The synaptic state recurrence (calcium / buffer / RRP / reserve / SNARE / complexin / energy) was
advanced under ``torch.no_grad()`` with detached scatter inputs, so NO gradient flowed through the
state — the kinetics could not be learned. yw9.2 adds an opt-in ``differentiable=True`` mode to
``SynapticPresyn.release_canonical`` that runs the SAME update math under autograd, so the advanced
state carries gradient w.r.t. this step's inputs (and, once the kinetics are Parameters in yw9.3,
the kinetic params). This enables BPTT through the recurrence across a chain of calls (the chunked
TBPTT of yw9.2.3 builds on it).

The mode is DEFAULT-OFF and changes only gradient tracking, never the forward value — so existing
behavior is byte-identical. These tests lock: (1) exact forward parity vs the detached default,
(2) the next state is differentiable (gradcheck), (3) gradients genuinely flow ACROSS calls (BPTT),
and (4) the default mode keeps the recurrence detached.

Run:  pytest tests/test_differentiable_recurrence.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn, build_presyn_state

B, H, T_KEY, K, T = 1, 2, 6, 3, 4


def _presyn():
    cfg = SynapticConfig(enable_presyn=True)
    return SynapticPresyn(d_head=8, cfg=cfg), cfg


def _inputs(dtype=torch.float64, seed=1):
    g = torch.Generator().manual_seed(seed)
    drive = torch.randn(B, H, T, K, generator=g, dtype=dtype) * 0.5 + 0.5
    idx = torch.randint(0, T_KEY, (B, H, T, K), generator=g)
    return drive, idx


def _fresh_state(presyn, cfg, dtype=torch.float64):
    presyn.ema_e.fill_(1.0)  # deterministic EMA so the test is a pure function of inputs
    return build_presyn_state(B, T_KEY, H, "cpu", dtype, cfg)


# --------------------------------------------------------------------------- #
# 1. FORWARD PARITY: differentiable mode changes only gradients, not values
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_forward_value_is_identical_to_detached_default():
    presyn, cfg = _presyn()
    drive, idx = _inputs()

    def run(differentiable):
        st = _fresh_state(presyn, cfg)
        e = presyn.release_canonical(st, drive, idx, train=False, differentiable=differentiable)
        state = {k: (v.detach().clone() if torch.is_tensor(v) else [x.detach().clone() for x in v])
                 for k, v in st.items()}
        return e.detach().clone(), state

    e_off, st_off = run(False)
    e_on, st_on = run(True)
    assert torch.equal(e_off, e_on), "the returned bias must be byte-identical"
    for key in ("C", "BUF", "RRP", "RES", "PR", "CL", "E"):
        assert torch.equal(st_off[key], st_on[key]), f"state[{key}] must be byte-identical"


# --------------------------------------------------------------------------- #
# 2. THE NEXT STATE IS DIFFERENTIABLE (gradcheck w.r.t. drive)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_gradcheck_next_state_wrt_drive():
    presyn, cfg = _presyn()
    drive, idx = _inputs()

    def step_state(drive):
        st = _fresh_state(presyn, cfg)
        presyn.release_canonical(st, drive, idx, train=False, differentiable=True)
        # next-state scalar (does not involve the ema_e normalization)
        return st["C"].sum() + st["RRP"].sum() + st["E"].sum() + st["BUF"].sum()

    d = drive.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(step_state, (d,), eps=1e-6, atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. BPTT: gradients flow ACROSS calls in differentiable mode, not in default
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_gradient_flows_across_calls_only_when_differentiable():
    presyn, cfg = _presyn()
    drive, idx = _inputs()

    def two_step(drive1, differentiable):
        st = _fresh_state(presyn, cfg)
        presyn.release_canonical(st, drive1, idx, train=False, differentiable=differentiable)
        drive2 = torch.full((B, H, T, K), 0.3, dtype=torch.float64)
        return presyn.release_canonical(st, drive2, idx, train=False, differentiable=differentiable).sum()

    # default: step-1 drive cannot affect step-2 output (state detached) -> no grad graph
    d1 = drive.clone().requires_grad_(True)
    out_off = two_step(d1, False)
    assert not out_off.requires_grad, "default mode must detach the recurrence"

    # differentiable: step-1 drive DOES affect step-2 output via the carried state
    d1 = drive.clone().requires_grad_(True)
    out_on = two_step(d1, True)
    (grad,) = torch.autograd.grad(out_on, d1, allow_unused=True)
    assert grad is not None and grad.abs().sum() > 0, "BPTT must flow drive1 -> state -> step-2"


# --------------------------------------------------------------------------- #
# 4. DEFAULT MODE leaves the advanced state detached (no grad graph on state)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_default_mode_state_is_detached():
    presyn, cfg = _presyn()
    drive, idx = _inputs()
    d = drive.clone().requires_grad_(True)
    st = _fresh_state(presyn, cfg)
    presyn.release_canonical(st, d, idx, train=False, differentiable=False)
    assert not st["C"].requires_grad and not st["RRP"].requires_grad, "default state must be detached"

    # ...while differentiable mode produces a state that does carry grad
    d = drive.clone().requires_grad_(True)
    st = _fresh_state(presyn, cfg)
    presyn.release_canonical(st, d, idx, train=False, differentiable=True)
    assert st["C"].requires_grad and st["RRP"].requires_grad, "differentiable state must carry grad"
