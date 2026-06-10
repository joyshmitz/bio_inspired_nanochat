"""
Per-sequence reset of fast weights and plasticity buffers — bead vg9.4.

W_fast / camkii / pp1 / bdnf / u_buf / v_buf are module buffers/params that persisted across
sequences and were never reset, so one sequence's writes leaked into the next. reset_sequence_state
gives a documented per-sequence/persistent contract:
  PER-SEQUENCE (reset): eligibility u_buf/v_buf, plasticity bookkeeping; CaMKII/PP1/BDNF gate
    (reset_consolidation, default True); fast weights w_fast/post.fast (reset_fast_weights, opt-in).
  PERSISTENT (never reset): slow weights, low-rank U/V, fixed projections, bias, presyn EMA.

Run:  pytest tests/test_sequence_reset.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticLinear

from _bio_testkit import make_tiny_synaptic, random_tokens, set_seed

IN, OUT, B = 16, 16, 8


def _driven_linear(*, R=4, steps=3, seed=0) -> SynapticLinear:
    set_seed(seed)
    cfg = SynapticConfig(enable_hebbian=True, rank_eligibility=R)
    lin = SynapticLinear(IN, OUT, cfg).eval()
    with torch.no_grad():
        for _ in range(steps):
            lin(torch.randn(B, IN), torch.ones(B), torch.ones(B))
    return lin


# --------------------------------------------------------------------------- #
# 1. The contract: per-sequence state resets, persistent state survives
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_reset_clears_eligibility_and_gate_but_keeps_slow():
    lin = _driven_linear()
    assert lin.u_buf.abs().sum() > 0, "eligibility should have accumulated before reset"
    assert not torch.allclose(lin.post.pp1, torch.full_like(lin.post.pp1, 0.5)), "PP1 should have drifted"

    # snapshot the PERSISTENT state
    w_slow0 = lin.w_slow.clone()
    proj0 = lin.proj_in.clone()
    slow0 = lin.post.slow.clone()
    u_lowrank0 = lin.post.U.clone()

    lin.reset_sequence_state()

    # PER-SEQUENCE state cleared
    assert lin.u_buf.abs().sum() == 0 and lin.v_buf.abs().sum() == 0
    assert lin.post.camkii.abs().sum() == 0
    assert torch.allclose(lin.post.pp1, torch.full_like(lin.post.pp1, 0.5))
    assert lin.post.bdnf.abs().sum() == 0
    assert lin._plasticity_pending is False and lin._last_gate_scale is None

    # PERSISTENT state untouched
    assert torch.equal(lin.w_slow, w_slow0), "slow weight must persist"
    assert torch.equal(lin.proj_in, proj0), "fixed projection must persist"
    assert torch.equal(lin.post.slow, slow0), "consolidated diagonal must persist"
    assert torch.equal(lin.post.U, u_lowrank0), "low-rank U must persist"


# --------------------------------------------------------------------------- #
# 2. Toggles: reset_fast_weights and reset_consolidation
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_default_does_not_touch_fast_weights_but_optin_zeros_them():
    lin = _driven_linear()
    wf_before = lin.w_fast.clone()
    lin.reset_sequence_state(reset_fast_weights=False)
    assert torch.equal(lin.w_fast, wf_before), "default reset must NOT zero the trained fast weight"

    lin.reset_sequence_state(reset_fast_weights=True)
    assert lin.w_fast.abs().sum() == 0, "strict mode must zero w_fast"
    assert lin.post.fast.abs().sum() == 0, "strict mode must zero post.fast"


@pytest.mark.unit
def test_consolidation_can_be_carried_across_sequences():
    lin = _driven_linear()
    pp1_before = lin.post.pp1.clone()
    bdnf_before = lin.post.bdnf.clone()
    lin.reset_sequence_state(reset_consolidation=False)  # cel mode: keep the gate state
    assert torch.equal(lin.post.pp1, pp1_before), "consolidation state must persist when opted in"
    assert torch.equal(lin.post.bdnf, bdnf_before)
    assert lin.u_buf.abs().sum() == 0, "but eligibility still resets every sequence"


# --------------------------------------------------------------------------- #
# 3. Model-level reset across all synaptic layers
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_model_reset_clears_all_layers_and_reports_count():
    model = make_tiny_synaptic(seed=0, train=True)
    with torch.no_grad():
        model(random_tokens(2, 16, vocab=model.config.vocab_size))

    lins = [m for m in model.modules() if isinstance(m, SynapticLinear) and m.u_buf is not None]
    assert lins and any(m.u_buf.abs().sum() > 0 for m in lins), "some layer should have state"

    n = model.reset_sequence_state()
    assert n == len([m for m in model.modules() if isinstance(m, SynapticLinear)])
    for m in lins:
        assert m.u_buf.abs().sum() == 0, "every layer's eligibility must reset"


@pytest.mark.unit
def test_reset_is_idempotent_and_finite():
    lin = _driven_linear()
    lin.reset_sequence_state()
    lin.reset_sequence_state()  # second reset is a safe no-op
    assert lin.u_buf.abs().sum() == 0
    assert torch.isfinite(lin.w_slow).all() and torch.isfinite(lin.post.slow).all()
