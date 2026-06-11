"""
Differentiable, conservation-preserving vesicle depletion/refill (bead yw9.2.2).

The RRP/RES/DELAY (readily-releasable pool / reserve / in-flight endocytosis queue) update in
``release_canonical`` is the nonlinear, hard-clamped, `torch.no_grad` part of the presynaptic
recurrence. `vesicle_depletion_refill` is its differentiable replacement: smooth softplus clamp
surrogates so gradients flow, and **explicit paired transfers** so the vesicle-conservation
invariant holds *structurally* —

    Δ(RRP + RES + Σdelay) = − released_eff · (1 − rec_rate)

regardless of the surrogate sharpness `β` or whether any clamp saturates. Total vesicles change
ONLY by the explicitly-modelled endocytosis recycling leak (exact conservation at ``rec_rate=1``).
See docs/differentiable_synaptic_dynamics_design.md §2.

Run:  pytest tests/test_vesicle_conservation.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import vesicle_depletion_refill, _soft_relu, _soft_min

PRIME = 0.075  # cfg.prime_rate default


def _pools(seed, shape=(4, 3), rrp_scale=6.0, res_scale=18.0, n_delay=3):
    g = torch.Generator().manual_seed(seed)
    rrp = torch.rand(shape, generator=g, dtype=torch.float64) * rrp_scale
    res = torch.rand(shape, generator=g, dtype=torch.float64) * res_scale
    delay = [torch.rand(shape, generator=g, dtype=torch.float64) for _ in range(n_delay)]
    return rrp, res, delay


def _total(rrp, res, delay):
    return rrp.sum() + res.sum() + sum(d.sum() for d in delay)


# --------------------------------------------------------------------------- #
# 1. CONSERVATION: Δtotal == −released_eff·(1−rec_rate), exactly, any β / saturation
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize("rec_rate", [0.0, 0.06, 0.5, 1.0])
@pytest.mark.parametrize("saturate", [False, True])
def test_conservation_is_exact(rec_rate, saturate):
    rrp, res, delay = _pools(seed=int(rec_rate * 100) + saturate)
    g = torch.Generator().manual_seed(7)
    # saturate=True: released can exceed RRP (forces the over-depletion clamp); β small to stress it.
    released = torch.rand((4, 3), generator=g, dtype=torch.float64) * (25.0 if saturate else 3.0)
    beta = 20.0 if saturate else 50.0

    t0 = _total(rrp, res, delay)
    rrp2, res2, delay2, diag = vesicle_depletion_refill(
        rrp, res, delay, released, prime_rate=PRIME, rec_rate=rec_rate, beta=beta
    )
    t1 = _total(rrp2, res2, delay2)
    expected = -diag["released_eff"].sum() * (1.0 - rec_rate)
    assert torch.allclose(t1 - t0, expected, atol=1e-10), "vesicle bookkeeping must be exact"


@pytest.mark.unit
def test_no_delay_buffer_keeps_queue_empty_and_conserves():
    # endo_delay==0: recycled vesicles have nowhere to queue, so they route straight back to the
    # reserve — the queue must stay EMPTY (not grow to length 1) and the budget must still hold.
    rrp, res, _ = _pools(seed=11, n_delay=0)
    released = torch.rand((4, 3), dtype=torch.float64) * 20.0
    t0 = rrp.sum() + res.sum()
    rrp2, res2, delay2, diag = vesicle_depletion_refill(
        rrp, res, [], released, prime_rate=PRIME, rec_rate=0.06, beta=20.0
    )
    assert delay2 == [], "no delay buffer => queue stays empty"
    expected = -diag["released_eff"].sum() * (1.0 - 0.06)
    assert torch.allclose((rrp2.sum() + res2.sum()) - t0, expected, atol=1e-10)


@pytest.mark.unit
def test_full_recycling_is_exactly_conserved():
    rrp, res, delay = _pools(seed=1)
    released = torch.rand((4, 3)).double() * 30.0  # large -> saturates depletion
    t0 = _total(rrp, res, delay)
    rrp2, res2, delay2, _ = vesicle_depletion_refill(
        rrp, res, delay, released, prime_rate=PRIME, rec_rate=1.0, beta=15.0
    )
    assert torch.allclose(_total(rrp2, res2, delay2), t0, atol=1e-10), "rec_rate=1 => no net loss"


# --------------------------------------------------------------------------- #
# 2. POOLS STAY NON-NEGATIVE (no over-depletion) even with huge release
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_pools_never_go_negative():
    rrp, res, delay = _pools(seed=2)
    released = torch.full((4, 3), 1e6, dtype=torch.float64)  # absurd over-release
    rrp2, res2, delay2, _ = vesicle_depletion_refill(
        rrp, res, delay, released, prime_rate=PRIME, rec_rate=0.06, beta=50.0
    )
    assert (rrp2 >= -1e-9).all() and (res2 >= -1e-9).all()
    assert all((d >= -1e-9).all() for d in delay2)


# --------------------------------------------------------------------------- #
# 3. DIFFERENTIABILITY (gradcheck through the whole step)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_gradcheck_through_depletion_refill():
    g = torch.Generator().manual_seed(3)
    rrp = (torch.rand(2, 2, generator=g, dtype=torch.float64) * 6).requires_grad_(True)
    res = (torch.rand(2, 2, generator=g, dtype=torch.float64) * 18).requires_grad_(True)
    rel = (torch.rand(2, 2, generator=g, dtype=torch.float64) * 4).requires_grad_(True)
    d = [(torch.rand(2, 2, generator=g, dtype=torch.float64)).requires_grad_(True) for _ in range(3)]

    def f(rrp, res, rel, d0, d1, d2):
        r, s, nd, _ = vesicle_depletion_refill(
            rrp, res, [d0, d1, d2], rel, prime_rate=PRIME, rec_rate=0.06, beta=20.0
        )
        return r.sum() + s.sum() + sum(x.sum() for x in nd)

    assert torch.autograd.gradcheck(f, (rrp, res, rel, d[0], d[1], d[2]), eps=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. FORWARD PARITY with the hard-clamp reference in the unsaturated regime
# --------------------------------------------------------------------------- #
def _reference_update(rrp, res, delay, add_vals, prime, rec):
    """Mirror of release_canonical's hard-clamped RRP/RES/DELAY update."""
    rrp_up = torch.clamp(rrp - add_vals, 0)
    res_up = res + delay[0]
    new_delay = delay[1:] + [add_vals * rec]
    take = torch.minimum(res_up, torch.ones_like(res_up))
    res_up = torch.clamp(res_up - prime * take, 0)
    rrp_up = torch.clamp(rrp_up + prime * take, 0, 30.0)
    return rrp_up, res_up, new_delay


@pytest.mark.unit
def test_matches_reference_in_unsaturated_regime():
    # Values chosen so no clamp activates: released < rrp, res > 1, rrp + primed < cap.
    rrp = torch.full((3,), 5.0, dtype=torch.float64)
    res = torch.full((3,), 10.0, dtype=torch.float64)
    delay = [torch.full((3,), 0.5, dtype=torch.float64) for _ in range(3)]
    released = torch.full((3,), 1.0, dtype=torch.float64)

    r1, s1, nd1, _ = vesicle_depletion_refill(
        rrp, res, delay, released, prime_rate=PRIME, rec_rate=0.06, beta=2000.0
    )
    r2, s2, nd2 = _reference_update(rrp, res, delay, released, PRIME, 0.06)
    assert torch.allclose(r1, r2, atol=1e-5) and torch.allclose(s1, s2, atol=1e-5)
    assert torch.allclose(nd1[-1], nd2[-1], atol=1e-12)


# --------------------------------------------------------------------------- #
# 5. RRP CAP routes the excess to the RESERVE (conserve, don't discard)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_rrp_cap_routes_excess_to_reserve():
    cap = 30.0
    rrp = torch.full((2,), cap - 0.01, dtype=torch.float64)  # near the cap
    res = torch.full((2,), 20.0, dtype=torch.float64)        # plenty of reserve -> primes hard
    delay = [torch.zeros(2, dtype=torch.float64) for _ in range(3)]
    released = torch.zeros(2, dtype=torch.float64)           # no release -> priming pushes over cap

    t0 = _total(rrp, res, delay)
    rrp2, res2, delay2, diag = vesicle_depletion_refill(
        rrp, res, delay, released, prime_rate=PRIME, rec_rate=0.06, rrp_cap=cap, beta=200.0
    )
    assert (rrp2 <= cap + 1e-3).all(), "RRP must respect the cap"
    assert (diag["over"] > 0).any(), "the cap must have activated"
    # released_eff == 0 here, so conservation is exact (no leak): the over-cap vesicles went to RES.
    assert torch.allclose(_total(rrp2, res2, delay2), t0, atol=1e-10)


# --------------------------------------------------------------------------- #
# 6. The smooth surrogates approach the hard clamps as β grows
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_soft_surrogates_converge_to_hard():
    x = torch.linspace(-3, 3, 50, dtype=torch.float64)
    assert (_soft_relu(x, 1000.0) - torch.clamp(x, min=0)).abs().max() < 1e-2
    assert (_soft_min(x, 1.0, 1000.0) - torch.minimum(x, torch.ones_like(x))).abs().max() < 1e-2
