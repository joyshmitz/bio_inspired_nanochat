"""Golden reference tests locking the canonical presyn dynamics — bead 8j9.4.

After the presyn unification (8j9.2), the single faithful source of truth for the
LIVE attention path is ``SynapticPresyn.release_canonical``. This module freezes
its numerics with committed golden artifacts so that future refactors — and the
Triton/Rust kernel backends still to be written (jyb.2, eqyk.13) — cannot silently
change the biophysics.

Two things are frozen:

1. The **release-probability formula** ``_faithful_release_prob`` (the documented
   Hill ``Syt(C)=C/(C+Kd)`` + complexin + Doc2 + bilinear-drive equation) over a
   fixed grid of inputs.
2. A multi-step **``release_canonical`` trajectory**: fixed config, fixed initial
   state, fixed per-step (drive, idx) → expected per-step bias ``e`` and the full
   evolved state (C/BUF/RRP/RES/PR/CL/E + the endocytosis DELAY queue) and EMA.

The golden file (``tests/golden/presyn_canonical_v1.npz``) stores BOTH the inputs
and the expected outputs, so it is self-contained and language-agnostic: the
backend-parity work (jyb) loads the same inputs, runs its kernel, and compares to
the same expected outputs via :func:`load_golden`.

NOTE on ``forward()``: the sequential ``SynapticPresyn.forward`` reference and the
parallel ``release_canonical`` implement the SAME biophysics but with deliberately
different execution models (mean-softplus influx + tau-refill vs. per-edge influx +
DELAY-queue), so they are NOT numerically identical and are intentionally not
cross-asserted here. ``release_canonical`` is the canonical live path and the one
the kernels must match.

Regenerate (only when a change to the dynamics is intended and reviewed):
    BIO_REGEN_GOLDEN=1 uv run --no-sync python -m pytest tests/test_presyn_golden.py -q
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
)

from _bio_testkit import set_seed

pytestmark = pytest.mark.unit

DEV = torch.device("cpu")
DT = torch.float32
GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "presyn_canonical_v1.npz"

# Case dimensions (small but exercises B/H, multi-step recurrence, the DELAY queue).
SEED = 1234
B, H, T, K, DH = 2, 3, 6, 4, 16
NUM_STEPS = 4
# Tight tolerance: CPU float32 elementwise + gather/scatter is deterministic; this
# catches semantic drift while tolerating cross-platform float rounding.
ATOL = 1e-6
RTOL = 1e-5

STATE_KEYS = ("C", "BUF", "RRP", "RES", "PR", "CL", "AMP", "E")
# Hill-formula grid axes.
RP_C = np.linspace(0.0, 4.0, 9, dtype=np.float32)
RP_PR = np.linspace(0.0, 1.0, 5, dtype=np.float32)
RP_CL = np.linspace(0.0, 1.0, 5, dtype=np.float32)
RP_DRIVE = np.linspace(-3.0, 3.0, 7, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Deterministic case construction (used only when regenerating the golden)
# --------------------------------------------------------------------------- #
def _causal_topk_idx() -> torch.Tensor:
    idx = torch.zeros(B, H, T, K, dtype=torch.long)
    for t in range(T):
        idx[:, :, t, :] = torch.randint(0, t + 1, (B, H, K))
    return idx


def _make_inputs() -> tuple[SynapticConfig, dict, list[tuple[torch.Tensor, torch.Tensor]]]:
    set_seed(SEED)
    cfg = SynapticConfig(enable_presyn=True, stochastic_train_frac=0.0)
    init_state = build_presyn_state(B, T, H, DEV, DT, cfg)
    steps = [(torch.randn(B, H, T, K, dtype=DT), _causal_topk_idx()) for _ in range(NUM_STEPS)]
    return cfg, init_state, steps


def _clone_state(state: dict) -> dict:
    out = {k: state[k].clone() for k in STATE_KEYS}
    out["DELAY"] = [d.clone() for d in state["DELAY"]]
    return out


def run_release_canonical(
    cfg: SynapticConfig,
    init_state: dict,
    steps: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[list[torch.Tensor], dict, float]:
    """Run the canonical path over the fixed steps. EMA is reset to 1.0 so the
    golden is independent of the buffer's construction default."""
    pre = SynapticPresyn(DH, cfg)
    pre.ema_e.fill_(1.0)
    state = _clone_state(init_state)
    es = [
        pre.release_canonical(state, drive, idx, train=False, apply_barrier=True)
        for drive, idx in steps
    ]
    return es, state, float(pre.ema_e.item())


def _faithful_prob_grid(cfg: SynapticConfig) -> np.ndarray:
    pre = SynapticPresyn(DH, cfg)
    c, pr, cl, dr = torch.meshgrid(
        torch.from_numpy(RP_C),
        torch.from_numpy(RP_PR),
        torch.from_numpy(RP_CL),
        torch.from_numpy(RP_DRIVE),
        indexing="ij",
    )
    p = pre._faithful_release_prob(c, pr, cl, dr)
    return p.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Golden (de)serialization
# --------------------------------------------------------------------------- #
def _state_into(d: dict, prefix: str, state: dict) -> None:
    for k in STATE_KEYS:
        d[f"{prefix}_{k}"] = state[k].detach().cpu().numpy()
    d[f"{prefix}_DELAY"] = np.stack([t.detach().cpu().numpy() for t in state["DELAY"]])


def _state_from(z, prefix: str) -> dict:
    state = {k: torch.from_numpy(np.asarray(z[f"{prefix}_{k}"])).to(DT) for k in STATE_KEYS}
    delay = np.asarray(z[f"{prefix}_DELAY"])
    state["DELAY"] = [torch.from_numpy(delay[i]).to(DT) for i in range(delay.shape[0])]
    return state


def _regenerate() -> None:
    cfg, init_state, steps = _make_inputs()
    es, final_state, ema = run_release_canonical(cfg, init_state, steps)
    blob: dict = {
        "config_json": np.array(json.dumps(dataclasses.asdict(cfg))),
        "ema_e": np.array(ema, dtype=np.float32),
        "rp_grid": _faithful_prob_grid(cfg),
    }
    _state_into(blob, "init", init_state)
    _state_into(blob, "final", final_state)
    for i, (drive, idx) in enumerate(steps):
        blob[f"drive_{i}"] = drive.detach().cpu().numpy()
        blob[f"idx_{i}"] = idx.detach().cpu().numpy()
        blob[f"e_{i}"] = es[i].detach().cpu().numpy()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GOLDEN_PATH, **blob)


def load_golden() -> dict:
    """Public entry point (reused by the jyb backend-parity harness).

    Returns a dict with: ``cfg`` (SynapticConfig), ``init_state`` (dict of tensors),
    ``steps`` (list of (drive, idx)), ``expected_e`` (list of tensors),
    ``expected_final`` (state dict), ``expected_ema`` (float), and ``rp_grid``.
    """
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"{GOLDEN_PATH} missing; regenerate with BIO_REGEN_GOLDEN=1."
        )
    z = np.load(GOLDEN_PATH, allow_pickle=False)
    cfg = SynapticConfig(**json.loads(str(z["config_json"])))
    steps = [
        (torch.from_numpy(np.asarray(z[f"drive_{i}"])).to(DT),
         torch.from_numpy(np.asarray(z[f"idx_{i}"])).to(torch.long))
        for i in range(NUM_STEPS)
    ]
    expected_e = [torch.from_numpy(np.asarray(z[f"e_{i}"])).to(DT) for i in range(NUM_STEPS)]
    return {
        "cfg": cfg,
        "init_state": _state_from(z, "init"),
        "steps": steps,
        "expected_e": expected_e,
        "expected_final": _state_from(z, "final"),
        "expected_ema": float(np.asarray(z["ema_e"])),
        "rp_grid": np.asarray(z["rp_grid"]),
    }


# Regenerate on import when requested, so a single pytest invocation refreshes it.
if os.environ.get("BIO_REGEN_GOLDEN") == "1":
    _regenerate()


@pytest.fixture(scope="module")
def golden() -> dict:
    return load_golden()


# --------------------------------------------------------------------------- #
# 1. The committed golden reproduces exactly (the contract jyb must satisfy)
# --------------------------------------------------------------------------- #
def test_release_canonical_e_matches_golden(golden: dict) -> None:
    es, _, _ = run_release_canonical(golden["cfg"], golden["init_state"], golden["steps"])
    assert len(es) == len(golden["expected_e"]) == NUM_STEPS
    for i, (got, want) in enumerate(zip(es, golden["expected_e"])):
        assert torch.allclose(got, want, atol=ATOL, rtol=RTOL), (
            f"step {i}: release bias drifted from golden "
            f"(max |Δ|={ (got - want).abs().max().item():.3e}); "
            "regenerate with BIO_REGEN_GOLDEN=1 only if the change is intended."
        )


def test_release_canonical_final_state_matches_golden(golden: dict) -> None:
    _, final, ema = run_release_canonical(golden["cfg"], golden["init_state"], golden["steps"])
    for k in STATE_KEYS:
        assert torch.allclose(final[k], golden["expected_final"][k], atol=ATOL, rtol=RTOL), (
            f"final state[{k}] drifted from golden"
        )
    got_delay = torch.stack(final["DELAY"])
    want_delay = torch.stack(golden["expected_final"]["DELAY"])
    assert torch.allclose(got_delay, want_delay, atol=ATOL, rtol=RTOL), "DELAY queue drifted"
    assert abs(ema - golden["expected_ema"]) < 1e-5, "EMA normalizer drifted from golden"


def test_faithful_release_prob_matches_golden(golden: dict) -> None:
    fresh = _faithful_prob_grid(golden["cfg"])
    assert np.allclose(fresh, golden["rp_grid"], atol=ATOL, rtol=RTOL), (
        "the Hill release-probability formula changed; this is the documented "
        "equation and must not drift silently"
    )


# --------------------------------------------------------------------------- #
# 2. Physical invariants (always true, independent of the stored numbers)
# --------------------------------------------------------------------------- #
def test_state_conservation_invariants(golden: dict) -> None:
    cfg = golden["cfg"]
    _, final, _ = run_release_canonical(cfg, golden["init_state"], golden["steps"])
    assert (final["C"] >= 0).all(), "calcium must stay non-negative"
    assert ((final["BUF"] >= 0) & (final["BUF"] <= 1)).all(), "buffer occupancy in [0,1]"
    assert ((final["PR"] >= 0) & (final["PR"] <= 1)).all(), "SNARE priming in [0,1]"
    assert ((final["CL"] >= 0) & (final["CL"] <= 1)).all(), "complexin clamp in [0,1]"
    assert ((final["RRP"] >= 0) & (final["RRP"] <= 30.0)).all(), "RRP within [0, 30] pool bound"
    assert ((final["E"] >= 0) & (final["E"] <= cfg.energy_max)).all(), "energy in [0, energy_max]"
    for k in STATE_KEYS:
        assert torch.isfinite(final[k]).all(), f"state[{k}] must stay finite"


def test_release_prob_is_a_valid_probability(golden: dict) -> None:
    grid = golden["rp_grid"]
    assert (grid >= 0).all() and (grid <= 1).all(), "release probability must lie in [0,1]"
    # Monotone non-decreasing in calcium (axis 0 of the grid).
    diffs = np.diff(grid, axis=0)
    assert (diffs >= -1e-6).all(), "release probability must be non-decreasing in calcium"


# --------------------------------------------------------------------------- #
# 3. Determinism (the golden is only meaningful if the path is reproducible)
# --------------------------------------------------------------------------- #
def test_canonical_path_is_deterministic(golden: dict) -> None:
    a, _, _ = run_release_canonical(golden["cfg"], golden["init_state"], golden["steps"])
    b, _, _ = run_release_canonical(golden["cfg"], golden["init_state"], golden["steps"])
    for x, y in zip(a, b):
        assert torch.equal(x, y), "deterministic (train=False) path must be bit-reproducible"
