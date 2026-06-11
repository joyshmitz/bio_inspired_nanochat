"""CMA-ES proxy objective fixes — bead 74f.1.

The Phase-1 proxy was a NO-GO: mean last-10 TRAIN loss, single fixed seed, no held-out
split — signal ~10-15x below seed noise, and (root cause found here) synaptic candidates
diverged to NaN -> PENALTY_LOSS because the eval loop never reset the per-sequence
plasticity state. These tests lock the fixes: per-batch reset keeps synaptic training
finite, a fixed held-out set is the objective, and a readiness gate measures whether the
proxy separates a known-good from a known-bad config above seed noise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynapticConfig
from bio_inspired_nanochat.synaptic import SynapticConfig
from scripts.tune_bio_params import (
    TOP10_PARAM_SPECS,
    CandidateEvalResult,
    encode_params,
    evaluate_candidate_detailed,
    generate_batch,
    multi_seed_objective,
    readiness_from_objectives,
)

pytestmark = pytest.mark.unit

SPECS = TOP10_PARAM_SPECS
DEFAULT_VEC = encode_params(SynapticConfig(), SPECS)
# Tiny, even-seq config so the copy task + a couple of steps run fast on CPU.
TINY = GPTSynapticConfig(
    sequence_len=16, vocab_size=64, n_layer=1, n_head=2, n_kv_head=2,
    n_embd=16, synapses=True, use_moe=False,
)


def test_generate_batch_seed_is_reproducible_and_held_out_differs():
    a, _ = generate_batch(4, 16, 64, "cpu", seed=999)
    b, _ = generate_batch(4, 16, 64, "cpu", seed=999)
    c, _ = generate_batch(4, 16, 64, "cpu", seed=1000)
    assert torch.equal(a, b), "a fixed seed must give a reproducible held-out batch"
    assert not torch.equal(a, c), "different seeds must give different batches"


def test_objective_prefers_finite_held_out_loss():
    assert CandidateEvalResult(2.0, 10, held_out_loss=1.0).objective == 1.0
    assert CandidateEvalResult(2.0, 10, held_out_loss=None).objective == 2.0
    assert CandidateEvalResult(2.0, 10, held_out_loss=float("nan")).objective == 2.0


def test_held_out_eval_is_finite_with_per_batch_reset():
    # The default synaptic config diverges to NaN over many batches WITHOUT a per-sequence
    # reset; with reset_state=True (the fix) it stays finite and yields a held-out loss.
    res = evaluate_candidate_detailed(
        DEFAULT_VEC, specs=SPECS, seed=1, steps=6, batch_size=8, device="cpu",
        lr=2e-3, weight_decay=0.0, timeout_seconds=None, max_retries=0,
        raise_on_error=True, model_config=TINY, held_out_batches=3, reset_state=True,
    )
    assert res.held_out_loss is not None and np.isfinite(res.held_out_loss)
    assert res.objective == res.held_out_loss


def test_multi_seed_objective_structure():
    out = multi_seed_objective(
        DEFAULT_VEC, specs=SPECS, seeds=(1, 2), steps=4, batch_size=8,
        device="cpu", model_config=TINY, held_out_batches=2,
    )
    assert set(out["per_seed"]) == {1, 2}
    assert out["n"] == 2 and np.isfinite(out["mean"]) and out["std"] >= 0.0


def test_readiness_gate_passes_on_clear_signal():
    good = {1: 0.10, 2: 0.11, 3: 0.09}
    bad = {1: 1.00, 2: 1.05, 3: 0.95}
    gate = readiness_from_objectives(good, bad, sigma_gate=3.0, rel_gate=0.005)
    assert gate["ready"] is True
    assert gate["sigma_separation"] > 3.0
    assert gate["relative_improvement"] > 0.005
    assert gate["paired_t_p"] < 0.05


def test_readiness_gate_rejects_noise_dominated_signal():
    # Models the bead's failure: signal ~1.5e-3 buried under seed-noise ~2e-2.
    good = {1: 1.000, 2: 1.020, 3: 0.980}
    bad = {1: 1.0015, 2: 1.0215, 3: 0.9815}
    gate = readiness_from_objectives(good, bad, sigma_gate=3.0, rel_gate=0.005)
    assert gate["ready"] is False
    assert gate["sigma_separation"] < 3.0
