"""
Logit softcap parity for GPTSynaptic (bead vg9.1).

The vanilla GPT head bounds logits via ``softcap*tanh(logits/softcap)`` (softcap=15);
GPTSynaptic previously did not, leaving logits unbounded — a stability regression made
worse by the synaptic attention's unbounded ``log(ε+release)`` bias. These tests pin
the parity behavior: logits are bounded on both the inference and the loss paths, and
the cap is cleanly toggleable (``logit_softcap=0`` disables it for ablation).

Run:  pytest tests/test_logit_softcap.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.gpt_synaptic import GPTSynapticConfig

from _bio_testkit import assert_finite, make_tiny_synaptic, random_tokens

SOFTCAP = 15.0


def _blow_up_head(model, factor: float = 1000.0):
    """Scale the lm_head so RAW (pre-cap) logits are enormous — this is what makes
    the softcap observable (otherwise small models produce already-small logits)."""
    with torch.no_grad():
        model.lm_head.weight.mul_(factor)


@pytest.mark.unit
def test_default_config_has_softcap_parity():
    cfg = GPTSynapticConfig(n_layer=1, n_embd=32, n_head=2, n_kv_head=2, vocab_size=64, sequence_len=16)
    assert cfg.logit_softcap == SOFTCAP, "default must match the vanilla GPT softcap (15)"


@pytest.mark.unit
def test_inference_logits_bounded_by_softcap():
    m = make_tiny_synaptic(seed=0)  # default softcap=15
    _blow_up_head(m)
    logits, _ = m(random_tokens(2, 16))
    amax = logits.abs().max().item()
    assert amax <= SOFTCAP + 1e-4, f"logits must be bounded by softcap, got |max|={amax}"
    assert amax > SOFTCAP - 1.0, "with a blown-up head the cap should be near-saturated"
    assert_finite(logits, "softcapped logits")


@pytest.mark.unit
def test_loss_path_logits_bounded_and_loss_finite():
    m = make_tiny_synaptic(seed=0, train=True)
    _blow_up_head(m)
    x, y = random_tokens(2, 16), random_tokens(2, 16)
    logits, loss = m(x, targets=y)
    assert logits.abs().max().item() <= SOFTCAP + 1e-4
    assert torch.isfinite(loss), "loss must be finite with softcap applied"


@pytest.mark.unit
def test_softcap_zero_disables_capping():
    m = make_tiny_synaptic(seed=0, logit_softcap=0.0)
    _blow_up_head(m)
    logits, _ = m(random_tokens(2, 16))
    assert logits.abs().max().item() > SOFTCAP, "softcap=0 must leave logits uncapped"


@pytest.mark.unit
def test_softcap_formula_is_bounded_monotone_and_near_identity_at_zero():
    # The mathematical property the model relies on: softcap*tanh(z/softcap) is a
    # smooth, strictly-monotone squash, bounded in (-softcap, softcap), and ~identity
    # for small z (so it doesn't distort already-reasonable logits).
    z = torch.linspace(-1000.0, 1000.0, 401)
    capped = SOFTCAP * torch.tanh(z / SOFTCAP)
    # Asymptotes to ±softcap (reaches it exactly in float32 at large |z|).
    assert capped.abs().max().item() <= SOFTCAP                     # bounded
    assert torch.all(capped.diff() >= 0)                           # monotone (flat at saturation)
    assert capped.diff()[len(z) // 2] > 0                           # strictly increasing through 0
    small = torch.linspace(-0.5, 0.5, 11)
    assert torch.allclose(SOFTCAP * torch.tanh(small / SOFTCAP), small, atol=2e-3)  # ~identity near 0
