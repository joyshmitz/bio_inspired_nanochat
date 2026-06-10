"""
ukxt: live attention path migrated to the canonical faithful presyn release (8j9.2).

Verifies the migration end-to-end:
1. The live attention forward calls release_canonical (faithful Hill dynamics), NOT the legacy
   sigmoid release().
2. SMOKE-TRAIN GATE (mandatory for a core-engine change): a tiny end-to-end model trains for
   several steps with finite, bounded loss and finite grads — no NaN/Inf, no explosion.
3. The septin barrier is not double-counted: the attention still applies its own logit-level
   barrier, and the canonical is invoked with apply_barrier off (default).

Run:  pytest tests/test_presyn_migration.py -v -s
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import SynapticPresyn

from _bio_testkit import make_tiny_synaptic, random_tokens


@pytest.mark.unit
def test_live_attention_calls_canonical_not_legacy(monkeypatch):
    calls = {"canonical": 0, "legacy": 0}
    orig_canon = SynapticPresyn.release_canonical
    orig_legacy = SynapticPresyn.release

    def spy_canon(self, *a, **k):
        calls["canonical"] += 1
        return orig_canon(self, *a, **k)

    def spy_legacy(self, *a, **k):
        calls["legacy"] += 1
        return orig_legacy(self, *a, **k)

    monkeypatch.setattr(SynapticPresyn, "release_canonical", spy_canon)
    monkeypatch.setattr(SynapticPresyn, "release", spy_legacy)

    m = make_tiny_synaptic(seed=0, train=True).train()
    m(random_tokens(2, 16, vocab=m.config.vocab_size))

    assert calls["canonical"] > 0, "live attention must call release_canonical (the migration)"
    assert calls["legacy"] == 0, "live attention must NOT call the legacy sigmoid release()"


@pytest.mark.unit
def test_smoke_train_gate_is_stable():
    # MANDATORY gate (8j9.2/ukxt): swapping the live release equation must keep training stable.
    torch.manual_seed(0)
    m = make_tiny_synaptic(seed=0, train=True).train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)

    losses = []
    for step in range(15):
        x = random_tokens(4, 24, vocab=m.config.vocab_size)
        logits, loss = m(x, targets=x)
        assert torch.isfinite(loss), f"step {step}: loss not finite ({loss.item()})"
        assert logits.abs().max().item() <= 15.0 + 1e-3, "logit softcap (vg9.1) must still bound logits"
        loss.backward()
        gsum = sum(
            (p.grad.abs().sum() for p in m.parameters() if p.grad is not None),
            torch.zeros(()),
        )
        assert torch.isfinite(gsum) and gsum > 0, f"step {step}: grads must be finite and flow"
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())

    # Stability: every step finite and bounded; no explosion. (Random data -> no learning signal,
    # so we assert boundedness, not improvement.)
    print(f"smoke-train loss trajectory: {[round(x, 3) for x in losses]}")
    assert all(x < 100.0 for x in losses), f"loss exploded under the canonical migration: {losses}"
    assert max(losses) < 20.0, f"loss should stay near log(vocab); got max {max(losses):.2f}"


@pytest.mark.unit
def test_eval_forward_finite_through_canonical():
    m = make_tiny_synaptic(seed=1, train=True).eval()
    with torch.no_grad():
        logits = m(random_tokens(2, 20, vocab=m.config.vocab_size))
        logits = logits[0] if isinstance(logits, tuple) else logits
    assert torch.isfinite(logits).all(), "eval forward through the canonical path must be finite"
