"""
Chunked truncated BPTT for the differentiable synaptic recurrence (bead yw9.2.3).

`chunked_recurrence` runs the differentiable presyn recurrence over a sequence of steps, detaching
the carried state every ``chunk_len`` steps. This truncates backprop to within a chunk (bounding
peak memory to ``chunk_len`` steps instead of the whole sequence) while leaving the forward values
untouched — so the differentiable recurrence (yw9.2) becomes usable on long sequences during
training. Detaching changes only the gradient graph, never the values.

These tests lock: gradient flows WITHIN a chunk and is cut at chunk boundaries; full BPTT
(``chunk_len=0``) flows across all steps; and the forward values are identical for any chunk length.

Run:  pytest tests/test_chunked_tbptt.py -v
"""

from __future__ import annotations

import pytest
import torch

from bio_inspired_nanochat.synaptic import (
    SynapticConfig,
    SynapticPresyn,
    build_presyn_state,
    chunked_recurrence,
)

B, H, T_KEY, K, T, N = 1, 2, 6, 3, 4, 6


def _setup():
    cfg = SynapticConfig(enable_presyn=True)
    presyn = SynapticPresyn(d_head=8, cfg=cfg)
    g = torch.Generator().manual_seed(2)
    drives = [
        (torch.randn(B, H, T, K, generator=g, dtype=torch.float64) * 0.4 + 0.5)
        for _ in range(N)
    ]
    idxs = [torch.randint(0, T_KEY, (B, H, T, K), generator=g) for _ in range(N)]
    return presyn, cfg, drives, idxs


def _run(presyn, cfg, drives, idxs, chunk_len):
    st = build_presyn_state(B, T_KEY, H, "cpu", torch.float64, cfg)
    presyn.ema_e.fill_(1.0)
    return chunked_recurrence(presyn, st, drives, idxs, chunk_len=chunk_len)


def _grad(outs, drives, j, i):
    g, = torch.autograd.grad(outs[j].sum(), drives[i], retain_graph=True, allow_unused=True)
    return 0.0 if g is None else g.abs().sum().item()


# --------------------------------------------------------------------------- #
# 1. Gradient is TRUNCATED at chunk boundaries, flows WITHIN a chunk
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_gradient_truncated_across_chunk_boundary():
    presyn, cfg, drives, idxs = _setup()
    for d in drives:
        d.requires_grad_(True)
    outs = _run(presyn, cfg, drives, idxs, chunk_len=3)  # chunks {0,1,2}, {3,4,5}

    assert _grad(outs, drives, 2, 0) > 0, "drive0 -> out2 (same chunk) must carry gradient"
    assert _grad(outs, drives, 5, 3) > 0, "drive3 -> out5 (same chunk) must carry gradient"
    assert _grad(outs, drives, 3, 3) > 0, "same-step dependence is always differentiable"
    assert _grad(outs, drives, 3, 0) == 0.0, "drive0 -> out3 must be CUT by the chunk-3 detach"
    assert _grad(outs, drives, 5, 2) == 0.0, "drive2 -> out5 must be cut across the boundary"


@pytest.mark.unit
def test_configurable_chunk_length_moves_the_boundary():
    presyn, cfg, drives, idxs = _setup()
    for d in drives:
        d.requires_grad_(True)
    outs = _run(presyn, cfg, drives, idxs, chunk_len=2)  # chunks {0,1},{2,3},{4,5}

    assert _grad(outs, drives, 1, 0) > 0, "within chunk {0,1}"
    assert _grad(outs, drives, 2, 1) == 0.0, "cut at the t=2 boundary"
    assert _grad(outs, drives, 3, 2) > 0, "within chunk {2,3}"


# --------------------------------------------------------------------------- #
# 2. Full BPTT (chunk_len=0) flows across the whole sequence
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_chunk_len_zero_is_full_bptt():
    presyn, cfg, drives, idxs = _setup()
    for d in drives:
        d.requires_grad_(True)
    outs = _run(presyn, cfg, drives, idxs, chunk_len=0)
    assert _grad(outs, drives, 5, 0) > 0, "with no truncation, drive0 must reach out5"


# --------------------------------------------------------------------------- #
# 3. Forward values are identical regardless of chunk length (detach ⇒ grad-only)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_forward_values_independent_of_chunk_length():
    presyn, cfg, drives, idxs = _setup()
    drives = [d.detach() for d in drives]
    with torch.no_grad():
        full = torch.stack(_run(presyn, cfg, drives, idxs, chunk_len=0))
        chunk3 = torch.stack(_run(presyn, cfg, drives, idxs, chunk_len=3))
        chunk1 = torch.stack(_run(presyn, cfg, drives, idxs, chunk_len=1))
    assert torch.equal(full, chunk3) and torch.equal(full, chunk1), "chunking must not change values"
