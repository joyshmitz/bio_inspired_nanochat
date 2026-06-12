"""Memory-budget estimator + throughput micro-benchmark for the scale-up (bead hwxb.2.2).

Predicts the VRAM footprint of a Phase-0 config BEFORE committing a multi-hour 2×4090 run,
so we know the headroom left for the synaptic per-key state once Phase 3 turns mechanisms ON.

What is EXACT (CPU-computable, unit-tested in tests/test_scaleup_memory.py):
  - param_bytes        : Σ numel·element_size over model.parameters()
  - buffer_bytes       : Σ over persistent registered buffers (eligibility traces, EMAs, …)
  - optimizer_bytes    : the moment state — AdamW keeps 2 (exp_avg, exp_avg_sq), Muon keeps 1
                         (momentum_buffer), each zeros_like(param). ZeRO-style Dist* optimizers
                         shard moment state, so per-rank ≈ total / world_size. (Excludes the
                         negligible per-param int64 `step` scalar.)

What is a ROUGH ESTIMATE (documented as such; depends on autocast / activation checkpointing /
the exact kernels, which only a real run on the 4090 pins down — that measured table is hwxb.2.2's
GPU-gated residual):
  - activation_bytes_est       : transformer activations ≈ B·T·d·L·bytes·k
  - synaptic_state_bytes_est   : per-key presyn buffers ≈ B·H·T·n_presyn·L·bytes (synaptic only)

Run it: `python -m scripts.scale_memory --depth 10 --tie-embeddings --batch 16 --seq 1024 --world-size 2`
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from bio_inspired_nanochat.torch_imports import torch
from bio_inspired_nanochat.muon import DistMuon, Muon

# Rough activation multiplier: residual stream + attention scores + MLP intermediates, in bf16.
# Deliberately conservative; real autocast/checkpointing shifts this. Tune against a measured run.
_ACT_MULT = 16
# Number of per-key presynaptic state buffers carried per layer (C/BUF/RRP/RES/PR/CL/E ≈ 7).
_N_PRESYN_BUFFERS = 7


@dataclass
class MemoryBudget:
    param_bytes: int
    buffer_bytes: int
    optimizer_bytes: int          # per-rank moment state
    activation_bytes_est: int
    synaptic_state_bytes_est: int
    world_size: int
    batch: int
    seq: int

    @property
    def persistent_bytes(self) -> int:
        """Resident model + optimizer state on one rank (params + buffers + moments)."""
        return self.param_bytes + self.buffer_bytes + self.optimizer_bytes

    @property
    def total_est_bytes(self) -> int:
        return self.persistent_bytes + self.activation_bytes_est + self.synaptic_state_bytes_est

    def as_gb(self) -> dict[str, float]:
        gb = 1024**3
        return {
            "param_gb": self.param_bytes / gb,
            "buffer_gb": self.buffer_bytes / gb,
            "optimizer_gb": self.optimizer_bytes / gb,
            "activation_est_gb": self.activation_bytes_est / gb,
            "synaptic_state_est_gb": self.synaptic_state_bytes_est / gb,
            "persistent_gb": self.persistent_bytes / gb,
            "total_est_gb": self.total_est_bytes / gb,
        }

    def headroom_gb(self, vram_gb: float) -> float:
        return vram_gb - self.total_est_bytes / 1024**3


# --------------------------------------------------------------------------- #
# Exact (CPU-computable) terms
# --------------------------------------------------------------------------- #
def param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def buffer_bytes(model: torch.nn.Module) -> int:
    """Persistent registered buffers only (RoPE cos/sin are persistent=False, so excluded)."""
    return sum(b.numel() * b.element_size() for b in model.buffers())


def optimizer_moment_bytes(model: torch.nn.Module, *, world_size: int = 1) -> int:
    """Per-rank optimizer moment-state bytes from the model's real param grouping.

    AdamW → 2 moments/param; Muon → 1/param; each the same dtype/shape as the param.
    Dist* optimizers shard moment state ZeRO-style, so we divide by world_size (the
    tiny non-shardable 0-D/odd params are not sharded, but they are negligible).
    """
    total = 0
    for opt in model.setup_optimizers():
        per_param = 1 if isinstance(opt, (Muon, DistMuon)) else 2
        for group in opt.param_groups:
            for p in group["params"]:
                total += per_param * p.numel() * p.element_size()
    return total // max(1, world_size)


# --------------------------------------------------------------------------- #
# Rough estimates
# --------------------------------------------------------------------------- #
def activation_bytes_est(config, *, batch: int, seq: int, bytes_per: int = 2) -> int:
    d = int(config.n_embd)
    layers = int(config.n_layer)
    return _ACT_MULT * batch * seq * d * layers * bytes_per


def synaptic_state_bytes_est(config, *, batch: int, seq: int, bytes_per: int = 4) -> int:
    """Per-key presyn buffers ≈ B·H·T·n_buffers·L. Zero for the vanilla model."""
    if not getattr(config, "synapses", False):
        return 0
    heads = int(config.n_head)
    layers = int(config.n_layer)
    return _N_PRESYN_BUFFERS * batch * heads * seq * layers * bytes_per


def estimate(model: torch.nn.Module, config, *, batch: int, seq: int, world_size: int = 1) -> MemoryBudget:
    return MemoryBudget(
        param_bytes=param_bytes(model),
        buffer_bytes=buffer_bytes(model),
        optimizer_bytes=optimizer_moment_bytes(model, world_size=world_size),
        activation_bytes_est=activation_bytes_est(config, batch=batch, seq=seq),
        synaptic_state_bytes_est=synaptic_state_bytes_est(config, batch=batch, seq=seq),
        world_size=world_size,
        batch=batch,
        seq=seq,
    )


# --------------------------------------------------------------------------- #
# Throughput micro-benchmark (runs on whatever device; ready for the 4090)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _noop():  # pragma: no cover
    pass


def measure_throughput(model, config, *, batch: int, seq: int, steps: int = 20, warmup: int = 5,
                       device: str = "cpu") -> dict:
    """Measure tok/s + step time over a few real forward/backward/step iterations.

    Device-agnostic: meaningful on a GPU, runnable (if slow) on CPU. Imports time lazily and
    avoids Date/Math.random restrictions by reading a monotonic clock per call.
    """
    import time

    synaptic = getattr(config, "synapses", False)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    vocab = int(config.vocab_size)
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randint(0, vocab, (batch, seq), generator=gen).to(device)
    y = torch.randint(0, vocab, (batch, seq), generator=gen).to(device)

    def _step():
        if synaptic:
            _, loss = model(x, y, None, train_mode=True)
        else:
            loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        _step()
    t0 = time.monotonic()
    for _ in range(steps):
        _step()
    dt = time.monotonic() - t0
    tok = batch * seq * steps
    return {"tok_per_sec": tok / dt if dt > 0 else float("inf"), "step_ms": 1000.0 * dt / steps, "steps": steps}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_model(depth: int, seq: int, synapses: bool, tie: bool):
    model_dim = depth * 64
    n_head = max(1, (model_dim + 127) // 128)
    common = dict(sequence_len=seq, vocab_size=65536, n_layer=depth, n_head=n_head,
                  n_kv_head=n_head, n_embd=model_dim, tie_embeddings=tie)
    if synapses:
        from bio_inspired_nanochat.gpt_synaptic import GPTSynaptic, GPTSynapticConfig

        cfg = GPTSynapticConfig(synapses=True, **common)  # ty: ignore[invalid-argument-type]
        model = GPTSynaptic(cfg)
    else:
        from bio_inspired_nanochat.gpt import GPT, GPTConfig

        cfg = GPTConfig(**common)  # ty: ignore[invalid-argument-type]
        model = GPT(cfg)
    model.init_weights()
    return model, cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Scale-up memory-budget estimate + throughput")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--synapses", action="store_true")
    p.add_argument("--tie-embeddings", action="store_true")
    p.add_argument("--vram-gb", type=float, default=24.0)
    p.add_argument("--throughput", action="store_true", help="also run the throughput micro-bench")
    args = p.parse_args(argv)

    model, cfg = _build_model(args.depth, args.seq, args.synapses, args.tie_embeddings)
    b = estimate(model, cfg, batch=args.batch, seq=args.seq, world_size=args.world_size)
    gb = b.as_gb()
    print(f"=== memory budget (depth={args.depth} synapses={args.synapses} tie={args.tie_embeddings} "
          f"batch={args.batch} seq={args.seq} world_size={args.world_size}) ===")
    for k, v in gb.items():
        print(f"  {k:24s} {v:8.3f} GB")
    print(f"  {'headroom_vs_' + str(args.vram_gb) + 'GB':24s} {b.headroom_gb(args.vram_gb):8.3f} GB")
    if args.throughput:
        tp = measure_throughput(model, cfg, batch=args.batch, seq=args.seq)
        print(f"  throughput: {tp['tok_per_sec']:.1f} tok/s  ({tp['step_ms']:.1f} ms/step)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
