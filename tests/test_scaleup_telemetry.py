"""Unit tests for the unified TrainingTelemetry layer (bead hwxb.7.1 / hwxb.7.2).

Fast, CPU-only. Verify the logger emits the canonical schema, mirrors scalars to
TensorBoard, is a strict no-op on non-master ranks (DDP rank-gating), and that the
JSONL stream round-trips through the ablation reader.
"""
from __future__ import annotations

import json

from bio_inspired_nanochat.torch_imports import torch
from bio_inspired_nanochat.run_logging import (
    EVENT_BIO,
    EVENT_EVAL,
    EVENT_TRAIN_STEP,
    TrainingTelemetry,
    read_run_events,
)


def _events(run_dir) -> list[dict]:
    path = run_dir / "events.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_train_step_emits_canonical_schema(tmp_path):
    with TrainingTelemetry(tmp_path, is_master=True, tensorboard=False, console=False) as tel:
        tel.log_train_step(
            10, loss=2.31, lr=3e-4, grad_norm=1.2, tok_per_sec=12345,
            step_ms=42.0, mfu=18.5, vram_gb=7.1, divergence_action="none",
        )
    steps = read_run_events(tmp_path, event=EVENT_TRAIN_STEP)
    assert len(steps) == 1
    rec = steps[0]
    assert rec["step"] == 10
    for key in ("loss", "lr", "grad_norm", "tok_per_sec", "step_ms", "mfu", "vram_gb", "divergence_action"):
        assert key in rec, f"missing canonical train_step field {key!r}"
    assert rec["loss"] == 2.31
    assert rec["divergence_action"] == "none"


def test_eval_and_bio_events(tmp_path):
    with TrainingTelemetry(tmp_path, is_master=True, tensorboard=False, console=False) as tel:
        tel.log_eval(250, val_bpb=1.05, core_metric=0.42, niah_acc=0.6)
        # bio telemetry: scalars + a per-layer kinetics dict + a tensor (-> summary)
        tel.log_bio(
            250,
            spectral_radius=0.87,
            fast_weight_mag=0.013,
            kinetics={"layer0": {"tau_c": 4.0}, "layer1": {"tau_c": 4.2}},
            neuromod={"nm/da": 0.1, "nm/ach": 0.3, "nm/ne": 0.0},
            calcium=torch.randn(3, 4),
        )
    evals = read_run_events(tmp_path, event=EVENT_EVAL)
    bios = read_run_events(tmp_path, event=EVENT_BIO)
    assert len(evals) == 1 and evals[0]["val_bpb"] == 1.05 and evals[0]["niah_acc"] == 0.6
    assert len(bios) == 1
    bio = bios[0]
    assert bio["spectral_radius"] == 0.87
    assert bio["neuromod"]["nm/da"] == 0.1
    assert bio["kinetics"]["layer1"]["tau_c"] == 4.2
    # a (3,4) tensor must be stored as a compact summary, not raw values
    assert bio["calcium"]["shape"] == [3, 4]
    assert "mean" in bio["calcium"]


def test_non_master_rank_is_strict_noop(tmp_path):
    tel = TrainingTelemetry(tmp_path, is_master=False, tensorboard=True)
    # Every method must be safe and produce nothing on non-master ranks.
    tel.log_train_step(1, loss=1.0, lr=1e-3)
    tel.log_eval(1, val_bpb=1.0)
    tel.log_bio(1, spectral_radius=0.5)
    tel.event("custom", foo=1)
    assert tel.should_log_heavy(0) is False
    tel.close()
    assert not (tmp_path / "events.jsonl").exists(), "non-master rank must not write events.jsonl"
    assert not (tmp_path / "tb").exists(), "non-master rank must not create a TB writer"


def test_heavy_cadence_gating(tmp_path):
    tel = TrainingTelemetry(tmp_path, is_master=True, tensorboard=False, heavy_every=100, console=False)
    assert tel.should_log_heavy(0) is True
    assert tel.should_log_heavy(100) is True
    assert tel.should_log_heavy(150) is False
    tel.close()
    # heavy_every=0 disables the helper
    tel2 = TrainingTelemetry(tmp_path / "b", is_master=True, tensorboard=False, heavy_every=0, console=False)
    assert tel2.should_log_heavy(0) is False
    tel2.close()


def test_tensorboard_scalars_written(tmp_path):
    pytest_tb = TrainingTelemetry(tmp_path, is_master=True, tensorboard=True, console=False)
    pytest_tb.log_train_step(5, loss=2.0, lr=1e-3, grad_norm=0.9)
    pytest_tb.log_bio(5, spectral_radius=0.8, neuromod={"nm/da": 0.2})
    pytest_tb.close()
    tb_dir = tmp_path / "tb"
    assert tb_dir.exists(), "TB log dir should exist on master with tensorboard=True"
    # at least one event file with non-trivial content
    files = list(tb_dir.glob("events.out.tfevents.*"))
    assert files, "no TensorBoard event file written"
    assert any(f.stat().st_size > 0 for f in files)
