"""
Unit tests for the structured run-logging infrastructure (bead eqyk.2).

Verifies the JSONL event stream is correct & parseable, tensor summaries never
raise on pathological input, per-step bio-state logging works, run_id correlation
holds, provenance is stamped, and the context manager records start/end/errors.

Run:  pytest tests/test_run_logging.py -v
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from bio_inspired_nanochat.run_logging import (
    RunLogger,
    gather_provenance,
    tensor_summary,
    _jsonable,
)


@pytest.mark.unit
def test_event_written_and_parseable(tmp_run_dir):
    rl = RunLogger(tmp_run_dir, name="t", console=False)
    rec = rl.event("hello", step=3, value=42, note="hi")
    rl.close()
    assert rec["event"] == "hello" and rec["step"] == 3 and rec["value"] == 42
    # The JSONL file must be valid JSON, one object per line.
    lines = (tmp_run_dir / "events.jsonl").read_text().splitlines()
    parsed = [json.loads(line) for line in lines]
    events = [p["event"] for p in parsed]
    assert events[0] == "run_start" and "hello" in events and events[-1] == "run_end"
    for p in parsed:
        assert {"ts", "run_id", "event", "level"} <= set(p)


@pytest.mark.unit
def test_run_id_correlates_all_events(tmp_run_dir):
    rl = RunLogger(tmp_run_dir, name="t", console=False)
    rl.event("a")
    rl.event("b")
    rl.close()
    ids = {e["run_id"] for e in rl.read_events()}
    assert len(ids) == 1 and rl.run_id in ids


@pytest.mark.unit
def test_tensor_summary_handles_pathological_input():
    s = tensor_summary(torch.tensor([float("nan"), float("inf"), 1.0, -2.0]))
    assert s["nan"] == 1 and s["inf"] == 1
    assert math.isfinite(s["mean"]) and s["min"] == -2.0 and s["max"] == 1.0
    # empty
    se = tensor_summary(torch.empty(0))
    assert se["numel"] == 0 and se["mean"] is None
    # all non-finite -> stats None, counts set
    sa = tensor_summary(torch.tensor([float("nan"), float("inf")]))
    assert sa["mean"] is None and sa["nan"] == 1 and sa["inf"] == 1
    # summary is JSON-serializable
    json.dumps(s)


@pytest.mark.unit
def test_log_bio_state_summarizes_named_tensors(tmp_run_dir):
    rl = RunLogger(tmp_run_dir, name="bio", console=False)
    rl.log_bio_state(step=5, calcium=torch.randn(2, 4), rrp=torch.ones(3))
    rl.close()
    bio = [e for e in rl.read_events() if e["event"] == "bio_state"][0]
    assert bio["step"] == 5
    assert set(bio["tensors"]) == {"calcium", "rrp"}
    assert bio["tensors"]["rrp"]["mean"] == 1.0
    assert bio["tensors"]["calcium"]["shape"] == [2, 4]


@pytest.mark.unit
def test_log_metrics_and_tensor(tmp_run_dir):
    rl = RunLogger(tmp_run_dir, name="m", console=False)
    rl.log_metrics(step=1, loss=2.5, lr=3e-4)
    rl.log_tensor("logits", torch.zeros(2, 3), step=1)
    rl.close()
    evs = {e["event"]: e for e in rl.read_events()}
    assert evs["metrics"]["loss"] == 2.5 and evs["metrics"]["lr"] == pytest.approx(3e-4)
    assert evs["tensor"]["summary"]["shape"] == [2, 3]


@pytest.mark.unit
def test_jsonable_coerces_nan_inf_and_tensors():
    assert _jsonable(float("nan")) == "nan"
    assert _jsonable(float("inf")) == "inf"
    assert _jsonable(1.5) == 1.5
    assert _jsonable(torch.tensor(3.0)) == 3.0  # scalar tensor -> number
    big = _jsonable(torch.ones(10))  # large tensor -> summary dict
    assert isinstance(big, dict) and big["mean"] == 1.0
    nested = _jsonable({"a": torch.tensor(2.0), "b": [torch.tensor(1.0), float("nan")]})
    assert nested["a"] == 2.0 and nested["b"][1] == "nan"
    json.dumps(_jsonable({"x": torch.randn(4), "y": float("inf")}))


@pytest.mark.unit
def test_provenance_is_stamped(tmp_run_dir):
    rl = RunLogger(tmp_run_dir, name="p", console=False, provenance={"config_hash": "abc123", "seed": 7})
    rl.close()
    start = [e for e in rl.read_events() if e["event"] == "run_start"][0]
    prov = start["provenance"]
    assert {"torch", "cuda", "host", "pid"} <= set(prov)
    assert prov["config_hash"] == "abc123" and prov["seed"] == 7
    assert isinstance(prov["cuda"], bool)


@pytest.mark.unit
def test_context_manager_logs_start_and_end(tmp_run_dir):
    with RunLogger(tmp_run_dir, name="cm", console=False) as rl:
        rl.event("work")
        path = rl.events_path
    events = [json.loads(line)["event"] for line in path.read_text().splitlines()]
    assert events[0] == "run_start" and events[-1] == "run_end" and "work" in events


@pytest.mark.unit
def test_context_manager_records_error(tmp_run_dir):
    path = tmp_run_dir / "events.jsonl"
    with pytest.raises(ValueError):
        with RunLogger(tmp_run_dir, name="err", console=False) as rl:
            rl.event("before")
            raise ValueError("boom")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert "run_error" in kinds
    err = [e for e in events if e["event"] == "run_error"][0]
    assert err["error_type"] == "ValueError" and "boom" in err["error"]
    assert kinds[-1] == "run_end"  # still closes cleanly


@pytest.mark.unit
def test_console_mirror_does_not_break(tmp_run_dir, caplog):
    # console=True must not raise and should still write JSONL.
    with caplog.at_level("INFO"):
        rl = RunLogger(tmp_run_dir, name="con", console=True)
        rl.log_metrics(step=0, loss=1.0)
        rl.close()
    assert len(rl.read_events()) >= 3  # run_start + metrics + run_end


@pytest.mark.unit
def test_gather_provenance_never_raises():
    prov = gather_provenance({"extra": torch.tensor(1.0)})
    assert "torch" in prov and prov["extra"] == 1.0
    json.dumps(prov)  # serializable
