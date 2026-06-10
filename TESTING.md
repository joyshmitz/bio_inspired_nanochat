# Testing Guide

> The test discipline for `bio_inspired_nanochat`. Established by bead **eqyk.1**
> (*Unit-test framework, conventions, fixtures & coverage gate*) — the foundation
> every later "unit tests + e2e + detailed logging" bead builds on.

The goal: **green = working.** Tests are deterministic, fast on CPU, portable to
GPU CI, and emit enough diagnostics that a failure tells you *why*.

---

## TL;DR — running tests

```bash
# One-time: install deps WITHOUT building the optional Rust extension.
# (The pure-Python package imports fine without it; rustbpe tests auto-skip.)
uv sync --no-install-project --extra cpu        # or --extra gpu on a CUDA box

# Fast unit gate (what CI runs on every PR):
.venv/bin/python -m pytest -m "unit" -q

# Everything except slow tests:
.venv/bin/python -m pytest -m "not slow" -q

# With coverage gate (CI / nightly):
.venv/bin/python -m pytest -m "not slow" --cov

# A single file, verbose:
.venv/bin/python -m pytest tests/test_framework_smoke.py -v
```

`pyproject.toml` sets `pythonpath = ["."]`, so `import bio_inspired_nanochat …`
resolves **without** a full editable/`maturin` install. You do not need to build
`rustbpe` to run the suite.

> **Why `--no-install-project`?** The project uses `maturin` as its build backend
> for the `rustbpe` Rust extension. A plain `uv sync`/`uv run` tries to build it
> and currently fails on a `python-source`/module-name config quirk (tracked
> separately — see bead `jyb.8` *Build and ship the Rust extension in CI*). Until
> that's fixed, install deps with `--no-install-project` and run via
> `.venv/bin/python -m pytest`. To get the Rust paths, run `uv run maturin develop`.

---

## Markers (test taxonomy)

Registered in `pyproject.toml`; `--strict-markers` makes a typo'd marker an error.

| Marker     | Meaning                                                        | Where it runs       |
|------------|----------------------------------------------------------------|---------------------|
| `unit`     | Fast, deterministic, no GPU/network/data downloads             | **PR gate**         |
| `e2e`      | End-to-end run exercising a whole flow                         | Nightly             |
| `slow`     | Long-running (`-m "not slow"` to skip)                         | Nightly             |
| `gpu`      | Requires CUDA; **auto-skipped** on CPU-only hosts              | GPU CI              |
| `golden`   | Compares against committed golden artifacts (`tests/golden/`)  | PR gate + nightly   |

`gpu`-marked tests are skipped automatically when `torch.cuda.is_available()` is
False (see `tests/conftest.py::pytest_collection_modifyitems`), so the same suite
is green on a laptop and on GPU CI.

---

## The test kit — `tests/_bio_testkit.py`

Dependency-light helpers (torch + stdlib only). Import directly, or use the
`conftest.py` fixtures that wrap them.

| Helper | Purpose |
|--------|---------|
| `set_seed(seed=0)` | Seed python/numpy/torch (+CUDA), return a seeded `torch.Generator`. |
| `tensor_stats(t)` → `TensorStats` | Never-raises summary: shape/dtype/mean/std/min/max/‖·‖/NaN/Inf. `str(...)` is one tidy log line. |
| `summarize(name, t)` | `"logits: (2,16,97) float32 mean=… ⚠ NaN=…"` — the detailed-logging primitive. |
| `assert_finite(t, name)` | Assert no NaN/Inf, with a diagnostic on failure. |
| `make_tiny_synaptic(seed, **cfg)` | Tiny CPU `GPTSynaptic` (sub-second forward). |
| `make_tiny_vanilla(seed, **cfg)` | Tiny CPU vanilla `GPT` (bio-vs-vanilla baseline). |
| `random_tokens(B, T, vocab, seed)` | Deterministic token ids. |
| `count_params(model)` | Parameter count. |
| `cuda_available()` / `rustbpe_available()` | Capability probes. |
| `assert_golden(name, t, atol, rtol)` | Compare against `tests/golden/<name>.npy`; bootstraps the golden on first run. |

### Golden artifacts
`assert_golden` locks numerical semantics (used by parity/property beads, e.g.
`eqyk.13`, `eqyk.14`, and the theory-thrust certificates). On first run — or with
`BIO_UPDATE_GOLDEN=1` after an **intentional** semantics change — it (re)writes the
golden and skips the comparison. Otherwise values must match within tolerance.
Commit the resulting `tests/golden/*.npy` files.

---

## Fixtures — `tests/conftest.py`

| Fixture | Gives you |
|---------|-----------|
| *(autouse)* `_per_test_determinism` | Re-seeds RNGs **before every test** so randomness can't leak between tests. |
| `seed` / `rng` | The canonical seed / a freshly-seeded `torch.Generator`. |
| `device` / `cuda_device` | CPU-or-CUDA / CUDA (skips if absent). |
| `tmp_run_dir` | A throwaway per-test run directory (checkpoints/logs/artifacts). |
| `tiny_synaptic_model` / `tiny_vanilla_model` | Ready-built tiny models. |
| `tiny_synaptic_model_factory` | `make_tiny_synaptic` for building several / patching config. |

---

## Conventions for new tests

1. **Name & locate:** `tests/test_<area>.py`, functions `test_*`. Put a module
   docstring naming the bead it serves (e.g. `(bead vg9.1)`), mirroring existing tests.
2. **Mark it:** at least `@pytest.mark.unit` (or `e2e`/`slow`/`gpu`/`golden`).
3. **Be deterministic:** use the `seed`/`rng` fixtures or `set_seed`; never depend
   on global RNG state from another test.
4. **Be diagnostic:** assert with `assert_finite` / include `tensor_stats` in
   messages so failures are self-explaining. Per the Definition of Done, a feature
   isn't done until tests pass **and** logs demonstrate correct behavior — the
   structured run-logging infra is bead `eqyk.2`; this kit is its test-side analog.
5. **Keep `unit` fast:** sub-second on CPU. Heavy/long runs are `slow`/`e2e`.
6. **Optional deps:** guard with `pytest.importorskip("…")` (see `test_rustbpe.py`),
   never a bare top-level import that breaks collection.

---

## Coverage gate

`[tool.coverage]` in `pyproject.toml` measures branch coverage of
`bio_inspired_nanochat` and enforces `fail_under` **only when `--cov` is passed**
(so the default fast `pytest` stays lightweight). The floor starts at **25%**
(current suite ≈ 34%) and should be **ratcheted upward, never lowered**, as beads
add their tests. CI runs `pytest -m "not slow" --cov`.

---

## What CI should run (see bead `eqyk.17`)

- **PR (fast):** `pytest -m "unit" --cov` + the coverage gate + `ruff` + `ty`.
- **Nightly (full):** `pytest -m "not slow"`, the e2e umbrellas
  (`eqyk.18`–`eqyk.22`), parity (`eqyk.13`), property (`eqyk.14`), perf-regression
  (`eqyk.15`), and the **validation report** (`eqyk.16`), uploading logs as artifacts.
