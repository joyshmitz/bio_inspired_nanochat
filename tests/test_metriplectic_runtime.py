"""
Runtime monitor + guards + deterministic fallback for the metriplectic integrator
(beads 0642.1.2.2 / 0642.1.2.3 / 0642.1.2.4).

Locks the runtime-certificate contract on top of the discrete-gradient integrator:

  - the free-energy Lyapunov MONITOR records E/S/F + entropy production per step and asserts F is
    non-increasing — the auditable evidence for the stability obligation (0642.1.2.2);
  - the conservation/entropy/degeneracy GUARDS pass for the structural operators, and a
    degeneracy-breaking (learned-style) operator trips the guard and deterministically falls back to
    the clamped-Euler baseline — never corrupting the run (0642.1.2.3);
  - the `metriplectic_integrator` toggle is default-off and registered with its prerequisite
    (0642.1.2.4).

Run:  pytest tests/test_metriplectic_runtime.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from bio_inspired_nanochat import metriplectic_integrator as mi
from bio_inspired_nanochat.ablation_registry import _BY_FIELD, validate_config
from bio_inspired_nanochat.synaptic import SynapticConfig

pytestmark = pytest.mark.unit

Z0 = np.array([1.0, 0.5, 0.0])


# --------------------------------------------------------------------------- #
# 1. The Lyapunov monitor (0642.1.2.2).
# --------------------------------------------------------------------------- #
def test_monitor_records_and_certifies_free_energy_lyapunov():
    traj, mon = mi.run_monitored(Z0, dt=0.1, steps=400)
    assert len(mon.records) == 400
    mon.assert_lyapunov()  # raises if F ever increases beyond tolerance
    s = mon.summary()
    assert s["lyapunov_ok"] is True
    assert s["max_energy_drift"] <= 1e-8, "energy must be conserved within the guard tolerance"
    assert s["min_entropy_production"] >= -1e-10, "entropy must be non-decreasing"
    assert s["n_fallbacks"] == 0, "the structural integrator must never need the fallback"


def test_monitor_detects_a_non_lyapunov_sequence():
    # Hand a fabricated increasing-F record stream to the monitor: it must flag it.
    mon = mi.LyapunovMonitor(tol=1e-9)
    for i, f in enumerate([1.0, 0.9, 0.95]):  # F goes back up at step 2
        mon.append(mi.StepRecord(i, 0, 0, f, 0, 0, 0, 0, False, ""))
    assert not mon.free_energy_nonincreasing()
    with pytest.raises(AssertionError, match="Lyapunov"):
        mon.assert_lyapunov()


# --------------------------------------------------------------------------- #
# 2. The guards + deterministic fallback (0642.1.2.3).
# --------------------------------------------------------------------------- #
def test_structural_operators_pass_all_guards():
    _, rec = mi.guarded_step(Z0, 0.1, 0, mi.GuardThresholds())
    assert rec.breach == "" and not rec.used_fallback
    assert rec.res_L_gradS < 1e-12 and rec.res_M_gradE < 1e-12, "structural degeneracy is exact"
    assert abs(rec.energy_drift) < 1e-10 and rec.entropy_production >= -1e-12


def test_degeneracy_breaking_operator_trips_guard_and_falls_back():
    # A "learned" friction that breaks M·∇E = 0 (an extra non-degenerate term).
    def bad_M(z, gC=mi.GAMMA_C, gB=mi.GAMMA_B):
        return mi.M_op(z, gC, gB) + np.diag([0.3, 0.3, 0.3])  # diag adds M·∇E ≠ 0

    res_ls, res_me = mi.degeneracy_residuals(Z0, M_fn=bad_M)
    assert res_me > 1e-2, "the broken operator must have a large M·∇E residual"

    _, rec = mi.guarded_step(Z0, 0.1, 0, mi.GuardThresholds(), M_fn=bad_M)
    assert rec.breach == "degeneracy" and rec.used_fallback, "must fall back on a degeneracy breach"


def test_energy_drift_guard_trips_on_a_loose_integrator():
    # Force a drift breach with an absurdly tight energy tolerance the exact step still satisfies
    # numerically — so instead inject a bad operator that makes the discrete step drift energy.
    def skew_breaking_L(omega=mi.OMEGA):
        return np.array([[0.2, omega, 0.0], [-omega, 0.0, 0.0], [0.0, 0.0, 0.0]])  # not skew

    _, rec = mi.guarded_step(Z0, 0.3, 0, mi.GuardThresholds(eps_D=1e9), L_fn=skew_breaking_L)
    # eps_D huge so the degeneracy guard does not pre-empt; the non-skew L drifts energy ⟹ fallback.
    assert rec.used_fallback and rec.breach in ("energy_drift", "entropy")


def test_run_monitored_with_a_bad_operator_uses_fallback_and_stays_safe():
    def bad_M(z, gC=mi.GAMMA_C, gB=mi.GAMMA_B):
        return mi.M_op(z, gC, gB) + np.diag([0.3, 0.3, 0.3])

    traj, mon = mi.run_monitored(Z0, dt=0.1, steps=100, M_fn=bad_M)
    s = mon.summary()
    assert s["n_fallbacks"] == 100, "every step must fall back when the operator is degenerate-broken"
    assert np.all(np.isfinite(traj)), "the safe fallback must keep the trajectory finite"


# --------------------------------------------------------------------------- #
# 3. The toggle + registry discipline (0642.1.2.4).
# --------------------------------------------------------------------------- #
def test_metriplectic_integrator_toggle_default_off_and_registered():
    assert SynapticConfig().metriplectic_integrator is False
    assert "metriplectic_integrator" in _BY_FIELD
    assert _BY_FIELD["metriplectic_integrator"].requires == ("enable_presyn",)


def test_validator_flags_integrator_without_presyn():
    errors, _ = validate_config(
        SynapticConfig(metriplectic_integrator=True, enable_presyn=False)
    )
    assert any("metriplectic_integrator" in e and "enable_presyn" in e for e in errors)


def test_default_config_still_validates_clean():
    assert validate_config(SynapticConfig()) == ([], [])
