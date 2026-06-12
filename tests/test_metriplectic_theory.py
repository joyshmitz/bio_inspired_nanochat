"""
Numerical corroboration of the metriplectic / GENERIC theory note (bead 0642.1.1).

Checks the EXACT algebraic facts and the qualitative dynamical facts of the constructed metriplectic
core z = (C, B, h) directly (the "symbolic + grid check" the bead asks for):

  - degeneracy   L·∇S = 0  and  M·∇E = 0     at a grid of random states (D1, D2);
  - structure    L + Lᵀ = 0 (skew)  and  eig(M) ≥ 0 (PSD)  on the grid;
  - conservation integrating ż = L∇E + M∇S conserves E (drift → 0 with the step), S is monotone
    non-decreasing, and F = E − T·S is non-increasing (Lyapunov);
  - boundedness  the trajectory stays inside the energy-shell bound C² + B² ≤ 2E₀ and relaxes to the
    MaxEnt equilibrium z* = (0, 0, E₀);
  - baseline     forward Euler (the vg9-style step) drifts E far more than RK4 — motivating the
    structure-preserving discrete-gradient integrator (0642.1.2.1).

See docs/theory/metriplectic.md.  Run:  pytest tests/test_metriplectic_theory.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# Default operators / parameters (ω = reversible exchange rate; γ_C, γ_B = leak/dissipation rates).
OMEGA, GAMMA_C, GAMMA_B, TEMP = 1.0, 0.2, 0.1, 0.5


def grad_E(z: np.ndarray) -> np.ndarray:
    C, B, _h = z
    return np.array([C, B, 1.0])


def grad_S(_z: np.ndarray) -> np.ndarray:
    return np.array([0.0, 0.0, 1.0])


def L_op(omega: float = OMEGA) -> np.ndarray:
    return np.array([[0.0, omega, 0.0], [-omega, 0.0, 0.0], [0.0, 0.0, 0.0]])


def M_op(z: np.ndarray, gC: float = GAMMA_C, gB: float = GAMMA_B) -> np.ndarray:
    C, B, _h = z
    u = np.array([1.0, 0.0, -C])
    v = np.array([0.0, 1.0, -B])
    return gC * np.outer(u, u) + gB * np.outer(v, v)


def field(z: np.ndarray, omega=OMEGA, gC=GAMMA_C, gB=GAMMA_B) -> np.ndarray:
    return L_op(omega) @ grad_E(z) + M_op(z, gC, gB) @ grad_S(z)


def energy(z: np.ndarray) -> float:
    C, B, h = z
    return 0.5 * (C * C + B * B) + h


def entropy(z: np.ndarray) -> float:
    return float(z[2])


def free_energy(z: np.ndarray, T: float = TEMP) -> float:
    return energy(z) - T * entropy(z)


def _grid(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-3.0, 3.0, size=(n, 3))


# --------------------------------------------------------------------------- #
# 1. Degeneracy conditions (D1, D2) hold at every state.
# --------------------------------------------------------------------------- #
def test_degeneracy_L_gradS_is_zero():
    for z in _grid():
        assert np.allclose(L_op() @ grad_S(z), 0.0, atol=1e-12), "D1: L·∇S must vanish"


def test_degeneracy_M_gradE_is_zero():
    for z in _grid():
        assert np.allclose(M_op(z) @ grad_E(z), 0.0, atol=1e-10), "D2: M·∇E must vanish"


# --------------------------------------------------------------------------- #
# 2. Structure: L skew-symmetric, M symmetric PSD.
# --------------------------------------------------------------------------- #
def test_L_is_skew_symmetric():
    L = L_op()
    assert np.allclose(L + L.T, 0.0), "L must be skew-symmetric (Lᵀ = −L)"


def test_M_is_symmetric_psd():
    for z in _grid():
        M = M_op(z)
        assert np.allclose(M, M.T, atol=1e-12), "M must be symmetric"
        eig = np.linalg.eigvalsh(M)
        assert eig.min() > -1e-10, f"M must be PSD; min eigenvalue {eig.min():.2e}"


# --------------------------------------------------------------------------- #
# 3. Conservation / production / Lyapunov along the continuous flow (RK4).
# --------------------------------------------------------------------------- #
def _rk4(z0: np.ndarray, dt: float, steps: int):
    z = z0.astype(np.float64).copy()
    traj = [z.copy()]
    for _ in range(steps):
        k1 = field(z)
        k2 = field(z + 0.5 * dt * k1)
        k3 = field(z + 0.5 * dt * k2)
        k4 = field(z + dt * k3)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(z.copy())
    return np.array(traj)


def test_energy_conserved_entropy_monotone_free_energy_lyapunov():
    z0 = np.array([1.0, 0.5, 0.0])
    E0 = energy(z0)
    traj = _rk4(z0, dt=0.01, steps=2000)
    Es = np.array([energy(z) for z in traj])
    Ss = np.array([entropy(z) for z in traj])
    Fs = np.array([free_energy(z) for z in traj])

    assert np.max(np.abs(Es - E0)) / E0 < 1e-6, "E must be conserved along the continuous flow"
    assert np.all(np.diff(Ss) >= -1e-9), "S must be non-decreasing (entropy production)"
    assert np.all(np.diff(Fs) <= 1e-9), "F = E − T·S must be non-increasing (Lyapunov)"


def test_trajectory_is_bounded_and_relaxes_to_maxent_equilibrium():
    z0 = np.array([1.0, 0.5, 0.0])
    E0 = energy(z0)
    # Long enough for the slow buffer leak (γ_B = 0.1) to drain the mechanical energy into heat.
    traj = _rk4(z0, dt=0.01, steps=8000)
    mech = traj[:, 0] ** 2 + traj[:, 1] ** 2
    assert np.all(mech <= 2 * E0 + 1e-6), "C² + B² must stay within the energy-shell bound 2E₀"
    # MaxEnt equilibrium: all mechanical energy → heat, z* = (0, 0, E₀).
    z_final = traj[-1]
    assert abs(z_final[0]) < 1e-3 and abs(z_final[1]) < 1e-3, "calcium must relax to 0"
    assert abs(z_final[2] - E0) < 1e-3, "heat must reach E₀ (S maximal on the shell)"


# --------------------------------------------------------------------------- #
# 4. Baseline contrast: forward Euler drifts E; RK4 (closer to the continuous flow) far less.
# --------------------------------------------------------------------------- #
def test_forward_euler_drifts_energy_more_than_rk4():
    z0 = np.array([1.0, 0.5, 0.0])
    E0 = energy(z0)
    dt, steps = 0.05, 400

    z = z0.copy()
    euler_max = 0.0
    for _ in range(steps):
        z = z + dt * field(z)
        euler_max = max(euler_max, abs(energy(z) - E0))

    traj = _rk4(z0, dt=dt, steps=steps)
    rk4_max = float(np.max(np.abs([energy(z) for z in traj] - E0 * np.ones(len(traj)))))

    assert euler_max > rk4_max, (
        f"forward Euler (vg9-style) should drift E more than RK4 "
        f"(euler {euler_max:.2e} vs rk4 {rk4_max:.2e})"
    )
    # Neither is exact — the discrete-gradient integrator (0642.1.2.1) is what conserves E exactly.
    assert euler_max > 1e-4, "the baseline Euler step must show a measurable energy drift"
