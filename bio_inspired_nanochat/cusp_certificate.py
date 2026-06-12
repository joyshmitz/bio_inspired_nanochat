"""Runtime retention certificate for the bistable CaMKII/PP1 latch (bead 0642.2.2.3).

Turns the *theory* of `docs/theory/singular_perturbation.md` into a runtime artifact: the closed-form
**hysteresis half-width** `δ* = (2/3√3)·(−a)^{3/2}` is the certified lower bound on how far the latch
control can drift before a latched memory is destroyed. We compute the cusp normal-form coefficients
`(a, b)` of the *actual* CaMKII equilibrium residual at a resting operating point, evaluate `δ*`, and
gate the claim on a timescale-separation (normal-hyperbolicity) check. When the latch is monostable
(`a ≥ 0`) or the fast calcium subsystem is not contractive enough to slave to the slow latch
(`ρ(M_cb) > cusp_eps_max`), the certificate is **dropped** and the model falls back to the heuristic
`sax.2` latch with no retention claim — the deterministic fail-closed discipline of the proof ledger
(§5/§7 of the note).

Scope: this module is the certificate + fallback gate (`0642.2.2.3`). The cusp-form latch *update*
and the minimum-energy write/erase pulses are `0642.2.2.1`; a full `ε = τ_fast/τ_slow` gauge is
`0642.10.1` — here `ε` is proxied by the fast-subsystem spectral radius `ρ(M_cb)`, exactly the
normal-hyperbolicity hypothesis (F1/F2) the certificate rests on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bio_inspired_nanochat.synaptic import SynapticConfig, cb_spectral_radius
from bio_inspired_nanochat.torch_imports import torch

_TWO_OVER_3SQRT3 = 2.0 / (3.0 * math.sqrt(3.0))


def hill_inflection(n: float, k: float) -> float:
    """Inflection of the self-excitation Hill `H(m)=m^n/(k^n+m^n)`: where `H''=0` (n>1)."""
    if n <= 1.0:
        return k
    return k * ((n - 1.0) / (n + 1.0)) ** (1.0 / n)


def _hill(m: float, n: float, k: float) -> float:
    mn = m ** n
    return mn / (k ** n + mn + 1e-12)


def _camkii_residual(m: float, cfg: SynapticConfig, drive: float, pp1: float) -> float:
    """The CaMKII equilibrium residual G(m) = α_ca·d·(1−m) − β_pp1·p·m + γ·H(m) (note §0/§3).

    Equilibria of the latch are the roots of G; the self-excitation Hill is the only nonconvex term,
    so the cusp lives at its inflection.
    """
    return (
        cfg.latch_alpha_ca * drive * (1.0 - m)
        - cfg.latch_beta_pp1 * pp1 * m
        + cfg.latch_gamma_auto * _hill(m, cfg.latch_hill_n, cfg.latch_hill_k)
    )


def resting_drive(cfg: SynapticConfig, c_rest: float) -> float:
    """The LTP drive `d(c)=σ(g·(c−θ_LTP))` at the resting calcium `c_rest`."""
    return 1.0 / (1.0 + math.exp(-cfg.latch_input_gain * (c_rest - cfg.camkii_thr)))


def cusp_coefficients(cfg: SynapticConfig, *, c_rest: float | None = None, h: float = 1e-3) -> tuple[float, float]:
    """The cusp normal-form coefficients `(a, b)` of `m̃³ + a·m̃ + b` at the resting operating point.

    `a = −C₁/|C₃|`, `b = −C₀/|C₃|` with `C₀=G(m_*)`, `C₁=G'(m_*)`, `C₃=G'''(m_*)/6` evaluated by
    central finite differences at the Hill inflection `m_*` (the cusp organizing center, where the
    quadratic term vanishes). PP1 sits at its basal floor at rest (the ON state's slaved value is
    floored there). `a < 0 ⟺ bistable` (self-excitation slope exceeds the linear decay).
    """
    if c_rest is None:
        c_rest = 0.5 * (cfg.latch_ltd_thr + cfg.camkii_thr)
    d = resting_drive(cfg, c_rest)
    p = cfg.latch_pp1_basal
    m_star = hill_inflection(cfg.latch_hill_n, cfg.latch_hill_k)

    def g(m: float) -> float:
        return _camkii_residual(m, cfg, d, p)

    c0 = g(m_star)
    c1 = (g(m_star + h) - g(m_star - h)) / (2.0 * h)
    c3 = (g(m_star + 2 * h) - 2 * g(m_star + h) + 2 * g(m_star - h) - g(m_star - 2 * h)) / (2.0 * h ** 3) / 6.0
    # A negligible cubic coefficient (relative to the linear scale) means there is no self-excitation
    # nonlinearity — the residual is affine, the system is monostable, and there is no cusp scaling.
    # Report the monostable boundary a=0 rather than a finite-difference-noise blow-up.
    if abs(c3) < 1e-6 * (1.0 + abs(c1) + abs(c0)):
        return 0.0, 0.0
    return -c1 / abs(c3), -c0 / abs(c3)


def retention_delta_star(a: float) -> float:
    """Closed-form hysteresis half-width `δ*(a)=(2/3√3)·(−a)^{3/2}` (0 for the monostable `a≥0`)."""
    return _TWO_OVER_3SQRT3 * (-a) ** 1.5 if a < 0.0 else 0.0


def epsilon_gauge(cfg: SynapticConfig, n_beta: int = 21) -> float:
    """Normal-hyperbolicity proxy: worst-case fast-subsystem spectral radius `ρ(M_cb)` over β∈[0,1].

    Strong timescale separation (small `ε`) requires the fast calcium↔buffer map to be a clean
    contraction, i.e. `ρ(M_cb)` comfortably below 1 (a clear gap). This reuses the `yw9.7`
    closed-form `cb_spectral_radius`.
    """
    rho_c = torch.tensor(math.exp(-1.0 / cfg.tau_c), dtype=torch.float64)
    rho_b = torch.tensor(math.exp(-1.0 / cfg.tau_buf), dtype=torch.float64)
    a_on = torch.tensor(float(cfg.alpha_buf_on), dtype=torch.float64)
    a_off = torch.tensor(float(cfg.alpha_buf_off), dtype=torch.float64)
    betas = torch.linspace(0.0, 1.0, n_beta, dtype=torch.float64)
    rho = cb_spectral_radius(rho_c, rho_b, a_on, a_off, betas)
    return float(rho.max())


@dataclass(frozen=True)
class RetentionCertificate:
    """The runtime retention certificate + its fail-closed verdict."""

    a: float                 # cusp splitting parameter (a < 0 ⟺ bistable)
    b: float                 # cusp bias parameter
    delta_star: float        # certified retention half-width (0 if uncertified)
    eps: float               # ε proxy = ρ(M_cb), the fast-subsystem spectral radius
    bistable: bool           # a < 0
    separated: bool          # ε ≤ cusp_eps_max (normal hyperbolicity holds)
    certified: bool          # bistable AND separated
    use_heuristic_fallback: bool  # not certified ⟹ fall back to the sax.2 heuristic latch
    reason: str


def certify_retention(cfg: SynapticConfig, *, c_rest: float | None = None) -> RetentionCertificate:
    """Compute the retention certificate for ``cfg`` and apply the deterministic fallback rule.

    Certified iff the latch is bistable (`a < 0`) AND the fast subsystem is contractive enough
    (`ρ(M_cb) ≤ cusp_eps_max`). Otherwise the certificate is void, `δ*` is dropped to 0, and the
    caller should use the heuristic `sax.2` latch (no retention claim).
    """
    a, b = cusp_coefficients(cfg, c_rest=c_rest)
    eps = epsilon_gauge(cfg)
    bistable = a < 0.0
    separated = eps <= cfg.cusp_eps_max
    certified = bool(bistable and separated)
    delta = retention_delta_star(a) if certified else 0.0
    if certified:
        reason = (f"certified: bistable (a={a:.3g}<0) and separated (ρ_fast={eps:.3g}≤"
                  f"{cfg.cusp_eps_max:g}); retention ≥ δ*={delta:.3g}")
    elif not bistable:
        reason = (f"uncertified: monostable (a={a:.3g}≥0) — heuristic sax.2 fallback, no retention "
                  f"claim")
    else:
        reason = (f"uncertified: insufficient timescale separation (ρ_fast={eps:.3g}>"
                  f"{cfg.cusp_eps_max:g}) — heuristic sax.2 fallback")
    return RetentionCertificate(
        a=a, b=b, delta_star=delta, eps=eps, bistable=bistable, separated=separated,
        certified=certified, use_heuristic_fallback=not certified, reason=reason,
    )
