# Stable-Recurrence Theory — spectral radius & contraction (bead yw9.7)

_Author: OrangeMill · 2026-06-11 · Epic `yw9` (Differentiable Synaptic Dynamics)._

Learning the synaptic kinetics (yw9.3) through the differentiable recurrence (yw9.2) risks
instability: if the effective state-transition has spectral radius ≥ 1, the state — and the
gradients backpropagated through it — blow up. This note derives the contraction conditions, shows
how the yw9.3 parameterization satisfies them, and specifies the runtime monitor
(`cb_spectral_radius` / `LearnableKinetics.spectral_radius`).

## 1. Which channels need a stability argument

From the per-step recurrence (see `release_canonical` and `docs/differentiable_synaptic_dynamics_design.md` §2):

- **Decoupled leaky integrators** — complexin `CL`, energy `E`, reserve `RES`, and the in-flight
  `DELAY` line are each `x_{t+1} = a_t x_t + b_t` with a per-channel decay `a_t`. With `0 ≤ a_t < 1`
  each is a scalar contraction (`|a_t| < 1`), trivially stable; the parallel scan over them inherits
  bounded prefix products `∏ a` (no blow-up). The vesicle pools `RRP`/`RES` are conservation-bounded
  (yw9.2.2). The only nontrivial subsystem is **calcium ↔ buffer**.

## 2. The calcium↔buffer transition matrix

The linear part of the `C`/`BUF` update (dropping the `clamp` projections and the exogenous influx,
which do not affect stability) is, with the bilinear coupling coefficient frozen at `β = (1 − BUF) ∈ [0,1]`:

```
C'   = (ρc − αon·β)·C + αoff·BUF
BUF' = (αon·β)·C      + (ρb − αoff)·BUF
```

i.e. `[C', BUF']ᵀ = M(β) · [C, BUF]ᵀ` with

```
            ⎡ ρc − αon·β     αoff      ⎤
   M(β)  =  ⎢                          ⎥
            ⎣  αon·β        ρb − αoff   ⎦
```

(`ρc = sigmoid(θ)` calcium decay, `ρb` buffer decay, `αon/αoff` the buffer on/off rates.)

## 3. Spectral radius (closed form) and the contraction condition

For a 2×2 `M = [[a,b],[c,d]]` with `tr = a+d`, `det = ad − bc`, `Δ = tr² − 4·det`:

- **Real eigenvalues** (`Δ ≥ 0`): `λ = (tr ± √Δ)/2`, so `ρ(M) = max(|tr+√Δ|, |tr−√Δ|)/2`.
- **Complex pair** (`Δ < 0`): `λ = (tr ± i√−Δ)/2` with `|λ|² = det`, so `ρ(M) = √det`.

`cb_spectral_radius(ρc, ρb, αon, αoff, β)` implements exactly this (differentiable, broadcasts over
`β`); it matches `torch.linalg.eigvals` to ~1e-6. The subsystem **cannot blow up ⟺ ρ(M) < 1** for all
`β ∈ [0,1]` (the discrete-time Jury/Schur conditions, `|det| < 1` and `|tr| < 1 + det`, are the
equivalent inequalities).

## 4. The guarantee, and its sharp edges

**Claim.** With the yw9.3 parameterization — `ρc, ρb = sigmoid(θ) ∈ (0,1)` and
`αon, αoff = ABUF_MAX·sigmoid(θ) ∈ [0, ABUF_MAX]` — the subsystem is **strictly contractive
(`ρ(M) < 1`) for every finite parameter value.** So no finite SGD update can destabilize the
forward or backward pass. This is why "learnable kinetics" is safe by construction.

**Why it holds.** `ρc, ρb` are strictly `< 1` for finite `θ` (sigmoid never reaches 1). At the two
boundaries of the parameter range the spectral radius **approaches** 1 but never reaches it:

- `ρc → 1` (or `ρb → 1`): at `β = 0`, `M` is upper-triangular with eigenvalues `ρc` and `ρb − αoff`;
  `ρ(M) → ρc → 1⁻`. (A decay of 1 is a pure integrator — marginally stable.)
- `ρc, ρb → 0` with `αon = αoff = ABUF_MAX`, `β = 1`: `M = [[−A, A],[A, −A]]` has eigenvalues
  `{0, −2A}`, so `ρ(M) → 2·ABUF_MAX`. With `ABUF_MAX = 0.5` this approaches `1⁻`.

So `ρ(M) < 1` is **strict but not uniform** — it is not bounded away from 1 as `θ → ±∞`. In
practice this is a non-issue: the realistic kinetics sit at `ρ(M) ≈ 0.85` (dominated by `ρc`), with
a wide margin, and a loss that rewarded `ρc → 1` (infinite calcium memory) or `ρ → 0` (instant
decay) would be pathological. SGD does not drive there.

**Optional uniform margin.** For a guaranteed margin `ρ(M) ≤ 1 − ε` one would additionally bound the
decays away from the endpoints, `ρ = ρmin + (ρmax − ρmin)·sigmoid(θ)` with `ρmax < 1`, and tighten
`2·ABUF_MAX ≤ ρmax`. This is left as a config option rather than the default, since it narrows the
expressivity of the learned kinetics for a guarantee that the runtime monitor already provides
cheaply.

## 5. Constraints implemented + the runtime monitor

- **Constraints (yw9.3):** decays via `sigmoid`, gains via `softplus`, buffer coupling via the
  bounded `ABUF_MAX·sigmoid` — exactly the conditions of §4.
- **Monitor:** `LearnableKinetics.spectral_radius(n_beta)` returns the worst-case `ρ(M)` over a grid
  of `β ∈ [0,1]`. It is differentiable, so it can be (a) logged as a per-step stability margin in
  telemetry, (b) asserted `< 1` in tests/CI, or (c) added as a soft penalty `max(0, ρ − (1−ε))²` to
  the loss if a uniform margin is ever wanted without re-parameterizing.

Tests: `tests/test_spectral_stability.py` (closed-form vs `eigvals`, init contractivity, strict
contraction for arbitrary finite parameters, differentiability, and the marginal-limit behavior).
