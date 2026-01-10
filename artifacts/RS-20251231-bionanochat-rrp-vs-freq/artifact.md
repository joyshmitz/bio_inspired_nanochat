---
session_id: "RS-20251231-bionanochat-rrp-vs-freq"
created_at: "2025-12-31T09:33:44.574Z"
updated_at: "2025-12-31T09:33:44.574Z"
version: 1
contributors:
  - agent: "BrownSnow"
    contributed_at: "2025-12-31T09:33:30.860057+00:00"
status: "active"
---

# Brenner Protocol Artifact: RS-20251231-bionanochat-rrp-vs-freq

## 1. Research Thread

**RT**: Is presynaptic vesicle depletion (RRP clamping) in Bio-Inspired Nanochat reducible to a token-frequency penalty, or does it implement edge-/attention-structure-dependent fatigue?

**Context**: We start with cheap mechanistic probes (<5 min) that can falsify the strict-equivalence view before running full decode/eval harnesses. Thread includes a micro-sim that holds token preference (logits) fixed while varying q/k geometry to test edge dependence.

**Why it matters**: If the mechanism is edge-dependent, it is not equivalent to a count-based frequency penalty and should be evaluated/controlled differently. If it is reducible, we should treat it as a parameterized penalty and focus on tuning + matched baselines.

**Anchors**: §99, §103, §160, [inference]

## 2. Hypothesis Slate

### H1: H1: Equivalence to tuned frequency penalty
**Claim**: Presynaptic depletion is functionally equivalent to an explicit (count-based) frequency penalty / logit bias once tuned to match repetition statistics.
**Mechanism**: RRP depletion primarily tracks repeated attention to the same keys; its effect can be reproduced by a penalty that depends only on token/key counts and a global decay timescale.
**Anchors**: §103, [inference]

### H2: H2: Edge-/context-dependent fatigue
**Claim**: Presynaptic depletion is not reducible to token-count penalties; it depends on attention edge structure (q·k geometry, distance barrier), producing context-conditional suppression.
**Mechanism**: Release depends on per-edge bilinear term (q·k) and distance-dependent barrier; thus two prompts with similar token counts can yield different fatigue dynamics.
**Anchors**: §160, [inference]

### H3: H3 (Third Alternative): Confounds / measurement artifacts
**Claim**: Any observed ‘wins’ are artifacts (metric confounds, sampling/seed mismatch, compute/capacity mismatch, or evaluation leakage), not evidence of a distinct mechanism.
**Mechanism**: Small evals are brittle; changes in sampling, logging, or hidden hyperparams can mimic improvements. Treat ambiguous results as quarantined until replicated under matched baselines.
**Anchors**: §103, [inference]
**Third alternative**: true

## 3. Predictions Table

| ID | Observation/Condition | H1 | H2 | H3 |
| --- | --- | --- | --- | --- |
| P1 | With identical causal logits (same token preference), changing q/k geometry changes syn_logit outputs in the presynaptic reference forward() micro-sim. | No material change (mean_abs_diff ≈ 0 within tolerance). | Material change (mean_abs_diff > 0) due to edge dependence. | Unstable / dominated by numeric quirks; replicate under controls before updating beliefs. |
| P2 | On toy decode prompts, a tuned explicit frequency penalty can match repetition metrics of presynaptic depletion on calibration prompts. | Yes; matching repetition metrics transfers to held-out prompts. | No; transfer fails on edge-structured prompts despite matched counts. | Apparent matches are seed/metric artifacts; require multi-seed replication. |
| P3 | Varying presynaptic barrier_strength (distance penalty) while holding token counts fixed produces systematic changes in where repetition breaks. | Little/no systematic effect beyond global penalty strength. | Systematic effect (distance matters in a way count-penalty can’t express). | Sensitive to implementation details; verify via unit-level invariants. |

## 4. Discriminative Tests

### T1: Presynaptic edge-dependence micro-sim (same token preference, different q/k) (Score: 10/12)
**Test ID**: T1
**Procedure**: In /data/projects/bio_inspired_nanochat, run a small CPU-only presynaptic forward-pass twice with identical logits but different q/k structure (uniform vs random). Compute mean |syn_logit_uniform - syn_logit_random|.

Command (example):
python -c 'import json, torch; from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn, build_presyn_state; torch.manual_seed(0); cfg=SynapticConfig(); B,H,T,D=1,1,8,4; pre=SynapticPresyn(D,cfg); logits=torch.full((B,H,T,T), -20.0);
for t in range(T): logits[0,0,t,:t+1] = -2.0; logits[0,0,t,0] = 2.0;
q1=torch.ones((B,H,T,D)); k1=torch.ones((B,H,T,D)); s1=build_presyn_state(B,T,H,device="cpu",dtype=torch.float32,cfg=cfg); y1,_=pre.forward(q1,k1,logits,s1);
torch.manual_seed(0); q2=torch.randn((B,H,T,D)); k2=torch.randn((B,H,T,D)); s2=build_presyn_state(B,T,H,device="cpu",dtype=torch.float32,cfg=cfg); y2,_=pre.forward(q2,k2,logits,s2);
print(json.dumps({"mean_abs_diff": float((y1-y2).abs().mean()), "y1_mean": float(y1.mean()), "y2_mean": float(y2.mean())}, indent=2))'
**Discriminates**: H1 vs H2 (is presynaptic depletion equivalent to a pure token-frequency penalty, or does it depend on attention edge structure?)
**Expected outcomes**:
- H1: With identical logits, changing q/k structure should not materially change syn_logit (mean_abs_diff ≈ 0 within numerical tolerance).
- H2: With identical logits, changing q/k structure produces material syn_logit differences (mean_abs_diff > 0), implying edge-/attention-structure dependence beyond token-count penalties.
- H3: If results are unstable across seeds or dominated by numerical artifacts, treat as inconclusive and quarantine.
**Potency check**: Repeat with identical q/k seeds (should yield mean_abs_diff ≈ 0). Optional: run with cfg.enable_presyn=False and verify syn_logit becomes effectively constant / uninformative.
**Evidence-per-week score**: LR=3, Cost=3, Speed=3, Ambiguity=1
**Status**: passed
**Last run**:
- Result ID: `f00fc4f9-ff6a-406b-9941-36952bd39607`
- Run at: 2025-12-31T09:27:32.830Z
- Exit code: 0
- Duration: 0.98s
- Summary: Test completed: exit 0 in 1.0s
- Result file: `artifacts/RS-20251231-bionanochat-rrp-vs-freq/experiments/T1/20251231T092732Z_f00fc4f9-ff6a-406b-9941-36952bd39607.json`

### T2: Toy decode: tuned freq-penalty baseline vs presynaptic depletion (Score: 7/12)
**Procedure**: Run a tiny generation (CPU, small prompt set) in 4 conditions: (A) presynaptic on, (B) presynaptic off, (C) presynaptic off + tuned freq_penalty to match repetition on calibration prompts, (D) presynaptic on + freq_penalty (check double-counting). Compare held-out repetition metrics + qualitative failure modes under fixed seed/sampling.
**Discriminates**: H1 vs H2 vs H3 (transfer of a tuned count-based penalty).
**Expected outcomes**:
- H1: C matches A on held-out prompts; D offers no additional benefit (or over-penalizes).
- H2: C fails to match A on edge-structured prompts despite calibration; D may over-penalize or reveal interactions.
- H3: Outcomes vary with seed/sampling/metric choice; replication needed before inference.
**Potency check**: Include a prompt where controlled repetition is required (copy-span). A pure freq penalty should degrade this predictably; selective fatigue should not (or should degrade differently).
**Evidence-per-week score**: LR=2, Cost=1, Speed=2, Ambiguity=2

## 5. Assumption Ledger

### A1: Assumption: micro-sim is a mechanistic probe (not full behavior)
**Statement**: The presynaptic reference micro-sim is only an existence proof of edge dependence in the formula; it is not a substitute for decode-level equivalence testing.
**Load**: If false, we may over-update from a toy calculation.
**Test**: Replicate with decode-level harness (T2) under matched sampling and prompt sets.

### A2: Scale/physics check: toy sizes vs real seq_len
**Statement**: Our micro-sim uses T=8, D=4 and synthetic logits; real runs use much larger seq_len and learned q/k distributions. Treat micro-sim as qualitative unless replicated at scale.
**Load**: If false, we might generalize incorrectly from a low-dimensional regime.
**Test**: Run the same probe with larger T (e.g., 128) and normalized q/k; verify effect persists and is not a small-T artifact.
**Scale check**: true
**Calculation**: Toy: T=8, D=4. Typical inference: seq_len ~ 4k-8k, d_head ~ O(64). We are ~500× smaller in T and ~16× smaller in D; many asymptotics do not transfer.

### A3: Assumption: q/k geometry variation is a proxy for edge structure
**Statement**: Varying q/k while holding logits fixed meaningfully probes edge-dependence in the presynaptic update (since release uses q·k), even though learned q/k in real models are not random.
**Load**: If false, our probe could be irrelevant to realistic regimes.
**Test**: Repeat using q/k drawn from an actual model forward pass (frozen) while controlling logits/attention masks.

## 6. Anomaly Register

None registered.

## 7. Adversarial Critique

### C1: Critique: micro-sim targets the equation, not end-to-end behavior
**Attack**: Even if synaptic.forward() is edge-dependent, end-to-end generation might still be mimicked by a tuned count-based penalty once the model adapts. This test could be ‘true but irrelevant’.
**Evidence**: Need decode-level matched-baseline experiments (T2) with fixed sampling, multi-seed replication, and prompts designed to separate edge-structure from token counts.
**Current status**: Active; treat T1 as mechanistic signal only.
**Real third alternative**: true

### C2: Critique: q/k randomization could be an artifact of scaling
**Attack**: The observed diff might be driven by unnormalized q/k norms or sigmoid saturation, not a robust qualitative property. If so, ‘edge dependence’ could vanish under normalization or realistic distributions.
**Evidence**: Repeat T1 with q/k normalized to fixed norms and sweep cfg parameters (e.g., barrier_strength, q_beta) to see if the effect is stable.
**Current status**: Active; add controls before strong inference.
