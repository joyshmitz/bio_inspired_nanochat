## Deltas

```delta
{
  "operation": "EDIT",
  "section": "research_thread",
  "target_id": null,
  "payload": {
    "statement": "Is presynaptic vesicle depletion (RRP clamping) in Bio-Inspired Nanochat reducible to a token-frequency penalty, or does it implement edge-/attention-structure-dependent fatigue?",
    "context": "We start with cheap mechanistic probes (<5 min) that can falsify the strict-equivalence view before running full decode/eval harnesses. Thread includes a micro-sim that holds token preference (logits) fixed while varying q/k geometry to test edge dependence.",
    "why_it_matters": "If the mechanism is edge-dependent, it is not equivalent to a count-based frequency penalty and should be evaluated/controlled differently. If it is reducible, we should treat it as a parameterized penalty and focus on tuning + matched baselines.",
    "anchors": ["§99", "§103", "§160", "[inference]"]
  },
  "rationale": "Initialize the research_thread section so compiled artifacts are self-contained and grounded (thread join-key + why it matters)."
}
```

```delta
{
  "operation": "ADD",
  "section": "hypothesis_slate",
  "target_id": null,
  "payload": {
    "name": "H1: Equivalence to tuned frequency penalty",
    "claim": "Presynaptic depletion is functionally equivalent to an explicit (count-based) frequency penalty / logit bias once tuned to match repetition statistics.",
    "mechanism": "RRP depletion primarily tracks repeated attention to the same keys; its effect can be reproduced by a penalty that depends only on token/key counts and a global decay timescale.",
    "anchors": ["§103", "[inference]"]
  },
  "rationale": "Baseline equivalence hypothesis from kickoff."
}
```

```delta
{
  "operation": "ADD",
  "section": "hypothesis_slate",
  "target_id": null,
  "payload": {
    "name": "H2: Edge-/context-dependent fatigue",
    "claim": "Presynaptic depletion is not reducible to token-count penalties; it depends on attention edge structure (q·k geometry, distance barrier), producing context-conditional suppression.",
    "mechanism": "Release depends on per-edge bilinear term (q·k) and distance-dependent barrier; thus two prompts with similar token counts can yield different fatigue dynamics.",
    "anchors": ["§160", "[inference]"]
  },
  "rationale": "Mechanistic alternative: if true, matched count-penalty baselines will fail on edge-structured prompts."
}
```

```delta
{
  "operation": "ADD",
  "section": "hypothesis_slate",
  "target_id": null,
  "payload": {
    "name": "H3 (Third Alternative): Confounds / measurement artifacts",
    "claim": "Any observed ‘wins’ are artifacts (metric confounds, sampling/seed mismatch, compute/capacity mismatch, or evaluation leakage), not evidence of a distinct mechanism.",
    "mechanism": "Small evals are brittle; changes in sampling, logging, or hidden hyperparams can mimic improvements. Treat ambiguous results as quarantined until replicated under matched baselines.",
    "third_alternative": true,
    "anchors": ["§103", "[inference]"]
  },
  "rationale": "Required third-alternative guardrail (§103): both mechanistic stories can be wrong."
}
```

```delta
{
  "operation": "ADD",
  "section": "predictions_table",
  "target_id": null,
  "payload": {
    "condition": "With identical causal logits (same token preference), changing q/k geometry changes syn_logit outputs in the presynaptic reference forward() micro-sim.",
    "predictions": {
      "H1": "No material change (mean_abs_diff ≈ 0 within tolerance).",
      "H2": "Material change (mean_abs_diff > 0) due to edge dependence.",
      "H3": "Unstable / dominated by numeric quirks; replicate under controls before updating beliefs."
    }
  },
  "rationale": "Direct prediction tied to T1."
}
```

```delta
{
  "operation": "ADD",
  "section": "predictions_table",
  "target_id": null,
  "payload": {
    "condition": "On toy decode prompts, a tuned explicit frequency penalty can match repetition metrics of presynaptic depletion on calibration prompts.",
    "predictions": {
      "H1": "Yes; matching repetition metrics transfers to held-out prompts.",
      "H2": "No; transfer fails on edge-structured prompts despite matched counts.",
      "H3": "Apparent matches are seed/metric artifacts; require multi-seed replication."
    }
  },
  "rationale": "Bridges to decode-level ‘matched baseline’ tests."
}
```

```delta
{
  "operation": "ADD",
  "section": "predictions_table",
  "target_id": null,
  "payload": {
    "condition": "Varying presynaptic barrier_strength (distance penalty) while holding token counts fixed produces systematic changes in where repetition breaks.",
    "predictions": {
      "H1": "Little/no systematic effect beyond global penalty strength.",
      "H2": "Systematic effect (distance matters in a way count-penalty can’t express).",
      "H3": "Sensitive to implementation details; verify via unit-level invariants."
    }
  },
  "rationale": "A second, cheap lever targeting edge structure."
}
```

```delta
{
  "operation": "ADD",
  "section": "discriminative_tests",
  "target_id": null,
  "payload": {
    "name": "Toy decode: tuned freq-penalty baseline vs presynaptic depletion",
    "procedure": "Run a tiny generation (CPU, small prompt set) in 4 conditions: (A) presynaptic on, (B) presynaptic off, (C) presynaptic off + tuned freq_penalty to match repetition on calibration prompts, (D) presynaptic on + freq_penalty (check double-counting). Compare held-out repetition metrics + qualitative failure modes under fixed seed/sampling.",
    "discriminates": "H1 vs H2 vs H3 (transfer of a tuned count-based penalty).",
    "expected_outcomes": {
      "H1": "C matches A on held-out prompts; D offers no additional benefit (or over-penalizes).",
      "H2": "C fails to match A on edge-structured prompts despite calibration; D may over-penalize or reveal interactions.",
      "H3": "Outcomes vary with seed/sampling/metric choice; replication needed before inference."
    },
    "potency_check": "Include a prompt where controlled repetition is required (copy-span). A pure freq penalty should degrade this predictably; selective fatigue should not (or should degrade differently).",
    "score": { "likelihood_ratio": 2, "cost": 1, "speed": 2, "ambiguity": 2 }
  },
  "rationale": "Second test to satisfy minimum test count and connect micro-sim to decode-level behavior."
}
```

```delta
{
  "operation": "EDIT",
  "section": "discriminative_tests",
  "target_id": "T1",
  "payload": {
    "score": { "likelihood_ratio": 3, "cost": 3, "speed": 3, "ambiguity": 1 }
  },
  "rationale": "Add score breakdown for T1 (cheap, fast, high LR; some ambiguity about external validity)."
}
```

```delta
{
  "operation": "ADD",
  "section": "assumption_ledger",
  "target_id": null,
  "payload": {
    "name": "Assumption: micro-sim is a mechanistic probe (not full behavior)",
    "statement": "The presynaptic reference micro-sim is only an existence proof of edge dependence in the formula; it is not a substitute for decode-level equivalence testing.",
    "load": "If false, we may over-update from a toy calculation.",
    "test": "Replicate with decode-level harness (T2) under matched sampling and prompt sets.",
    "anchors": ["§99", "[inference]"]
  },
  "rationale": "Keep interpretation disciplined."
}
```

```delta
{
  "operation": "ADD",
  "section": "assumption_ledger",
  "target_id": null,
  "payload": {
    "name": "Scale/physics check: toy sizes vs real seq_len",
    "statement": "Our micro-sim uses T=8, D=4 and synthetic logits; real runs use much larger seq_len and learned q/k distributions. Treat micro-sim as qualitative unless replicated at scale.",
    "load": "If false, we might generalize incorrectly from a low-dimensional regime.",
    "test": "Run the same probe with larger T (e.g., 128) and normalized q/k; verify effect persists and is not a small-T artifact.",
    "scale_check": true,
    "calculation": "Toy: T=8, D=4. Typical inference: seq_len ~ 4k-8k, d_head ~ O(64). We are ~500× smaller in T and ~16× smaller in D; many asymptotics do not transfer.",
    "anchors": ["§208", "[inference]"]
  },
  "rationale": "Explicit scale warning to avoid premature claims."
}
```

```delta
{
  "operation": "ADD",
  "section": "assumption_ledger",
  "target_id": null,
  "payload": {
    "name": "Assumption: q/k geometry variation is a proxy for edge structure",
    "statement": "Varying q/k while holding logits fixed meaningfully probes edge-dependence in the presynaptic update (since release uses q·k), even though learned q/k in real models are not random.",
    "load": "If false, our probe could be irrelevant to realistic regimes.",
    "test": "Repeat using q/k drawn from an actual model forward pass (frozen) while controlling logits/attention masks.",
    "anchors": ["§160", "[inference]"]
  },
  "rationale": "Make the proxy explicit so we can challenge it."
}
```

```delta
{
  "operation": "ADD",
  "section": "adversarial_critique",
  "target_id": null,
  "payload": {
    "name": "Critique: micro-sim targets the equation, not end-to-end behavior",
    "attack": "Even if synaptic.forward() is edge-dependent, end-to-end generation might still be mimicked by a tuned count-based penalty once the model adapts. This test could be ‘true but irrelevant’.",
    "evidence": "Need decode-level matched-baseline experiments (T2) with fixed sampling, multi-seed replication, and prompts designed to separate edge-structure from token counts.",
    "current_status": "Active; treat T1 as mechanistic signal only.",
    "real_third_alternative": true,
    "anchors": ["§103", "[inference]"]
  },
  "rationale": "Force relevance discipline and block over-interpretation."
}
```

```delta
{
  "operation": "ADD",
  "section": "adversarial_critique",
  "target_id": null,
  "payload": {
    "name": "Critique: q/k randomization could be an artifact of scaling",
    "attack": "The observed diff might be driven by unnormalized q/k norms or sigmoid saturation, not a robust qualitative property. If so, ‘edge dependence’ could vanish under normalization or realistic distributions.",
    "evidence": "Repeat T1 with q/k normalized to fixed norms and sweep cfg parameters (e.g., barrier_strength, q_beta) to see if the effect is stable.",
    "current_status": "Active; add controls before strong inference.",
    "anchors": ["§99", "[inference]"]
  },
  "rationale": "Second critique to satisfy minimum and propose a concrete control."
}
```
