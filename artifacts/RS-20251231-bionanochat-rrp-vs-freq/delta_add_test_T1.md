## Deltas

```delta
{
  "operation": "ADD",
  "section": "discriminative_tests",
  "target_id": null,
  "payload": {
    "name": "Presynaptic edge-dependence micro-sim (same token preference, different q/k)",
    "procedure": "In /data/projects/bio_inspired_nanochat, run a small CPU-only presynaptic forward-pass twice with identical logits but different q/k structure (uniform vs random). Compute mean |syn_logit_uniform - syn_logit_random|.\n\nCommand (example):\npython -c 'import json, torch; from bio_inspired_nanochat.synaptic import SynapticConfig, SynapticPresyn, build_presyn_state; torch.manual_seed(0); cfg=SynapticConfig(); B,H,T,D=1,1,8,4; pre=SynapticPresyn(D,cfg); logits=torch.full((B,H,T,T), -20.0);\nfor t in range(T): logits[0,0,t,:t+1] = -2.0; logits[0,0,t,0] = 2.0;\nq1=torch.ones((B,H,T,D)); k1=torch.ones((B,H,T,D)); s1=build_presyn_state(B,T,H,device=\"cpu\",dtype=torch.float32,cfg=cfg); y1,_=pre.forward(q1,k1,logits,s1);\ntorch.manual_seed(0); q2=torch.randn((B,H,T,D)); k2=torch.randn((B,H,T,D)); s2=build_presyn_state(B,T,H,device=\"cpu\",dtype=torch.float32,cfg=cfg); y2,_=pre.forward(q2,k2,logits,s2);\nprint(json.dumps({\"mean_abs_diff\": float((y1-y2).abs().mean()), \"y1_mean\": float(y1.mean()), \"y2_mean\": float(y2.mean())}, indent=2))'\n",
    "discriminates": "H1 vs H2 (is presynaptic depletion equivalent to a pure token-frequency penalty, or does it depend on attention edge structure?)",
    "expected_outcomes": {
      "H1": "With identical logits, changing q/k structure should not materially change syn_logit (mean_abs_diff ≈ 0 within numerical tolerance).",
      "H2": "With identical logits, changing q/k structure produces material syn_logit differences (mean_abs_diff > 0), implying edge-/attention-structure dependence beyond token-count penalties.",
      "H3": "If results are unstable across seeds or dominated by numerical artifacts, treat as inconclusive and quarantine."
    },
    "potency_check": "Repeat with identical q/k seeds (should yield mean_abs_diff ≈ 0). Optional: run with cfg.enable_presyn=False and verify syn_logit becomes effectively constant / uninformative."
  },
  "rationale": "Cheap (<1 min) mechanistic probe: if presynaptic logit adjustments depend on q/k geometry (edges) even when token preference is held fixed, it is not equivalent to a simple frequency penalty defined only by token counts."
}
```
