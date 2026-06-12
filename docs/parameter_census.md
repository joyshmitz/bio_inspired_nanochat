# Parameter Census — `SynapticConfig`

> **Generated** by `scripts/param_census.py` (bead `bio_inspired_nanochat-8j9.6`). Do not hand-edit; re-run `uv run python -m scripts.param_census`. Machine-readable companion: [`parameter_census.json`](./parameter_census.json).

`SynapticConfig` has **92 fields** — **92 LIVE** (read by runtime code) and **0 DEAD** (declared, read by nothing). This is the ground truth behind the README's *“48-parameter genome”* framing, which conflated three different counts.

## What the counts actually are

- **The learned genome is 4-D, not 48.** The biological 'genome' is the learned per-expert Xi vector (xi_dim=4: [alpha_fatigue, alpha_energy, camkii_gain, pp1_gain]), a torch parameter decoded to phenotype kinetics -- NOT the SynapticConfig hyperparameters. Every field here is a fixed hyperparameter, not a learned weight.

- **The wired search space is 10 params**, not 48. CMA-ES Phase 1 (`TOP10_PARAM_SPECS` in `scripts/tune_bio_params.py`) tunes: `alpha_ca`, `complexin_bias`, `doc2_gain`, `lambda_loge`, `nsf_recover`, `prime_rate`, `syt_fast_kd`, `syt_slow_kd`, `tau_c`, `unprime_per_release`. The 48-/82-parameter figures are the *aspirational* two-phase plan, not shipping code.

- **The config surface is 92 hyperparameters**, every one of which is read by runtime code — `8j9.5` pruned the last dead fields (`enabled`, `camkii_down`, `router_sim_threshold`, `native_presyn`, `native_metrics`, `native_plasticity`).


## Dead fields (read by nothing)

None — every `SynapticConfig` field is read on some runtime path (invariant enforced by `tests/test_param_census.py`).


## Full census by subsystem


### `general` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `rank_eligibility` | `8` | LIVE |  | bio_inspired_nanochat/synaptic.py:1056 |
| `attn_topk` | `32` | LIVE |  | bio_inspired_nanochat/synaptic.py:1757 |
| `stochastic_train_frac` | `0.12` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:170 |
| `stochastic_mode` | `normal_reparam` | LIVE |  | bio_inspired_nanochat/synaptic.py:795 |
| `stochastic_tau` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:794 |
| `stochastic_count_cap` | `8` | LIVE |  | bio_inspired_nanochat/synaptic.py:791 |

### `presynaptic` (12/12 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `tau_c` | `6.0` | LIVE | ✓ | bio_inspired_nanochat/cusp_certificate.py:100 |
| `learnable_kinetics` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:664 |
| `differentiable_recurrence` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:191 |
| `recurrence_block_size` | `64` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:192 |
| `recurrence_chunk_len` | `0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:196 |
| `doc2_gain` | `0.08` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:52 |
| `prime_rate` | `0.075` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:874 |
| `unprime_per_release` | `0.05` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:879 |
| `nsf_recover` | `0.08` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:880 |
| `rec_rate` | `0.06` | LIVE |  | bio_inspired_nanochat/synaptic.py:869 |
| `endo_delay` | `3` | LIVE |  | bio_inspired_nanochat/synaptic.py:867 |
| `metriplectic_integrator` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:672 |

### `initial_state` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `init_rrp` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1541 |
| `init_reserve` | `18.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1542 |
| `init_snare` | `0.7` | LIVE |  | bio_inspired_nanochat/synaptic.py:1543 |
| `init_clamp` | `0.6` | LIVE |  | bio_inspired_nanochat/synaptic.py:1544 |
| `init_amp` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1547 |
| `init_energy` | `0.85` | LIVE |  | bio_inspired_nanochat/synaptic.py:1545 |

### `energy` (3/3 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `energy_fill` | `0.02` | LIVE |  | bio_inspired_nanochat/synaptic.py:888 |
| `energy_use` | `0.02` | LIVE |  | bio_inspired_nanochat/synaptic.py:889 |
| `energy_max` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:888 |

### `attention` (4/4 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `lambda_loge` | `1.0` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:1802 |
| `barrier_strength` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:816 |
| `epsilon` | `1e-06` | LIVE |  | bio_inspired_nanochat/synaptic.py:999 |
| `loge_bias_clamp` | `10.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1805 |

### `kernel_compat` (18/18 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `tau_buf` | `4.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:101 |
| `tau_prime` | `5.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:935 |
| `tau_rrp` | `40.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:936 |
| `tau_energy` | `50.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:937 |
| `alpha_ca` | `0.55` | LIVE | ✓ | bio_inspired_nanochat/synaptic.py:590 |
| `alpha_buf_on` | `0.1` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:102 |
| `alpha_buf_off` | `0.1` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:103 |
| `alpha_prime` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:974 |
| `alpha_unprime` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:1008 |
| `alpha_refill` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:975 |
| `energy_in` | `0.01` | LIVE |  | bio_inspired_nanochat/synaptic.py:977 |
| `energy_cost_rel` | `0.015` | LIVE |  | bio_inspired_nanochat/synaptic.py:1011 |
| `energy_cost_pump` | `0.01` | LIVE |  | bio_inspired_nanochat/synaptic.py:1012 |
| `syt_fast_kd` | `0.4` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:50 |
| `syt_slow_kd` | `1.0` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:51 |
| `complexin_bias` | `0.0` | LIVE | ✓ | bio_inspired_nanochat/flex_synaptic.py:54 |
| `qmax` | `2.0` | LIVE |  | bio_inspired_nanochat/flex_synaptic.py:57 |
| `q_beta` | `1.0` | LIVE |  | bio_inspired_nanochat/flex_synaptic.py:57 |

### `postsynaptic` (17/17 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `post_fast_decay` | `0.95` | LIVE |  | bio_inspired_nanochat/synaptic.py:1217 |
| `post_fast_lr` | `0.0015` | LIVE |  | bio_inspired_nanochat/synaptic.py:1217 |
| `post_slow_lr` | `0.0005` | LIVE |  | bio_inspired_nanochat/synaptic.py:1205 |
| `post_trace_decay` | `0.96` | LIVE |  | bio_inspired_nanochat/synaptic.py:1337 |
| `fast_weight_normalized` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:1371 |
| `fast_weight_eta` | `0.5` | LIVE |  | bio_inspired_nanochat/synaptic.py:1383 |
| `fast_weight_max_norm` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1385 |
| `camkii_up` | `0.05` | LIVE |  | bio_inspired_nanochat/synaptic.py:1117 |
| `pp1_tau` | `0.985` | LIVE |  | bio_inspired_nanochat/synaptic.py:1118 |
| `camkii_thr` | `1.0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:181 |
| `pp1_thr` | `0.7` | LIVE |  | bio_inspired_nanochat/synaptic.py:1115 |
| `bdnf_tau` | `0.985` | LIVE |  | bio_inspired_nanochat/synaptic.py:1132 |
| `bdnf_scale` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1197 |
| `bdnf_gamma` | `0.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1197 |
| `bdnf_hebb_accumulate` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1129 |
| `bdnf_max` | `10.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1138 |
| `plasticity_during_training` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1496 |

### `latch` (14/14 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `bistable_latch` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:174 |
| `latch_ltd_thr` | `0.5` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:181 |
| `latch_input_gain` | `12.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:57 |
| `latch_alpha_ca` | `0.6` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:49 |
| `latch_beta_pp1` | `1.0` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:50 |
| `latch_gamma_auto` | `0.45` | LIVE |  | bio_inspired_nanochat/cusp_certificate.py:51 |
| `latch_hill_n` | `6.0` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:175 |
| `latch_hill_k` | `0.6` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:177 |
| `latch_alpha_pp1` | `0.5` | LIVE |  | bio_inspired_nanochat/synaptic.py:1109 |
| `latch_beta_camkii` | `0.3` | LIVE |  | bio_inspired_nanochat/synaptic.py:1109 |
| `latch_pp1_basal` | `0.3` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:179 |
| `latch_gate_beta` | `6.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:1187 |
| `cusp_latch` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:186 |
| `cusp_eps_max` | `0.98` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:186 |

### `structural` (6/6 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `structural_interval` | `50000` | LIVE |  | bio_inspired_nanochat/synaptic.py:2124 |
| `structural_tau_util` | `0.2` | LIVE |  | bio_inspired_nanochat/synaptic.py:2113 |
| `structural_age_bias` | `1.0` | LIVE |  | bio_inspired_nanochat/synaptic.py:2123 |
| `router_embed_dim` | `24` | LIVE |  | bio_inspired_nanochat/gpt_synaptic.py:488 |
| `router_contrastive_lr` | `0.0001` | LIVE |  | bio_inspired_nanochat/synaptic.py:2081 |
| `router_contrastive_push` | `0.1` | LIVE |  | bio_inspired_nanochat/synaptic.py:2081 |

### `genetics` (1/1 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `xi_dim` | `4` | LIVE |  | bio_inspired_nanochat/synaptic.py:1954 |

### `toggle` (4/4 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `enable_presyn` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:735 |
| `enable_hebbian` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:1274 |
| `enable_metabolism` | `True` | LIVE |  | bio_inspired_nanochat/synaptic.py:2005 |
| `use_flex_attention` | `False` | LIVE |  | bio_inspired_nanochat/ablation_registry.py:202 |

### `native_toggle` (1/1 live)

| Field | Default | Status | Tuned | Read at |
|---|---|---|---|---|
| `native_genetics` | `False` | LIVE |  | bio_inspired_nanochat/synaptic.py:2016 |

---
*Status = LIVE when read by a runtime module (`bio_inspired_nanochat/**` or the Rust kernel), DEAD otherwise. “Read at” shows the first runtime/Rust read site; full evidence is in the JSON.*
