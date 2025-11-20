import torch
import numpy as np
import pytest
from bio_inspired_nanochat.synaptic import SynapticConfig, build_presyn_state
import torch.nn.functional as F

# Try to import rustbpe_native, skip if not available
try:
    from rustbpe import rustbpe as rustbpe_native
    rustbpe = rustbpe_native # For backwards compatibility with test calls
except ImportError:
    rustbpe = None

def softplus(x):
    return torch.log1p(torch.exp(x))

def sigmoid(x):
    return torch.sigmoid(x)

def presyn_step_python_ref(q, k, logits, state, cfg):
    B, H, T, D = q.shape
    
    # Unpack state
    c = state["C"].clone()
    buf = state["BUF"].clone()
    rrp = state["RRP"].clone()
    res = state["RES"].clone()
    pr = state["PR"].clone()
    cl = state["CL"].clone()
    e_st = state["E"].clone()
    
    # Constants
    rho_c = np.exp(-1.0 / cfg.tau_c)
    rho_b = np.exp(-1.0 / cfg.tau_buf)
    rho_p = np.exp(-1.0 / cfg.tau_prime)
    rho_r = np.exp(-1.0 / cfg.tau_rrp)
    rho_e = np.exp(-1.0 / cfg.tau_energy)
    sqrt_d = np.sqrt(D)
    
    syn_logit = torch.zeros_like(logits)
    
    for t in range(T):
        # 1. Compute Influx
        # logits slice (B, H, t, 0:t+1)
        log_t = logits[:, :, t, :t+1]
        clamped = log_t.clamp(-20.0, 20.0)
        drive = softplus(clamped)
        sum_drive = drive.sum(dim=-1) # (B, H)
        influx = sum_drive / (t + 1)
        
        # 2. Update State
        c_prev = c[:, :, t]
        buf_prev = buf[:, :, t]
        
        c_next = rho_c * c_prev + cfg.alpha_ca * influx - cfg.alpha_buf_on * c_prev * (1.0 - buf_prev) + cfg.alpha_buf_off * buf_prev
        buf_next = rho_b * buf_prev + cfg.alpha_buf_on * c_prev * (1.0 - buf_prev) - cfg.alpha_buf_off * buf_prev
        
        c_next = c_next.clamp(min=0.0)
        buf_next = buf_next.clamp(0.0, 1.0)
        
        # Update c and buf for this timestep
        
        pr_val = pr[:, :, t]
        rrp_val = rrp[:, :, t]
        res_val = res[:, :, t]
        e_val = e_st[:, :, t]
        
        pr_mid = (rho_p * pr_val + cfg.alpha_prime * (1.0 - pr_val)).clamp(0.0, 1.0)
        rrp_refill = (rho_r * rrp_val + cfg.alpha_refill * res_val).clamp(0.0, 1.0)
        res_mid = (res_val - cfg.alpha_refill * res_val).clamp(0.0, 1.0)
        e_mid = (rho_e * e_val + cfg.energy_in).clamp(0.0, 1.6)
        
        # 3. Compute Release
        q_t = q[:, :, t, :] # (B, H, D)
        
        c_val = c_next # Use updated C
        fast = c_val / (c_val + cfg.syt_fast_kd)
        slow = c_val / (c_val + cfg.syt_slow_kd)
        syt = 0.7 * fast + 0.3 * slow
        
        cl_val = cl[:, :, t]
        fuse_logit_base = 3.0 * syt + 2.0 * pr_mid - 2.0 * (cl_val + cfg.complexin_bias)
        fuse_base = sigmoid(fuse_logit_base)
        
        # Loop over j
        k_j = k[:, :, :t+1, :] # (B, H, t+1, D)
        # Dot product
        # (B, H, D) * (B, H, t+1, D) -> (B, H, t+1)
        dot = torch.einsum("bhd,bhjd->bhj", q_t, k_j)
        d_bilin = sigmoid(dot / sqrt_d)
        
        fuse_p = fuse_base.unsqueeze(-1) * d_bilin # (B, H, t+1)
        avail = rrp_refill.unsqueeze(-1)
        
        rr = (fuse_p * avail).clamp(0.0, 1.0)
        row_sum = rr.sum(dim=-1) # (B, H)
        
        scale = torch.ones_like(row_sum)
        mask_scale = row_sum > cfg.epsilon
        scale[mask_scale] = (rrp_refill[mask_scale] / row_sum[mask_scale]).clamp(max=1.0)
        
        rel = rr * scale.unsqueeze(-1) # (B, H, t+1)
        used = rel.sum(dim=-1)
        
        # 4. Update Final State
        rrp_n = (rrp_refill - used).clamp(0.0, 1.0)
        res_n = (res_mid + used).clamp(0.0, 1.0)
        pr_n = (pr_mid - cfg.alpha_unprime * used).clamp(0.0, 1.0)
        e_n = (e_mid - cfg.energy_cost_rel * used - cfg.energy_cost_pump * (1.0 - res_n)).clamp(0.0, 1.6)
        
        qamp = sigmoid(cfg.q_beta * (e_n - 0.5)) * cfg.qmax
        
        # 5. Syn Logit
        # dist
        j_indices = torch.arange(t + 1, device=q.device).float()
        t_float = float(t)
        dist = (t_float - j_indices).abs() / max(1.0, float(T))
        dist = dist.view(1, 1, -1)
        
        val = (rel * qamp.unsqueeze(-1)).clamp(min=cfg.epsilon).log() - cfg.barrier_strength * dist
        
        syn_logit[:, :, t, :t+1] = val
        syn_logit[:, :, t, t+1:] = np.log(cfg.epsilon)
        
        # Store state back
        c[:, :, t] = c_next
        buf[:, :, t] = buf_next
        rrp[:, :, t] = rrp_n
        res[:, :, t] = res_n
        pr[:, :, t] = pr_n
        e_st[:, :, t] = e_n
        
    new_state = {
        "C": c, "BUF": buf, "RRP": rrp, "RES": res, "PR": pr, "CL": cl, "E": e_st
    }
    return syn_logit, new_state

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_presyn_step_cpu_parity():
    B, H, T, D = 2, 4, 32, 16
    # Use config compatible with Rust implementation
    cfg = SynapticConfig(native_presyn=True)
    
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    logits = torch.randn(B, H, T, T)
    mask = torch.tril(torch.ones(T, T)).bool()
    logits.masked_fill_(~mask.view(1, 1, T, T), -20.0) # Use -20 instead of -inf for softplus stability in test
    
    # Build initial state tensors (random for robustness)
    state = {
        "C": torch.rand(B, H, T),
        "BUF": torch.rand(B, H, T),
        "RRP": torch.rand(B, H, T),
        "RES": torch.rand(B, H, T),
        "PR": torch.rand(B, H, T),
        "CL": torch.rand(B, H, T),
        "E": torch.rand(B, H, T),
    }
    
    # Run Rust version
    q_np = q.numpy()
    k_np = k.numpy()
    logits_np = logits.numpy()
    state_np = {k: v.numpy() for k, v in state.items()}
    
    syn_logit_rust, state_new_rust = rustbpe.presyn_step_cpu(q_np, k_np, logits_np, state_np, cfg)
    
    # Run Python Reference
    syn_logit_py, state_new_py = presyn_step_python_ref(q, k, logits, state, cfg)
    
    print("Comparing syn_logit...")
    mask_expanded = mask.view(1, 1, T, T).expand(B, H, T, T)
    diff = torch.abs(torch.from_numpy(syn_logit_rust) - syn_logit_py)
    diff_masked = diff[mask_expanded]
    print(f"Max diff: {diff_masked.max().item()}")
    
    assert torch.allclose(torch.from_numpy(syn_logit_rust)[mask_expanded], syn_logit_py[mask_expanded], atol=1e-4, rtol=1e-4)
    
    print("Comparing state...")
    for k in state:
        if k == "CL": continue # CL is constant
        diff = torch.abs(torch.from_numpy(state_new_rust[k]) - state_new_py[k])
        print(f"State {k} max diff: {diff.max().item()}")
        assert torch.allclose(torch.from_numpy(state_new_rust[k]), state_new_py[k], atol=1e-4, rtol=1e-4)

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_moe_stats_cpu_parity():
    B, T, k = 2, 128, 2
    E = 8
    
    idx = torch.randint(0, E, (B, T, k))
    gates = torch.rand(B, T, k)
    
    # Python reference
    me = torch.zeros(E)
    pe = torch.zeros(E)
    for e in range(E):
        mask = idx == e
        sel = mask.any(dim=-1)
        me[e] = sel.sum()
        pe[e] = gates.masked_select(mask).sum()
        
    # Rust version
    idx_np = idx.numpy().astype("int64")
    gates_np = gates.numpy()
    
    counts_rust, probs_rust = rustbpe.accumulate_router_stats_cpu(idx_np, gates_np, E)
    
    print("Comparing MoE stats...")
    print(f"Counts max diff: {np.abs(counts_rust - me.numpy()).max()}")
    print(f"Probs max diff: {np.abs(probs_rust - pe.numpy()).max()}")
    
    assert np.allclose(counts_rust, me.numpy(), atol=1e-5)
    assert np.allclose(probs_rust, pe.numpy(), atol=1e-4)

@pytest.mark.skipif(rustbpe is None, reason="rustbpe not installed")
def test_metabolism_cpu_parity():
    E = 8
    fatigue = torch.rand(E)
    energy = torch.rand(E)
    alpha_fatigue = torch.rand(E) * 0.1
    alpha_energy = torch.rand(E) * 0.1
    util = torch.rand(E)
    
    # Python reference
    f_py = fatigue.clone()
    e_py = energy.clone()
    f_py.mul_(1.0 - alpha_fatigue).add_(alpha_fatigue * util)
    e_py.mul_(1.0 - alpha_energy).add_(alpha_energy * (1.0 - util))
    
    # Rust version
    f_rust, e_rust = rustbpe.update_metabolism_cpu(
        fatigue.numpy(), energy.numpy(), alpha_fatigue.numpy(), alpha_energy.numpy(), util.numpy()
    )
    
    print("Comparing Metabolism...")
    print(f"Fatigue max diff: {np.abs(f_rust - f_py.numpy()).max()}")
    print(f"Energy max diff: {np.abs(e_rust - e_py.numpy()).max()}")
    
    assert np.allclose(f_rust, f_py.numpy(), atol=1e-5)
    assert np.allclose(e_rust, e_py.numpy(), atol=1e-5)

if __name__ == "__main__":
    if rustbpe:
        test_presyn_step_cpu_parity()
        test_moe_stats_cpu_parity()
        test_metabolism_cpu_parity()
        print("All tests passed!")
    else:
        print("rustbpe not installed, skipping tests")