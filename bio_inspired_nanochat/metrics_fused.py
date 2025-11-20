import torch
import torch.distributed as dist

def accumulate_metrics_fused(moe_ctx_list, num_experts):
    """
    Optimized reduction of MoE metrics using PyTorch primitives that map to efficient CUDA kernels.
    Avoids iterating over batch items in Python.
    
    Args:
        moe_ctx_list: List of dicts containing {'gates': Tensor, 'indices': Tensor}
                      gates: (B, T, k), indices: (B, T, k)
        num_experts: int
        
    Returns:
        counts: (E,)
        prob_sums: (E,)
    """
    device = moe_ctx_list[0]['gates'].device
    dtype = moe_ctx_list[0]['gates'].dtype
    
    total_counts = torch.zeros(num_experts, device=device, dtype=dtype)
    total_probs = torch.zeros(num_experts, device=device, dtype=dtype)
    
    for ctx in moe_ctx_list:
        gates = ctx['gates'] # (B, T, k)
        indices = ctx['indices'] # (B, T, k)
        
        # Flatten
        gates_flat = gates.view(-1)
        indices_flat = indices.view(-1)
        
        # Scatter Add (Atomic accumulation)
        # We want to sum 'gates' into 'probs' at 'indices'
        # index_add_ expects indices to be same size as source dim
        
        # Handle out-of-bound indices (if any, though they shouldn't be)
        mask = (indices_flat >= 0) & (indices_flat < num_experts)
        valid_indices = indices_flat[mask]
        valid_gates = gates_flat[mask]
        
        # Sum probabilities
        total_probs.index_add_(0, valid_indices, valid_gates)
        
        # Count occurrences (sum of 1.0s)
        ones = torch.ones_like(valid_gates)
        total_counts.index_add_(0, valid_indices, ones)
        
    return total_counts, total_probs

def async_log_metrics(logger, model, step):
    """
    Extracts and logs metrics asynchronously to avoid blocking training loop.
    Should be called via a ThreadPool or similar if extreme perf is needed,
    but even running this optimized reduction sync is fast.
    """
    if logger is None:
        return

    moe_stats = []
    for name, module in model.named_modules():
        if hasattr(module, 'last_ctx') and module.last_ctx:
            moe_stats.append(module.last_ctx)
            
    if not moe_stats:
        return
        
    # Assume homogenous MOE for now
    E = 8 # default, should extract from module
    
    counts, probs = accumulate_metrics_fused(moe_stats, E)
    
    # Sync for logging (this is the only stall, can be threaded)
    counts_cpu = counts.cpu().numpy()
    probs_cpu = probs.cpu().numpy()
    
    # ... Logging logic ...
