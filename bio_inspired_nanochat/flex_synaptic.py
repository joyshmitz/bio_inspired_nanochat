"""
Fused Synaptic Attention using PyTorch FlexAttention.
Requires PyTorch >= 2.5.0

This module replaces the memory-heavy (B, H, T, T) materialization of biological biases
with an on-the-fly "score mod" that fuses directly into the FlashAttention kernel.

Key Benefits:
1. O(N) memory usage (vs O(N^2)).
2. FlashAttention speeds.
3. Automatic backward pass differentiation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def _compile_synaptic_bias(presyn_state, cfg):
    """
    Creates a compiled score_mod for flex_attention.
    
    The biological bias is:
    bias[b, h, q, k] = log(release_frac[b, h, k] * qamp[b, h, k] + eps) - barrier(q, k)
    
    Note: 
    - presyn_state tensors (RRP, Calcium, etc) are (B, H, T).
    - They are pre-computed *before* attention (the "release" step).
    - Here we just map them to the (q, k) grid efficiently.
    """
    
    # We capture these tensors by closure. They must be contiguous.
    # State tensors are (B, H, T).
    # release_frac and qamp are derived from state.
    
    # Pre-compute the per-token modulation factors to keep the score_mod lightweight.
    # This effectively vectorizes the "Pass 3" of the original Triton kernel.
    
    # 1. Compute Release Probability & QAmp from updated state
    # We assume presyn_state has been updated by the dynamics step already.
    
    # Unpack state
    # In flex_attention, score_mod receives (score, b, h, q_idx, kv_idx).
    # We need to look up values from our pre-computed state tensors.
    
    # To make this fast, we pre-calculate a single (B, H, T) tensor representing 
    # the "availability" of the key token.
    
    # From synaptic.py logic:
    # syn_logit = log(release_frac * qamp + eps) - barrier
    # release_frac = (fuse_p * rrp_refill) * scale
    
    # This dependency on (q, k) interaction (dot product inside fuse_p) 
    # makes it hard to purely pre-calculate per K.
    # However, biological release is often modeled as Pre-synaptic (Key) property.
    # If we simplify fuse_p to be primarily driven by Calcium (which is per-token),
    # we can pre-compute a "Synaptic Availability" vector S[k].
    
    # COMPLEXITY: The original math has `dot(q, k)` inside the release logic.
    # This couples Q and K biologically.
    # To use FlexAttention efficiently, we usually want decoupled terms or simple arithmetic.
    # If we keep the full dot product dependency, we are essentially re-implementing attention.
    
    # APPROXIMATION for Speed:
    # The `d_bilin` term (sigmoid(q.k)) modulates release.
    # Since FlexAttention computes `score = q.k`, we can use `score` directly!
    
    # Let's define the score mod.
    
    def synaptic_score_mod(score, b, h, q_idx, kv_idx):
        # score is (q @ k) / sqrt(d) scaling is done by flex_attention? 
        # flex_attention usually takes raw scores? No, it usually expects scaled scores if we want.
        # But typically score = q @ k.
        
        # 1. Retrieve pre-computed biological factors for the KEY (kv_idx)
        # We need a way to pass the state tensors into this compiled function.
        # FlexAttention supports capturing tensors.
        
        # We need to index into tensors of shape (B, H, T).
        # Triton/FlexAttn indexing: tensor[b, h, kv_idx]
        
        # Constants
        barrier_strength = cfg.barrier_strength
        epsilon = cfg.epsilon
        
        # Load state values for the Key (Presynaptic terminal)
        # These must be pre-calculated "potentials"
        # Let's assume we pre-calculate a (B, H, T) tensor `base_release_prob` 
        # which combines Calcium, RRP, and Complexin *ignoring* the specific Q.
        # bias_k = presyn_state['base_potential'][b, h, kv_idx]
        
        # The original formula:
        # fuse_p = sigmoid(fuse_logit_base) * sigmoid(score)
        # release = fuse_p * rrp_refill * scale
        
        # Decomposed:
        # release = sigmoid(fuse_logit_base) * rrp_refill * scale * sigmoid(score)
        # Let `KeyFactor` = sigmoid(fuse_logit_base) * rrp_refill * scale
        # This `KeyFactor` is purely a function of the Key's biological history!
        # We can pre-compute `KeyFactor` (B, H, T) in O(T).
        
        # Then inside attention:
        # release_frac = KeyFactor[k] * sigmoid(score)
        # term = log(release_frac * qamp + eps)
        
        # This is expressible in FlexAttention!
        
        # Capture pre-computed tensors
        # key_factor: (B, H, T)
        # qamp: (B, H, T) -> Wait, QAmp depends on Energy of the KEY? Yes (e_new of key).
        
        kf = key_factor[b, h, kv_idx]
        qa = qamp[b, h, kv_idx]
        
        # Sigmoid(score) approximation or actual op?
        # score is q@k. 
        # We typically divide by sqrt(d) before sigmoid.
        # flex_attention passes `score` which is usually unscaled? 
        # Documentation says "score is the attention score".
        # We can apply scaling inside mod if needed.
        
        # release = kf * torch.sigmoid(score)
        # val = release * qa
        # log_val = torch.log(val + epsilon)
        
        # Barrier penalty
        dist = torch.abs(q_idx - kv_idx)
        # We need T to normalize distance. Can capture T as scalar.
        dist_pen = barrier_strength * (dist / T_float)
        
        return score + log_val - dist_pen

    return synaptic_score_mod

class SynapticFlexAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # We need to ensure state tensors are registered buffers or similar?
        # No, they are passed per forward pass.

    def precompute_bio_factors(self, state, cfg):
        """
        Computes the O(N) biological factors needed for the O(N^2) attention map.
        Returns:
            key_factor: (B, H, T) - The 'readiness' of the key to release.
            qamp: (B, H, T) - The quantal amplitude of the key.
        """
        # Extract state
        c = state['c'] # Calcium
        rrp = state['rrp'] # Vesicle pool
        cl = state['cl'] # Clamp
        sn = state['sn'] # Priming (SNARE)
        amp = state['amp'] # AMPA/Quantal size
        
        # 1. Mix Prob (Synaptotagmin & Complexin)
        # p1 = sigmoid(syt1_slope * (c - 0.55))
        p1 = torch.sigmoid(cfg.syt1_slope * (c - 0.55))
        # p7 = sigmoid(syt7_slope * (c - 0.25))
        p7 = torch.sigmoid(cfg.syt7_slope * (c - 0.25))
        # p = p1*0.8 + p7*0.2 + doc2_gain * sigmoid(4*(c-0.12))
        p = p1 * 0.8 + p7 * 0.2 + cfg.doc2_gain * torch.sigmoid(4 * (c - 0.12))
        # Clamp factor: 1 / (1 + exp((thresh - c)*8))
        clamp_factor = 1.0 / (1.0 + torch.exp((cfg.cpx_thresh - c) * 8.0))
        
        # mix_prob = p * clamp_factor * sn
        mix_prob = p * clamp_factor * sn
        mix_prob = torch.clamp(mix_prob, 0, 0.999)
        
        # 2. Key Factor
        # release = mix_prob * rrp
        key_factor = mix_prob * rrp
        
        # 3. QAmp
        # Corresponds to 'amp' state
        qamp = amp
        
        return key_factor, qamp

    def forward(self, q, k, v, presyn_state, block_mask=None):
        """
        q, k, v: (B, H, T, D)
        presyn_state: Dict of (B, H, T)
        """
        B, H, T, D = q.shape
        
        # 1. Pre-compute biological factors O(N)
        key_factor, qamp = self.precompute_bio_factors(presyn_state, self.config)
        
        # 2. Define the score modifier
        # We need to wrap this in a closure to capture tensors
        # Capture constants to avoid lookup inside
        barrier_strength = self.config.barrier_strength
        epsilon = self.config.epsilon
        
        def score_mod(score, b, h, q_idx, kv_idx):
            # Scaling score (FlexAttention typically expects users to handle scale if modifying?)
            # Default attention is QK^T. We usually want QK^T / sqrt(D).
            scaled_score = score / (D ** 0.5)
            
            # Bio modulation
            # kf[b, h, kv_idx]
            kf_val = key_factor[b, h, kv_idx]
            qa_val = qamp[b, h, kv_idx]
            
            # release = kf * sigmoid(scaled_score)
            release = kf_val * torch.sigmoid(scaled_score)
            
            # log_term = log(release * qa + eps)
            bio_bias = torch.log(release * qa_val + epsilon)
            
            # Barrier
            # Ensure T is treated as tensor for division if symbolic
            dist = (q_idx - kv_idx).abs() / T
            barrier = barrier_strength * dist
            
            return scaled_score + bio_bias - barrier

        # 3. Run FlexAttention
        # block_mask can be used for causal masking
        out = flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)
        
        return out
