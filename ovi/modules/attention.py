# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# --- MODIFICATIONS ---
# This file has been modified from the original version to implement
# automatic selection of the best available attention backend, prioritizing
# in the order of Sage > Flash > PyTorch SDPA.

import torch
import warnings

# --- Backend Availability Checks ---
try:
    from sageattention import sageattn
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    sageattn = None
    SAGE_ATTENTION_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_AVAILABLE = False


# --- Automatic Backend Selection (Runs Once on Import) ---
_SELECTED_BACKEND = "sdpa" # Default
if SAGE_ATTENTION_AVAILABLE:
    _SELECTED_BACKEND = "sage"
elif FLASH_ATTN_AVAILABLE:
    _SELECTED_BACKEND = "flash"

# --- Inform User of Selected Backend (Runs Once on Import) ---
if _SELECTED_BACKEND == "sage":
    print("✅  Using SAGE Attention (highest performance)", flush=True)
elif _SELECTED_BACKEND == "flash":
    print("✅  Using Flash Attention", flush=True)
else:
    print("Using PyTorch SDPA (Fallback)", flush=True)
    print("For better performance, install flash-attn and/or sageattention.", flush=True)


__all__ = [
    'attention',
    'attention_with_weights',
]


def attention(
    q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
    softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1),
    deterministic=False, dtype=torch.bfloat16, **kwargs # use kwargs to swallow unused args
):
    """
    Main attention dispatcher. Automatically uses the backend selected at startup.
    """
    # --- SAGE ATTENTION BACKEND ---
    if _SELECTED_BACKEND == "sage":
        out_dtype = q.dtype
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        if q_scale is not None:
            q = q * q_scale
        
        out = sageattn(q, k, v, tensor_layout="NHD", is_causal=causal, sm_scale=softmax_scale)
        return out.to(out_dtype)

    # --- FLASH ATTENTION BACKEND ---
    elif _SELECTED_BACKEND == "flash":
        # The original code called flash_attn_varlen_func, we replicate that here.
        b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype
        
        def half(x):
            return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(dtype)

        # preprocess query
        if q_lens is None:
            q = half(q.flatten(0, 1))
            q_lens = torch.tensor(
                [lq] * b, dtype=torch.int32).to(
                    device=q.device, non_blocking=True)
        else:
            q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

        # preprocess key, value
        if k_lens is None:
            k = half(k.flatten(0, 1))
            v = half(v.flatten(0, 1))
            k_lens = torch.tensor(
                [lk] * b, dtype=torch.int32).to(
                    device=k.device, non_blocking=True)
        else:
            k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
            v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if q_scale is not None:
            q = q * q_scale

        x = flash_attn.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq, max_seqlen_k=lk, dropout_p=dropout_p,
            softmax_scale=softmax_scale, causal=causal, window_size=window_size, deterministic=deterministic
        ).unflatten(0, (b, lq))

        return x.type(out_dtype)

    # --- PYTORCH SDPA BACKEND (FALLBACK) ---
    else:
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out


def attention_with_weights(
    q, k, v, q_lens=None, k_lens=None, softmax_scale=None, q_scale=None,
    causal=False, average_for_q=False, total_video_latent_frames=21
):
    """
    Compute attention with explicit attention weights for visualization.
    This function is unchanged.
    """
    out_dtype = q.dtype
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    
    if q_lens is None: q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else: q_lens = q_lens.to(q.device)
        
    if k_lens is None: k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else: k_lens = k_lens.to(k.device)
    
    if q_scale is not None: q = q * q_scale
    
    scale = softmax_scale if softmax_scale is not None else (q.size(-1) ** -0.5)
    scores = torch.einsum('blhd,bshd->bhls', q, k) * scale
    
    if causal:
        mask = torch.triu(torch.ones(lq, lk, device=q.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    k_mask = torch.arange(lk, device=k.device).unsqueeze(0) >= k_lens.unsqueeze(1)
    scores.masked_fill_(k_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
    
    q_mask = torch.arange(lq, device=q.device).unsqueeze(0) >= q_lens.unsqueeze(1)
    scores.masked_fill_(q_mask.unsqueeze(1).unsqueeze(3), float('-inf'))
    
    attn_weights = torch.softmax(scores, dim=-1)
    assert attn_weights.shape[0] == 1, "Batch size > 1 not supported for attention visualization."
    
    if average_for_q:
        avg_attn_weights = torch.max(attn_weights, dim=3)[0].mean(dim=(0, 1))
    else:
        B, H, Lq, Lk = attn_weights.shape
        per_frame_seq_len = Lk // total_video_latent_frames
        per_frame_aud_len = Lq // total_video_latent_frames
        avg_attn_weights = torch.zeros((Lk,), device=attn_weights.device, dtype=attn_weights.dtype)
        eps = 1e-8
        for i in range(total_video_latent_frames):
            start_idx_v, end_idx_v = i * per_frame_seq_len, (i + 1) * per_frame_seq_len
            start_idx_a, end_idx_a = i * per_frame_aud_len, (i + 1) * per_frame_aud_len
            attn_chunk = attn_weights[0, :, start_idx_a:end_idx_a, start_idx_v:end_idx_v]
            p = attn_chunk / (attn_chunk.sum(dim=-1, keepdim=True) + eps)
            entropy = -(p * (p + eps).log()).sum(dim=-1).mean(dim=1)
            saliency = 1.0 / (entropy + 1e-6)
            head_w = saliency / (saliency.sum() + eps)
            per_head = torch.amax(attn_chunk, dim=1)
            weighted = (per_head * head_w[:, None]).sum(dim=0)
            avg_attn_weights[start_idx_v:end_idx_v] = weighted
            
    out = torch.einsum('bhls,bshd->blhd', attn_weights, v)
    return out.to(out_dtype), avg_attn_weights.to(out_dtype)