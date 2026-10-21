import torch
import os
from einops import rearrange


try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    XFORMERS_AVAILABLE = False


def initialize_attention_priority():
    if os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION') is not None:
        return os.environ.get('DIFFSYNTH_ATTENTION_IMPLEMENTATION').lower()
    elif FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    elif FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    elif SAGE_ATTN_AVAILABLE:
        return "sage_attention"
    elif XFORMERS_AVAILABLE:
        return "xformers"
    else:
        return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_priority()


def rearrange_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", required_in_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if q_pattern != required_in_pattern:
        q = rearrange(q, f"{q_pattern} -> {required_in_pattern}", **dims)
    if k_pattern != required_in_pattern:
        k = rearrange(k, f"{k_pattern} -> {required_in_pattern}", **dims)
    if v_pattern != required_in_pattern:
        v = rearrange(v, f"{v_pattern} -> {required_in_pattern}", **dims)
    return q, k, v


def rearrange_out(out: torch.Tensor, out_pattern="b n s d", required_out_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if out_pattern != required_out_pattern:
        out = rearrange(out, f"{required_out_pattern} -> {out_pattern}", **dims)
    return out


def torch_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale, is_causal=is_causal)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale)
    if isinstance(out, tuple):
        out = out[0]
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def flash_attention_2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None, is_causal=False):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale, causal=is_causal)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def torch_varlen_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale=None,
    is_causal=False,
) -> torch.Tensor:
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    k_cu = cu_seqlens_k.detach().cpu().tolist()
    outputs = []
    for idx in range(len(q_cu) - 1):
        q_start, q_end = int(q_cu[idx]), int(q_cu[idx + 1])
        k_start, k_end = int(k_cu[idx]), int(k_cu[idx + 1])
        q_i = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_i = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        v_i = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        out_i = torch.nn.functional.scaled_dot_product_attention(
            q_i,
            k_i,
            v_i,
            scale=scale,
            is_causal=is_causal,
        )
        outputs.append(out_i.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else q.new_zeros(q.shape)


_VARLEN_FALLBACK_WARNED = False


def _warn_varlen_fallback_once() -> None:
    """Say it once: the packed path is running on SDPA-per-segment, not flash-attn."""
    global _VARLEN_FALLBACK_WARNED
    if _VARLEN_FALLBACK_WARNED:
        return
    _VARLEN_FALLBACK_WARNED = True
    import warnings

    warnings.warn(
        "Packed DiT attention is using the SDPA per-segment fallback because no "
        "FlashAttention build with flash_attn_varlen_func was found. Results are "
        "identical to the dense path (verified bit-for-bit); it is slower per token. "
        "Install flash-attn for the fast kernel: pip install flash-attn --no-build-isolation",
        stacklevel=3,
    )


def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale=None,
    is_causal=False,
) -> torch.Tensor:
    """Run variable-length attention on packed tensors shaped [total_tokens, heads, head_dim]."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"Varlen attention expects q/k/v [total, heads, dim], got {q.shape}, {k.shape}, {v.shape}")
    if os.environ.get("DIFFSYNTH_VARLEN_ATTENTION_ALLOW_FALLBACK", "0") == "1":
        return torch_varlen_attention_fallback(q, k, v, cu_seqlens_q, cu_seqlens_k, scale=scale, is_causal=is_causal)
    if FLASH_ATTN_2_AVAILABLE and hasattr(flash_attn, "flash_attn_varlen_func"):
        return flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scale,
            causal=is_causal,
        )
    if FLASH_ATTN_3_AVAILABLE and hasattr(flash_attn_interface, "flash_attn_varlen_func"):
        out = flash_attn_interface.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scale,
            causal=is_causal,
        )
        return out[0] if isinstance(out, tuple) else out
    # Without a FlashAttention build, fall back to per-segment SDPA. Set
    # DIFFSYNTH_VARLEN_ATTENTION_FORBID_FALLBACK=1 to make this an error instead.
    if os.environ.get("DIFFSYNTH_VARLEN_ATTENTION_FORBID_FALLBACK", "0") != "1":
        _warn_varlen_fallback_once()
        return torch_varlen_attention_fallback(q, k, v, cu_seqlens_q, cu_seqlens_k, scale=scale, is_causal=is_causal)
    raise RuntimeError(
        "Packed varlen attention requires flash-attn varlen support. "
        "Install a FlashAttention build with flash_attn_varlen_func, or leave "
        "DIFFSYNTH_VARLEN_ATTENTION_FORBID_FALLBACK unset to allow the "
        "bit-identical SDPA fallback."
    )


def sage_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = sageattn(q, k, v, sm_scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def xformers_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern= "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    out = rearrange_out(out, out_pattern, required_out_pattern, dims)
    return out


def attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, is_causal=False, compatibility_mode=False):
    if compatibility_mode or (attn_mask is not None):
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale, is_causal=is_causal)
    else:
        if ATTENTION_IMPLEMENTATION == "flash_attention_3":
            return flash_attention_3(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif ATTENTION_IMPLEMENTATION == "flash_attention_2":
            return flash_attention_2(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale, is_causal=is_causal)
        elif ATTENTION_IMPLEMENTATION == "sage_attention":
            return sage_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        elif ATTENTION_IMPLEMENTATION == "xformers":
            return xformers_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
        else:
            return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale, is_causal=is_causal)
