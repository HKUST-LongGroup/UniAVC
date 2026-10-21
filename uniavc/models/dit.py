import math
import functools
import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Tuple, Callable
import numpy as np
import torch
from einops import rearrange
from .common import rms_norm
from ..core.attention.attention import attention_forward, flash_attention_varlen
from ..core import gradient_checkpoint_forward

# Token-role embedding ablation. Gated at the consumption site, not by zeroing the parameter:
# `assign=True` load rebinds the model attribute but not the preprocessors' captured copy.
# Set DISABLE_TOKEN_TYPE_EMBEDDING=1 to omit the term.
def _token_role_embedding_disabled() -> bool:
    """True when the token-role (target/reference) embedding must be omitted from the forward."""
    return os.environ.get("DISABLE_TOKEN_TYPE_EMBEDDING", "0").strip().lower() in ("1", "true", "on", "yes")


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
    ):
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = torch.nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = torch.nn.SiLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim

        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class PixArtAlphaCombinedTimestepSizeEmbeddings(torch.nn.Module):
    """
    For PixArt-Alpha.
    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
    ):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)
        return timesteps_emb











ADALN_NUM_BASE_PARAMS = 6
# Cross-attention AdaLN adds 3 more (scale, shift, gate) for the CA norm.
ADALN_NUM_CROSS_ATTN_PARAMS = 3


def adaln_embedding_coefficient(cross_attention_adaln: bool) -> int:
    """Total number of AdaLN parameters per block."""
    return ADALN_NUM_BASE_PARAMS + (ADALN_NUM_CROSS_ATTN_PARAMS if cross_attention_adaln else 0)


class AdaLayerNormSingle(torch.nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).
    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).
    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(embedding_dim, embedding_coefficient * embedding_dim, bias=True)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class RopeType(Enum):
    INTERLEAVED = "interleaved"
    SPLIT = "split"


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    rope_type: RopeType = RopeType.INTERLEAVED,
) -> torch.Tensor:
    if rope_type == RopeType.INTERLEAVED:
        return apply_interleaved_rotary_emb(input_tensor, *freqs_cis)
    elif rope_type == RopeType.SPLIT:
        return apply_split_rotary_emb(input_tensor, *freqs_cis)
    else:
        raise ValueError(f"Invalid rope type: {rope_type}")



def apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs

    return out


def apply_split_rotary_emb(
    input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    needs_reshape = False
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]

    output = split_input * cos_freqs.unsqueeze(-2)
    first_half_output = output[..., :1, :]
    second_half_output = output[..., 1:, :]

    first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)
    second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)

    output = rearrange(output, "... d r -> ... (d r)")
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)

    return output


@functools.lru_cache(maxsize=5)
def generate_freq_grid_np(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta

    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            np.log(start) / np.log(theta),
            np.log(end) / np.log(theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32)


@functools.lru_cache(maxsize=5)
def generate_freq_grid_pytorch(
    positional_embedding_theta: float, positional_embedding_max_pos_count: int, inner_dim: int
) -> torch.Tensor:
    theta = positional_embedding_theta
    start = 1
    end = theta
    n_elem = 2 * positional_embedding_max_pos_count

    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
        )
    )
    indices = indices.to(dtype=torch.float32)

    indices = indices * math.pi / 2

    return indices


def get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int]) -> torch.Tensor:
    n_pos_dims = indices_grid.shape[1]
    assert n_pos_dims == len(max_pos), (
        f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})"
    )
    fractional_positions = torch.stack(
        [indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)],
        dim=-1,
    )
    return fractional_positions


def generate_freqs(
    indices: torch.Tensor, indices_grid: torch.Tensor, max_pos: list[int], use_middle_indices_grid: bool
) -> torch.Tensor:
    if use_middle_indices_grid:
        assert len(indices_grid.shape) == 4
        assert indices_grid.shape[-1] == 2
        indices_grid_start, indices_grid_end = indices_grid[..., 0], indices_grid[..., 1]
        indices_grid = (indices_grid_start + indices_grid_end) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]

    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)

    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    return freqs


def split_freqs_cis(freqs: torch.Tensor, pad_size: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()

    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)

    # Reshape freqs to be compatible with multi-head attention
    b = cos_freq.shape[0]
    t = cos_freq.shape[1]

    cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1)
    sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1)

    cos_freq = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)
    sin_freq = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)
    return cos_freq, sin_freq


def interleaved_freqs_cis(freqs: torch.Tensor, pad_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq


def precompute_freqs_cis(
    indices_grid: torch.Tensor,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | None = None,
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    rope_type: RopeType = RopeType.INTERLEAVED,
    freq_grid_generator: Callable[[float, int, int, torch.device], torch.Tensor] = generate_freq_grid_pytorch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    indices = freq_grid_generator(theta, indices_grid.shape[1], dim)
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)

    if rope_type == RopeType.SPLIT:
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        # 2 because of cos and sin by 3 for (t, x, y), 1 for temporal only
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
    return cos_freq.to(out_dtype), sin_freq.to(out_dtype)


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: RopeType = RopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        # (B, T, 1) zeroes queries that had no legal key; softmax over a fully-masked row
        # averages uniformly rather than returning zero. None = every row has a key.
        row_valid: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe, self.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

        # Reshape for attention_forward using unflatten
        q = q.unflatten(-1, (self.heads, self.dim_head))
        k = k.unflatten(-1, (self.heads, self.dim_head))
        v = v.unflatten(-1, (self.heads, self.dim_head))

        out = attention_forward(
            q=q,
            k=k,
            v=v,
            q_pattern="b s n d",
            k_pattern="b s n d",
            v_pattern="b s n d",
            out_pattern="b s n d",
            attn_mask=mask
        )

        # Reshape back to original format
        out = out.flatten(2, 3)

        # Queries with no legal key contribute nothing (see row_valid). Applied before the
        # gate and to_out so the packed path's zeros flow through the identical tail.
        if row_valid is not None:
            out = out * row_valid.to(out.dtype)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            # Reshape to (B, T, H, D) for per-head gating
            out = out.view(b, t, self.heads, self.dim_head)
            # Apply gating: 2 * sigmoid(x) so that zero-init gives identity (2 * 0.5 = 1.0)
            gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
            out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
            # Reshape back to (B, T, H*D)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)

    def forward_packed_self(
        self,
        x: torch.Tensor,
        *,
        pe: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Packed self-attention expects x [sum_N, D], got {tuple(x.shape)}")
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q.unsqueeze(0), pe, self.rope_type).squeeze(0)
            k = apply_rotary_emb(k.unsqueeze(0), pe, self.rope_type).squeeze(0)

        q = q.unflatten(-1, (self.heads, self.dim_head))
        k = k.unflatten(-1, (self.heads, self.dim_head))
        v = v.unflatten(-1, (self.heads, self.dim_head))

        out = flash_attention_varlen(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=int(max_seqlen),
            max_seqlen_k=int(max_seqlen),
        )
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
        out = out.flatten(1, 2)
        return self.to_out(out)

    def forward_packed_cross(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Packed dual-stream cross-attention.

        Queries come from ``x`` (one packed modality, segmented by ``cu_seqlens_q``) and
        keys/values from ``context`` (the other packed modality, segmented by
        ``cu_seqlens_k``). flash varlen with distinct q/k cu_seqlens makes this
        block-diagonal: query-segment ``i`` attends only to key-segment ``i``. The two
        ``cu_seqlens`` must have the same number of segments (same N samples).
        """
        if x.ndim != 2 or context.ndim != 2:
            raise ValueError(f"Packed cross-attention expects x/context [sum_N, D], got {tuple(x.shape)}, {tuple(context.shape)}")
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q.unsqueeze(0), pe, self.rope_type).squeeze(0)
        if k_pe is not None:
            k = apply_rotary_emb(k.unsqueeze(0), k_pe, self.rope_type).squeeze(0)

        q = q.unflatten(-1, (self.heads, self.dim_head))
        k = k.unflatten(-1, (self.heads, self.dim_head))
        v = v.unflatten(-1, (self.heads, self.dim_head))

        out = flash_attention_varlen(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
        )
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
        out = out.flatten(1, 2)
        return self.to_out(out)

    def forward_packed_self_split_ref(
        self,
        x: torch.Tensor,
        *,
        pe: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        target_token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Packed split self-attention expects x [sum_N, D], got {tuple(x.shape)}")
        target_token_lengths = torch.as_tensor(target_token_lengths, device=x.device, dtype=torch.long)
        if target_token_lengths.ndim != 1:
            raise ValueError("target_token_lengths must be a 1-D tensor or sequence")

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q.unsqueeze(0), pe, self.rope_type).squeeze(0)
            k = apply_rotary_emb(k.unsqueeze(0), pe, self.rope_type).squeeze(0)

        q = q.unflatten(-1, (self.heads, self.dim_head))
        k = k.unflatten(-1, (self.heads, self.dim_head))
        v = v.unflatten(-1, (self.heads, self.dim_head))

        cu = cu_seqlens.detach().cpu().tolist()
        if target_token_lengths.numel() != len(cu) - 1:
            raise ValueError(
                f"target_token_lengths must contain {len(cu) - 1} values, got {target_token_lengths.numel()}"
            )

        outputs = []
        for idx, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
            start = int(start)
            end = int(end)
            segment_len = end - start
            target_len = int(target_token_lengths[idx].item())
            if target_len <= 0 or target_len > segment_len:
                raise ValueError("Each target_token_lengths value must be in the range [1, segment_length]")

            if target_len == segment_len:
                out_i = flash_attention_varlen(
                    q=q[start:end],
                    k=k[start:end],
                    v=v[start:end],
                    cu_seqlens_q=torch.tensor([0, segment_len], device=x.device, dtype=torch.int32),
                    cu_seqlens_k=torch.tensor([0, segment_len], device=x.device, dtype=torch.int32),
                    max_seqlen_q=int(max_seqlen),
                    max_seqlen_k=int(max_seqlen),
                )
                outputs.append(out_i)
                continue

            split = start + target_len
            source_len = end - split
            target_out = flash_attention_varlen(
                q=q[start:split],
                k=k[start:end],
                v=v[start:end],
                cu_seqlens_q=torch.tensor([0, target_len], device=x.device, dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, segment_len], device=x.device, dtype=torch.int32),
                max_seqlen_q=target_len,
                max_seqlen_k=segment_len,
            )
            source_out = flash_attention_varlen(
                q=q[split:end],
                k=k[split:end],
                v=v[split:end],
                cu_seqlens_q=torch.tensor([0, source_len], device=x.device, dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, source_len], device=x.device, dtype=torch.int32),
                max_seqlen_q=source_len,
                max_seqlen_k=source_len,
            )
            outputs.append(torch.cat([target_out, source_out], dim=0))

        out = torch.cat(outputs, dim=0) if outputs else x.new_zeros((0, self.heads, self.dim_head))
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
        out = out.flatten(1, 2)
        return self.to_out(out)

    def forward_packed_cross_split_ref(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        q_target_token_lengths: torch.Tensor | None,
        k_target_token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Reference-aware variant of ``forward_packed_cross``.

        Both packed streams lay each segment out as ``[target | reference]``. Plain
        cross-attention lets a CLEAN reference query read the OTHER modality's NOISED
        target, which is the cross-modal analogue of the leak
        ``forward_packed_self_split_ref`` closes within a stream. Here, per segment:

            target queries    -> whole key segment       [target | reference]
            reference queries -> key reference span only [reference]

        Reference-to-reference is deliberately kept (both sides are clean and mutually
        informative), matching the self-attention precedent above. Segments whose query
        side carries no reference (``q_target_len == q_seg_len``, e.g. pure t2av) collapse
        to a single full attention call, so generation-only batches are numerically
        identical to ``forward_packed_cross``. When the KEY side carries no reference but
        the query side does, reference queries have no legal keys at all and receive
        zeros rather than an empty-range flash-attn call.
        """
        if x.ndim != 2 or context.ndim != 2:
            raise ValueError(
                f"Packed cross-attention expects x/context [sum_N, D], got {tuple(x.shape)}, {tuple(context.shape)}"
            )
        k_target_token_lengths = torch.as_tensor(k_target_token_lengths, device=x.device, dtype=torch.long)
        if k_target_token_lengths.ndim != 1:
            raise ValueError("k_target_token_lengths must be a 1-D tensor or sequence")
        if q_target_token_lengths is not None:
            q_target_token_lengths = torch.as_tensor(q_target_token_lengths, device=x.device, dtype=torch.long)
            if q_target_token_lengths.ndim != 1:
                raise ValueError("q_target_token_lengths must be a 1-D tensor or sequence")

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q.unsqueeze(0), pe, self.rope_type).squeeze(0)
        if k_pe is not None:
            k = apply_rotary_emb(k.unsqueeze(0), k_pe, self.rope_type).squeeze(0)

        q = q.unflatten(-1, (self.heads, self.dim_head))
        k = k.unflatten(-1, (self.heads, self.dim_head))
        v = v.unflatten(-1, (self.heads, self.dim_head))

        cu_q = cu_seqlens_q.detach().cpu().tolist()
        cu_k = cu_seqlens_k.detach().cpu().tolist()
        num_segments = len(cu_q) - 1
        if len(cu_k) - 1 != num_segments:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same number of segments")
        if k_target_token_lengths.numel() != num_segments:
            raise ValueError(
                f"k_target_token_lengths must contain {num_segments} values, got {k_target_token_lengths.numel()}"
            )
        if q_target_token_lengths is not None and q_target_token_lengths.numel() != num_segments:
            raise ValueError(
                f"q_target_token_lengths must contain {num_segments} values, got {q_target_token_lengths.numel()}"
            )

        def _attend(q_part: torch.Tensor, k_part: torch.Tensor, v_part: torch.Tensor) -> torch.Tensor:
            q_len = int(q_part.shape[0])
            k_len = int(k_part.shape[0])
            return flash_attention_varlen(
                q=q_part,
                k=k_part,
                v=v_part,
                cu_seqlens_q=torch.tensor([0, q_len], device=x.device, dtype=torch.int32),
                cu_seqlens_k=torch.tensor([0, k_len], device=x.device, dtype=torch.int32),
                max_seqlen_q=q_len,
                max_seqlen_k=k_len,
            )

        outputs = []
        for idx in range(num_segments):
            q_start, q_end = int(cu_q[idx]), int(cu_q[idx + 1])
            k_start, k_end = int(cu_k[idx]), int(cu_k[idx + 1])
            q_seg_len = q_end - q_start
            k_seg_len = k_end - k_start
            k_target_len = int(k_target_token_lengths[idx].item())
            if not 0 < k_target_len <= k_seg_len:
                raise ValueError("Each k_target_token_lengths value must be in the range [1, key segment length]")
            q_target_len = q_seg_len if q_target_token_lengths is None else int(q_target_token_lengths[idx].item())
            if not 0 < q_target_len <= q_seg_len:
                raise ValueError("Each q_target_token_lengths value must be in the range [1, query segment length]")

            # No query-side reference (pure generation): one unrestricted call, exactly
            # what forward_packed_cross would compute for this segment.
            if q_target_len == q_seg_len:
                outputs.append(_attend(q[q_start:q_end], k[k_start:k_end], v[k_start:k_end]))
                continue

            q_split = q_start + q_target_len
            target_out = _attend(q[q_start:q_split], k[k_start:k_end], v[k_start:k_end])
            if k_target_len == k_seg_len:
                # Key side has no reference span -> reference queries have nothing legal
                # to read. Emit zeros; the residual add downstream makes this a no-op.
                ref_out = target_out.new_zeros((q_seg_len - q_target_len,) + target_out.shape[1:])
            else:
                k_split = k_start + k_target_len
                ref_out = _attend(q[q_split:q_end], k[k_split:k_end], v[k_split:k_end])
            outputs.append(torch.cat([target_out, ref_out], dim=0))

        out = torch.cat(outputs, dim=0) if outputs else x.new_zeros((0, self.heads, self.dim_head))
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
        out = out.flatten(1, 2)
        return self.to_out(out)


class PixArtAlphaTextProjection(torch.nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.
    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features: int, hidden_size: int, out_features: int | None = None, act_fn: str = "gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = torch.nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = torch.nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = torch.nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states



@dataclass(frozen=True)
class PackedTransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: torch.Tensor
    visual_cu_seqlens: torch.Tensor
    max_visual_seqlen: int
    segment_ids: torch.Tensor
    enabled: bool
    prompt_timestep: torch.Tensor | None = None
    target_token_lengths: torch.Tensor | None = None
    # True walls the clean in-context reference off from the noised target, as in training.
    split_ref_attention: bool = True
    # True extends that wall to AV cross-attention: a clean reference token does not read the
    # other modality's noised target. No effect without a reference.
    split_ref_cross_attention: bool = True
    # AV cross-attention (joint packed t2av/tav2av). cross_positional_embeddings is
    # per-token (time axis); the scale/shift/gate timesteps are PER-SAMPLE [num_samples, ...]
    # and expanded to per-token via segment_ids inside the block.
    cross_positional_embeddings: torch.Tensor | None = None
    cross_scale_shift_timestep: torch.Tensor | None = None
    cross_gate_timestep: torch.Tensor | None = None
    # Per-sample [num_samples] mask on the AV cross-attention write into this stream.
    # None = all ones.
    av_cross_gate_mask: torch.Tensor | None = None






def checkpoint_packed_av_block(
    block,
    video: PackedTransformerArgs,
    audio: PackedTransformerArgs,
    *,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
) -> tuple[PackedTransformerArgs, PackedTransformerArgs]:
    if not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
        return block.forward_packed_av(video, audio)

    def custom_forward(vx: torch.Tensor, ax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_out, a_out = block.forward_packed_av(replace(video, x=vx), replace(audio, x=ax))
        return v_out.x, a_out.x

    vx, ax = gradient_checkpoint_forward(
        custom_forward,
        use_gradient_checkpointing,
        use_gradient_checkpointing_offload,
        video.x,
        audio.x,
    )
    return replace(video, x=vx), replace(audio, x=ax)


class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: RopeType,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
        visual_token_type_embedding: torch.nn.Parameter | None = None,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.caption_projection = caption_projection
        self.prompt_adaln = prompt_adaln
        self.visual_token_type_embedding = visual_token_type_embedding

    def _prepare_timestep(
        self, timestep: torch.Tensor, adaln: AdaLayerNormSingle, batch_size: int, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings."""
        timestep_scaled = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return timestep, embedded_timestep


    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        """Prepare attention mask."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask.to(x_dtype) - 1).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max


    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare positional embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        return pe


    def prepare_packed(
        self,
        *,
        visual_tokens: torch.Tensor,
        positions: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        timesteps: torch.Tensor,
        sigma: torch.Tensor,
        visual_cu_seqlens: torch.Tensor,
        max_visual_seqlen: int,
        segment_ids: torch.Tensor,
        target_token_lengths: torch.Tensor | None = None,
        split_ref_attention: bool = True,
    ) -> PackedTransformerArgs:
        if visual_tokens.ndim != 2:
            raise ValueError(f"Packed visual tokens must have shape [sum_N, D], got {tuple(visual_tokens.shape)}")
        batch_size = int(visual_cu_seqlens.numel() - 1)
        if batch_size <= 0:
            raise ValueError("Packed visual batch must contain at least one segment")
        if int(visual_cu_seqlens[-1].item()) != int(visual_tokens.shape[0]):
            raise ValueError("visual_cu_seqlens[-1] must equal the number of packed visual tokens")
        if segment_ids.shape != (visual_tokens.shape[0],):
            raise ValueError(f"segment_ids must have shape [{visual_tokens.shape[0]}], got {tuple(segment_ids.shape)}")
        if context.shape[0] != batch_size:
            raise ValueError(f"Packed context batch {context.shape[0]} does not match {batch_size} visual segments")

        x = self.patchify_proj(visual_tokens)
        segment_lengths = visual_cu_seqlens[1:] - visual_cu_seqlens[:-1]
        target_token_lengths_tensor = None
        if target_token_lengths is not None:
            target_token_lengths_tensor = torch.as_tensor(target_token_lengths, device=visual_tokens.device, dtype=torch.long)
            if target_token_lengths_tensor.ndim != 1:
                raise ValueError("target_token_lengths must be a 1-D tensor or sequence")
            if target_token_lengths_tensor.numel() != segment_lengths.numel():
                raise ValueError(
                    f"target_token_lengths must contain {segment_lengths.numel()} values, "
                    f"got {target_token_lengths_tensor.numel()}"
                )
            if torch.any(target_token_lengths_tensor <= 0) or torch.any(target_token_lengths_tensor > segment_lengths.to(device=target_token_lengths_tensor.device)):
                raise ValueError("Each target_token_lengths value must be in the range [1, segment_length]")
        # DISABLE_TOKEN_TYPE_EMBEDDING ablates this term (see _token_role_embedding_disabled).
        if (
            target_token_lengths_tensor is not None
            and self.visual_token_type_embedding is not None
            and not _token_role_embedding_disabled()
        ):
            type_ids = torch.zeros(visual_tokens.shape[0], device=visual_tokens.device, dtype=torch.long)
            cu = visual_cu_seqlens.detach().cpu().tolist()
            for idx, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
                split = int(start) + int(target_token_lengths_tensor[idx].item())
                type_ids[split:int(end)] = 1
            type_embedding = self.visual_token_type_embedding.to(device=x.device, dtype=x.dtype)
            x = x + type_embedding[type_ids]
        raw_timesteps = timesteps.reshape(-1)
        raw_sigma = sigma.reshape(-1)
        if raw_timesteps.numel() != batch_size:
            raise ValueError(f"Packed timesteps must contain {batch_size} values, got {raw_timesteps.numel()}")
        if raw_sigma.numel() != batch_size:
            raise ValueError(f"Packed sigma must contain {batch_size} values, got {raw_sigma.numel()}")

        token_timestep = raw_timesteps[segment_ids].reshape(1, -1, 1)
        if target_token_lengths_tensor is not None:
            cu = visual_cu_seqlens.detach().cpu().tolist()
            for idx, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
                split = int(start) + int(target_token_lengths_tensor[idx].item())
                token_timestep[:, split:int(end), :] = 0
        timestep, embedded_timestep = self._prepare_timestep(
            token_timestep,
            self.adaln,
            1,
            visual_tokens.dtype,
        )
        timestep = timestep.squeeze(0)
        embedded_timestep = embedded_timestep.squeeze(0)

        prompt_timestep = None
        if self.prompt_adaln is not None:
            prompt_timestep, _ = self._prepare_timestep(
                raw_sigma.reshape(batch_size, 1, 1),
                self.prompt_adaln,
                batch_size,
                visual_tokens.dtype,
            )

        if self.caption_projection is not None:
            context = self.caption_projection(context)
        context = context.view(batch_size, -1, x.shape[-1])
        attention_mask = self._prepare_attention_mask(context_mask, visual_tokens.dtype)
        if positions.ndim == 3:
            positions = positions.unsqueeze(0)
        pe = self._prepare_positional_embeddings(
            positions=positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=visual_tokens.dtype,
        )
        return PackedTransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            visual_cu_seqlens=visual_cu_seqlens.to(device=visual_tokens.device, dtype=torch.int32),
            max_visual_seqlen=int(max_visual_seqlen),
            segment_ids=segment_ids.to(device=visual_tokens.device, dtype=torch.long),
            enabled=True,
            prompt_timestep=prompt_timestep,
            target_token_lengths=target_token_lengths_tensor,
            split_ref_attention=bool(split_ref_attention),
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: RopeType,
        av_ca_timestep_scale_multiplier: int,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
        visual_token_type_embedding: torch.nn.Parameter | None = None,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            caption_projection=caption_projection,
            prompt_adaln=prompt_adaln,
            visual_token_type_embedding=visual_token_type_embedding,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier


    def prepare_packed(self, **kwargs) -> PackedTransformerArgs:
        return self.simple_preprocessor.prepare_packed(**kwargs)

    def prepare_packed_cross_args(
        self,
        *,
        own_positions: torch.Tensor,
        cross_sigma: torch.Tensor,
        num_samples: int,
        x_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build packed AV cross-attention args for one modality.

        Returns ``(cross_positional_embeddings, cross_scale_shift_timestep,
        cross_gate_timestep)``. ``cross_positional_embeddings`` is per-token (time axis
        of this modality's own packed positions); the two timestep tensors are
        PER-SAMPLE ``[num_samples, ...]`` built from the CROSS modality's scalar sigma
        (matching the dense ``prepare`` convention, where the cross timestep comes from
        the other modality). In training audio/video share one sigma, so this is exact.
        """
        positions = own_positions
        if positions.ndim == 3:
            positions = positions.unsqueeze(0)
        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=x_dtype,
        )
        cross_timestep = cross_sigma.reshape(num_samples, 1)
        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=cross_timestep,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=num_samples,
            hidden_dtype=x_dtype,
        )
        # [num_samples, 1, P*dim] -> [num_samples, P*dim] (per-sample, expanded by segment_ids later)
        return cross_pe, cross_scale_shift_timestep.reshape(num_samples, -1), cross_gate_timestep.reshape(num_samples, -1)

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor | None,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross attention timestep embeddings."""
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])
        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        return scale_shift_timestep, gate_noise_timestep


@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: RopeType = RopeType.INTERLEAVED,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.idx = idx
        if video is not None:
            self.attn1 = Attention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = Attention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = Attention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = Attention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = Attention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = Attention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )

        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))

        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values


    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: Attention,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation."""
        if cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
                self.norm_eps,
            )
        return attn(rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask)

    def _get_packed_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        timestep: torch.Tensor,
        indices: slice,
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        values = (
            scale_shift_table[indices].to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(timestep.shape[0], num_ada_params, -1)[:, indices, :]
        ).unbind(dim=1)
        return values

    def _pad_packed_visual_tokens(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
        cu = cu_seqlens.detach().cpu().tolist()
        spans = [(int(start), int(end)) for start, end in zip(cu[:-1], cu[1:])]
        if not spans:
            return x.new_zeros((0, 0, x.shape[-1])), timesteps.new_zeros((0, 0, timesteps.shape[-1])), spans
        if len(spans) == 1:
            # Batch 1: the single span is already max_len, so return views instead of
            # copying ~1.93 GiB of timesteps.
            return x.unsqueeze(0), timesteps.unsqueeze(0), spans
        max_len = max(end - start for start, end in spans)
        padded_x = x.new_zeros((len(spans), max_len, x.shape[-1]))
        padded_timesteps = timesteps.new_zeros((len(spans), max_len, timesteps.shape[-1]))
        for idx, (start, end) in enumerate(spans):
            length = end - start
            padded_x[idx, :length] = x[start:end]
            padded_timesteps[idx, :length] = timesteps[start:end]
        return padded_x, padded_timesteps, spans

    def _unpad_packed_visual_tokens(self, padded: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
        if not spans:
            return padded.new_zeros((0, padded.shape[-1]))
        if len(spans) == 1:
            # Mirrors the single-span fast path above: slicing a view beats cat.
            return padded[0, : spans[0][1] - spans[0][0]]
        return torch.cat([padded[idx, : end - start] for idx, (start, end) in enumerate(spans)], dim=0)

    def _apply_packed_text_cross_attention(
        self,
        vx: torch.Tensor,
        video: PackedTransformerArgs,
        *,
        attn: "Attention | None" = None,
        scale_shift_table: torch.Tensor | None = None,
        prompt_scale_shift_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Defaults target the video branch; pass audio modules for the audio branch.
        if attn is None:
            attn = self.attn2
        if scale_shift_table is None:
            scale_shift_table = self.scale_shift_table
        if prompt_scale_shift_table is None:
            prompt_scale_shift_table = getattr(self, "prompt_scale_shift_table", None)
        padded_vx, padded_timesteps, spans = self._pad_packed_visual_tokens(
            vx,
            video.timesteps,
            video.visual_cu_seqlens,
        )
        if not spans:
            return vx.new_zeros(vx.shape)
        padded_out = self._apply_text_cross_attention(
            padded_vx,
            video.context,
            attn,
            scale_shift_table,
            prompt_scale_shift_table,
            padded_timesteps,
            video.prompt_timestep,
            video.context_mask,
            cross_attention_adaln=self.cross_attention_adaln,
        )
        return self._unpad_packed_visual_tokens(padded_out, spans)



    def _get_packed_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        segment_ids: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Packed analogue of ``get_av_ca_ada_values``.

        The AV-CA scale/shift/gate are PER-SAMPLE (built from each sample's scalar
        cross sigma). Compute them per sample, then expand to per-token via
        ``segment_ids``. ``scale_shift_timestep`` is ``[num_samples, num_scale_shift_values*dim]``
        and ``gate_timestep`` is ``[num_samples, dim]``.
        """
        dim = scale_shift_table.shape[1]
        num_samples = scale_shift_timestep.shape[0]
        table = scale_shift_table.to(device=scale_shift_timestep.device, dtype=scale_shift_timestep.dtype)
        ss = table[:num_scale_shift_values].unsqueeze(0) + scale_shift_timestep.reshape(num_samples, num_scale_shift_values, dim)
        ss = ss[:, scale_shift_indices, :]  # [num_samples, 2, dim]
        scale_ps, shift_ps = ss.unbind(dim=1)  # [num_samples, dim] each
        gate_table = table[num_scale_shift_values:]  # [1, dim]
        gate_ps = (gate_table.unsqueeze(0) + gate_timestep.reshape(num_samples, gate_table.shape[0], dim))[:, 0, :]
        return scale_ps[segment_ids], shift_ps[segment_ids], gate_ps[segment_ids]

    @staticmethod
    def _av_gate_mask(args: PackedTransformerArgs) -> float | torch.Tensor:
        """Per-token multiplier for the AV cross-attention write INTO ``args``' stream.

        Returns the scalar ``1.0`` when no mask is set (so the arithmetic is a no-op and
        numerics are unchanged), else the per-sample mask expanded to per-token via
        ``segment_ids`` and shaped ``[sum_N, 1]`` to broadcast over the feature dim.
        """
        mask = args.av_cross_gate_mask
        if mask is None:
            return 1.0
        mask = mask.to(device=args.x.device, dtype=args.x.dtype)
        return mask[args.segment_ids].unsqueeze(-1)

    @staticmethod
    def _packed_av_cross(
        attn,
        q_x: torch.Tensor,
        k_x: torch.Tensor,
        *,
        query: PackedTransformerArgs,
        key: PackedTransformerArgs,
    ) -> torch.Tensor:
        """Run one direction of the AV cross-attention, reference-aware when configured.

        Falls back to the plain ``forward_packed_cross`` whenever the split-ref cross
        pattern is disabled or the KEY stream carries no reference to wall off, so those
        cases stay bit-identical to the pre-masking behaviour.
        """
        common = dict(
            pe=query.cross_positional_embeddings, k_pe=key.cross_positional_embeddings,
            cu_seqlens_q=query.visual_cu_seqlens, cu_seqlens_k=key.visual_cu_seqlens,
            max_seqlen_q=query.max_visual_seqlen, max_seqlen_k=key.max_visual_seqlen,
        )
        if not query.split_ref_cross_attention or key.target_token_lengths is None:
            return attn.forward_packed_cross(q_x, k_x, **common)
        return attn.forward_packed_cross_split_ref(
            q_x, k_x,
            q_target_token_lengths=query.target_token_lengths,
            k_target_token_lengths=key.target_token_lengths,
            **common,
        )

    def forward_packed_av(
        self,
        video: PackedTransformerArgs,
        audio: PackedTransformerArgs,
    ) -> tuple[PackedTransformerArgs, PackedTransformerArgs]:
        """Packed joint audio+video block (t2av / tav2av).

        Dual-stream: video and audio each run packed varlen self-attention + text
        cross-attention, then bidirectional AV cross-attention where query-segment ``i``
        of one modality attends to key-segment ``i`` of the other (block-diagonal via
        distinct q/k cu_seqlens). Self-attention is FULL within each segment for pure
        generation; for editing (``target_token_lengths`` set) each stream uses the
        target/source split-ref path so the clean source reference does not attend to
        the noised target. Perturbations are an inference-only feature, not applied here.
        """
        if video is None or audio is None:
            raise ValueError("forward_packed_av requires both video and audio packed args.")
        vx = video.x
        ax = audio.x
        if vx.ndim != 2 or ax.ndim != 2:
            raise ValueError("forward_packed_av expects packed [sum_N, D] tensors.")

        # --- Video self-attention + text cross-attention ---
        vshift_msa, vscale_msa, vgate_msa = self._get_packed_ada_values(self.scale_shift_table, video.timesteps, slice(0, 3))
        norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa
        if video.target_token_lengths is None or not video.split_ref_attention:
            video_self_attn = self.attn1.forward_packed_self(
                norm_vx, pe=video.positional_embeddings,
                cu_seqlens=video.visual_cu_seqlens, max_seqlen=video.max_visual_seqlen,
            )
        else:
            # Match packed tv2v: clean source-video tokens do not attend to the noised target.
            video_self_attn = self.attn1.forward_packed_self_split_ref(
                norm_vx, pe=video.positional_embeddings,
                cu_seqlens=video.visual_cu_seqlens, max_seqlen=video.max_visual_seqlen,
                target_token_lengths=video.target_token_lengths,
            )
        vx = vx + video_self_attn * vgate_msa
        vx = vx + self._apply_packed_text_cross_attention(vx, video)

        # --- Audio self-attention + text cross-attention ---
        ashift_msa, ascale_msa, agate_msa = self._get_packed_ada_values(self.audio_scale_shift_table, audio.timesteps, slice(0, 3))
        norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
        if audio.target_token_lengths is None or not audio.split_ref_attention:
            audio_self_attn = self.audio_attn1.forward_packed_self(
                norm_ax, pe=audio.positional_embeddings,
                cu_seqlens=audio.visual_cu_seqlens, max_seqlen=audio.max_visual_seqlen,
            )
        else:
            # Match packed ta2a: clean source-audio tokens do not attend to the noised target.
            audio_self_attn = self.audio_attn1.forward_packed_self_split_ref(
                norm_ax, pe=audio.positional_embeddings,
                cu_seqlens=audio.visual_cu_seqlens, max_seqlen=audio.max_visual_seqlen,
                target_token_lengths=audio.target_token_lengths,
            )
        ax = ax + audio_self_attn * agate_msa
        ax = ax + self._apply_packed_text_cross_attention(
            ax, audio,
            attn=self.audio_attn2,
            scale_shift_table=self.audio_scale_shift_table,
            prompt_scale_shift_table=getattr(self, "audio_prompt_scale_shift_table", None),
        )

        # --- Audio<->Video cross-attention (block-diagonal, dual cu_seqlens) ---
        vx_norm3 = rms_norm(vx, eps=self.norm_eps)
        ax_norm3 = rms_norm(ax, eps=self.norm_eps)

        # a2v: video queries attend audio keys; writes video. (video table slice 0:2, video gate)
        scale_v_a2v, shift_v_a2v, gate_a2v = self._get_packed_av_ca_ada_values(
            self.scale_shift_table_a2v_ca_video, video.cross_scale_shift_timestep, video.cross_gate_timestep, video.segment_ids, slice(0, 2),
        )
        scale_a_a2v, shift_a_a2v, _ = self._get_packed_av_ca_ada_values(
            self.scale_shift_table_a2v_ca_audio, audio.cross_scale_shift_timestep, audio.cross_gate_timestep, audio.segment_ids, slice(0, 2),
        )
        a2v = self._packed_av_cross(
            self.audio_to_video_attn,
            vx_norm3 * (1 + scale_v_a2v) + shift_v_a2v,
            ax_norm3 * (1 + scale_a_a2v) + shift_a_a2v,
            query=video, key=audio,
        )
        vx = vx + a2v * gate_a2v * self._av_gate_mask(video)

        # v2a: audio queries attend video keys; writes audio. (audio table slice 2:4, audio gate)
        scale_a_v2a, shift_a_v2a, gate_v2a = self._get_packed_av_ca_ada_values(
            self.scale_shift_table_a2v_ca_audio, audio.cross_scale_shift_timestep, audio.cross_gate_timestep, audio.segment_ids, slice(2, 4),
        )
        scale_v_v2a, shift_v_v2a, _ = self._get_packed_av_ca_ada_values(
            self.scale_shift_table_a2v_ca_video, video.cross_scale_shift_timestep, video.cross_gate_timestep, video.segment_ids, slice(2, 4),
        )
        v2a = self._packed_av_cross(
            self.video_to_audio_attn,
            ax_norm3 * (1 + scale_a_v2a) + shift_a_v2a,
            vx_norm3 * (1 + scale_v_v2a) + shift_v_v2a,
            query=audio, key=video,
        )
        ax = ax + v2a * gate_v2a * self._av_gate_mask(audio)

        # --- Feed-forward ---
        vshift_mlp, vscale_mlp, vgate_mlp = self._get_packed_ada_values(self.scale_shift_table, video.timesteps, slice(3, 6))
        vx = vx + self.ff(rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp) * vgate_mlp
        ashift_mlp, ascale_mlp, agate_mlp = self._get_packed_ada_values(self.audio_scale_shift_table, audio.timesteps, slice(3, 6))
        ax = ax + self.audio_ff(rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp) * agate_mlp

        return replace(video, x=vx), replace(audio, x=ax)



def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: Attention,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate


class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelType(Enum):
    AudioVideo = "audio-video"
    VideoOnly = "video-only"
    AudioOnly = "audio-only"

    def is_video_enabled(self) -> bool:
        return self in (ModelType.AudioVideo, ModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (ModelType.AudioVideo, ModelType.AudioOnly)


class DiTModel(torch.nn.Module):
    """
    Audio-video diffusion transformer.

    The joint backbone: two modality streams whose blocks cross-attend, with
    per-stream AdaLN modulation.
    """
    _repeated_blocks = ["BasicAVTransformerBlock"]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        key = prefix + "visual_token_type_embedding"
        if hasattr(self, "visual_token_type_embedding") and key not in state_dict:
            param = self.visual_token_type_embedding
            state_dict[key] = torch.zeros(tuple(param.shape), dtype=param.dtype, device="cpu")
        audio_key = prefix + "audio_token_type_embedding"
        if hasattr(self, "audio_token_type_embedding") and audio_key not in state_dict:
            audio_param = self.audio_token_type_embedding
            state_dict[audio_key] = torch.zeros(tuple(audio_param.shape), dtype=audio_param.dtype, device="cpu")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: ModelType = ModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        caption_channels: int = 3840,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = [20, 2048, 2048],
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = [20],
        av_ca_timestep_scale_multiplier: int = 1000,
        rope_type: RopeType = RopeType.SPLIT,
        double_precision_rope: bool = True,
        apply_gated_attention: bool = False,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        self.cross_attention_adaln = cross_attention_adaln
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                caption_channels=caption_channels,
                norm_eps=norm_eps,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            apply_gated_attention=apply_gated_attention,
        )

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        self.visual_token_type_embedding = torch.nn.Parameter(torch.zeros(2, self.inner_dim))
        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)
        self.prompt_adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None

        # Video caption projection
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels,
                hidden_size=self.inner_dim,
            )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        caption_channels: int,
        norm_eps: float,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        # Audio counterpart of ``visual_token_type_embedding``: row 0 target, row 1 reference.
        self.audio_token_type_embedding = torch.nn.Parameter(torch.zeros(2, self.audio_inner_dim))

        self.audio_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)
        self.audio_prompt_adaln_single = AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None

        # Audio caption projection
        if caption_channels is not None:
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels,
                hidden_size=self.audio_inner_dim,
            )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize the input preprocessors."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
                visual_token_type_embedding=getattr(self, "visual_token_type_embedding", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
                visual_token_type_embedding=getattr(self, "audio_token_type_embedding", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
                visual_token_type_embedding=getattr(self, "visual_token_type_embedding", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
                visual_token_type_embedding=getattr(self, "audio_token_type_embedding", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize the transformer blocks."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable



    def _process_output_packed(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        scale_shift_values = (
            scale_shift_table[None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, None, :]
        )
        shift, scale = scale_shift_values[:, 0], scale_shift_values[:, 1]
        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x







    def forward_packed_av(
        self,
        *,
        video_latents,
        video_positions,
        video_context,
        video_timesteps,
        video_cu_seqlens,
        audio_latents,
        audio_positions,
        audio_context,
        audio_timesteps,
        audio_cu_seqlens,
        sigma,
        video_context_mask=None,
        audio_context_mask=None,
        max_video_seqlen=None,
        max_audio_seqlen=None,
        video_segment_ids=None,
        audio_segment_ids=None,
        video_target_token_lengths=None,
        audio_target_token_lengths=None,
        split_ref_attention=True,
        split_ref_cross_attention=True,
        audio_sigma=None,
        video_av_cross_gate_mask=None,
        audio_av_cross_gate_mask=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
    ):
        """Packed-token forward for JOINT-AV tasks (t2av / tav2av).

        Packs N samples into a video stream (``video_cu_seqlens``) and an audio stream
        (``audio_cu_seqlens``) that share the SAME N samples. Each block runs packed
        self-attention per modality plus block-diagonal a2v/v2a cross-attention
        (query-segment i attends to key-segment i across modalities). Returns
        ``(video_noise_pred, audio_noise_pred)`` as packed ``[sum_N, out]`` tensors.

        ``sigma`` is the video branch's noise level; ``audio_sigma`` defaults to it (the
        shared-sigma convention used by joint tav2av/t2av). Pass a different
        ``audio_sigma`` for SEQUENTIAL denoising, where one modality is already finished
        (its target is clean, sigma ~= 0) while the other runs the normal schedule. Note
        each branch's AV-CA AdaLN args are built from the OTHER branch's sigma, so the two
        are crossed here -- that crossing is a no-op while the sigmas are equal but is
        load-bearing once they differ.

        ``video_av_cross_gate_mask`` / ``audio_av_cross_gate_mask`` are per-sample [N]
        multipliers on the AV cross-attention write INTO that stream. Zero one to make a
        branch immune to the other (single-modality editing: tav2v/tav2a). ``None`` =>
        unmasked, numerically identical to before.

        ``split_ref_cross_attention`` (default True) additionally walls the clean
        in-context reference of each stream off from the OTHER stream's noised target in
        the a2v/v2a cross-attention -- the cross-modal counterpart of
        ``split_ref_attention``. It is a no-op for samples with no reference, so pure
        t2av batches are unaffected; pass False to restore the old unmasked coupling.
        """
        if not (self.model_type.is_video_enabled() and self.model_type.is_audio_enabled()):
            raise ValueError("Packed audio-video forward requires the joint model.")

        def _segments(cu, device):
            cu = cu.to(device=device, dtype=torch.int32)
            lengths = (cu[1:] - cu[:-1]).to(dtype=torch.long)
            return cu, lengths

        video_cu_seqlens, v_lengths = _segments(video_cu_seqlens, video_latents.device)
        audio_cu_seqlens, a_lengths = _segments(audio_cu_seqlens, audio_latents.device)
        num_samples = int(v_lengths.numel())
        if int(a_lengths.numel()) != num_samples:
            raise ValueError("Video and audio packed streams must contain the same number of samples.")
        if max_video_seqlen is None:
            max_video_seqlen = int(v_lengths.max().item()) if v_lengths.numel() else 0
        if max_audio_seqlen is None:
            max_audio_seqlen = int(a_lengths.max().item()) if a_lengths.numel() else 0
        if video_segment_ids is None:
            video_segment_ids = torch.repeat_interleave(
                torch.arange(num_samples, device=video_latents.device, dtype=torch.long), v_lengths
            )
        if audio_segment_ids is None:
            audio_segment_ids = torch.repeat_interleave(
                torch.arange(num_samples, device=audio_latents.device, dtype=torch.long), a_lengths
            )
        if video_context.ndim != 3:
            video_context = video_context.unsqueeze(0)
        if audio_context.ndim != 3:
            audio_context = audio_context.unsqueeze(0)

        cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
        self._init_preprocessors(cross_pe_max_pos)

        video = self.video_args_preprocessor.prepare_packed(
            visual_tokens=video_latents,
            positions=video_positions,
            context=video_context,
            context_mask=video_context_mask,
            timesteps=video_timesteps,
            sigma=sigma,
            visual_cu_seqlens=video_cu_seqlens,
            max_visual_seqlen=max_video_seqlen,
            segment_ids=video_segment_ids,
            target_token_lengths=video_target_token_lengths,
            split_ref_attention=split_ref_attention,
        )
        audio_sigma_eff = sigma if audio_sigma is None else audio_sigma
        audio = self.audio_args_preprocessor.prepare_packed(
            visual_tokens=audio_latents,
            positions=audio_positions,
            context=audio_context,
            context_mask=audio_context_mask,
            timesteps=audio_timesteps,
            sigma=audio_sigma_eff,
            visual_cu_seqlens=audio_cu_seqlens,
            max_visual_seqlen=max_audio_seqlen,
            segment_ids=audio_segment_ids,
            target_token_lengths=audio_target_token_lengths,
            split_ref_attention=split_ref_attention,
        )

        # Each modality's cross timestep comes from the OTHER modality's sigma.
        v_cross_pe, v_cross_ss, v_cross_gate = self.video_args_preprocessor.prepare_packed_cross_args(
            own_positions=video_positions, cross_sigma=audio_sigma_eff, num_samples=num_samples, x_dtype=video_latents.dtype,
        )
        a_cross_pe, a_cross_ss, a_cross_gate = self.audio_args_preprocessor.prepare_packed_cross_args(
            own_positions=audio_positions, cross_sigma=sigma, num_samples=num_samples, x_dtype=audio_latents.dtype,
        )
        def _gate_mask(value, device):
            if value is None:
                return None
            mask = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)
            if mask.numel() != num_samples:
                raise ValueError(
                    f"av_cross_gate_mask must have {num_samples} values (one per sample), got {mask.numel()}"
                )
            return mask

        video = replace(
            video,
            cross_positional_embeddings=v_cross_pe,
            cross_scale_shift_timestep=v_cross_ss,
            cross_gate_timestep=v_cross_gate,
            av_cross_gate_mask=_gate_mask(video_av_cross_gate_mask, video_latents.device),
            split_ref_cross_attention=split_ref_cross_attention,
        )
        audio = replace(
            audio,
            cross_positional_embeddings=a_cross_pe,
            cross_scale_shift_timestep=a_cross_ss,
            cross_gate_timestep=a_cross_gate,
            av_cross_gate_mask=_gate_mask(audio_av_cross_gate_mask, audio_latents.device),
            split_ref_cross_attention=split_ref_cross_attention,
        )

        for block in self.transformer_blocks:
            video, audio = checkpoint_packed_av_block(
                block,
                video,
                audio,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )

        video_out = self._process_output_packed(
            self.scale_shift_table, self.norm_out, self.proj_out, video.x, video.embedded_timestep,
        )
        audio_out = self._process_output_packed(
            self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out, audio.x, audio.embedded_timestep,
        )
        return video_out, audio_out
