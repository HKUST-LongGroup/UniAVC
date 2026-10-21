"""Shared helpers for UniAVC inference.

Vendored and adapted from DiffSynth-Studio's LTX-2.3 multitask example code
(Apache-2.0). Kept deliberately small: only what the four released tasks need.
"""

from __future__ import annotations

from typing import Any

import torch

from uniavc.models.common import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    VideoLatentShape,
    get_pixel_coords,
)

# Reference blocks are shifted forward on the time axis so the DiT can tell the
# clean reference apart from what it is generating. Without this the source is
# silently ignored -- see `_video_in_context_reference` in the original code.
REF_ROPE_TIME_OFFSET = 100.0

# Task -> (mechanism, given branch). Mirrors task_schema.AV_EDIT_MECHANISM.
TASK_MECHANISM: dict[str, tuple[str | None, str | None]] = {
    "tav2av": (None, None),
    "t2av": (None, None),
    "ta2v": ("pinned", "audio"),
    "tv2a": ("pinned", "video"),
}

# Generation tasks that read the given signal from the TARGET slot (their rows
# carry no source latents), as opposed to the edit pair which reads `source_*`.
TARGET_SLOT_TASKS = {"ta2v", "tv2a"}


def batched_video(latents: torch.Tensor) -> torch.Tensor:
    """[C,F,H,W] -> [B,C,F,H,W]."""
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    if latents.ndim != 5:
        raise ValueError(f"expected video latents [C,F,H,W] or [B,C,F,H,W], got {tuple(latents.shape)}")
    return latents.contiguous()


def batched_audio(latents: torch.Tensor) -> torch.Tensor:
    """[C,T,F] -> [B,C,T,F]."""
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)
    if latents.ndim != 4:
        raise ValueError(f"expected audio latents [C,T,F] or [B,C,T,F], got {tuple(latents.shape)}")
    return latents.contiguous()


def batched_context(context: torch.Tensor) -> torch.Tensor:
    """[S,D] -> [B,S,D]."""
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context.ndim != 3:
        raise ValueError(f"expected context [S,D] or [B,S,D], got {tuple(context.shape)}")
    return context.contiguous()


def video_positions_for_latents(pipe, latents: torch.Tensor, frame_rate: float) -> torch.Tensor:
    """Patch-grid bounds -> pixel coords -> seconds, on the latent's own device."""
    shape = VideoLatentShape.from_torch_shape(latents.shape)
    latent_coords = pipe.video_patchifier.get_patch_grid_bounds(
        output_shape=shape, device=latents.device
    )
    positions = get_pixel_coords(latent_coords, VIDEO_SCALE_FACTORS, True).float()
    positions[:, 0, ...] = positions[:, 0, ...] / frame_rate
    return positions.to(dtype=latents.dtype, device=latents.device)


def audio_positions_for_latents(pipe, latents: torch.Tensor) -> torch.Tensor:
    return pipe.audio_patchifier.get_patch_grid_bounds(
        AudioLatentShape.from_torch_shape(latents.shape), device=latents.device
    )


def offset_time_axis(positions: torch.Tensor, offset: float = REF_ROPE_TIME_OFFSET) -> torch.Tensor:
    """Shift the temporal axis of in-context reference positions forward.

    Without this the reference sits on the target's own time coordinates and the DiT
    ignores the source clip entirely.

    Layouts that occur:
      * `[B, 3, T, 2]` video and `[B, 1, T, 2]` audio -- trailing pair is
        `[start, end)`, and the axis dim (the 3 or the 1) is at `ndim - 3`.
      * `[B, 3, T]` or `[3, T]` -- no trailing pair, so the axis dim is at `ndim - 2`.

    In every case the time axis is component 0 of the axis dim.
    """
    shifted = positions.clone()
    axis_dim = shifted.ndim - 3 if shifted.shape[-1] == 2 else shifted.ndim - 2
    index: list[Any] = [slice(None)] * shifted.ndim
    index[axis_dim] = 0
    key = tuple(index)
    shifted[key] = shifted[key] + float(offset)
    return shifted


def no_grad_encode(fn):
    """Encode helpers must not retain the autograd graph.

    `.detach()` on the result is too late -- the graph is already built, and this
    was the cause of a 78 GiB OOM in the original code.
    """
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    return wrapper
