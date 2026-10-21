"""Encoding user media into latents, and decoding latents back to files."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from uniavc.core.data import UnifiedDataset
from uniavc.models.common import (
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from uniavc.models.text_encoder import _convert_to_additive_mask, _to_binary_mask
from uniavc.utils.data.audio import convert_to_stereo, read_audio

# VAE divisibility: spatial axes in multiples of 32, frame count 8k+1.
SPATIAL_DIVISION = 32
TIME_DIVISION = 8

# The released geometry: 1280x704, 121 frames at 24 fps.
DEFAULT_HEIGHT = 704
DEFAULT_WIDTH = 1280
DEFAULT_NUM_FRAMES = 121
DEFAULT_FRAME_RATE = 24.0

# Mel padding makes a 121-frame clip encode to 127 audio latents; the duration formula gives
# 126. Generation has no shape to inherit, so it pins this.
TRAIN_AUDIO_FRAMES = 127


def floor_to_multiple(value: int, factor: int = SPATIAL_DIVISION) -> int:
    """Floor to a VAE-safe multiple, never rounding up."""
    return max(factor, int(value) // factor * factor)


def safe_frame_count(num_frames: int) -> int:
    """Round a pixel frame count down to the nearest 8k+1."""
    num_frames = max(1, int(num_frames))
    if num_frames == 1:
        return 1
    return ((num_frames - 1) // TIME_DIVISION) * TIME_DIVISION + 1


# ------------------------------------------------------------------ encoding


@torch.no_grad()
def encode_video(pipe, frames: list[Image.Image], *, tiled: bool = True) -> torch.Tensor:
    """PIL frames -> video latents [C, F, H, W] on CPU.

    Loads only the video encoder for the duration. Under `--offload` this is what
    keeps the DiT (39 GiB) off the card while the frames are encoded; without
    `load_models_to_device` every class stays resident and the DiT forward then has
    nothing left for its activations.
    """
    pipe.load_models_to_device(["video_vae_encoder"])
    video = pipe.preprocess_video(frames)
    latents = pipe.video_vae_encoder.encode(video, tiled=tiled)
    return latents.detach().squeeze(0).cpu()


@torch.no_grad()
def encode_audio(pipe, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Waveform [C, T] -> audio latents [C, T, F] on CPU."""
    pipe.load_models_to_device(["audio_vae_encoder"])
    waveform = convert_to_stereo(waveform).to(device=pipe.device)
    mel = pipe.audio_processor.waveform_to_mel(
        waveform.unsqueeze(0), waveform_sample_rate=sample_rate
    ).to(dtype=pipe.torch_dtype, device=pipe.device)
    return pipe.audio_vae_encoder(mel).detach().squeeze(0).cpu()


@torch.no_grad()
def encode_video_context(pipe, text: str) -> torch.Tensor:
    """Prompt -> the DiT's video conditioning tensor [S, D] on CPU."""
    pipe.load_models_to_device(["text_encoder", "text_encoder_post_modules"])
    token_pairs = pipe.tokenizer.tokenize_with_weights(text)["gemma"]
    input_ids = torch.tensor([[t[0] for t in token_pairs]], device=pipe.device)
    attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=pipe.device)
    outputs = pipe.text_encoder(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
    video_feats, _ = pipe.text_encoder_post_modules.feature_extractor_linear(
        outputs.hidden_states, attention_mask, padding_side="left"
    )
    additive = _convert_to_additive_mask(attention_mask, video_feats.dtype)
    context, mask = pipe.text_encoder_post_modules.embeddings_connector(video_feats, additive)
    context, _ = _to_binary_mask(context, mask)
    return context.detach().squeeze(0).cpu()


@torch.no_grad()
def encode_audio_context(pipe, text: str) -> torch.Tensor:
    """Prompt -> the DiT's audio conditioning tensor [S, D] on CPU.

    The post-modules use *separated* connectors, so `feature_extractor_linear`
    returns `(video_feats, audio_feats)` and the audio side must take the second
    element of that pair, then run its own `audio_embeddings_connector`.

    Apply `_to_binary_mask` exactly once, on the connector's own mask. Routing this
    through `process_hidden_states` instead would binarize twice and invert
    keep/drop, zeroing every real token.
    """
    pipe.load_models_to_device(["text_encoder", "text_encoder_post_modules"])
    token_pairs = pipe.tokenizer.tokenize_with_weights(text)["gemma"]
    input_ids = torch.tensor([[t[0] for t in token_pairs]], device=pipe.device)
    attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=pipe.device)
    outputs = pipe.text_encoder(
        input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
    )
    _, audio_feats = pipe.text_encoder_post_modules.feature_extractor_linear(
        outputs.hidden_states, attention_mask, padding_side="left"
    )
    additive = _convert_to_additive_mask(attention_mask, audio_feats.dtype)
    context, mask = pipe.text_encoder_post_modules.audio_embeddings_connector(audio_feats, additive)
    context, _ = _to_binary_mask(context, mask)
    return context.detach().squeeze(0).cpu()


# ---------------------------------------------------------------- media input


def load_video_frames(path: str, *, height: int, width: int, num_frames: int, frame_rate: float):
    """Decode an mp4 to resized PIL frames on the model grid."""
    operator = UnifiedDataset.default_video_operator(
        base_path="",
        max_pixels=0,  # keep the caller's resolution; do not shrink it back
        height=height,
        width=width,
        height_division_factor=SPATIAL_DIVISION,
        width_division_factor=SPATIAL_DIVISION,
        num_frames=num_frames,
        time_division_factor=TIME_DIVISION,
        time_division_remainder=1,
        frame_rate=frame_rate,
        fix_frame_rate=True,
    )
    return operator(path)


def load_waveform(path: str, *, duration: float | None):
    """Read a file's audio track as [C, T] plus its sample rate.

    When `duration` is known the waveform is pinned to exactly that many seconds --
    trimming overflow and zero-padding shortfall -- so the audio and video branches
    always agree on length. Without this a few milliseconds of difference between
    two decodes becomes a mismatched latent frame count.
    """
    # `auto` walks torchcodec -> PyAV -> libsndfile; PyAV is what reads mp4 containers.
    try:
        waveform, sample_rate = read_audio(
            path, start_time=0, duration=duration, backend="auto"
        )
    except Exception as exc:
        raise RuntimeError(f"could not decode audio from {path!r}: {exc}") from exc

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    sample_rate = int(sample_rate)
    if duration is not None and duration > 0:
        target = int(round(float(duration) * sample_rate))
        current = waveform.shape[-1]
        if current > target:
            waveform = waveform[..., :target]
        elif current < target:
            waveform = torch.nn.functional.pad(waveform, (0, target - current))
    return waveform, sample_rate


def resolve_video_geometry(
    path: str, *, height: int, width: int, num_frames: int
) -> tuple[int, int, int, float]:
    """Pick the encode geometry for a source clip.

    Source clips are encoded at their own resolution and frame rate, floored to the
    VAE grid, with the frame count pinned to `num_frames`. Pinning frames rather
    than taking the clip's own count keeps the token budget from varying with the
    input, which is what made longer clips OOM on the attention.
    """
    import imageio.v2 as imageio

    try:
        reader = imageio.get_reader(path)
        meta = reader.get_meta_data()
        reader.close()
        src_h = int(meta.get("size", (height, width))[1])
        src_w = int(meta.get("size", (height, width))[0])
        src_fps = float(meta.get("fps") or DEFAULT_FRAME_RATE)
    except Exception:
        return height, width, num_frames, DEFAULT_FRAME_RATE

    if src_h <= 0 or src_w <= 0:
        return height, width, num_frames, DEFAULT_FRAME_RATE
    return floor_to_multiple(src_h), floor_to_multiple(src_w), num_frames, src_fps


# ------------------------------------------------------------------ decoding


@torch.no_grad()
def decode_video(pipe, latents: torch.Tensor, *, tiled: bool = True) -> list[Image.Image]:
    """Video latents [B,C,F,H,W] -> PIL frames."""
    latents = latents.to(device=pipe.device, dtype=pipe.torch_dtype)
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    pipe.load_models_to_device(["video_vae_decoder"])
    decoded = pipe.video_vae_decoder.decode(latents, tiled=tiled)
    return pipe.vae_output_to_video(decoded)


@torch.no_grad()
def decode_audio(pipe, latents: torch.Tensor):
    """Audio latents [B,C,T,F] -> ([C, T] waveform, sample rate)."""
    latents = latents.to(device=pipe.device, dtype=pipe.torch_dtype)
    pipe.load_models_to_device(["audio_vae_decoder", "audio_vocoder"])
    decoded = pipe.audio_vocoder(pipe.audio_vae_decoder(latents))
    waveform = pipe.output_audio_format_check(decoded)
    sample_rate = int(getattr(pipe.audio_vocoder, "output_sampling_rate", 48000) or 48000)
    return waveform, sample_rate


def save_video_audio(frames: list[Image.Image], waveform, output_path: str, *, fps: float, sample_rate: int) -> None:
    """Mux frames and a waveform into an mp4."""
    from uniavc.utils.data.audio_video import write_video_audio

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_video_audio(
        frames, waveform, output_path,
        fps=int(round(float(fps))),
        audio_sample_rate=int(sample_rate),
    )


# ------------------------------------------------------------------- shapes


def generation_audio_latent_shape(
    num_frames: int, frame_rate: float, *, width: int | None = None, height: int | None = None
) -> tuple[int, ...]:
    """Audio latent shape for a branch that is *generated* rather than inherited.

    When a source audio track exists the shape comes from encoding it, which yields
    the trained 127 frames for a 121-frame clip. With nothing to encode (t2av, and
    tv2a's audio side) the duration formula would give 126 instead, so the trained
    value is pinned whenever the geometry is the one it was packed at.
    """
    duration = float(num_frames) / float(frame_rate)
    at_training_geometry = (
        int(num_frames) == DEFAULT_NUM_FRAMES
        and abs(float(frame_rate) - DEFAULT_FRAME_RATE) < 1e-6
        and (width is None or int(width) == DEFAULT_WIDTH)
        and (height is None or int(height) == DEFAULT_HEIGHT)
    )
    return audio_latent_shape(
        duration, pinned_frames=TRAIN_AUDIO_FRAMES if at_training_geometry else None
    )


def video_latent_shape(height: int, width: int, num_frames: int) -> tuple[int, ...]:
    pixel = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=DEFAULT_FRAME_RATE)
    return tuple(VideoLatentShape.from_pixel_shape(pixel, latent_channels=128).to_torch_shape())


def audio_latent_shape(duration: float, *, pinned_frames: int | None = None) -> tuple[int, ...]:
    """Audio latent shape for a clip length.

    Pass `pinned_frames` when there is no source audio to inherit a shape from
    (t2av generation): the training convention is 127 frames for a 121-frame clip,
    where the duration formula yields 126.
    """
    if pinned_frames is not None and int(pinned_frames) > 0:
        latent = AudioLatentShape(batch=1, channels=8, frames=int(pinned_frames), mel_bins=16)
        return tuple(latent.to_torch_shape())
    return tuple(AudioLatentShape.from_duration(batch=1, duration=duration).to_torch_shape())
