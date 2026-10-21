"""Shared command-line runner for the four released inference tasks.

Each `inference/infer_<task>.py` script is a thin wrapper that fixes the task and
its input semantics, then calls `run()`.
"""

from __future__ import annotations

import argparse
import time

from uniavc.inference import media as media_io
from uniavc.inference.common import batched_audio, batched_video
from uniavc.inference.pipeline import load_pipeline
from uniavc.inference.sampler import sample_joint_av

# (needs an input video, needs an input audio file). Every task outputs both modalities; this
# only says what the caller must supply.
TASK_INPUTS = {
    "tav2av": (True, False),   # audio defaults to the input video's own track
    "t2av": (False, False),    # no conditioning media at all
    "ta2v": (False, True),     # the given audio is the condition
    "tv2a": (True, False),     # the given video is the condition
}

# Tasks that take `--audio` at all, required or not (tav2av falls back to the --video track).
AUDIO_TASKS = {"ta2v", "tav2av"}

# Per-task defaults, ordered (video_text, audio_text, video_reference, audio_reference).
# tav2av leans less on the instruction for audio and harder on the source audio; the
# generation tasks have no source latents, so their reference axis is inert.
TASK_CFG_DEFAULTS: dict[str, tuple[float, float, float, float]] = {
    "tav2av": (3.0, 2.0, 1.5, 3.0),
    "t2av":   (6.0, 6.0, 1.0, 1.0),
    "ta2v":   (6.0, 6.0, 1.0, 1.0),
    "tv2a":   (6.0, 6.0, 1.0, 1.0),
}

# Tasks that anchor guidance on the built-in negative prompt rather than a zeroed context.
NEGATIVE_PROMPT_TASKS = ("t2av", "ta2v", "tv2a")

# The LTX-2.3 built-in negative prompt, verbatim.
NEGATIVE_PROMPT_DEFAULT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted "
    "proportions, unnatural skin tones, deformed facial features, asymmetrical face, "
    "missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts "
    "around text, inconsistent perspective, camera shake, incorrect depth of field, "
    "background too sharp, background clutter, distracting reflections, harsh shadows, "
    "inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, "
    "unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, "
    "exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted "
    "audio, distorted voice, robotic voice, echo, background noise, off-sync audio, "
    "incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, "
    "flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI "
    "artifacts."
)


def build_parser(task: str, description: str) -> argparse.ArgumentParser:
    needs_video, needs_audio = TASK_INPUTS[task]

    p = argparse.ArgumentParser(
        prog=f"infer_{task}",
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--prompt", required=True, help="Text prompt or edit instruction.")
    if needs_video:
        p.add_argument(
            "--video", required=True,
            help="Input mp4. Supplies both the video and, unless --audio is given, its audio.",
        )
    if task in AUDIO_TASKS:
        # ta2v requires audio; tav2av defaults to the video's own track.
        p.add_argument(
            "--audio", required=(task == "ta2v"),
            help="Input audio file. Defaults to the --video track."
            if task != "ta2v" else "Input audio to condition video generation on.",
        )

    p.add_argument("--output", "-o", default=f"{task}_output.mp4", help="Output mp4 path.")

    # Both models are read from ./models; there is no model path to pass.
    p.add_argument("--device", default="cuda", help="Torch device.")
    # On by default: 43 GiB + 22 GiB leaves no room for activations on an 80 GB card.
    p.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep weights on CPU between phases, loading only what each step "
                        "needs (default). --no-offload keeps everything resident, which "
                        "needs a much larger card.")

    # Generation geometry.
    p.add_argument("--height", type=int, default=media_io.DEFAULT_HEIGHT)
    p.add_argument("--width", type=int, default=media_io.DEFAULT_WIDTH)
    p.add_argument("--num-frames", type=int, default=media_io.DEFAULT_NUM_FRAMES)
    p.add_argument("--frame-rate", type=float, default=media_io.DEFAULT_FRAME_RATE)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tiled", action="store_true", help="Decode without VAE tiling.")

    # Per-branch guidance defaults; there is no shared scale and no cross-modal axis.
    vt, at, vr, ar = TASK_CFG_DEFAULTS[task]
    p.add_argument("--video-text-cfg-scale", type=float, default=vt,
                   help="Instruction guidance on the video branch.")
    p.add_argument("--audio-text-cfg-scale", type=float, default=at,
                   help="Instruction guidance on the audio branch.")
    p.add_argument("--video-reference-cfg-scale", type=float, default=vr,
                   help="Source adherence on the video branch. Inert without an input video.")
    p.add_argument("--audio-reference-cfg-scale", type=float, default=ar,
                   help="Source adherence on the audio branch. Inert without an input audio track.")
    p.add_argument("--negative-prompt",
                   default=NEGATIVE_PROMPT_DEFAULT if task in NEGATIVE_PROMPT_TASKS else None,
                   help="Negative prompt for guidance. The joint and audio-to-video generation "
                        "tasks default to the built-in LTX-2.3 list; the edit tasks default to "
                        "the training null (zero context).")

    if task == "tav2av":
        p.add_argument("--post-refine", choices=["video", "audio"], default=None,
                       help="After a joint pass, re-denoise only this branch against the other.")
        p.add_argument("--refine-lambda", type=float, default=0.5,
                       help="Noise fraction re-added before the refinement pass. Must be > 0.")
        p.add_argument("--refine-steps", type=int, default=None,
                       help="Steps in the refinement pass. Default: round(refine-lambda * "
                            "num-inference-steps, floored at 2), which keeps per-step noise "
                            "resolution on par with the main pass. Pass a value to override.")
        p.add_argument("--intent-aware", choices=["video", "audio"], default=None,
                       help="Hold this branch intact by conditioning instead of copying: its "
                            "instruction guidance is pinned to 1.0 while the other branch follows "
                            "the prompt.")
    return p


def run(task: str, argv: list[str] | None = None, description: str = "") -> None:
    args = build_parser(task, description).parse_args(argv)
    started = time.time()

    pipe = load_pipeline(
        device=args.device,
        offload=args.offload,
    )

    # ---- encode the prompt ---------------------------------------------------
    video_context = media_io.encode_video_context(pipe, args.prompt)
    audio_context = media_io.encode_audio_context(pipe, args.prompt)
    negative_video = negative_audio = None
    if args.negative_prompt:
        negative_video = media_io.encode_video_context(pipe, args.negative_prompt)
        negative_audio = media_io.encode_audio_context(pipe, args.negative_prompt)

    # ---- encode the input media ---------------------------------------------
    needs_video, _ = TASK_INPUTS[task]
    src_video = src_audio = None
    given_video = given_audio = None
    video_shape = audio_shape = None
    frame_rate = args.frame_rate

    if needs_video:
        height, width, num_frames, frame_rate = media_io.resolve_video_geometry(
            args.video, height=args.height, width=args.width, num_frames=args.num_frames
        )
        frames = media_io.load_video_frames(
            args.video, height=height, width=width, num_frames=num_frames, frame_rate=frame_rate
        )
        latents = media_io.encode_video(pipe, frames, tiled=not args.no_tiled)
        duration = len(frames) / frame_rate
        # encode_video returns [C,F,H,W]; the sampler works in batched [B,C,F,H,W].
        video_shape = batched_video(latents).shape

        if task == "tv2a":
            # The video is the given signal; generated audio takes its duration.
            given_video = latents
            audio_shape = media_io.generation_audio_latent_shape(num_frames, frame_rate)
        else:
            src_video = latents
            audio_source = args.audio or args.video
            waveform, sample_rate = media_io.load_waveform(audio_source, duration=duration)
            audio_latents = media_io.encode_audio(pipe, waveform, sample_rate)
            src_audio = audio_latents
            audio_shape = batched_audio(audio_latents).shape

    elif task == "ta2v":
        # The audio is the given signal; the video geometry is a free choice, so trim the audio
        # to match and keep the branches aligned.
        height, width = args.height, args.width
        num_frames = args.num_frames
        duration = num_frames / frame_rate
        waveform, sample_rate = media_io.load_waveform(args.audio, duration=duration)
        given_audio = media_io.encode_audio(pipe, waveform, sample_rate)
        audio_shape = batched_audio(given_audio).shape
        video_shape = media_io.video_latent_shape(height, width, num_frames)

    else:  # t2av: no conditioning media at all
        duration = args.num_frames / frame_rate
        video_shape = media_io.video_latent_shape(args.height, args.width, args.num_frames)
        audio_shape = media_io.audio_latent_shape(
            duration, pinned_frames=media_io.TRAIN_AUDIO_FRAMES
        )

    # ---- sample --------------------------------------------------------------
    latents = sample_joint_av(
        pipe,
        task=task,
        video_context=video_context,
        audio_context=audio_context,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        frame_rate=frame_rate,
        source_video_latents=src_video,
        source_audio_latents=src_audio,
        given_video_latents=given_video,
        given_audio_latents=given_audio,
        video_shape=video_shape,
        audio_shape=audio_shape,
        video_text_cfg_scale=args.video_text_cfg_scale,
        audio_text_cfg_scale=args.audio_text_cfg_scale,
        video_reference_cfg_scale=args.video_reference_cfg_scale,
        audio_reference_cfg_scale=args.audio_reference_cfg_scale,
        post_refine=getattr(args, "post_refine", None),
        refine_lambda=getattr(args, "refine_lambda", 0.5),
        refine_steps=getattr(args, "refine_steps", None),
        intent_aware=getattr(args, "intent_aware", None),
        negative_video_context=negative_video,
        negative_audio_context=negative_audio,
    )

    # ---- decode and write ----------------------------------------------------
    frames = media_io.decode_video(pipe, latents["video_latents"], tiled=not args.no_tiled)
    waveform, sample_rate = media_io.decode_audio(pipe, latents["audio_latents"])
    media_io.save_video_audio(
        frames, waveform, args.output, fps=frame_rate, sample_rate=sample_rate
    )

    print(
        f"done in {time.time() - started:.1f}s -> {args.output} "
        f"({len(frames)} frames @ {frame_rate:g} fps, audio {sample_rate} Hz)",
        flush=True,
    )
