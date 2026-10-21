"""Joint audio-video sampling.

Fixed rather than exposed, because they are properties of the checkpoint:
the dual scheduler at shifts 9.0 / 2.0, per-branch guidance, and no cross-modal
AV-coupling axis (it was inert at 1.0 in every released run).
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from tqdm import tqdm

from uniavc.inference.common import (
    TASK_MECHANISM,
    audio_positions_for_latents,
    batched_audio,
    batched_context,
    batched_video,
    offset_time_axis,
    video_positions_for_latents,
)

# Training shifts. The AV cross-attention modulation reads the *other* branch's sigma,
# so a wrong pair degrades output with no error.
DEFAULT_VIDEO_SHIFT = 9.0
DEFAULT_AUDIO_SHIFT = 2.0


# --------------------------------------------------------------------------- CFG


def interaction_cfg_scale(scale_text: float, scale_ref: float) -> float:
    """Coefficient on the text x reference interaction term.

    Both axes guided -> the quadrature sum; exactly one -> the larger, so the
    interaction only reshapes a prediction that already exists on the other axis.
    """
    text_guided = abs(scale_text - 1.0) > 1e-6
    ref_guided = abs(scale_ref - 1.0) > 1e-6
    if text_guided and ref_guided:
        return (scale_text ** 2 + scale_ref ** 2) ** 0.5
    if text_guided:
        return scale_text
    if ref_guided:
        return scale_ref
    return 1.0


def additive_cfg(
    uncond: torch.Tensor,
    text_only: torch.Tensor | None,
    ref_only: torch.Tensor | None,
    full: torch.Tensor,
    *,
    scale_text: float,
    scale_ref: float,
) -> torch.Tensor:
    """Additive CFG: every delta measured against the fully-unconditional anchor.

    `e = u + s_t*(T - u) + s_r*(R - u) + s_b*(F - T - R + u)`

    At `s_t = s_r = 1` this returns `full` exactly, which is the cfg-free
    prediction -- the property the interaction weight is chosen to preserve.
    """
    if text_only is None or ref_only is None:
        # A single live axis. `full` is the only other corner available.
        scale = scale_text if text_only is not None else scale_ref
        return uncond + scale * (full - uncond)

    delta_text = text_only - uncond
    delta_ref = ref_only - uncond
    delta_interaction = full - text_only - ref_only + uncond
    scale_both = interaction_cfg_scale(scale_text, scale_ref)

    return uncond + scale_text * delta_text + scale_ref * delta_ref + scale_both * delta_interaction


# ----------------------------------------------------------------- the sampler


def _build_stages(*, mechanism: str | None, branch: str | None) -> list[tuple[str, ...]]:
    """Which branches each denoising stage steps.

    Every released task produces both modalities, so audio is always present.
    """
    if mechanism == "post-refine":
        stages = [("video", "audio"), (branch,)]
    elif mechanism == "pinned":
        # The given branch is never stepped; only the other one denoises.
        stages = [("audio" if branch == "video" else "video",)]
    else:
        stages = [("video", "audio")]
    return stages


@torch.no_grad()
def sample_joint_av(
    pipe,
    *,
    task: str,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    num_inference_steps: int,
    seed: int,
    frame_rate: float = 24.0,
    # Conditioning media. Edit tasks pass both; the generation pair passes the single given
    # modality.
    source_video_latents: torch.Tensor | None = None,
    source_audio_latents: torch.Tensor | None = None,
    given_video_latents: torch.Tensor | None = None,
    given_audio_latents: torch.Tensor | None = None,
    # Output shapes; default to the conditioning latent shapes.
    video_shape: tuple[int, ...] | None = None,
    audio_shape: tuple[int, ...] | None = None,
    # Per-branch guidance; see TASK_CFG_DEFAULTS in cli.py.
    video_text_cfg_scale: float = 3.0,
    audio_text_cfg_scale: float = 2.0,
    video_reference_cfg_scale: float = 1.5,
    audio_reference_cfg_scale: float = 3.0,
    # Post-refinement: re-noise one branch off its own result, re-denoise it, keep the other clean.
    post_refine: str | None = None,
    refine_lambda: float = 0.5,
    refine_steps: int | None = None,
    # Intent-aware editing: pin one branch's text guidance to 1.0, leaving its context intact.
    intent_aware: str | None = None,
    negative_video_context: torch.Tensor | None = None,
    negative_audio_context: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Denoise a joint audio-video latent pair.

    The loaded weights keep torch's default `requires_grad=True`, so `no_grad` is what
    stops every step retaining a graph for a backward that never runs.

    Returns `{"video_latents": [B,C,F,H,W], "audio_latents": [B,C,T,F]}`.
    """
    mechanism, branch = TASK_MECHANISM[task]

    # Which stream is given. The edit pair reads it from `source_*`; the generation pair has
    # no source latents and reads it from the target slot.
    pinned_clean: str | None = None
    if mechanism == "pinned":
        pinned_clean = branch
    elif post_refine:
        mechanism = "post-refine"
        branch = post_refine

    # ---- noise, positions, contexts ------------------------------------------
    if video_shape is None:
        base = given_video_latents if given_video_latents is not None else source_video_latents
        if base is None:
            raise ValueError(f"task {task!r} needs a video shape or a video reference to infer one")
        video_shape = batched_video(base).shape
    if audio_shape is None:
        base = given_audio_latents if given_audio_latents is not None else source_audio_latents
        if base is None:
            raise ValueError(f"task {task!r} needs an audio shape or an audio reference to infer one")
        audio_shape = batched_audio(base).shape

    device, dtype = pipe.device, pipe.torch_dtype

    video_latents = pipe.generate_noise(torch.Size(video_shape), seed=seed, rand_device="cpu").to(
        device=device, dtype=dtype
    )
    audio_latents = pipe.generate_noise(torch.Size(audio_shape), seed=seed, rand_device="cpu").to(
        device=device, dtype=dtype
    )

    # A pinned branch's noise is replaced outright, never `add_noise`: the grid floors at
    # sigma 0.1, so `add_noise(..., t=0)` would silently mean 10% noise.
    if pinned_clean == "video":
        given = given_video_latents if given_video_latents is not None else source_video_latents
        video_latents = batched_video(given).to(device=device, dtype=dtype)
    elif pinned_clean == "audio":
        given = given_audio_latents if given_audio_latents is not None else source_audio_latents
        audio_latents = batched_audio(given).to(device=device, dtype=dtype)

    video_positions = video_positions_for_latents(pipe, video_latents, frame_rate)
    audio_positions = audio_positions_for_latents(pipe, audio_latents)

    video_context = batched_context(video_context).to(device=device, dtype=dtype)
    audio_context = batched_context(audio_context).to(device=device, dtype=dtype)

    # In-context references are appended as clean tokens with a shifted time axis; without
    # the shift the DiT ignores the source entirely. Only tav2av carries them.
    ref_video = ref_video_positions = None
    ref_audio = ref_audio_positions = None
    if task == "tav2av":
        if source_video_latents is not None:
            ref_video = batched_video(source_video_latents).to(device=device, dtype=dtype)
            ref_video_positions = offset_time_axis(video_positions_for_latents(pipe, ref_video, frame_rate))
        if source_audio_latents is not None:
            ref_audio = batched_audio(source_audio_latents).to(device=device, dtype=dtype)
            ref_audio_positions = offset_time_axis(audio_positions_for_latents(pipe, ref_audio))

    # Intent-aware pins the preserved branch's text guidance to 1.0; the reference axis stays
    # at full strength.
    def _scale(value: float, which: str) -> float:
        return 1.0 if intent_aware == which else float(value)

    scale_v_text = _scale(video_text_cfg_scale, "video")
    scale_a_text = _scale(audio_text_cfg_scale, "audio")
    scale_v_ref = _scale(video_reference_cfg_scale, "video")
    scale_a_ref = _scale(audio_reference_cfg_scale, "audio")

    use_text_cfg = scale_v_text > 1.0 or scale_a_text > 1.0
    use_ref_cfg = (scale_v_ref > 1.0 or scale_a_ref > 1.0) and (
        ref_video is not None or ref_audio is not None
    )

    # ---- CFG anchors ----------------------------------------------------------
    zero_video_context = torch.zeros_like(video_context) if video_context is not None else None
    zero_audio_context = torch.zeros_like(audio_context) if audio_context is not None else None
    zero_ref_video = torch.zeros_like(ref_video) if ref_video is not None else None
    zero_ref_audio = torch.zeros_like(ref_audio) if ref_audio is not None else None

    uncond_video_context = zero_video_context
    uncond_audio_context = zero_audio_context
    if negative_video_context is not None and intent_aware != "video":
        uncond_video_context = batched_context(negative_video_context).to(device=device, dtype=dtype)
    if negative_audio_context is not None and intent_aware != "audio":
        uncond_audio_context = batched_context(negative_audio_context).to(device=device, dtype=dtype)

    # Each branch walks its own shifted grid at the trained pair. Sharing one grid would feed
    # the AV cross-attention modulation a sigma pair the network never saw.
    pipe.dual_scheduler = True
    pipe.scheduler.set_timesteps(num_inference_steps, shift=DEFAULT_VIDEO_SHIFT)
    pipe.scheduler_audio.set_timesteps(num_inference_steps, shift=DEFAULT_AUDIO_SHIFT)
    stages = _build_stages(mechanism=mechanism, branch=branch)
    sched_timesteps = pipe.scheduler.timesteps

    max_timestep = sched_timesteps[0].reshape(1).to(dtype=dtype, device=device)
    finished_timestep = torch.zeros_like(max_timestep)

    # A pinned-clean branch is given, not produced: reported finished from step zero.
    done: set[str] = {pinned_clean} if pinned_clean else set()

    # The refine grid rescales the main one (sigma * lambda, index for index) so both finish
    # on the same step.
    refine_schedulers: dict[str, Any] = {}
    lam = float(refine_lambda)
    if mechanism == "post-refine":
        if lam <= 0.0:
            raise ValueError(
                "post-refine needs --refine-lambda > 0: it is the fraction of noise added "
                "back to the stage-1 result before stage 2 re-denoises it. A pass at 0 "
                "would re-denoise nothing and destroy the stage-1 output."
            )
        if len(stages) > 1:
            # The refine pass enters the trajectory late, so its step count scales with lambda.
            main_steps = len(pipe.scheduler.timesteps)
            if refine_steps is None:
                # Floor at 2: one step from lambda to 0 is not a refinement pass.
                refine_steps = max(2, round(lam * main_steps))
            refine_steps = max(1, min(int(refine_steps), main_steps))

            # Rebuild the refining branch's own grid (audio uses shift 2.0).
            is_audio = branch == "audio"
            base_scheduler = pipe.scheduler_audio if is_audio else pipe.scheduler
            sched = copy.copy(base_scheduler)
            sched.set_timesteps(
                refine_steps, shift=DEFAULT_AUDIO_SHIFT if is_audio else DEFAULT_VIDEO_SHIFT
            )
            sched.sigmas = sched.sigmas * lam
            sched.timesteps = sched.timesteps * lam
            refine_schedulers[branch] = sched
            print(
                f"post-refine: {branch} branch, lambda={lam:g}, {refine_steps} steps "
                f"(main pass: {main_steps})",
                flush=True,
            )

    # ---- the forward ----------------------------------------------------------
    def predict(v_ctx, a_ctx, v_ref, a_ref):
        return pipe.model_fn(
            pipe.dit,
            video_latents=video_latents,
            video_context=v_ctx,
            video_positions=video_positions,
            video_patchifier=pipe.video_patchifier,
            audio_latents=audio_latents,
            audio_context=a_ctx,
            audio_positions=audio_positions,
            audio_patchifier=pipe.audio_patchifier,
            timestep=model_timestep,
            in_context_video_latents=v_ref,
            in_context_video_positions=ref_video_positions if v_ref is not None else None,
            in_context_audio_latents=a_ref,
            in_context_audio_positions=ref_audio_positions if a_ref is not None else None,
            split_ref_attention=True,
            split_ref_cross_attention=True,
            audio_timestep=audio_model_timestep,
        )

    def predict_cfg():
        """Apply guidance and return (video_pred, audio_pred)."""
        # cfg-free: a single forward at the real conditioning.
        if not use_text_cfg and not use_ref_cfg:
            return predict(video_context, audio_context, ref_video, ref_audio)

        # One live axis: two corners, anchored on the fully-unconditional one.
        if not (use_text_cfg and use_ref_cfg):
            if use_text_cfg:
                # Keep the reference in BOTH corners: this axis asks "what does the instruction
                # change", not "how do I get to a source at all".
                (v_u, a_u), (v_c, a_c) = [
                    predict(vc, ac, ref_video, ref_audio)
                    for vc, ac in (
                        (uncond_video_context, uncond_audio_context),
                        (video_context, audio_context),
                    )
                ]
                return (
                    v_u + scale_v_text * (v_c - v_u),
                    a_u + scale_a_text * (a_c - a_u),
                )
            (v_u, a_u), (v_c, a_c) = [
                predict(video_context, audio_context, vr, ar)
                for vr, ar in (
                    (zero_ref_video, zero_ref_audio),
                    (ref_video, ref_audio),
                )
            ]
            return (
                v_u + scale_v_ref * (v_c - v_u),
                a_u + scale_a_ref * (a_c - a_u),
            )

        # Both axes live: four conditioning corners.
        v_uncond, a_uncond = predict(uncond_video_context, uncond_audio_context, zero_ref_video, zero_ref_audio)
        v_text, a_text = predict(video_context, audio_context, zero_ref_video, zero_ref_audio)
        v_ref, a_ref = predict(uncond_video_context, uncond_audio_context, ref_video, ref_audio)
        v_full, a_full = predict(video_context, audio_context, ref_video, ref_audio)

        v_pred = additive_cfg(
            v_uncond, v_text, v_ref, v_full,
            scale_text=scale_v_text, scale_ref=scale_v_ref,
        )
        a_pred = additive_cfg(
            a_uncond, a_text, a_ref, a_full,
            scale_text=scale_a_text, scale_ref=scale_a_ref,
        )
        return v_pred, a_pred

    # ---- denoise --------------------------------------------------------------
    pipe.load_models_to_device(["dit"])

    for stage_idx, stage in enumerate(stages):
        # Stage 2 re-noises off the stage-1 result with an offset seed, so the added noise is
        # uncorrelated with this sample's initial noise.
        if refine_schedulers and stage_idx > 0:
            renoise_seed = seed + (9_000_001 if branch == "video" else 9_000_002)
            if branch == "video":
                video_latents = (1.0 - lam) * video_latents + lam * pipe.generate_noise(
                    video_latents.shape, seed=renoise_seed, rand_device="cpu"
                ).to(device=device, dtype=dtype)
            else:
                audio_latents = (1.0 - lam) * audio_latents + lam * pipe.generate_noise(
                    audio_latents.shape, seed=renoise_seed, rand_device="cpu"
                ).to(device=device, dtype=dtype)

        # The refine grid is never longer than the main one, so `progress_id` stays in range.
        stage_grid = (
            refine_schedulers[branch].timesteps
            if refine_schedulers and stage_idx > 0 and branch in refine_schedulers
            else sched_timesteps
        )
        stage_desc = f"denoise {stage_idx + 1}/{len(stages)}"
        for progress_id, timestep in enumerate(tqdm(stage_grid, desc=stage_desc, leave=True)):
            stage_timestep = timestep.reshape(1).to(dtype=dtype, device=device)
            # Audio reads its own grid at the same index, keeping the (sigma_v, sigma_a) pair on
            # the trained curve.
            audio_stage_timestep = (
                pipe.scheduler_audio.timesteps[progress_id].reshape(1).to(dtype=dtype, device=device)
            )

            def branch_timestep(name: str):
                if refine_schedulers and name in refine_schedulers and stage_idx > 0:
                    return refine_schedulers[name].timesteps[progress_id].reshape(1).to(
                        dtype=dtype, device=device
                    )
                if name in stage:
                    return audio_stage_timestep if name == "audio" else stage_timestep
                return finished_timestep if name in done else max_timestep

            model_timestep = branch_timestep("video")
            # AV cross-attention reads the OTHER branch's sigma, so the pair is a network input
            # and must always be stated.
            audio_model_timestep = branch_timestep("audio")

            video_pred, audio_pred = predict_cfg()

            # A refining branch steps on its own rescaled grid, not the main one.
            def scheduler_for(name: str):
                if refine_schedulers and name in refine_schedulers and stage_idx > 0:
                    return refine_schedulers[name]
                # Audio steps on scheduler_audio: step() derives the update from the grid it gets.
                return pipe.scheduler_audio if name == "audio" else pipe.scheduler

            if "video" in stage:
                video_latents = pipe.step(
                    scheduler_for("video"), video_latents,
                    progress_id=progress_id, noise_pred=video_pred,
                )
            if "audio" in stage and audio_pred is not None:
                audio_latents = pipe.step(
                    scheduler_for("audio"), audio_latents,
                    progress_id=progress_id, noise_pred=audio_pred,
                )

        done.update(stage)

    return {
        "video_latents": video_latents.detach(),
        "audio_latents": audio_latents.detach(),
    }
