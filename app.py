"""UniAVC Gradio demo: one tab per task, plus post-refine and intent-aware on the
editing tab."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from uniavc.inference import media as media_io
from uniavc.inference.common import batched_audio, batched_video
from uniavc.inference.cli import (
    NEGATIVE_PROMPT_DEFAULT, NEGATIVE_PROMPT_TASKS, TASK_CFG_DEFAULTS, TASK_INPUTS,
)
from uniavc.inference.pipeline import load_pipeline
from uniavc.inference.sampler import sample_joint_av

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
LOGO = ROOT / "assets" / "logo.svg"

# Model weights are read from ./models (see uniavc/inference/pipeline.py).
DEVICE = os.environ.get("UNIAVC_DEVICE", "cuda")

# The pipeline is built once and reused across requests.
_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        _pipe = load_pipeline(
            device=DEVICE,
            offload=os.environ.get("UNIAVC_OFFLOAD", "1") == "1",
        )
    return _pipe


def _generate(
    *,
    task: str,
    prompt: str,
    video_path: str | None = None,
    audio_path: str | None = None,
    num_inference_steps: int = 30,
    cfg_scales: tuple[float, float, float, float] | None = None,
    seed: int = 42,
    post_refine: str | None = None,
    refine_lambda: float = 0.5,
    intent_aware: str | None = None,
    progress=gr.Progress(),
) -> str:
    """Encode inputs, sample, decode, and return the output mp4 path."""
    pipe = get_pipe()
    needs_video, _ = TASK_INPUTS[task]

    progress(0.05, desc="Encoding prompt")
    video_context = media_io.encode_video_context(pipe, prompt)
    audio_context = media_io.encode_audio_context(pipe, prompt)
    # The generation tasks guide against the built-in LTX-2.3 negative prompt; the edit
    # tasks anchor on the training null.
    negative_video = negative_audio = None
    if task in NEGATIVE_PROMPT_TASKS:
        negative_video = media_io.encode_video_context(pipe, NEGATIVE_PROMPT_DEFAULT)
        negative_audio = media_io.encode_audio_context(pipe, NEGATIVE_PROMPT_DEFAULT)

    src_video = src_audio = given_video = given_audio = None
    video_shape = audio_shape = None
    frame_rate = media_io.DEFAULT_FRAME_RATE

    if needs_video:
        progress(0.15, desc="Encoding input video")
        height, width, num_frames, frame_rate = media_io.resolve_video_geometry(
            video_path, height=media_io.DEFAULT_HEIGHT, width=media_io.DEFAULT_WIDTH,
            num_frames=media_io.DEFAULT_NUM_FRAMES,
        )
        frames = media_io.load_video_frames(
            video_path, height=height, width=width, num_frames=num_frames, frame_rate=frame_rate
        )
        latents = media_io.encode_video(pipe, frames)
        duration = len(frames) / frame_rate
        # encode_video returns [C,F,H,W]; the sampler works in batched [B,C,F,H,W].
        video_shape = batched_video(latents).shape

        if task == "tv2a":
            given_video = latents
            audio_shape = media_io.generation_audio_latent_shape(num_frames, frame_rate)
        else:
            src_video = latents
            waveform, sample_rate = media_io.load_waveform(
                audio_path or video_path, duration=duration
            )
            src_audio = media_io.encode_audio(pipe, waveform, sample_rate)
            audio_shape = batched_audio(src_audio).shape

    elif task == "ta2v":
        progress(0.15, desc="Encoding input audio")
        duration = media_io.DEFAULT_NUM_FRAMES / frame_rate
        waveform, sample_rate = media_io.load_waveform(audio_path, duration=duration)
        given_audio = media_io.encode_audio(pipe, waveform, sample_rate)
        audio_shape = batched_audio(given_audio).shape
        video_shape = media_io.video_latent_shape(
            media_io.DEFAULT_HEIGHT, media_io.DEFAULT_WIDTH, media_io.DEFAULT_NUM_FRAMES
        )

    else:  # t2av
        duration = media_io.DEFAULT_NUM_FRAMES / frame_rate
        video_shape = media_io.video_latent_shape(
            media_io.DEFAULT_HEIGHT, media_io.DEFAULT_WIDTH, media_io.DEFAULT_NUM_FRAMES
        )
        audio_shape = media_io.audio_latent_shape(
            duration, pinned_frames=media_io.TRAIN_AUDIO_FRAMES
        )

    progress(0.3, desc="Sampling")
    latents = sample_joint_av(
        pipe,
        task=task,
        video_context=video_context,
        audio_context=audio_context,
        num_inference_steps=num_inference_steps,
        seed=seed,
        frame_rate=frame_rate,
        source_video_latents=src_video,
        source_audio_latents=src_audio,
        given_video_latents=given_video,
        given_audio_latents=given_audio,
        video_shape=video_shape,
        audio_shape=audio_shape,
        video_text_cfg_scale=(cfg_scales or TASK_CFG_DEFAULTS[task])[0],
        audio_text_cfg_scale=(cfg_scales or TASK_CFG_DEFAULTS[task])[1],
        video_reference_cfg_scale=(cfg_scales or TASK_CFG_DEFAULTS[task])[2],
        audio_reference_cfg_scale=(cfg_scales or TASK_CFG_DEFAULTS[task])[3],
        post_refine=post_refine,
        refine_lambda=refine_lambda,
        intent_aware=intent_aware,
        negative_video_context=negative_video,
        negative_audio_context=negative_audio,
    )

    progress(0.85, desc="Decoding")
    frames = media_io.decode_video(pipe, latents["video_latents"])
    waveform, sample_rate = media_io.decode_audio(pipe, latents["audio_latents"])

    out = Path(tempfile.mkdtemp()) / f"{task}.mp4"
    media_io.save_video_audio(frames, waveform, str(out), fps=frame_rate, sample_rate=sample_rate)
    progress(1.0, desc="Done")
    return str(out)


# --------------------------------------------------------------------- examples

def _sample(name: str) -> str | None:
    """Absolute path to a bundled clip, or None when the file is absent."""
    path = SAMPLES / name
    return str(path) if path.exists() else None


# (card caption, source file, instruction). The prompts are the ones in the README.
TAV2AV_EXAMPLES = [
    ("Subject editing", "sample_1.mp4",
     "Replace the man's dark brown T-shirt with a blue cycling jersey with white stripes "
     "on the sleeves. Keep the pull-up bar, the beige wall and the lighting unchanged."),
    ("Subject addition", "sample_2.mp4",
     'Include a man in a grey coat seated in the front seat, saying: "We should have left '
     'an hour ago. Nobody checked the traffic or the weather."'),
    ("Subject removal", "sample_3.mp4",
     "Remove the table lamp from the foreground and fill in the window and curtains behind "
     "it. Keep the man and his dialogue unchanged."),
    ("Speech editing", "sample_4.mp4",
     'Have her say: "I did not expect the city to be this quiet at dawn."'),
    ("Background editing", "sample_5.mp4",
     "Replace the background with a warm, sunlit kitchen with wooden cabinets. Keep the "
     "man, his clothing and his speech unchanged."),
]

T2AV_EXAMPLES = [
    ("Man in a cafe",
     "A man in a grey shirt sits at a wooden table in a quiet cafe, a window with soft "
     'daylight behind him and a few empty chairs further back. He says: "We can start '
     'whenever you are ready, but let me finish this first." Static medium shot.'),
    ("Woman in a studio",
     "A woman with short dark hair stands against a seamless blue-grey studio backdrop and "
     'speaks to camera. She says: "Take your time, there is no rush at all." Static '
     "medium close-up, soft key light from the front left."),
    ("Street at dusk",
     "A young man in a denim jacket leans against a low brick wall on a quiet residential "
     'street at dusk, warm window light behind him. He says: "I thought you were not '
     'coming back tonight." Static medium shot, shallow depth of field.'),
]

# Audio -> video: the soundtrack comes from each clip, the description says what to draw.
TA2V_EXAMPLES = [
    ("Woman on the phone", "sample_6.mp4",
     "Medium close-up of a blonde woman in glasses and a purple blouse, holding a phone to "
     "her ear in a dark-panelled office, a framed print on the wall behind her, soft "
     "side light, static shot."),
    ("Man at a console", "sample_8.mp4",
     "A man with long hair and glasses in a plaid shirt speaks to camera in front of a wall "
     "of glowing control panels, a starfield on an old CRT monitor beside him, cool "
     "blue-green light, static medium shot."),
    ("Man in a shirt and tie", "sample_7.mp4",
     "A man in a pale shirt and dark tie speaks to camera in a bright office corridor, "
     "glass partitions and a window behind him, even overhead light, static medium "
     "close-up."),
]

# Video -> audio: the picture is given, the description says what it should sound like.
TV2A_EXAMPLES = [
    ("Kitchen", "sample_9.mp4",
     "A quiet domestic kitchen. A refrigerator hums, crockery clinks faintly, and the man "
     "speaks softly in a close, dry voice."),
    ("Garden", "sample_10.mp4",
     "An outdoor garden in daylight. Birds sing and a light breeze moves through the "
     "leaves while the man speaks at a conversational volume, his voice close and "
     "unreflected."),
    ("Break room", "sample_11.mp4",
     "An office break room with faint background chatter, a chair scraping, and the man "
     "speaking at a conversational volume, close and slightly reverberant."),
]



# Cards take the clips' 16:9 shape instead of Gradio's square crop, and the strip is
# exactly as tall as one row, so nothing scrolls.
CSS = """
.uni-cards .gallery-container { height: auto !important; }
.uni-cards .grid-wrap { overflow: visible !important; }
/* Gallery pins min-height 450px on wide screens when height is "auto" -- the blank
   band under the cards. */
.uni-cards .fixed-height { min-height: 0 !important; max-height: none !important; }
.uni-cards .grid-container { grid-auto-rows: auto !important; }
.uni-cards .thumbnail-item {
  aspect-ratio: 16 / 9 !important;
  height: auto !important;
}
.uni-cards .thumbnail-item video,
.uni-cards .thumbnail-item img {
  object-fit: cover !important;
}
"""

# --------------------------------------------------------------------- theme

# The project page's palette: near-black canvas, hairline panels, wordmark gradient as
# the single accent.
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.zinc,
    font=[gr.themes.GoogleFont("Noto Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0a0b0e",
    body_background_fill_dark="#0a0b0e",
    body_text_color="#eef0f4",
    body_text_color_dark="#eef0f4",
    body_text_color_subdued="#a8aeba",
    body_text_color_subdued_dark="#a8aeba",
    background_fill_primary="#101218",
    background_fill_primary_dark="#101218",
    background_fill_secondary="#15181f",
    background_fill_secondary_dark="#15181f",
    block_background_fill="#101218",
    block_background_fill_dark="#101218",
    block_border_color="rgba(255,255,255,0.10)",
    block_border_color_dark="rgba(255,255,255,0.10)",
    block_radius="16px",
    block_label_background_fill="#15181f",
    block_label_background_fill_dark="#15181f",
    block_label_text_color="#a8aeba",
    block_label_text_color_dark="#a8aeba",
    block_title_text_color="#eef0f4",
    block_title_text_color_dark="#eef0f4",
    input_background_fill="#0a0b0e",
    input_background_fill_dark="#0a0b0e",
    input_border_color="rgba(255,255,255,0.10)",
    input_border_color_dark="rgba(255,255,255,0.10)",
    button_primary_background_fill=(
        "linear-gradient(100deg, #5aa2ff 0%, #a78bfa 48%, #45d789 100%)"
    ),
    button_primary_background_fill_dark=(
        "linear-gradient(100deg, #5aa2ff 0%, #a78bfa 48%, #45d789 100%)"
    ),
    button_primary_text_color="#08090c",
    button_primary_text_color_dark="#08090c",
    button_secondary_background_fill="#15181f",
    button_secondary_background_fill_dark="#15181f",
    button_secondary_text_color="#eef0f4",
    button_secondary_text_color_dark="#eef0f4",
    slider_color="#a78bfa",
    slider_color_dark="#a78bfa",
)


def _logo_html() -> str:
    """The wordmark, inlined so it keeps its gradient and is not restyled by the theme."""
    if not LOGO.exists():
        return ""
    svg = LOGO.read_text()
    return (
        '<div style="display:flex;justify-content:center;padding:26px 0 8px 0;">'
        f'<div style="width:min(620px,80%);">{svg}</div></div>'
    )


# --------------------------------------------------------------------- widgets

def _steps() -> gr.Slider:
    return gr.Slider(10, 50, value=30, step=1, label="Denoising steps")


def _seed() -> gr.Number:
    return gr.Number(value=42, label="Seed", precision=0)


def _guidance(task: str):
    """Guidance sliders, defaulted per task from TASK_CFG_DEFAULTS.

    The reference axes only exist on tav2av: it is the only task that packs a source as
    an in-context reference. Every other task reads its given signal from the target
    slot, so both reference scales are inert there and are not shown.
    """
    vt, at, vr, ar = TASK_CFG_DEFAULTS[task]
    with gr.Accordion("Guidance", open=False):
        v_text = gr.Slider(1.0, 8.0, value=vt, step=0.1, label="Video text guidance")
        a_text = gr.Slider(1.0, 8.0, value=at, step=0.1, label="Audio text guidance")
        v_ref = a_ref = None
        if task == "tav2av":
            v_ref = gr.Slider(1.0, 8.0, value=vr, step=0.1, label="Video reference guidance")
            a_ref = gr.Slider(1.0, 8.0, value=ar, step=0.1, label="Audio reference guidance")
    return v_text, a_text, v_ref, a_ref


with gr.Blocks(title="UniAVC") as demo:
    gr.HTML(_logo_html())

    with gr.Tab("🎬 Joint editing"):
        je_card_rows = [
            (path, caption)
            for caption, name, _ in TAV2AV_EXAMPLES
            if (path := _sample(name))
        ]
        je_prompts = {
            path: prompt for (_, name, prompt), (path, _) in zip(TAV2AV_EXAMPLES, je_card_rows)
        }
        je_gallery = gr.Gallery(
            value=je_card_rows,
            label="Examples",
            columns=5,
            rows=1,
            height="auto",
            object_fit="cover",
            elem_classes=["uni-cards"],
            allow_preview=False,
            show_label=False,
            visible=bool(je_card_rows),
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                je_video = gr.Video(label="Input clip", sources=["upload"])
                je_prompt = gr.Textbox(
                    label="Edit instruction", lines=3,
                    placeholder="Replace the background with a dim wood-panelled study, "
                                "keeping the person unchanged.",
                )
                je_steps, je_seed = _steps(), _seed()
                je_vt, je_at, je_vr, je_ar = _guidance("tav2av")
                with gr.Accordion("Self-refinement", open=False):
                    je_refine_branch = gr.Radio(
                        ["off", "video", "audio"], value="off", label="Branch to refine"
                    )
                    je_lambda = gr.Slider(
                        0.05, 1.0, value=0.5, step=0.05, label="Refinement lambda"
                    )
                with gr.Accordion("Intent-aware editing", open=False):
                    je_intent_branch = gr.Radio(
                        ["off", "video", "audio"], value="off", label="Branch to preserve"
                    )
                je_run = gr.Button("Edit", variant="primary")
            with gr.Column(scale=1):
                je_out = gr.Video(label="Result")

        def _pick_je(evt: gr.SelectData):
            path = je_card_rows[evt.index][0]
            return [path, je_prompts[path]]

        je_gallery.select(_pick_je, None, [je_video, je_prompt])

        je_run.click(
            lambda v, p, s, sd, vt, at, vr, ar, b, lam, ib: _generate(
                task="tav2av", prompt=p, video_path=v,
                num_inference_steps=int(s),
                cfg_scales=(vt, at, vr, ar),
                seed=int(sd),
                post_refine=None if b == "off" else b,
                refine_lambda=lam,
                intent_aware=None if ib == "off" else ib,
            ),
            inputs=[je_video, je_prompt, je_steps, je_seed,
                    je_vt, je_at, je_vr, je_ar,
                    je_refine_branch, je_lambda, je_intent_branch],
            outputs=je_out,
        )

    with gr.Tab("✨ Joint generation"):
        tg_labels = gr.Radio(
            [label for label, _ in T2AV_EXAMPLES],
            label="Example prompts",
            value=None,
            interactive=True,
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                tg_prompt = gr.Textbox(
                    label="Prompt", lines=4,
                    placeholder="A girl is very happy. She says: I enjoy working with UniAVC.",
                )
                tg_steps, tg_seed = _steps(), _seed()
                tg_vt, tg_at, _, _ = _guidance("t2av")
                tg_run = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                tg_out = gr.Video(label="Result")

        # The radio itself must be the input: the handler reads its value.
        tg_labels.change(
            lambda label: next((prompt for name, prompt in T2AV_EXAMPLES if name == label), ""),
            tg_labels,
            tg_prompt,
        )

        tg_run.click(
            lambda p, s, sd, vt, at: _generate(
                task="t2av", prompt=p, num_inference_steps=int(s),
                cfg_scales=(vt, at, *TASK_CFG_DEFAULTS["t2av"][2:]), seed=int(sd),
            ),
            inputs=[tg_prompt, tg_steps, tg_seed, tg_vt, tg_at],
            outputs=tg_out,
        )

    with gr.Tab("🔊 Audio → video"):
        av_rows = [(path, label) for label, name, _ in TA2V_EXAMPLES if (path := _sample(name))]
        av_prompts = {path: prompt for (_, name, prompt), (path, _) in zip(TA2V_EXAMPLES, av_rows)}
        av_gallery = gr.Gallery(
            value=av_rows,
            label="Examples",
            columns=3,
            rows=1,
            height="auto",
            object_fit="cover",
            elem_classes=["uni-cards"],
            allow_preview=False,
            show_label=False,
            visible=bool(av_rows),
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                av_audio = gr.Audio(label="Input audio", type="filepath")
                av_prompt = gr.Textbox(
                    label="Prompt", lines=3,
                    placeholder="A woman speaks directly to the camera in a sunlit studio.",
                )
                av_steps, av_seed = _steps(), _seed()
                av_vt, av_at, _, _ = _guidance("ta2v")
                av_run = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                av_out = gr.Video(label="Result")

        if av_rows:
            def _pick_av(evt: gr.SelectData):
                path = av_rows[evt.index][0]
                return [path, av_prompts[path]]

            av_gallery.select(_pick_av, None, [av_audio, av_prompt])

        av_run.click(
            lambda a, p, s, sd, vt, at: _generate(
                task="ta2v", prompt=p, audio_path=a, num_inference_steps=int(s),
                cfg_scales=(vt, at, *TASK_CFG_DEFAULTS["ta2v"][2:]), seed=int(sd),
            ),
            inputs=[av_audio, av_prompt, av_steps, av_seed, av_vt, av_at],
            outputs=av_out,
        )

    with gr.Tab("🎥 Video → audio"):
        va_rows = [(path, label) for label, name, _ in TV2A_EXAMPLES if (path := _sample(name))]
        va_prompts = {path: prompt for (_, name, prompt), (path, _) in zip(TV2A_EXAMPLES, va_rows)}
        va_gallery = gr.Gallery(
            value=va_rows,
            label="Examples",
            columns=3,
            rows=1,
            height="auto",
            object_fit="cover",
            elem_classes=["uni-cards"],
            allow_preview=False,
            show_label=False,
            visible=bool(va_rows),
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                va_video = gr.Video(label="Input video", sources=["upload"])
                va_prompt = gr.Textbox(
                    label="Prompt", lines=3,
                    placeholder="Ambient audio of gentle rainfall on a window, indoors.",
                )
                va_steps, va_seed = _steps(), _seed()
                va_vt, va_at, _, _ = _guidance("tv2a")
                va_run = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                va_out = gr.Video(label="Result")

        if va_rows:
            def _pick_va(evt: gr.SelectData):
                path = va_rows[evt.index][0]
                return [path, va_prompts[path]]

            va_gallery.select(_pick_va, None, [va_video, va_prompt])

        va_run.click(
            lambda v, p, s, sd, vt, at: _generate(
                task="tv2a", prompt=p, video_path=v, num_inference_steps=int(s),
                cfg_scales=(vt, at, *TASK_CFG_DEFAULTS["tv2a"][2:]), seed=int(sd),
            ),
            inputs=[va_video, va_prompt, va_steps, va_seed, va_vt, va_at],
            outputs=va_out,
        )


if __name__ == "__main__":
    launch_kwargs = {
        "server_name": os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        "server_port": int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    }
    # Gradio 6 moved `theme` from the Blocks constructor to launch().
    if int(gr.__version__.split(".")[0]) >= 6:
        demo.queue().launch(theme=THEME, css=CSS, **launch_kwargs)
    else:
        demo.theme = THEME
        demo.css = CSS
        demo.queue().launch(**launch_kwargs)
