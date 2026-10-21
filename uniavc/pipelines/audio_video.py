"""Joint audio-video pipeline: model slots, loading, and the packed DiT forward."""

import torch
from typing import Union

from ..core.device.npu_compatible_device import get_device_type
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline
from ..diffusion.flow_match import FlowMatchScheduler
from ..models.audio_vae import AudioDecoder, AudioEncoder, AudioPatchifier, AudioProcessor, AudioVocoder
from ..models.dit import DiTModel
from ..models.text_encoder import GemmaTokenizer, TextEncoder, TextEncoderPostModules
from ..models.upsampler import LatentUpsampler
from ..models.video_vae import VideoDecoder, VideoEncoder, VideoLatentPatchifier


class AudioVideoPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=32,
            width_division_factor=32,
            time_division_factor=8,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler("UniAVC")
        # Audio branch's own grid; only read when `dual_scheduler` is on.
        self.scheduler_audio = FlowMatchScheduler("UniAVC")
        self.dual_scheduler = False
        self.text_encoder: TextEncoder = None
        self.tokenizer: GemmaTokenizer = None
        self.text_encoder_post_modules: TextEncoderPostModules = None
        self.dit: DiTModel = None
        self.video_vae_encoder: VideoEncoder = None
        self.video_vae_decoder: VideoDecoder = None
        self.audio_vae_encoder: AudioEncoder = None
        self.audio_vae_decoder: AudioDecoder = None
        self.audio_vocoder: AudioVocoder = None
        self.upsampler: LatentUpsampler = None

        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)
        self.audio_processor = AudioProcessor()
        self.model_fn = model_fn

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
    ):
        pipe = AudioVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs)

        pipe.text_encoder = model_pool.fetch_model("text_encoder")
        tokenizer_config.download_if_necessary()
        pipe.tokenizer = GemmaTokenizer(tokenizer_path=tokenizer_config.path)
        pipe.text_encoder_post_modules = model_pool.fetch_model("text_encoder_post_modules")
        pipe.dit = model_pool.fetch_model("dit")
        pipe.video_vae_encoder = model_pool.fetch_model("video_vae_encoder")
        pipe.video_vae_decoder = model_pool.fetch_model("video_vae_decoder")
        pipe.audio_vae_decoder = model_pool.fetch_model("audio_vae_decoder")
        pipe.audio_vocoder = model_pool.fetch_model("audio_vocoder")
        pipe.upsampler = model_pool.fetch_model("latent_upsampler")
        pipe.audio_vae_encoder = model_pool.fetch_model("audio_vae_encoder")

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe


def model_fn(
    dit: DiTModel,
    video_latents=None,
    video_context=None,
    video_positions=None,
    video_patchifier=None,
    audio_latents=None,
    audio_context=None,
    audio_positions=None,
    audio_patchifier=None,
    timestep=None,
    in_context_video_latents=None,
    in_context_video_positions=None,
    in_context_audio_latents=None,
    in_context_audio_positions=None,
    split_ref_attention=True,
    split_ref_cross_attention=True,
    audio_timestep=None,
):
    """One DiT forward over a joint AV pair, returning unpatchified noise predictions.

    The in-context reference of each stream is appended as clean (sigma 0) tokens; the
    target/reference split travels to the DiT as `target_token_lengths`, which is what lets
    it wall the reference off from the noised target without a dense mask.
    """
    timestep = timestep.float() / 1000.
    audio_sigma = None if audio_timestep is None else audio_timestep.float() / 1000.

    _, _, f, h, w = video_latents.shape
    video_latents = video_patchifier.patchify(video_latents)
    seq_len_video = video_latents.shape[1]

    if in_context_video_latents is not None:
        ref = video_patchifier.patchify(in_context_video_latents)
        video_latents = torch.cat([video_latents, ref], dim=1)
        video_positions = torch.cat([video_positions, in_context_video_positions], dim=2)

    if audio_latents is not None:
        _, c_a, _, mel_bins = audio_latents.shape
        audio_latents = audio_patchifier.patchify(audio_latents)
        seq_len_audio = audio_latents.shape[1]
        if in_context_audio_latents is not None:
            ref = audio_patchifier.patchify(in_context_audio_latents)
            audio_latents = torch.cat([audio_latents, ref], dim=1)
            audio_positions = torch.cat([audio_positions, in_context_audio_positions], dim=2)
    else:
        seq_len_audio = None

    # Passed unconditionally, matching training: the packer always supplies these, so a
    # reference-free row has target length == segment length and row 0 of the token-role
    # embedding lands on every token. Gating on "a reference exists" would drop that constant
    # per-token bias from every block.
    video_target_token_length = seq_len_video
    audio_target_token_length = seq_len_audio

    # One sample per stream: cu_seqlens is just [0, N].
    cu_video = torch.tensor([0, video_latents.shape[1]], device=video_latents.device, dtype=torch.int32)
    cu_audio = None
    if audio_latents is not None:
        cu_audio = torch.tensor([0, audio_latents.shape[1]], device=audio_latents.device, dtype=torch.int32)

    vx, ax = dit.forward_packed_av(
        video_latents=video_latents.squeeze(0),
        video_positions=video_positions if video_positions.ndim == 4 else video_positions.unsqueeze(0),
        video_context=video_context,
        video_timesteps=timestep.reshape(-1),
        video_cu_seqlens=cu_video,
        audio_latents=None if audio_latents is None else audio_latents.squeeze(0),
        audio_positions=None if audio_positions is None else (
            audio_positions if audio_positions.ndim == 4 else audio_positions.unsqueeze(0)
        ),
        audio_context=audio_context,
        audio_timesteps=None if audio_latents is None else (timestep if audio_sigma is None else audio_sigma).reshape(-1),
        audio_cu_seqlens=cu_audio,
        sigma=timestep.reshape(-1),
        audio_sigma=None if audio_sigma is None else audio_sigma.reshape(-1),
        video_target_token_lengths=None if video_target_token_length is None else [video_target_token_length],
        audio_target_token_lengths=None if audio_target_token_length is None else [audio_target_token_length],
        split_ref_attention=split_ref_attention,
        split_ref_cross_attention=split_ref_cross_attention,
    )

    vx = video_patchifier.unpatchify_video(vx.unsqueeze(0)[:, :seq_len_video, ...], f, h, w)
    ax = None if ax is None else audio_patchifier.unpatchify_audio(
        ax.unsqueeze(0)[:, :seq_len_audio, ...], c_a, mel_bins
    )
    return vx, ax
