"""Model loading.

One `model.safetensors` holds the DiT plus the base model's VAEs, vocoder and text
post-modules; the registry identifies it by the md5 of its sorted key:shape list, so
renaming is harmless but a new merge needs MODEL_CONFIGS stanzas. Gemma-3 is read
from its own repository.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import torch

from uniavc.core.loader import ModelConfig
from uniavc.pipelines.audio_video import AudioVideoPipeline

# One file holds every model; the registry maps its hash to each class that reads it.
UNIAVC_WEIGHTS_FILE = "model.safetensors"

# Where `scripts/download_weights.sh` puts the weights.
UNIAVC_CHECKPOINT = str(Path(__file__).resolve().parents[2] / "models" / "UniAVC" / UNIAVC_WEIGHTS_FILE)

# Gemma-3 text conditioning, also under models/.
GEMMA = str(Path(__file__).resolve().parents[2] / "models" / "gemma-3-12b-it-qat-q4_0-unquantized")


def _model_config(source: str, filename: str, **kwargs) -> ModelConfig:
    """Resolve a weight file from a local directory, a single file, or a Hub repo id."""
    candidate = Path(source)
    if candidate.exists():
        if candidate.is_dir():
            matches = sorted(glob.glob(str(candidate / filename)))
            if not matches:
                raise FileNotFoundError(f"no file matched {candidate / filename!r}")
            path = matches[0] if len(matches) == 1 else matches
        else:
            path = str(candidate)
        return ModelConfig(path=path, **kwargs)
    return ModelConfig(model_id=source, origin_file_pattern=filename, **kwargs)


def _checkpoint_config(**kwargs) -> ModelConfig:
    """Resolve the UniAVC checkpoint from its local copy under `models/`."""
    if not Path(UNIAVC_CHECKPOINT).exists():
        raise FileNotFoundError(
            f"no checkpoint at {UNIAVC_CHECKPOINT}; run `bash scripts/download_weights.sh`"
        )
    return ModelConfig(path=UNIAVC_CHECKPOINT, **kwargs)


def vram_config(offload: bool, device: str = "cuda", dtype=torch.bfloat16) -> dict[str, Any]:
    """Placement config. `offload=True` keeps weights on CPU between uses."""
    if offload:
        return {
            "offload_dtype": dtype, "offload_device": "cpu",
            "onload_dtype": dtype, "onload_device": device,
            "preparing_dtype": dtype, "preparing_device": device,
            "computation_dtype": dtype, "computation_device": device,
        }
    return {
        "offload_dtype": None, "offload_device": None,
        "onload_dtype": dtype, "onload_device": device,
        "preparing_dtype": dtype, "preparing_device": device,
        "computation_dtype": dtype, "computation_device": device,
    }


def load_pipeline(
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    offload: bool = True,
) -> AudioVideoPipeline:
    """Build the joint audio-video pipeline from the local weights under `models/`.

    Both model paths are fixed: the checkpoint and Gemma-3 are read from
    `models/`, where `scripts/download_weights.sh` places them.
    """
    cfg = vram_config(offload, device, dtype)

    # Loading ~46 GB is the quiet part of a run; report each phase.
    print(f"loading weights from {UNIAVC_CHECKPOINT} (offload={offload}) ...", flush=True)
    model_configs = [_checkpoint_config(**cfg)]
    print(f"checkpoint loaded; loading text encoder from {GEMMA} ...", flush=True)
    model_configs.append(_model_config(GEMMA, "model-*.safetensors", **cfg))

    pipe = AudioVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=GEMMA),
    )

    pipe.scheduler.set_timesteps(30)
    pipe.eval()
    print("models ready.", flush=True)
    return pipe
