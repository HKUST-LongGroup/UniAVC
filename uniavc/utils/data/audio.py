import wave
from pathlib import Path

import torch


def convert_to_mono(audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to mono by averaging channels.
    Supports [C, T] or [B, C, T]. Output shape: [1, T] or [B, 1, T].
    """
    return audio_tensor.mean(dim=-2, keepdim=True)


def convert_to_stereo(audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert audio to stereo.
    Supports [C, T] or [B, C, T]. Duplicate mono, keep stereo.
    """
    if audio_tensor.size(-2) == 1:
        return audio_tensor.repeat(1, 2, 1) if audio_tensor.dim() == 3 else audio_tensor.repeat(2, 1)
    return audio_tensor


def resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    """Resample waveform to target sample rate if needed."""
    if source_rate == target_rate:
        return waveform
    import torchaudio

    resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
    return resampled.to(dtype=waveform.dtype)


def read_audio_with_torchcodec(
    path: str,
    start_time: float = 0,
    duration: float | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Read audio from file natively using torchcodec, with optional start time and duration.
    
    Args:
        path (str): The file path to the audio file.
        start_time (float, optional): The start time in seconds to read from. Defaults to 0.
        duration (float | None, optional): The duration in seconds to read. If None, reads until the end. Defaults to None.
        
    Returns:
        tuple[torch.Tensor, int]: A tuple containing the audio tensor and the sample rate.
            The audio tensor shape is [C, T] where C is the number of channels and T is the number of audio frames.
    """
    from torchcodec.decoders import AudioDecoder
    decoder = AudioDecoder(path)
    stop_seconds = None if duration is None else start_time + duration
    waveform = decoder.get_samples_played_in_range(start_seconds=start_time, stop_seconds=stop_seconds).data
    return waveform, decoder.metadata.sample_rate


def read_audio_with_soundfile(
    path: str,
    start_time: float = 0,
    duration: float | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Read audio from file using libsndfile (the ``soundfile`` package), bypassing FFmpeg.

    This is the fallback path for plain audio containers (wav/flac/ogg) that the
    torchcodec/FFmpeg backend mis-decodes. FFmpeg 4 in particular raises
    "The frame has N channels, expected M" on mono float PCM wavs whose header
    declares an unknown channel layout (e.g. the ISSE ta2a sources). libsndfile
    reads those natively.

    Args:
        path (str): The file path to the audio file.
        start_time (float, optional): The start time in seconds to read from. Defaults to 0.
        duration (float | None, optional): The duration in seconds to read. If None, reads until the end. Defaults to None.

    Returns:
        tuple[torch.Tensor, int]: A tuple containing the audio tensor and the sample rate.
            The audio tensor shape is [C, T] where C is the number of channels and T is the number of audio frames.
    """
    import soundfile as sf

    with sf.SoundFile(path) as f:
        sample_rate = int(f.samplerate)
        start_frame = int(round(start_time * sample_rate)) if start_time else 0
        if start_frame:
            f.seek(start_frame)
        num_frames = -1 if duration is None else int(round(duration * sample_rate))
        # always_2d -> [T, C]; float32 matches torchcodec's output dtype.
        data = f.read(frames=num_frames, dtype="float32", always_2d=True)

    # [T, C] -> [C, T]; .copy() because torch.from_numpy needs a contiguous, writable buffer.
    waveform = torch.from_numpy(data.T.copy())
    return waveform, sample_rate


def read_audio_with_av(
    path: str,
    start_time: float = 0,
    duration: float | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Read the audio track of any FFmpeg-readable file (mp4/mkv/wav/...) via PyAV.

    This is the container-capable fallback. It exists because the video-reading
    tasks take their audio from the input *video* -- `--video clip.mp4` with no
    `--audio` means "use this clip's own track" -- and neither of the other two
    backends can be relied on for that: torchcodec is an optional install, and
    torchaudio 2.8 dropped its FFmpeg decoder, so its `load` raises
    "Couldn't find appropriate backend" for an mp4 on a machine where the
    torchcodec extension is absent. PyAV is already a hard dependency (the
    output muxer uses it), so routing container audio through it costs nothing.

    Args:
        path (str): The file path to read from.
        start_time (float, optional): Seconds to skip from the start. Defaults to 0.
        duration (float | None, optional): Seconds to read. None reads to the end.

    Returns:
        tuple[torch.Tensor, int]: waveform [C, T] as float32, and the sample rate.
            Channels are whatever the file holds; callers widen to stereo as needed.
    """
    import av
    import numpy as np

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError(f"{path!r} has no audio stream")

        sample_rate = int(stream.codec_context.sample_rate)
        # Decode to packed planar float32 at the file's own rate. Asking for a layout instead of
        # "mono" here would drop channels; "stereo" makes av duplicate mono, which matches the other
        # backends' [C, T] contract.
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout=stream.codec_context.layout.name or "stereo",
            rate=sample_rate,
        )

        start_sample = int(round(start_time * sample_rate)) if start_time else 0
        stop_sample = (
            None if duration is None else start_sample + int(round(duration * sample_rate))
        )

        chunks = []
        total = 0
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                data = resampled.to_ndarray()
                if stop_sample is not None and total + data.shape[-1] > stop_sample:
                    data = data[..., : stop_sample - total]
                if data.shape[-1]:
                    chunks.append(data)
                    total += data.shape[-1]
            if stop_sample is not None and total >= stop_sample:
                break
        # Flush the resampler's internal buffer.
        if stop_sample is None or total < stop_sample:
            for resampled in resampler.resample(None):
                data = resampled.to_ndarray()
                if stop_sample is not None and total + data.shape[-1] > stop_sample:
                    data = data[..., : stop_sample - total]
                if data.shape[-1]:
                    chunks.append(data)
                    total += data.shape[-1]

    if not chunks:
        raise RuntimeError(f"decoded no audio samples from {path!r}")
    waveform = torch.from_numpy(np.concatenate(chunks, axis=-1).astype("float32"))
    if start_sample and duration is None:
        pass  # trimming handled by the caller's pinning step
    return waveform, sample_rate


def _read_audio_trying_all(
    path: str,
    start_time: float = 0,
    duration: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Try each backend in order of capability and return the first that decodes.

    torchcodec is fastest where it is installed; PyAV covers every container the
    output muxer already handles; libsndfile is the last resort for plain audio.
    """
    errors = []
    for name, reader in (
        ("torchcodec", read_audio_with_torchcodec),
        ("av", read_audio_with_av),
        ("soundfile", read_audio_with_soundfile),
    ):
        try:
            return reader(path, start_time, duration)
        except Exception as exc:            # noqa: BLE001 - report all three on failure
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        f"no audio backend could read {path!r}; tried " + "; ".join(errors)
    )


def read_audio(
    path: str,
    start_time: float = 0,
    duration: float | None = None,
    resample: bool = False,
    resample_rate: int = 48000,
    backend: str = "torchcodec",
) -> tuple[torch.Tensor, int]:
    """
    Read audio from file, with optional start time, duration, and resampling.
    
    Args:
        path (str): The file path to the audio file.
        start_time (float, optional): The start time in seconds to read from. Defaults to 0.
        duration (float | None, optional): The duration in seconds to read. If None, reads until the end. Defaults to None.
        resample (bool, optional): Whether to resample the audio to a different sample rate. Defaults to False.
        resample_rate (int, optional): The target sample rate for resampling if resample is True. Defaults to 48000.
        backend (str, optional): The audio backend to use for reading. One of
            "torchcodec" (default), "soundfile", "av", or "auto". "torchcodec"
            falls back to soundfile for plain audio containers (wav/flac/ogg/aiff)
            when FFmpeg fails to decode. "auto" tries torchcodec, then av (which
            handles every container FFmpeg does, including mp4), then soundfile.
        
    Returns:
        tuple[torch.Tensor, int]: A tuple containing the audio tensor and the sample rate.
            The audio tensor shape is [C, T] where C is the number of channels and T is the number of audio frames.
    """
    if backend == "torchcodec":
        try:
            waveform, sample_rate = read_audio_with_torchcodec(path, start_time, duration)
        except RuntimeError as exc:
            # torchcodec linked against FFmpeg 4 raises "The frame has 0 channels" on some valid
            # mono float PCM wavs (unknown channel layout); libsndfile decodes those without FFmpeg.
            # Only fall back for plain audio containers.
            if str(path).lower().endswith((".wav", ".flac", ".ogg", ".aiff", ".aif")):
                waveform, sample_rate = read_audio_with_soundfile(path, start_time, duration)
            else:
                raise exc
    elif backend == "soundfile":
        waveform, sample_rate = read_audio_with_soundfile(path, start_time, duration)
    elif backend == "av":
        waveform, sample_rate = read_audio_with_av(path, start_time, duration)
    elif backend == "auto":
        waveform, sample_rate = _read_audio_trying_all(path, start_time, duration)
    else:
        raise ValueError(f"Unsupported audio backend: {backend}")

    if resample:
        waveform = resample_waveform(waveform, sample_rate, resample_rate)
        sample_rate = resample_rate

    return waveform, sample_rate


def save_audio_with_wave(waveform: torch.Tensor, sample_rate: int, save_path: str) -> None:
    """Save a waveform as 16-bit PCM WAV without FFmpeg/TorchCodec."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2:
        raise ValueError(f"Expected waveform shape [C, T], got {tuple(waveform.shape)}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    pcm = (
        waveform.detach()
        .cpu()
        .float()
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .transpose(0, 1)
        .contiguous()
    )

    with wave.open(str(save_path), "wb") as wav_file:
        wav_file.setnchannels(int(waveform.shape[0]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.numpy().tobytes())


def save_audio(waveform: torch.Tensor, sample_rate: int, save_path: str, backend: str = "torchcodec"):
    """
    Save audio tensor to file.
    
    Args:
        waveform (torch.Tensor): The audio tensor to save. Shape can be [C, T] or [B, C, T].
        sample_rate (int): The sample rate of the audio.
        save_path (str): The file path to save the audio to.
        backend (str, optional): The audio backend to use for saving. Defaults to "torchcodec".
    """
    if waveform.dim() == 3:
        waveform = waveform[0]
    waveform = waveform.cpu()

    if backend == "torchcodec":
        from torchcodec.encoders import AudioEncoder
        encoder = AudioEncoder(waveform, sample_rate=sample_rate)
        encoder.to_file(dest=save_path)
    elif backend in {"wave", "wav", "stdlib"}:
        save_audio_with_wave(waveform, sample_rate, save_path)
    else:
        raise ValueError(f"Unsupported audio backend: {backend}")
