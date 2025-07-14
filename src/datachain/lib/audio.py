import posixpath
from typing import TYPE_CHECKING, Optional, Union

from datachain.lib.file import FileError

if TYPE_CHECKING:
    from numpy import ndarray

    from datachain.lib.file import Audio, AudioFile, File

try:
    import torchaudio
except ImportError as exc:
    raise ImportError(
        "Missing dependencies for processing audio.\n"
        "To install run:\n\n"
        "  pip install 'datachain[audio]'\n"
    ) from exc


def audio_info(file: "Union[File, AudioFile]") -> "Audio":
    """Extract metadata like sample rate, channels, duration, and format."""
    from datachain.lib.file import Audio

    file = file.as_audio_file()

    try:
        with file.open() as f:
            info = torchaudio.info(f)

            sample_rate = int(info.sample_rate)
            channels = int(info.num_channels)
            frames = int(info.num_frames)
            duration = float(frames / sample_rate) if sample_rate > 0 else 0.0

            # Get format information
            format_name = getattr(info, "format", "")
            codec_name = getattr(info, "encoding", "")
            bit_rate = getattr(info, "bits_per_sample", 0) * sample_rate * channels

    except Exception as exc:
        raise FileError(
            "unable to extract metadata from audio file", file.source, file.path
        ) from exc

    return Audio(
        sample_rate=sample_rate,
        channels=channels,
        duration=duration,
        samples=frames,
        format=format_name,
        codec=codec_name,
        bit_rate=bit_rate,
    )


def audio_fragment_np(
    audio: "AudioFile", start: float = 0, duration: Optional[float] = None
) -> "tuple[ndarray, int]":
    """Load audio fragment as numpy array.
    Multi-channel audio is transposed to (samples, channels)."""
    if start < 0:
        raise ValueError("start must be a non-negative float")

    if duration is not None and duration <= 0:
        raise ValueError("duration must be a positive float")

    if hasattr(audio, "as_audio_file"):
        audio = audio.as_audio_file()

    try:
        with audio.open() as f:
            info = torchaudio.info(f)
            sample_rate = info.sample_rate

            frame_offset = int(start * sample_rate)
            num_frames = int(duration * sample_rate) if duration is not None else -1

            # Reset file pointer to the beginning
            # This is important to ensure we read from the correct position later
            f.seek(0)

            waveform, sr = torchaudio.load(
                f, frame_offset=frame_offset, num_frames=num_frames
            )

            audio_np = waveform.numpy()

            if audio_np.shape[0] > 1:
                audio_np = audio_np.T
            else:
                audio_np = audio_np.squeeze()

            return audio_np, int(sr)
    except Exception as exc:
        raise FileError(
            "unable to read audio fragment", audio.source, audio.path
        ) from exc


def audio_fragment_bytes(
    audio: "AudioFile",
    start: float = 0,
    duration: Optional[float] = None,
    format: str = "wav",
) -> bytes:
    """Convert audio fragment to bytes using soundfile."""
    y, sr = audio_fragment_np(audio, start, duration)

    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format=format)
    return buffer.getvalue()


def save_audio_fragment(
    audio: "AudioFile",
    start: float,
    end: float,
    output: str,
    format: Optional[str] = None,
) -> "AudioFile":
    """Save audio fragment with timestamped filename.
    Supports local and remote storage upload."""
    if start < 0 or end < 0 or start >= end:
        raise ValueError(f"Invalid time range: ({start:.3f}, {end:.3f})")

    if format is None:
        format = audio.get_file_ext()

    duration = end - start
    start_ms = int(start * 1000)
    end_ms = int(end * 1000)
    output_file = posixpath.join(
        output, f"{audio.get_file_stem()}_{start_ms:06d}_{end_ms:06d}.{format}"
    )

    try:
        audio_bytes = audio_fragment_bytes(audio, start, duration, format)

        from datachain.lib.file import AudioFile

        return AudioFile.upload(audio_bytes, output_file, catalog=audio._catalog)

    except Exception as exc:
        raise FileError(
            "unable to save audio fragment", audio.source, audio.path
        ) from exc
