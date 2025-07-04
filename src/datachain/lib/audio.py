from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING

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


def audio_info(file: File | AudioFile) -> Audio:
    """
    Returns audio file information.

    Args:
        file (AudioFile): Audio file object.

    Returns:
        Audio: Audio file information.
    """
    # Import here to avoid circular imports
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


def audio_segment_np(
    audio: AudioFile, start: float = 0, duration: float | None = None
) -> tuple[ndarray, int]:
    """
    Reads audio segment from a file and returns as numpy array.

    Args:
        audio (AudioFile): Audio file object.
        start (float): Start time in seconds (default: 0).
        duration (float, optional): Duration in seconds. If None, reads to end.

    Returns:
        tuple[ndarray, int]: Audio data and sample rate.
    """
    if start < 0:
        raise ValueError("start must be a non-negative float")

    if duration is not None and duration <= 0:
        raise ValueError("duration must be a positive float")

    # Ensure we have an AudioFile instance
    if hasattr(audio, "as_audio_file"):
        audio = audio.as_audio_file()

    try:
        with audio.open() as f:
            info = torchaudio.info(f)
            sample_rate = info.sample_rate

            # Calculate frame offset and number of frames
            frame_offset = int(start * sample_rate)
            num_frames = int(duration * sample_rate) if duration is not None else -1

            # Reset position before loading (critical for FileWrapper)
            f.seek(0)

            # Load the audio segment
            waveform, sr = torchaudio.load(
                f, frame_offset=frame_offset, num_frames=num_frames
            )

            audio_np = waveform.numpy()

            # If stereo, take the mean across channels or return multi-channel
            if audio_np.shape[0] > 1:
                # For compatibility, we can either return multi-channel or mono
                # Here returning multi-channel as (samples, channels)
                audio_np = audio_np.T
            else:
                # Mono: shape (samples,)
                audio_np = audio_np.squeeze()

            return audio_np, int(sr)
    except Exception as exc:
        raise FileError(
            "unable to read audio segment", audio.source, audio.path
        ) from exc


def audio_segment_bytes(
    audio: AudioFile,
    start: float = 0,
    duration: float | None = None,
    format: str = "wav",
) -> bytes:
    """
    Reads audio segment from a file and returns as audio bytes.

    Args:
        audio (AudioFile): Audio file object.
        start (float): Start time in seconds (default: 0).
        duration (float, optional): Duration in seconds. If None, reads to end.
        format (str): Audio format (default: 'wav').

    Returns:
        bytes: Audio segment as bytes.
    """
    y, sr = audio_segment_np(audio, start, duration)

    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format=format)
    return buffer.getvalue()


def save_audio_fragment(
    audio: AudioFile,
    start: float,
    end: float,
    output: str,
    format: str | None = None,
) -> AudioFile:
    """
    Saves audio interval as a new audio file. If output is a remote path,
    the audio file will be uploaded to the remote storage.

    Args:
        audio (AudioFile): Audio file object.
        start (float): Start time in seconds.
        end (float): End time in seconds.
        output (str): Output path, can be a local path or a remote path.
        format (str, optional): Output format (default: None). If not provided,
                                the format will be inferred from the audio fragment
                                file extension.

    Returns:
        AudioFile: Audio fragment model.
    """
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
        audio_bytes = audio_segment_bytes(audio, start, duration, format)

        from datachain.lib.file import AudioFile

        return AudioFile.upload(audio_bytes, output_file, catalog=audio._catalog)

    except Exception as exc:
        raise FileError(
            "unable to save audio fragment", audio.source, audio.path
        ) from exc
