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


def audio_info(file: "File | AudioFile") -> "Audio":
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

            codec_name = getattr(info, "encoding", "")
            file_ext = file.get_file_ext().lower()
            format_name = _encoding_to_format(codec_name, file_ext)

            bits_per_sample = getattr(info, "bits_per_sample", 0)
            bit_rate = (
                bits_per_sample * sample_rate * channels if bits_per_sample > 0 else -1
            )

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


def _encoding_to_format(encoding: str, file_ext: str) -> str:
    """
    Map torchaudio encoding to a format name.

    Args:
        encoding: The encoding string from torchaudio.info()
        file_ext: The file extension as a fallback

    Returns:
        Format name as a string
    """
    # Direct mapping for formats that match exactly
    encoding_map = {
        "FLAC": "flac",
        "MP3": "mp3",
        "VORBIS": "ogg",
        "AMR_WB": "amr",
        "AMR_NB": "amr",
        "OPUS": "opus",
        "GSM": "gsm",
    }

    if encoding in encoding_map:
        return encoding_map[encoding]

    # For PCM variants, use file extension to determine format
    if encoding.startswith("PCM_"):
        # Common PCM formats by extension
        pcm_formats = {
            "wav": "wav",
            "aiff": "aiff",
            "au": "au",
            "raw": "raw",
        }
        return pcm_formats.get(file_ext, "wav")  # Default to wav for PCM

    # Fallback to file extension if encoding is unknown
    return file_ext if file_ext else "unknown"


def audio_to_np(
    audio: "AudioFile", start: float = 0, duration: float | None = None
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


def audio_to_bytes(
    audio: "AudioFile",
    format: str = "wav",
    start: float = 0,
    duration: float | None = None,
) -> bytes:
    """Convert audio to bytes using soundfile.

    If duration is None, converts from start to end of file.
    If start is 0 and duration is None, converts entire file."""
    y, sr = audio_to_np(audio, start, duration)

    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format=format)
    return buffer.getvalue()


def save_audio(
    audio: "AudioFile",
    output: str,
    format: str | None = None,
    start: float = 0,
    end: float | None = None,
) -> "AudioFile":
    """Save audio file or extract fragment to specified format.

    Args:
        audio: Source AudioFile object
        output: Output directory path
        format: Output format ('wav', 'mp3', etc). Defaults to source format
        start: Start time in seconds (>= 0). Defaults to 0
        end: End time in seconds. If None, extracts to end of file

    Returns:
        AudioFile: New audio file with format conversion/extraction applied

    Examples:
        save_audio(audio, "/path", "mp3")                       # Entire file to MP3
        save_audio(audio, "s3://bucket/path", "wav", start=2.5) # From 2.5s to end
        save_audio(audio, "/path", "flac", start=1, end=3)      # Extract 1-3s fragment
    """
    if format is None:
        format = audio.get_file_ext()

    # Validate start time
    if start < 0:
        raise ValueError(
            f"Can't save audio for '{audio.path}', "
            f"start time must be non-negative: {start:.3f}"
        )

    # Handle full file conversion when end is None and start is 0
    if end is None and start == 0:
        output_file = posixpath.join(output, f"{audio.get_file_stem()}.{format}")
        try:
            audio_bytes = audio_to_bytes(audio, format, start=0, duration=None)
        except Exception as exc:
            raise FileError(
                "unable to convert audio file", audio.source, audio.path
            ) from exc
    elif end is None:
        # Extract from start to end of file
        output_file = posixpath.join(
            output, f"{audio.get_file_stem()}_{int(start * 1000):06d}_end.{format}"
        )
        try:
            audio_bytes = audio_to_bytes(audio, format, start=start, duration=None)
        except Exception as exc:
            raise FileError(
                "unable to save audio fragment", audio.source, audio.path
            ) from exc
    else:
        # Fragment extraction mode with specific end time
        if end < 0 or start >= end:
            raise ValueError(
                f"Can't save audio for '{audio.path}', "
                f"invalid time range: ({start:.3f}, {end:.3f})"
            )

        duration = end - start
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        output_file = posixpath.join(
            output, f"{audio.get_file_stem()}_{start_ms:06d}_{end_ms:06d}.{format}"
        )

        try:
            audio_bytes = audio_to_bytes(audio, format, start, duration)
        except Exception as exc:
            raise FileError(
                "unable to save audio fragment", audio.source, audio.path
            ) from exc

    from datachain.lib.file import AudioFile

    return AudioFile.upload(audio_bytes, output_file, catalog=audio._catalog)
