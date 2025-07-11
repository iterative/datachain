"""
Audio processing utilities for DataChain.

This module provides functionality for audio file metadata extraction,
segment processing, and streaming support with torchaudio integration.
"""

import posixpath
from typing import TYPE_CHECKING, Literal, Optional, Union

from datachain.lib.file import FileError

if TYPE_CHECKING:
    from numpy import ndarray

    from datachain.lib.file import Audio, AudioFile, File

# Audio format type definitions for better type safety
AudioFormat = Literal["wav", "flac", "ogg", "mp3", "aac", "m4a", "wma"]
SUPPORTED_FORMATS = frozenset(["wav", "flac", "ogg", "mp3", "aac", "m4a", "wma"])

# Memory management constants
DEFAULT_MAX_MEMORY_MB = 512  # Default max memory for audio segments
BYTES_PER_FLOAT32 = 4  # Memory calculation for float32 arrays

try:
    import torchaudio
    import soundfile as sf
except ImportError as exc:
    raise ImportError(
        "Missing dependencies for processing audio.\n"
        "To install run:\n\n"
        "  pip install 'datachain[audio]'\n"
    ) from exc


def audio_info(file: "Union[File, AudioFile]") -> "Audio":
    """
    Extract metadata like sample rate, channels, duration, and format.
    
    This function reads audio file metadata without loading the entire file
    into memory, making it efficient for large audio files.
    
    Args:
        file: Audio file object to extract metadata from
        
    Returns:
        Audio: Metadata object containing sample rate, channels, duration, etc.
        
    Raises:
        FileError: If unable to extract metadata from the audio file
        
    Performance Notes:
        - Only reads file headers, not the entire audio data
        - Suitable for streaming operations on large files
    """
    from datachain.lib.file import Audio

    file = file.as_audio_file()

    try:
        with file.open() as f:
            info = torchaudio.info(f)

            sample_rate = int(info.sample_rate)
            channels = int(info.num_channels)
            frames = int(info.num_frames)
            duration = float(frames / sample_rate) if sample_rate > 0 else 0.0

            # Get format information with better error handling
            format_name = getattr(info, "format", "unknown")
            codec_name = getattr(info, "encoding", "unknown")
            
            # Calculate bit rate more safely
            bits_per_sample = getattr(info, "bits_per_sample", 0)
            if bits_per_sample > 0 and sample_rate > 0 and channels > 0:
                bit_rate = bits_per_sample * sample_rate * channels
            else:
                bit_rate = -1

    except Exception as exc:
        raise FileError(
            f"unable to extract metadata from audio file (format: {format_name if 'format_name' in locals() else 'unknown'}, "
            f"error: {str(exc)})", 
            file.source, 
            file.path
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
    audio: "AudioFile", 
    start: float = 0, 
    duration: Optional[float] = None,
    max_memory_mb: Optional[int] = None
) -> "tuple[ndarray, int]":
    """
    Load audio segment as numpy array with memory management.
    
    Multi-channel audio is transposed to (samples, channels) format.
    For very large segments, considers memory constraints.
    
    Args:
        audio: The audio file to read from
        start: Start time in seconds (default: 0)
        duration: Duration in seconds (default: None for entire file)
        max_memory_mb: Maximum memory to use in MB (default: 512MB)
        
    Returns:
        Tuple of (audio_data, sample_rate)
        
    Raises:
        ValueError: If start is negative or duration is non-positive
        FileError: If unable to read the audio file
        MemoryError: If segment would exceed memory limit
        
    Performance Notes:
        - Loads entire segment into memory
        - For very large segments, consider using smaller chunks
        - Memory usage approximately: duration * sample_rate * channels * 4 bytes
    """
    if start < 0:
        raise ValueError("start must be a non-negative float")

    if duration is not None and duration <= 0:
        raise ValueError("duration must be a positive float")

    if hasattr(audio, "as_audio_file"):
        audio = audio.as_audio_file()

    # Memory management check
    if max_memory_mb is None:
        max_memory_mb = DEFAULT_MAX_MEMORY_MB

    try:
        with audio.open() as f:
            info = torchaudio.info(f)
            sample_rate = info.sample_rate
            channels = info.num_channels

            # Calculate memory requirements
            if duration is not None:
                estimated_mb = (duration * sample_rate * channels * BYTES_PER_FLOAT32) / (1024 * 1024)
                if estimated_mb > max_memory_mb:
                    raise MemoryError(
                        f"Segment would require {estimated_mb:.1f}MB but limit is {max_memory_mb}MB. "
                        f"Consider using a smaller duration or increasing max_memory_mb."
                    )

            frame_offset = int(start * sample_rate)
            num_frames = int(duration * sample_rate) if duration is not None else -1

            # Reset file pointer to the beginning
            # This is important to ensure we read from the correct position later
            f.seek(0)

            waveform, sr = torchaudio.load(
                f, frame_offset=frame_offset, num_frames=num_frames
            )

            audio_np = waveform.numpy()

            # Transpose multi-channel audio to (samples, channels)
            if audio_np.shape[0] > 1:
                audio_np = audio_np.T
            else:
                audio_np = audio_np.squeeze()

            return audio_np, int(sr)
            
    except MemoryError:
        raise  # Re-raise memory errors as-is
    except Exception as exc:
        raise FileError(
            f"unable to read audio segment (start: {start:.3f}s, duration: {duration}s, "
            f"error: {str(exc)})", 
            audio.source, 
            audio.path
        ) from exc


def audio_segment_bytes(
    audio: "AudioFile",
    start: float = 0,
    duration: Optional[float] = None,
    format: AudioFormat = "wav",
) -> bytes:
    """
    Convert audio segment to bytes using soundfile.
    
    Args:
        audio: The audio file to read from
        start: Start time in seconds (default: 0)
        duration: Duration in seconds (default: None for entire file)
        format: Output audio format (default: "wav")
        
    Returns:
        Audio segment as bytes in the specified format
        
    Raises:
        ValueError: If format is not supported
        FileError: If unable to process the audio segment
        
    Performance Notes:
        - Loads segment into memory then converts to bytes
        - For large segments, consider chunked processing
    """
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}. Supported formats: {sorted(SUPPORTED_FORMATS)}")
    
    try:
        y, sr = audio_segment_np(audio, start, duration)

        import io
        buffer = io.BytesIO()
        sf.write(buffer, y, sr, format=format)
        return buffer.getvalue()
        
    except Exception as exc:
        raise FileError(
            f"unable to convert audio segment to bytes (format: {format}, "
            f"error: {str(exc)})", 
            audio.source, 
            audio.path
        ) from exc


def save_audio_fragment(
    audio: "AudioFile",
    start: float,
    end: float,
    output: str,
    format: Optional[str] = None,
) -> "AudioFile":
    """
    Save audio fragment with timestamped filename.
    
    Supports local and remote storage upload with proper error handling.
    
    Args:
        audio: Source audio file
        start: Start time in seconds
        end: End time in seconds  
        output: Output directory path
        format: Output format (default: infer from source file)
        
    Returns:
        AudioFile: The saved audio fragment
        
    Raises:
        ValueError: If time range is invalid or format is unsupported
        FileError: If unable to save the audio fragment
        
    Performance Notes:
        - Processes entire fragment in memory
        - Generates timestamped filenames automatically
        - Supports both local and remote storage
    """
    if start < 0 or end < 0 or start >= end:
        raise ValueError(f"Invalid time range: ({start:.3f}, {end:.3f})")

    if format is None:
        format = audio.get_file_ext()
    
    if format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}. Supported formats: {sorted(SUPPORTED_FORMATS)}")

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
            f"unable to save audio fragment (start: {start:.3f}s, end: {end:.3f}s, "
            f"format: {format}, error: {str(exc)})", 
            audio.source, 
            audio.path
        ) from exc


def estimate_memory_usage(duration: float, sample_rate: int, channels: int) -> float:
    """
    Estimate memory usage for an audio segment in MB.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of channels
        
    Returns:
        Estimated memory usage in MB
    """
    return (duration * sample_rate * channels * BYTES_PER_FLOAT32) / (1024 * 1024)


def validate_audio_format(format: str) -> AudioFormat:
    """
    Validate and normalize audio format string.
    
    Args:
        format: Format string to validate
        
    Returns:
        Validated format string
        
    Raises:
        ValueError: If format is not supported
    """
    format_lower = format.lower()
    if format_lower not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {format}. Supported formats: {sorted(SUPPORTED_FORMATS)}")
    return format_lower  # type: ignore[return-value]
