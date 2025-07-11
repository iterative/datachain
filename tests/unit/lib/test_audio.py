"""Unit tests for audio processing functionality."""

import io
from unittest.mock import patch
import pytest
import numpy as np
import soundfile as sf

from datachain.lib.audio import (
    audio_info,
    audio_segment_bytes,
    audio_segment_np,
    save_audio_fragment,
    estimate_memory_usage,
    validate_audio_format,
    SUPPORTED_FORMATS,
    DEFAULT_MAX_MEMORY_MB,
)
from datachain.lib.file import Audio, AudioFile, FileError


def generate_test_wav(
    duration: float = 1.0, sample_rate: int = 16000, frequency: float = 440.0
) -> bytes:
    """Generate a simple sine wave WAV file as bytes."""
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="wav")
    return buffer.getvalue()


@pytest.fixture
def audio_file(tmp_path, catalog):
    """Create a temporary audio file for testing."""
    audio_data = generate_test_wav(duration=2.0, sample_rate=16000)
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(audio_data)

    file = AudioFile(path=str(audio_path), source="file://")
    # _set_stream is required to enable file.open() operations
    file._set_stream(catalog, caching_enabled=False)
    return file


@pytest.fixture
def stereo_audio_file(tmp_path, catalog):
    """Create a temporary stereo audio file for testing."""
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, False)

    # Create stereo signal (left and right channels)
    left_channel = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right_channel = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    stereo_data = np.column_stack([left_channel, right_channel])

    audio_path = tmp_path / "stereo_test.wav"
    sf.write(audio_path, stereo_data, sample_rate, format="wav")

    # Create a real AudioFile object
    file = AudioFile(path=str(audio_path), source="file://")
    # _set_stream is required to enable file.open() operations
    file._set_stream(catalog, caching_enabled=False)
    return file


class TestAudioInfo:
    """Test cases for audio_info function."""

    def test_audio_info_basic(self, audio_file):
        """Test basic audio metadata extraction."""
        result = audio_info(audio_file)

        assert isinstance(result, Audio)
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert abs(result.duration - 2.0) < 0.1  # Allow small tolerance
        assert result.samples == 32000
        assert result.format != ""
        assert result.bit_rate > 0

    def test_audio_info_stereo(self, stereo_audio_file):
        """Test audio info extraction for stereo files."""
        result = audio_info(stereo_audio_file)

        assert isinstance(result, Audio)
        assert result.sample_rate == 16000
        assert result.channels == 2  # Stereo
        assert result.samples == 16000
        assert result.format != ""

    def test_audio_info_error_handling(self, audio_file):
        """Test audio_info handles file errors properly."""
        with patch(
            "datachain.lib.audio.torchaudio.info", side_effect=Exception("Test error")
        ):
            with pytest.raises(
                FileError, match="unable to extract metadata from audio file"
            ):
                audio_info(audio_file)

    def test_audio_info_with_file_interface(self, audio_file):
        """Test audio_info with File interface conversion."""
        # Test that it works with File objects that get converted to AudioFile
        from datachain.lib.file import File
        
        regular_file = File(path=audio_file.path, source=audio_file.source)
        regular_file._set_stream(audio_file._catalog, caching_enabled=False)
        
        result = audio_info(regular_file)
        assert isinstance(result, Audio)
        assert result.sample_rate == 16000


class TestAudioSegmentNp:
    """Test cases for audio_segment_np function."""

    def test_audio_segment_np_full(self, audio_file):
        """Test loading full audio segment as numpy array."""
        audio_np, sr = audio_segment_np(audio_file)

        assert isinstance(audio_np, np.ndarray)
        assert sr == 16000
        assert len(audio_np.shape) == 1  # Mono audio should be 1D
        assert len(audio_np) == 32000  # 2 seconds at 16kHz

    def test_audio_segment_np_partial(self, audio_file):
        """Test loading partial audio segment."""
        # Load 0.5 seconds starting from 0.5 seconds
        audio_np, sr = audio_segment_np(audio_file, start=0.5, duration=0.5)

        assert isinstance(audio_np, np.ndarray)
        assert sr == 16000
        assert len(audio_np) == 8000  # 0.5 seconds at 16kHz

    def test_audio_segment_np_validation(self, audio_file):
        """Test input validation for audio_segment_np."""
        with pytest.raises(ValueError, match="start must be a non-negative float"):
            audio_segment_np(audio_file, start=-1.0)

        with pytest.raises(ValueError, match="duration must be a positive float"):
            audio_segment_np(audio_file, duration=0.0)

        with pytest.raises(ValueError, match="duration must be a positive float"):
            audio_segment_np(audio_file, duration=-1.0)

    def test_audio_segment_np_memory_limit(self, audio_file):
        """Test memory limit enforcement."""
        # Test with very small memory limit
        with pytest.raises(MemoryError, match="Segment would require"):
            audio_segment_np(audio_file, duration=2.0, max_memory_mb=0.1)

    def test_audio_segment_np_multichannel(self, stereo_audio_file):
        """Test multichannel audio handling."""
        audio_np, sr = audio_segment_np(stereo_audio_file)

        # Should be transposed to (samples, channels)
        assert audio_np.shape == (16000, 2)  # 1 second at 16kHz, 2 channels
        assert sr == 16000

    def test_audio_segment_np_error_handling(self, audio_file):
        """Test error handling in audio_segment_np."""
        with patch(
            "datachain.lib.audio.torchaudio.info", side_effect=Exception("Test error")
        ):
            with pytest.raises(FileError, match="unable to read audio segment"):
                audio_segment_np(audio_file)

    @pytest.mark.parametrize("start,duration", [(0.0, 1.0), (0.5, 0.5), (1.0, 1.0)])
    def test_audio_segment_np_different_durations(self, audio_file, start, duration):
        """Test audio loading with different start times and durations."""
        audio_np, sr = audio_segment_np(audio_file, start=start, duration=duration)

        assert isinstance(audio_np, np.ndarray)
        assert sr == 16000
        expected_samples = int(duration * 16000)
        assert len(audio_np) == expected_samples


class TestAudioSegmentBytes:
    """Test cases for audio_segment_bytes function."""

    def test_audio_segment_bytes_basic(self, audio_file):
        """Test converting audio segment to bytes."""
        audio_bytes = audio_segment_bytes(audio_file, start=0.0, duration=1.0)

        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0

        # Verify it's a valid WAV file by loading it back
        buffer = io.BytesIO(audio_bytes)
        data, sr = sf.read(buffer)
        assert sr == 16000
        assert len(data) == 16000  # 1 second at 16kHz

    def test_audio_segment_bytes_format_validation(self, audio_file):
        """Test format validation in audio_segment_bytes."""
        with pytest.raises(ValueError, match="Unsupported format"):
            audio_segment_bytes(audio_file, format="unsupported")

    @pytest.mark.parametrize("format_type", ["wav", "flac", "ogg"])
    def test_audio_segment_bytes_formats(self, audio_file, format_type):
        """Test audio conversion to different formats."""
        audio_bytes = audio_segment_bytes(audio_file, format=format_type)

        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0

    def test_audio_segment_bytes_error_handling(self, audio_file):
        """Test error handling in audio_segment_bytes."""
        with patch(
            "datachain.lib.audio.audio_segment_np", side_effect=Exception("Test error")
        ):
            with pytest.raises(FileError, match="unable to convert audio segment to bytes"):
                audio_segment_bytes(audio_file, format="wav")


class TestSaveAudioFragment:
    """Test cases for save_audio_fragment function."""

    def test_save_audio_fragment_basic(self, audio_file, tmp_path):
        """Test saving audio fragment."""
        with patch("datachain.lib.file.AudioFile.upload") as mock_upload:
            # Mock the upload to return a new AudioFile
            mock_uploaded_file = AudioFile(
                path="test_audio_000500_001500.wav", source="file://"
            )
            mock_upload.return_value = mock_uploaded_file

            result = save_audio_fragment(
                audio_file, start=0.5, end=1.5, output=str(tmp_path), format="wav"
            )

            # Verify AudioFile.upload was called
            mock_upload.assert_called_once()
            call_args = mock_upload.call_args

            # Check the uploaded data is bytes
            assert isinstance(call_args[0][0], bytes)

            # Check the filename pattern
            filename = call_args[0][1]
            assert "test_audio_000500_001500.wav" in filename

            # Check catalog is passed
            assert call_args[1]["catalog"] == audio_file._catalog

            assert result == mock_uploaded_file

    def test_save_audio_fragment_validation(self, audio_file, tmp_path):
        """Test input validation for save_audio_fragment."""
        with pytest.raises(ValueError, match="Invalid time range"):
            save_audio_fragment(audio_file, start=-1.0, end=1.0, output=str(tmp_path))

        with pytest.raises(ValueError, match="Invalid time range"):
            save_audio_fragment(audio_file, start=2.0, end=1.0, output=str(tmp_path))

        with pytest.raises(ValueError, match="Invalid time range"):
            save_audio_fragment(audio_file, start=1.0, end=1.0, output=str(tmp_path))

    def test_save_audio_fragment_format_validation(self, audio_file, tmp_path):
        """Test format validation in save_audio_fragment."""
        with pytest.raises(ValueError, match="Unsupported format"):
            save_audio_fragment(
                audio_file, start=0.0, end=1.0, output=str(tmp_path), format="unsupported"
            )

    def test_save_audio_fragment_auto_format(self, tmp_path, catalog):
        """Test saving audio fragment with auto-detected format."""
        audio_data = generate_test_wav(duration=1.0, sample_rate=16000)
        audio_path = tmp_path / "test_audio.flac"
        buffer = io.BytesIO(audio_data)
        temp_data, sr = sf.read(buffer)
        sf.write(audio_path, temp_data, sr, format="flac")

        audio_file = AudioFile(path=str(audio_path), source="file://")
        audio_file._set_stream(catalog, caching_enabled=False)

        with patch("datachain.lib.file.AudioFile.upload") as mock_upload:
            mock_upload.return_value = AudioFile(path="test_output.flac", source="file://")

            save_audio_fragment(audio_file, start=0.0, end=1.0, output=str(tmp_path))

            # Should use format from file extension
            call_args = mock_upload.call_args
            filename = call_args[0][1]
            assert filename.endswith(".flac")

    def test_save_audio_fragment_error_handling(self, audio_file, tmp_path):
        """Test error handling in save_audio_fragment."""
        with patch(
            "datachain.lib.audio.audio_segment_bytes", side_effect=Exception("Test error")
        ):
            with pytest.raises(FileError, match="unable to save audio fragment"):
                save_audio_fragment(audio_file, start=0.0, end=1.0, output=str(tmp_path))


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        # Test typical audio parameters
        memory_mb = estimate_memory_usage(
            duration=10.0, sample_rate=44100, channels=2
        )
        
        # Should be approximately 10 * 44100 * 2 * 4 bytes = ~3.37 MB
        assert 3.0 < memory_mb < 4.0

    def test_validate_audio_format_valid(self):
        """Test valid format validation."""
        for fmt in SUPPORTED_FORMATS:
            result = validate_audio_format(fmt)
            assert result == fmt
            
            # Test uppercase
            result = validate_audio_format(fmt.upper())
            assert result == fmt

    def test_validate_audio_format_invalid(self):
        """Test invalid format validation."""
        with pytest.raises(ValueError, match="Unsupported format"):
            validate_audio_format("unsupported")

    def test_supported_formats_constant(self):
        """Test that SUPPORTED_FORMATS contains expected formats."""
        expected_formats = {"wav", "flac", "ogg", "mp3", "aac", "m4a", "wma"}
        assert SUPPORTED_FORMATS == expected_formats

    def test_default_memory_limit(self):
        """Test default memory limit constant."""
        assert DEFAULT_MAX_MEMORY_MB == 512


class TestPerformanceConsiderations:
    """Test cases for performance-related functionality."""

    def test_memory_limit_enforcement(self, audio_file):
        """Test that memory limits are properly enforced."""
        # This should not raise an error with default limit
        audio_segment_np(audio_file, duration=0.1)
        
        # This should raise an error with very small limit
        with pytest.raises(MemoryError):
            audio_segment_np(audio_file, duration=2.0, max_memory_mb=0.001)

    def test_large_audio_handling(self, tmp_path, catalog):
        """Test handling of larger audio files."""
        # Create a larger audio file (5 seconds)
        audio_data = generate_test_wav(duration=5.0, sample_rate=44100)
        audio_path = tmp_path / "large_audio.wav"
        audio_path.write_bytes(audio_data)

        audio_file = AudioFile(path=str(audio_path), source="file://")
        audio_file._set_stream(catalog, caching_enabled=False)

        # Test that we can still process segments
        audio_np, sr = audio_segment_np(audio_file, start=1.0, duration=1.0)
        assert len(audio_np) == 44100  # 1 second at 44.1kHz
        assert sr == 44100

    def test_format_conversion_performance(self, audio_file):
        """Test that format conversion works efficiently."""
        # Test multiple format conversions
        formats = ["wav", "flac", "ogg"]
        
        for fmt in formats:
            audio_bytes = audio_segment_bytes(audio_file, duration=0.1, format=fmt)
            assert isinstance(audio_bytes, bytes)
            assert len(audio_bytes) > 0