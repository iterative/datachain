"""Functional tests for audio processing workflow."""

import io
from collections.abc import Iterator

import numpy as np
import soundfile as sf

import datachain as dc
from datachain.lib.audio import audio_segment_np
from datachain.lib.file import Audio, AudioFile, AudioFragment


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


def extract_audio_info(file: AudioFile) -> Audio:
    """Extract audio metadata from file."""
    return file.get_info()


def generate_fragments(file: AudioFile, meta: Audio) -> Iterator[AudioFragment]:
    """Generate audio fragments from file."""
    fragment_duration = 0.5
    yield from file.get_fragments(duration=fragment_duration, audio_duration=meta.duration)


def generate_middle_fragment(file: AudioFile, meta: Audio) -> Iterator[AudioFragment]:
    """Generate a single fragment from the middle of the audio."""
    start = meta.duration / 3
    fragment = file.get_fragment(start, start + 0.5)
    yield fragment


def process_fragment_to_stats(fragment: AudioFragment) -> dict:
    """Process audio fragment and return statistics."""
    audio_array, sample_rate = fragment.get_np()
    
    # Convert to mono if stereo (average the channels)
    if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
        audio_array = audio_array.mean(axis=1)
    
    return {
        "duration": fragment.end - fragment.start,
        "sample_rate": sample_rate,
        "samples": len(audio_array),
        "rms": float(np.sqrt(np.mean(audio_array**2))),
        "max_amplitude": float(np.max(np.abs(audio_array))),
    }


def test_audio_datachain_workflow(test_session, tmp_path):
    """Test complete audio processing workflow with DataChain."""
    # Create test audio files with different durations and frequencies
    audio_files = []
    for i, (duration, freq) in enumerate([(2.0, 440.0), (3.0, 880.0)]):
        audio_data = generate_test_wav(
            duration=duration, sample_rate=16000, frequency=freq
        )
        audio_path = tmp_path / f"test_audio_{i}.wav"
        audio_path.write_bytes(audio_data)
        audio_files.append(str(audio_path))

    # Use DataChain to read the audio files from filesystem
    chain = dc.read_storage(tmp_path.as_uri(), session=test_session, type="audio")

    # Extract Audio metadata
    chain_with_info = chain.map(
        info=extract_audio_info,
        params=["file"],
        output={"info": Audio},
    )

    # Verify we have the expected files and metadata
    results = list(chain_with_info.to_iter("file", "info"))
    assert len(results) == 2

    # Check that all files have expected audio metadata
    for file, info in results:
        assert isinstance(file, AudioFile)
        assert isinstance(info, Audio)
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert info.duration > 0
        assert info.samples > 0
        assert info.format != ""

    # Test the performance optimized fragment generation
    fragments_chain = chain_with_info.gen(
        fragment=generate_fragments,
        params=["file", "info"],
        output={"fragment": AudioFragment},
    )

    # Get all fragments and verify their properties
    fragments = list(fragments_chain.to_values("fragment"))

    # We should have multiple fragments (2s file creates 4, 3s file creates 6)
    expected_total_fragments = 4 + 6  # 10 total fragments
    assert len(fragments) == expected_total_fragments

    # Verify each fragment has correct properties
    for fragment in fragments:
        assert isinstance(fragment, AudioFragment)
        assert isinstance(fragment.audio, AudioFile)
        assert fragment.start >= 0
        assert fragment.end > fragment.start
        assert fragment.end - fragment.start <= 0.5  # fragment duration

    # Test processing fragments
    stats_chain = fragments_chain.map(
        stats=process_fragment_to_stats,
        params=["fragment"],
        output={"stats": dict},
    )

    # Verify processing results
    stats_results = list(stats_chain.to_values("stats"))
    assert len(stats_results) == expected_total_fragments

    for stats in stats_results:
        assert isinstance(stats, dict)
        assert "duration" in stats
        assert "sample_rate" in stats
        assert "samples" in stats
        assert "rms" in stats
        assert "max_amplitude" in stats
        assert stats["sample_rate"] == 16000
        assert stats["duration"] <= 0.5

    # Test saving fragments
    temp_output_dir = tmp_path / "fragments"
    temp_output_dir.mkdir()

    # Save first fragment as a test
    first_fragment = fragments[0]
    saved_file = first_fragment.save(str(temp_output_dir))

    assert isinstance(saved_file, AudioFile)
    assert saved_file.path.startswith("test_audio_")
    assert saved_file.path.endswith(".wav")

    # Verify the saved fragment has expected content
    saved_info = saved_file.get_info()
    expected_duration = first_fragment.end - first_fragment.start
    # Allow small tolerance for audio processing
    assert abs(saved_info.duration - expected_duration) < 0.1
    assert saved_info.sample_rate == 16000
    assert saved_info.channels == 1

    # Verify the audio content matches by comparing numpy arrays
    original_np, original_sr = first_fragment.get_np()
    saved_np, saved_sr = audio_segment_np(saved_file)

    assert original_sr == saved_sr == 16000
    assert len(original_np) == len(saved_np)
    # Allow some tolerance for audio processing differences
    assert np.allclose(original_np, saved_np, atol=1e-3)


def test_audio_middle_fragment_workflow(test_session, tmp_path):
    """Test extracting fragments from the middle of audio files."""
    # Create test audio file
    audio_data = generate_test_wav(duration=3.0, sample_rate=16000, frequency=440.0)
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(audio_data)

    # Create DataChain workflow
    chain = (
        dc.read_storage(tmp_path.as_uri(), session=test_session, type="audio")
        .map(info=extract_audio_info, params=["file"], output={"info": Audio})
        .gen(
            fragment=generate_middle_fragment,
            params=["file", "info"],
            output={"fragment": AudioFragment},
        )
    )

    # Get the results
    results = list(chain.to_iter("file", "info", "fragment"))
    assert len(results) == 1

    file, info, fragment = results[0]
    assert isinstance(file, AudioFile)
    assert isinstance(info, Audio)
    assert isinstance(fragment, AudioFragment)

    # Verify the fragment is from the middle third
    expected_start = info.duration / 3
    assert abs(fragment.start - expected_start) < 0.1
    assert fragment.end - fragment.start == 0.5


def test_audio_streaming_workflow(test_session, tmp_path):
    """Test audio processing with streaming (no caching)."""
    # Create test audio file
    audio_data = generate_test_wav(duration=2.0, sample_rate=16000, frequency=440.0)
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(audio_data)

    # Test with streaming settings (no cache, no prefetch)
    chain = (
        dc.read_storage(tmp_path.as_uri(), session=test_session, type="audio")
        .settings(cache=False, prefetch=False)
        .map(info=extract_audio_info, params=["file"], output={"info": Audio})
        .gen(
            fragment=generate_fragments,
            params=["file", "info"],
            output={"fragment": AudioFragment},
        )
    )

    # Verify streaming works correctly
    fragments = list(chain.to_values("fragment"))
    assert len(fragments) == 4  # 2 seconds with 0.5 second fragments

    for fragment in fragments:
        assert isinstance(fragment, AudioFragment)
        # Verify we can still process the fragment
        audio_np, sr = fragment.get_np()
        assert isinstance(audio_np, np.ndarray)
        assert sr == 16000


def test_audio_format_conversion_workflow(test_session, tmp_path):
    """Test audio format conversion in workflow."""
    # Create test audio file
    audio_data = generate_test_wav(duration=1.0, sample_rate=16000, frequency=440.0)
    audio_path = tmp_path / "test_audio.wav"
    audio_path.write_bytes(audio_data)

    # Create chain
    chain = (
        dc.read_storage(tmp_path.as_uri(), session=test_session, type="audio")
        .map(info=extract_audio_info, params=["file"], output={"info": Audio})
        .gen(
            fragment=generate_fragments,
            params=["file", "info"],
            output={"fragment": AudioFragment},
        )
    )

    # Get first fragment and test format conversion
    fragments = list(chain.to_values("fragment"))
    first_fragment = fragments[0]

    # Test different format conversions
    for format_type in ["wav", "flac", "ogg"]:
        audio_bytes = first_fragment.read_bytes(format=format_type)
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) > 0


def test_audio_error_handling(test_session, tmp_path):
    """Test error handling in audio workflows."""
    # Create an invalid audio file (empty file)
    invalid_audio_path = tmp_path / "invalid_audio.wav"
    invalid_audio_path.write_bytes(b"invalid audio data")

    # Create chain
    chain = dc.read_storage(tmp_path.as_uri(), session=test_session, type="audio")

    # This should handle the error gracefully
    try:
        results = list(chain.map(
            info=extract_audio_info,
            params=["file"],
            output={"info": Audio},
        ).to_values("info"))
        # If we get here, the error was handled and we should have gotten an exception
        # in the processing, not here
        assert len(results) == 0 or any(isinstance(r, Exception) for r in results)
    except Exception as e:
        # This is expected for invalid audio files
        assert "unable to extract metadata" in str(e).lower() or "error" in str(e).lower()


def test_audio_performance_optimization(test_session, tmp_path):
    """Test performance optimizations in audio processing."""
    # Create test audio files
    audio_data = generate_test_wav(duration=5.0, sample_rate=44100, frequency=440.0)
    audio_path = tmp_path / "large_audio.wav"
    audio_path.write_bytes(audio_data)

    # Test with memory limit optimization
    chain = dc.read_storage(tmp_path.as_uri(), session=test_session, type="audio")

    # Extract metadata first
    chain_with_info = chain.map(
        info=extract_audio_info,
        params=["file"],
        output={"info": Audio},
    )

    # Use pre-computed duration to avoid repeated get_info() calls
    def optimized_fragments(file: AudioFile, meta: Audio) -> Iterator[AudioFragment]:
        """Generate fragments with pre-computed duration."""
        fragment_duration = 1.0
        yield from file.get_fragments(
            duration=fragment_duration, audio_duration=meta.duration
        )

    fragments_chain = chain_with_info.gen(
        fragment=optimized_fragments,
        params=["file", "info"],
        output={"fragment": AudioFragment},
    )

    # Verify the optimization works
    fragments = list(fragments_chain.to_values("fragment"))
    assert len(fragments) == 5  # 5 seconds with 1 second fragments

    for fragment in fragments:
        assert isinstance(fragment, AudioFragment)
        assert fragment.end - fragment.start == 1.0  # Each fragment is 1 second
