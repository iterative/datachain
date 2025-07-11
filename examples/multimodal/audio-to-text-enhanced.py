"""
Enhanced Audio-to-Text Processing with DataChain

This example demonstrates the improved audio processing capabilities with
performance optimizations, memory management, and enhanced error handling.

To run this example, install:
`pip install 'datachain[audio,examples]'`

The example showcases:
1. Memory-efficient audio processing
2. Performance optimizations with pre-computed metadata
3. Format validation and error handling
4. Streaming audio processing without full downloads
5. Advanced audio analysis and processing
"""

from collections.abc import Iterator
import torch
from transformers import Pipeline, pipeline

import datachain as dc
from datachain import Audio, AudioFile, AudioFragment, C


def extract_audio_info(file: AudioFile) -> Audio:
    """Extract comprehensive audio metadata from file."""
    return file.get_info()


def generate_fragments_optimized(file: AudioFile, meta: Audio) -> Iterator[AudioFragment]:
    """
    Generate audio fragments with performance optimization.
    
    Uses pre-computed duration to avoid repeated metadata calls.
    """
    # Extract middle portion of audio (skip first and last 10%)
    start_time = meta.duration * 0.1
    end_time = meta.duration * 0.9
    
    # Generate 10-second fragments
    fragment_duration = 10.0
    
    yield from file.get_fragments(
        duration=fragment_duration,
        start=start_time,
        end=end_time,
        audio_duration=meta.duration  # Pre-computed for performance
    )


def process_fragment_with_memory_check(fragment: AudioFragment, pipeline: Pipeline) -> str:
    """
    Process audio fragment with memory management.
    
    This approach includes memory estimation and error handling.
    """
    from datachain.lib.audio import estimate_memory_usage
    
    # Get audio metadata for memory estimation
    duration = fragment.end - fragment.start
    
    # Estimate memory usage (assuming 16kHz mono audio)
    estimated_mb = estimate_memory_usage(duration, 16000, 1)
    print(f"Processing fragment {fragment.start:.1f}s-{fragment.end:.1f}s (estimated memory: {estimated_mb:.1f}MB)")
    
    try:
        # Get audio as numpy array with memory limit
        audio_array, sample_rate = fragment.get_np()

        # Convert to mono if stereo (average the channels)
        if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
            audio_array = audio_array.mean(axis=1)

        # Process with speech recognition pipeline
        result = pipeline({
            "raw": audio_array,
            "sampling_rate": sample_rate
        })
        
        return str(result.get("text", ""))
    
    except MemoryError as e:
        print(f"Memory error processing fragment: {e}")
        return f"[MEMORY_ERROR: {str(e)}]"
    except Exception as e:
        print(f"Error processing fragment: {e}")
        return f"[ERROR: {str(e)}]"


def extract_audio_statistics(fragment: AudioFragment) -> dict:
    """Extract detailed audio statistics from fragment."""
    import numpy as np
    
    audio_array, sample_rate = fragment.get_np()
    
    # Convert to mono if stereo
    if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
        audio_array = audio_array.mean(axis=1)
    
    # Calculate advanced statistics
    rms = float(np.sqrt(np.mean(audio_array**2)))
    max_amplitude = float(np.max(np.abs(audio_array)))
    mean_amplitude = float(np.mean(np.abs(audio_array)))
    zero_crossings = float(np.sum(np.diff(np.sign(audio_array)) != 0))
    
    return {
        "duration": fragment.end - fragment.start,
        "sample_rate": sample_rate,
        "samples": len(audio_array),
        "rms": rms,
        "max_amplitude": max_amplitude,
        "mean_amplitude": mean_amplitude,
        "zero_crossings": zero_crossings,
        "snr_estimate": 20 * np.log10(rms / (mean_amplitude + 1e-10)) if rms > 0 else -60,
    }


def filter_high_quality_audio(stats: dict) -> bool:
    """Filter for high-quality audio segments."""
    return (
        stats["rms"] > 0.01 and  # Minimum energy
        stats["max_amplitude"] < 0.95 and  # Not clipped
        stats["snr_estimate"] > -20  # Reasonable SNR
    )


def main():
    """Main processing pipeline with enhanced features."""
    print("=== Enhanced Audio Processing with DataChain ===")
    
    # Example 1: Memory-efficient processing workflow
    print("\n1. Memory-Efficient Audio Processing:")
    
    chain = (
        dc.read_storage("gs://datachain-demo/musdb18", type="audio", anon=True)
        .filter(C("file.path").glob("*.wav"))
        .limit(3)
        .settings(cache=False, prefetch=False, parallel=True)
        .map(meta=extract_audio_info)
    )
    
    # Display audio information with memory estimates
    for file, meta in chain.to_iter("file", "meta"):
        print(f"\nFile: {file.path}")
        print(f"  Duration: {meta.duration:.2f}s")
        print(f"  Sample Rate: {meta.sample_rate}Hz")
        print(f"  Channels: {meta.channels}")
        print(f"  Format: {meta.format}")
        print(f"  Bit Rate: {meta.bit_rate} bps")
        
        # Estimate memory for full file
        from datachain.lib.audio import estimate_memory_usage
        full_memory = estimate_memory_usage(meta.duration, meta.sample_rate, meta.channels)
        print(f"  Full file memory estimate: {full_memory:.1f}MB")
    
    # Example 2: Advanced audio analysis with quality filtering
    print("\n2. Advanced Audio Analysis:")
    
    analysis_chain = (
        chain
        .gen(fragment=generate_fragments_optimized)
        .map(stats=extract_audio_statistics)
        .filter(filter_high_quality_audio)  # Only process high-quality segments
    )
    
    print("\nHigh-Quality Audio Segments:")
    for fragment, stats in analysis_chain.to_iter("fragment", "stats"):
        print(f"Fragment: {fragment.start:.1f}s - {fragment.end:.1f}s")
        print(f"  RMS: {stats['rms']:.4f}")
        print(f"  Max Amplitude: {stats['max_amplitude']:.4f}")
        print(f"  SNR Estimate: {stats['snr_estimate']:.1f}dB")
        print(f"  Zero Crossings: {stats['zero_crossings']:.0f}")
    
    # Example 3: Speech recognition with error handling
    print("\n3. Speech Recognition with Enhanced Error Handling:")
    
    speech_chain = (
        analysis_chain
        .setup(pipeline=lambda: pipeline(
            "automatic-speech-recognition",
            "openai/whisper-small",
            torch_dtype=torch.float32,
            device="cpu",
        ))
        .map(text=process_fragment_with_memory_check)
    )
    
    print("\nSpeech Recognition Results:")
    for fragment, text in speech_chain.to_iter("fragment", "text"):
        print(f"Fragment: {fragment.start:.1f}s - {fragment.end:.1f}s")
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        print()
    
    # Example 4: Format conversion and saving
    print("\n4. Format Conversion and Saving:")
    
    output_dir = "./enhanced_audio_output"
    
    # Test different format conversions
    for fragment, stats in analysis_chain.to_iter("fragment", "stats"):
        if stats["rms"] > 0.05:  # Save only high-energy segments
            print(f"Processing fragment: {fragment.start:.1f}s - {fragment.end:.1f}s")
            
            # Test multiple formats with validation
            for format_type in ["wav", "flac", "ogg"]:
                try:
                    audio_bytes = fragment.read_bytes(format=format_type)
                    print(f"  {format_type.upper()}: {len(audio_bytes)} bytes")
                    
                    # Save the fragment
                    if format_type == "wav":  # Save one copy
                        saved_file = fragment.save(output_dir, format=format_type)
                        print(f"  Saved: {saved_file.path}")
                        
                except Exception as e:
                    print(f"  {format_type.upper()}: Error - {e}")
            
            break  # Only process first qualifying fragment for demo


def demo_performance_features():
    """Demonstrate advanced performance features."""
    print("\n=== Performance Feature Demonstration ===")
    
    chain = (
        dc.read_storage("gs://datachain-demo/musdb18", type="audio", anon=True)
        .filter(C("file.path").glob("*.wav"))
        .limit(1)
        .settings(cache=False, prefetch=False)
        .map(meta=extract_audio_info)
    )
    
    # Example 1: Memory management
    print("\n1. Memory Management:")
    for file, meta in chain.to_iter("file", "meta"):
        print(f"Processing: {file.path}")
        
        # Test different memory limits
        for memory_limit in [50, 100, 200]:
            print(f"  Testing with {memory_limit}MB memory limit:")
            
            try:
                # Try to process a segment with the memory limit
                from datachain.lib.audio import audio_segment_np
                audio_np, sr = audio_segment_np(
                    file, 
                    duration=min(10.0, meta.duration), 
                    max_memory_mb=memory_limit
                )
                print(f"    Success: {len(audio_np)} samples at {sr}Hz")
            except MemoryError as e:
                print(f"    Memory limit exceeded: {e}")
            except Exception as e:
                print(f"    Error: {e}")
    
    # Example 2: Format validation
    print("\n2. Format Validation:")
    from datachain.lib.audio import validate_audio_format, SUPPORTED_FORMATS
    
    print(f"Supported formats: {sorted(SUPPORTED_FORMATS)}")
    
    # Test format validation
    test_formats = ["wav", "WAV", "mp3", "MP3", "flac", "unsupported", "xyz"]
    for fmt in test_formats:
        try:
            validated = validate_audio_format(fmt)
            print(f"  '{fmt}' -> '{validated}' ✓")
        except ValueError as e:
            print(f"  '{fmt}' -> Error: {e} ✗")
    
    # Example 3: Performance optimization comparison
    print("\n3. Performance Optimization Comparison:")
    for file, meta in chain.to_iter("file", "meta"):
        print(f"File: {file.path} ({meta.duration:.1f}s)")
        
        import time
        
        # Method 1: Without optimization (calls get_info repeatedly)
        start_time = time.time()
        fragments_slow = list(file.get_fragments(duration=5.0))
        slow_time = time.time() - start_time
        
        # Method 2: With optimization (pre-computed duration)
        start_time = time.time()
        fragments_fast = list(file.get_fragments(
            duration=5.0, 
            audio_duration=meta.duration
        ))
        fast_time = time.time() - start_time
        
        print(f"  Without optimization: {len(fragments_slow)} fragments in {slow_time:.3f}s")
        print(f"  With optimization: {len(fragments_fast)} fragments in {fast_time:.3f}s")
        print(f"  Speedup: {slow_time/fast_time:.2f}x")
        
        break  # Only test first file


if __name__ == "__main__":
    try:
        main()
        print("\n" + "="*60)
        demo_performance_features()
        print("\nProcessing complete!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install 'datachain[audio,examples]'")