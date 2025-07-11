"""
Audio-to-Text Processing with DataChain

This example demonstrates how to use DataChain models like Audio, AudioFile,
and AudioFragment to efficiently access audio files, chunk them into
fragments, and process them with ML models to extract text.

To run this example, install:
`pip install 'datachain[audio,examples]'`

The example showcases:
1. Reading audio files from storage
2. Extracting audio metadata
3. Generating audio fragments
4. Processing fragments with speech recognition models
5. Streaming audio processing without full downloads
"""

from collections.abc import Iterator
import torch
from transformers import Pipeline, pipeline

import datachain as dc
from datachain import Audio, AudioFile, AudioFragment, C


def extract_audio_info(file: AudioFile) -> Audio:
    """Extract comprehensive audio metadata from file."""
    return file.get_info()


def generate_fragments(file: AudioFile, meta: Audio) -> Iterator[AudioFragment]:
    """
    Generate audio fragments from file with performance optimization.
    
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
        audio_duration=meta.duration
    )


def process_fragment_to_text(fragment: AudioFragment, pipeline: Pipeline) -> str:
    """
    Process audio fragment using direct numpy conversion.
    
    This approach is memory-efficient and works with streaming.
    """
    # Get audio as numpy array
    audio_array, sample_rate = fragment.get_np()

    # Convert to mono if stereo (average the channels)
    if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
        audio_array = audio_array.mean(axis=1)

    # Process with speech recognition pipeline
    # Pass the numpy array with exact sampling rate from fragment
    result = pipeline({
        "raw": audio_array,
        "sampling_rate": sample_rate
    })
    
    return str(result.get("text", ""))


def extract_audio_statistics(fragment: AudioFragment) -> dict:
    """Extract basic audio statistics from fragment."""
    import numpy as np
    
    audio_array, sample_rate = fragment.get_np()
    
    # Convert to mono if stereo
    if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
        audio_array = audio_array.mean(axis=1)
    
    return {
        "duration": fragment.end - fragment.start,
        "sample_rate": sample_rate,
        "samples": len(audio_array),
        "rms": float(np.sqrt(np.mean(audio_array**2))),
        "max_amplitude": float(np.max(np.abs(audio_array))),
        "mean_amplitude": float(np.mean(np.abs(audio_array))),
    }


def main():
    """Main processing pipeline."""
    # Example 1: Basic audio processing workflow
    print("=== Basic Audio Processing Workflow ===")
    
    # We disable caching and prefetching to ensure that we read only bytes
    # that we need for processing. Methods like `get_info` and `get_fragment`
    # don't require reading the entire file.
    chain = (
        dc.read_storage("gs://datachain-demo/musdb18", type="audio", anon=True)
        .filter(C("file.path").glob("*.wav"))
        .limit(5)
        .settings(cache=False, prefetch=False, parallel=True)
        .map(meta=extract_audio_info)
        .gen(fragment=generate_fragments)
    )
    
    # Display basic information about the audio files
    print("\nAudio Files and Metadata:")
    for file, meta in chain.to_iter("file", "meta"):
        print(f"File: {file.path}")
        print(f"  Duration: {meta.duration:.2f}s")
        print(f"  Sample Rate: {meta.sample_rate}Hz")
        print(f"  Channels: {meta.channels}")
        print(f"  Format: {meta.format}")
        print(f"  Bit Rate: {meta.bit_rate} bps")
        print()
    
    # Example 2: Audio-to-text processing with ML models
    print("=== Audio-to-Text Processing ===")
    
    # Setup the speech recognition pipeline
    speech_pipeline = (
        chain
        .setup(
            pipeline=lambda: pipeline(
                "automatic-speech-recognition",
                "openai/whisper-small",
                torch_dtype=torch.float32,
                device="cpu",
            )
        )
        .map(text=process_fragment_to_text)
    )
    
    # Process fragments and extract text
    print("\nProcessing audio fragments:")
    for fragment, text in speech_pipeline.to_iter("fragment", "text"):
        print(f"Fragment: {fragment.start:.1f}s - {fragment.end:.1f}s")
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        print()
    
    # Example 3: Audio analysis workflow
    print("=== Audio Analysis Workflow ===")
    
    analysis_chain = (
        chain
        .map(stats=extract_audio_statistics)
        .filter(lambda stats: stats["rms"] > 0.01)  # Filter out very quiet segments
    )
    
    # Analyze audio characteristics
    print("\nAudio Fragment Analysis:")
    for fragment, stats in analysis_chain.to_iter("fragment", "stats"):
        print(f"Fragment: {fragment.start:.1f}s - {fragment.end:.1f}s")
        print(f"  Duration: {stats['duration']:.2f}s")
        print(f"  RMS: {stats['rms']:.4f}")
        print(f"  Max Amplitude: {stats['max_amplitude']:.4f}")
        print(f"  Samples: {stats['samples']}")
        print()
    
    # Example 4: Saving processed audio fragments
    print("=== Saving Audio Fragments ===")
    
    # Save interesting fragments (high RMS) to local storage
    output_dir = "./audio_fragments"
    
    for fragment, stats in analysis_chain.to_iter("fragment", "stats"):
        if stats["rms"] > 0.05:  # Save only high-energy fragments
            saved_file = fragment.save(output_dir, format="wav")
            print(f"Saved fragment: {saved_file.path}")
    
    print("\nProcessing complete!")


def demo_advanced_features():
    """Demonstrate advanced audio processing features."""
    print("=== Advanced Audio Processing Features ===")
    
    # Create a simple audio processing chain
    chain = (
        dc.read_storage("gs://datachain-demo/musdb18", type="audio", anon=True)
        .filter(C("file.path").glob("*.wav"))
        .limit(2)
        .settings(cache=False, prefetch=False)
        .map(meta=extract_audio_info)
    )
    
    # Example 1: Memory-efficient processing
    print("\n1. Memory-Efficient Processing:")
    for file, meta in chain.to_iter("file", "meta"):
        print(f"Processing: {file.path}")
        
        # Process in smaller chunks to manage memory
        for fragment in file.get_fragments(duration=5.0, audio_duration=meta.duration):
            # Estimate memory usage
            from datachain.lib.audio import estimate_memory_usage
            memory_mb = estimate_memory_usage(
                fragment.end - fragment.start, 
                meta.sample_rate, 
                meta.channels
            )
            print(f"  Fragment {fragment.start:.1f}s-{fragment.end:.1f}s: ~{memory_mb:.1f}MB")
    
    # Example 2: Format conversion
    print("\n2. Format Conversion:")
    for file, meta in chain.to_iter("file", "meta"):
        # Get a small fragment
        fragment = file.get_fragment(0, min(5.0, meta.duration))
        
        # Convert to different formats
        for fmt in ["wav", "flac", "ogg"]:
            try:
                audio_bytes = fragment.read_bytes(format=fmt)
                print(f"  {fmt.upper()}: {len(audio_bytes)} bytes")
            except Exception as e:
                print(f"  {fmt.upper()}: Error - {e}")
    
    # Example 3: Audio quality assessment
    print("\n3. Audio Quality Assessment:")
    for file, meta in chain.to_iter("file", "meta"):
        # Sample a few fragments to assess quality
        for i in range(3):
            start = (meta.duration / 4) * i
            end = min(start + 2.0, meta.duration)
            
            if end > start:
                fragment = file.get_fragment(start, end)
                stats = extract_audio_statistics(fragment)
                
                # Simple quality assessment
                quality = "High" if stats["rms"] > 0.1 else "Medium" if stats["rms"] > 0.01 else "Low"
                print(f"  Fragment {start:.1f}s-{end:.1f}s: {quality} quality (RMS: {stats['rms']:.4f})")


if __name__ == "__main__":
    try:
        main()
        print("\n" + "="*50)
        demo_advanced_features()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install 'datachain[audio,examples]'")