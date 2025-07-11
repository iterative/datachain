# Audio Processing with DataChain

DataChain provides comprehensive support for audio file processing with streaming capabilities, allowing you to work with large audio datasets efficiently without downloading entire files.

## Overview

The audio processing functionality in DataChain consists of several key components:

- **AudioFile**: A data model for handling audio files with metadata extraction and fragment generation
- **AudioFragment**: A data model representing specific time segments within audio files
- **Audio**: A data model containing audio metadata (sample rate, channels, duration, etc.)
- **Streaming Support**: Process audio segments without downloading entire files

## Key Features

### 1. Streaming Audio Processing

DataChain's audio processing is designed for streaming, meaning you can process audio segments without downloading entire files:

```python
import datachain as dc
from datachain import AudioFile, AudioFragment, Audio

# Process audio with streaming (no full downloads)
chain = (
    dc.read_storage("s3://my-audio-bucket/", type="audio")
    .settings(cache=False, prefetch=False)  # Enable streaming
    .map(info=lambda file: file.get_info())
    .gen(fragment=lambda file, info: file.get_fragments(duration=10.0, audio_duration=info.duration))
)
```

### 2. Memory-Efficient Processing

The audio processing system includes memory management features to handle large audio files:

```python
from datachain.lib.audio import audio_segment_np, estimate_memory_usage

def process_large_audio(file: AudioFile, meta: Audio):
    # Estimate memory usage before processing
    memory_mb = estimate_memory_usage(
        duration=10.0, 
        sample_rate=meta.sample_rate, 
        channels=meta.channels
    )
    
    if memory_mb > 100:  # More than 100MB
        # Process in smaller chunks
        for fragment in file.get_fragments(duration=5.0, audio_duration=meta.duration):
            yield fragment
    else:
        # Process normally
        for fragment in file.get_fragments(duration=10.0, audio_duration=meta.duration):
            yield fragment
```

### 3. Format Support and Validation

DataChain supports multiple audio formats with proper validation:

```python
from datachain.lib.audio import validate_audio_format, SUPPORTED_FORMATS

# Supported formats: wav, flac, ogg, mp3, aac, m4a, wma
print(f"Supported formats: {SUPPORTED_FORMATS}")

# Validate format before processing
try:
    format_type = validate_audio_format("wav")
    audio_bytes = fragment.read_bytes(format=format_type)
except ValueError as e:
    print(f"Invalid format: {e}")
```

## Performance Optimizations

### 1. Pre-computed Metadata

To avoid repeated metadata extraction, pass pre-computed duration to fragment generation:

```python
def optimized_fragments(file: AudioFile, meta: Audio):
    """Generate fragments with pre-computed duration."""
    # Use pre-computed duration instead of calling get_info() repeatedly
    yield from file.get_fragments(
        duration=10.0,
        audio_duration=meta.duration  # Pre-computed duration
    )

# Usage in chain
chain = (
    dc.read_storage("s3://audio-bucket/", type="audio")
    .map(meta=lambda file: file.get_info())  # Extract metadata once
    .gen(fragment=optimized_fragments)  # Use pre-computed metadata
)
```

### 2. Memory Limits

Configure memory limits to prevent out-of-memory errors:

```python
from datachain.lib.audio import audio_segment_np

# Set memory limit for audio processing
audio_np, sr = audio_segment_np(
    audio_file, 
    duration=30.0,
    max_memory_mb=512  # Limit to 512MB
)
```

### 3. Chunked Processing

For very large audio files, process in chunks:

```python
def process_audio_chunks(file: AudioFile, meta: Audio):
    """Process audio in manageable chunks."""
    chunk_duration = 30.0  # 30-second chunks
    
    for fragment in file.get_fragments(duration=chunk_duration, audio_duration=meta.duration):
        # Process each chunk
        audio_np, sr = fragment.get_np()
        
        # Your processing logic here
        yield process_audio_chunk(audio_np, sr)
```

## Usage Examples

### Basic Audio Processing

```python
import datachain as dc
from datachain import AudioFile, Audio

def extract_info(file: AudioFile) -> Audio:
    return file.get_info()

# Basic workflow
chain = (
    dc.read_storage("gs://audio-dataset/", type="audio")
    .limit(10)
    .map(info=extract_info)
)

# Display audio information
for file, info in chain.to_iter("file", "info"):
    print(f"File: {file.path}")
    print(f"Duration: {info.duration:.2f}s")
    print(f"Sample Rate: {info.sample_rate}Hz")
    print(f"Channels: {info.channels}")
```

### Audio Fragment Processing

```python
from collections.abc import Iterator
from datachain import AudioFragment

def generate_fragments(file: AudioFile, meta: Audio) -> Iterator[AudioFragment]:
    """Generate 10-second fragments from audio."""
    yield from file.get_fragments(duration=10.0, audio_duration=meta.duration)

def process_fragment(fragment: AudioFragment) -> dict:
    """Process audio fragment and return statistics."""
    audio_np, sr = fragment.get_np()
    
    # Convert to mono if stereo
    if len(audio_np.shape) > 1 and audio_np.shape[1] > 1:
        audio_np = audio_np.mean(axis=1)
    
    return {
        "duration": fragment.end - fragment.start,
        "rms": float(np.sqrt(np.mean(audio_np**2))),
        "max_amplitude": float(np.max(np.abs(audio_np))),
    }

# Processing workflow
chain = (
    dc.read_storage("gs://audio-dataset/", type="audio")
    .map(meta=extract_info)
    .gen(fragment=generate_fragments)
    .map(stats=process_fragment)
)
```

### Audio-to-Text Processing

```python
import torch
from transformers import pipeline

def setup_speech_pipeline():
    """Setup speech recognition pipeline."""
    return pipeline(
        "automatic-speech-recognition",
        "openai/whisper-small",
        torch_dtype=torch.float32,
        device="cpu"
    )

def audio_to_text(fragment: AudioFragment, pipeline) -> str:
    """Convert audio fragment to text."""
    audio_np, sr = fragment.get_np()
    
    # Convert to mono if needed
    if len(audio_np.shape) > 1 and audio_np.shape[1] > 1:
        audio_np = audio_np.mean(axis=1)
    
    # Process with speech recognition
    result = pipeline({
        "raw": audio_np,
        "sampling_rate": sr
    })
    
    return result.get("text", "")

# Speech recognition workflow
chain = (
    dc.read_storage("gs://audio-dataset/", type="audio")
    .settings(cache=False, prefetch=False)  # Streaming mode
    .map(meta=extract_info)
    .gen(fragment=generate_fragments)
    .setup(pipeline=setup_speech_pipeline)
    .map(text=audio_to_text)
)
```

### Saving Audio Fragments

```python
def save_interesting_fragments(fragment: AudioFragment, stats: dict) -> str:
    """Save fragments that meet certain criteria."""
    if stats["rms"] > 0.05:  # High energy fragments
        saved_file = fragment.save("./output/", format="wav")
        return saved_file.path
    return ""

# Save workflow
chain = (
    dc.read_storage("gs://audio-dataset/", type="audio")
    .map(meta=extract_info)
    .gen(fragment=generate_fragments)
    .map(stats=process_fragment)
    .map(saved_path=save_interesting_fragments)
    .filter(lambda saved_path: saved_path != "")
)
```

## Best Practices

### 1. Use Streaming for Large Datasets

Always use streaming mode for large audio datasets:

```python
# Good: Streaming mode
chain = dc.read_storage("s3://large-audio-dataset/", type="audio").settings(cache=False, prefetch=False)

# Avoid: Full caching for large datasets
# chain = dc.read_storage("s3://large-audio-dataset/", type="audio").settings(cache=True)
```

### 2. Pre-compute Metadata

Extract metadata once and reuse:

```python
# Good: Extract metadata once
chain = (
    dc.read_storage("s3://audio/", type="audio")
    .map(meta=lambda file: file.get_info())
    .gen(fragment=lambda file, meta: file.get_fragments(duration=10.0, audio_duration=meta.duration))
)

# Avoid: Repeated metadata extraction
# chain = (
#     dc.read_storage("s3://audio/", type="audio")
#     .gen(fragment=lambda file: file.get_fragments(duration=10.0))  # Calls get_info() internally
# )
```

### 3. Handle Memory Limits

Set appropriate memory limits for your environment:

```python
# For memory-constrained environments
audio_np, sr = audio_segment_np(file, duration=10.0, max_memory_mb=256)

# For high-memory environments
audio_np, sr = audio_segment_np(file, duration=60.0, max_memory_mb=2048)
```

### 4. Validate Formats

Always validate audio formats before processing:

```python
from datachain.lib.audio import validate_audio_format

def safe_format_conversion(fragment: AudioFragment, format: str) -> bytes:
    try:
        validated_format = validate_audio_format(format)
        return fragment.read_bytes(format=validated_format)
    except ValueError as e:
        print(f"Invalid format {format}: {e}")
        return b""
```

### 5. Error Handling

Implement proper error handling for audio processing:

```python
def robust_audio_processing(file: AudioFile) -> Audio:
    """Robust audio metadata extraction with error handling."""
    try:
        return file.get_info()
    except Exception as e:
        print(f"Error processing {file.path}: {e}")
        # Return default metadata
        return Audio(
            sample_rate=-1,
            channels=-1,
            duration=-1.0,
            samples=-1,
            format="unknown",
            codec="unknown",
            bit_rate=-1
        )
```

## Performance Considerations

### Memory Usage

Audio processing can be memory-intensive. Consider these factors:

- **Sample Rate**: Higher sample rates use more memory
- **Channels**: Stereo audio uses twice the memory of mono
- **Duration**: Longer segments use proportionally more memory
- **Format**: Uncompressed formats use more memory

### Streaming vs. Caching

| Mode | Pros | Cons | Use Case |
|------|------|------|----------|
| Streaming | Low memory usage, fast startup | Repeated access slower | Large datasets, one-time processing |
| Caching | Fast repeated access | High memory/disk usage | Small datasets, multiple processing passes |

### Performance Tips

1. **Use appropriate fragment sizes**: 5-30 seconds is often optimal
2. **Process in parallel**: Use `parallel=True` for CPU-bound tasks
3. **Monitor memory usage**: Use memory limits to prevent OOM errors
4. **Choose appropriate formats**: Use compressed formats when possible
5. **Batch processing**: Process multiple fragments together when possible

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce fragment duration or set memory limits
2. **Format Errors**: Validate formats before processing
3. **Slow Processing**: Enable streaming mode and parallel processing
4. **Audio Quality Issues**: Check sample rates and format compatibility

### Debug Tips

```python
# Enable debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor memory usage
from datachain.lib.audio import estimate_memory_usage
memory_mb = estimate_memory_usage(duration=10.0, sample_rate=44100, channels=2)
print(f"Estimated memory usage: {memory_mb:.1f}MB")

# Check supported formats
from datachain.lib.audio import SUPPORTED_FORMATS
print(f"Supported formats: {SUPPORTED_FORMATS}")
```

## API Reference

### AudioFile Methods

- `get_info() -> Audio`: Extract audio metadata
- `get_fragment(start: float, end: float) -> AudioFragment`: Get specific time segment
- `get_fragments(duration: float, start: float = 0, end: Optional[float] = None, audio_duration: Optional[float] = None) -> Iterator[AudioFragment]`: Generate fragments

### AudioFragment Methods

- `get_np() -> tuple[ndarray, int]`: Get audio as numpy array with sample rate
- `read_bytes(format: str = "wav") -> bytes`: Get audio as bytes in specified format
- `save(output: str, format: Optional[str] = None) -> AudioFile`: Save fragment to file

### Audio Model

- `sample_rate: int`: Sample rate in Hz
- `channels: int`: Number of audio channels
- `duration: float`: Duration in seconds
- `samples: int`: Total number of samples
- `format: str`: Audio format (e.g., "wav", "mp3")
- `codec: str`: Audio codec
- `bit_rate: int`: Bit rate in bits per second

### Utility Functions

- `estimate_memory_usage(duration: float, sample_rate: int, channels: int) -> float`: Estimate memory usage in MB
- `validate_audio_format(format: str) -> str`: Validate and normalize audio format
- `audio_info(file: Union[File, AudioFile]) -> Audio`: Extract audio metadata
- `audio_segment_np(audio: AudioFile, start: float = 0, duration: Optional[float] = None, max_memory_mb: Optional[int] = None) -> tuple[ndarray, int]`: Load audio segment as numpy array
- `audio_segment_bytes(audio: AudioFile, start: float = 0, duration: Optional[float] = None, format: str = "wav") -> bytes`: Convert audio segment to bytes