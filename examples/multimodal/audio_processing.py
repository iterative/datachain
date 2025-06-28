#!/usr/bin/env python3
"""
Audio Processing with DataChain

This example demonstrates how to use DataChain to process audio files,
extract metadata, and perform basic audio analysis. It shows:

1. Reading audio files from storage
2. Extracting audio metadata (duration, sample rate, channels)
3. Computing audio features (energy, spectral features)
4. Filtering and processing audio datasets
5. Saving processed results

Prerequisites:
    pip install librosa soundfile

Example usage:
    python audio_processing.py
"""

import warnings
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from pydantic import BaseModel

import datachain as dc
from datachain import C, File
from datachain.lib.data_model import ModelStore

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


class AudioMetadata(BaseModel):
    """Audio metadata extracted from files."""
    duration: float
    sample_rate: int
    channels: int
    file_size: int
    format: str


class AudioFeatures(BaseModel):
    """Audio features computed from audio content."""
    rms_energy: float
    spectral_centroid: float
    spectral_rolloff: float
    zero_crossing_rate: float
    tempo: Optional[float] = None


# Register models with DataChain
AudioMetadataFeature = ModelStore.register(AudioMetadata)
AudioFeaturesFeature = ModelStore.register(AudioFeatures)


def extract_audio_metadata(file: File) -> AudioMetadata:
    """Extract basic metadata from audio file."""
    try:
        # Read audio file info without loading the full audio data
        info = sf.info(file.get_local_path())
        
        return AudioMetadata(
            duration=info.duration,
            sample_rate=info.samplerate,
            channels=info.channels,
            file_size=file.size,
            format=info.format
        )
    except Exception as e:
        # Return default values if file cannot be processed
        print(f"Warning: Could not process {file.name}: {e}")
        return AudioMetadata(
            duration=0.0,
            sample_rate=0,
            channels=0,
            file_size=file.size,
            format="unknown"
        )


def extract_audio_features(file: File) -> AudioFeatures:
    """Extract audio features using librosa."""
    try:
        # Load audio file
        y, sr = librosa.load(file.get_local_path(), sr=None)
        
        # Compute features
        rms = librosa.feature.rms(y=y)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Compute tempo (can be expensive, so make it optional)
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_value = float(tempo) if tempo is not None else None
        except Exception:
            tempo_value = None
        
        return AudioFeatures(
            rms_energy=float(np.mean(rms)),
            spectral_centroid=float(np.mean(spectral_centroid)),
            spectral_rolloff=float(np.mean(spectral_rolloff)),
            zero_crossing_rate=float(np.mean(zcr)),
            tempo=tempo_value
        )
        
    except Exception as e:
        print(f"Warning: Could not extract features from {file.name}: {e}")
        return AudioFeatures(
            rms_energy=0.0,
            spectral_centroid=0.0,
            spectral_rolloff=0.0,
            zero_crossing_rate=0.0,
            tempo=None
        )


def is_valid_audio_duration(metadata: AudioMetadata) -> bool:
    """Filter function to check if audio has valid duration."""
    return metadata.duration > 0.1  # At least 100ms


def main():
    """Main audio processing pipeline."""
    
    # Example: Process audio files from a directory
    # Replace with your audio file source
    audio_source = "audio_files/"  # Local directory with audio files
    # audio_source = "s3://your-bucket/audio/"  # S3 bucket with audio files
    
    print("=== DataChain Audio Processing Example ===\n")
    
    # Step 1: Read audio files from storage
    print("Step 1: Reading audio files...")
    audio_chain = (
        dc.read_storage(audio_source)
        .filter(C("file.path").glob("*.wav") | 
                C("file.path").glob("*.mp3") | 
                C("file.path").glob("*.flac"))
    )
    
    total_files = audio_chain.count()
    print(f"Found {total_files} audio files")
    
    if total_files == 0:
        print("No audio files found. Please add some audio files to the 'audio_files/' directory.")
        print("Supported formats: .wav, .mp3, .flac")
        return
    
    # Step 2: Extract metadata
    print("\nStep 2: Extracting audio metadata...")
    metadata_chain = audio_chain.map(
        metadata=extract_audio_metadata,
        output=AudioMetadataFeature
    )
    
    # Step 3: Filter valid audio files
    print("Step 3: Filtering valid audio files...")
    valid_audio_chain = metadata_chain.filter(
        lambda metadata: is_valid_audio_duration(metadata)
    )
    
    valid_count = valid_audio_chain.count()
    print(f"Valid audio files: {valid_count}")
    
    # Step 4: Extract audio features for valid files
    print("Step 4: Extracting audio features...")
    features_chain = valid_audio_chain.map(
        features=extract_audio_features,
        output=AudioFeaturesFeature
    )
    
    # Step 5: Save processed dataset
    print("Step 5: Saving processed dataset...")
    processed_chain = features_chain.save(name="processed_audio")
    
    # Step 6: Display results
    print("\n=== Results ===")
    print(f"Processed {processed_chain.count()} audio files")
    
    print("\nAudio Metadata Summary:")
    processed_chain.show(limit=5)
    
    # Step 7: Demonstrate filtering and analysis
    print("\n=== Audio Analysis ===")
    
    # Find high-energy audio files
    high_energy = processed_chain.filter(C("features.rms_energy") > 0.1)
    print(f"High-energy audio files: {high_energy.count()}")
    
    # Find files with detectable tempo
    with_tempo = processed_chain.filter(C("features.tempo").is_not_null())
    print(f"Files with detectable tempo: {with_tempo.count()}")
    
    # Show statistics
    if processed_chain.count() > 0:
        print("\nAudio Statistics:")
        print("- Average duration:", 
              processed_chain.select("metadata.duration").to_pandas()["metadata.duration"].mean())
        print("- Average RMS energy:", 
              processed_chain.select("features.rms_energy").to_pandas()["features.rms_energy"].mean())
        
        # Show sample of processed files
        print("\nSample processed files:")
        sample_chain = processed_chain.select(
            "file.name", 
            "metadata.duration", 
            "metadata.sample_rate",
            "features.rms_energy",
            "features.tempo"
        ).limit(3)
        sample_chain.show()


def create_sample_data():
    """Create sample audio data for testing (requires additional setup)."""
    import os
    
    # This function would create sample audio files for testing
    # In a real scenario, you would have your own audio files
    sample_dir = "audio_files"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        print(f"Created {sample_dir} directory.")
        print("Please add some audio files (.wav, .mp3, .flac) to this directory to run the example.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error running audio processing example: {e}")
        print("\nTo run this example:")
        print("1. Install dependencies: pip install librosa soundfile")
        print("2. Add audio files to 'audio_files/' directory")
        print("3. Run: python audio_processing.py")
        
        # Create sample directory if it doesn't exist
        create_sample_data()