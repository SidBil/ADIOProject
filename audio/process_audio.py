#!/usr/bin/env python3
"""
Process audio files in the speech folder: convert to mono, 16kHz, and normalize loudness.
"""

import subprocess
from pathlib import Path

# Configuration
SPEECH_FOLDER = "speech"
OUTPUT_FOLDER = "processed"
TARGET_SAMPLE_RATE = 16000  # 16kHz


def preprocess_audio(input_path: Path, output_path: Path):
    """
    Convert audio to:
    - mono
    - 16kHz
    - normalized loudness
    """
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output if exists
        "-i", str(input_path),  # input file
        "-ac", "1",  # mono
        "-ar", str(TARGET_SAMPLE_RATE),  # sample rate
        "-af", "loudnorm",  # loudness normalization
        str(output_path)  # output file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Processed: {input_path.name} → {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing {input_path.name}: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Error: ffmpeg not found. Please install ffmpeg.")
        return False


def main():
    """Process all audio files in the speech folder."""
    speech_path = Path(SPEECH_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    if not speech_path.exists():
        print(f"Error: '{SPEECH_FOLDER}' folder not found.")
        return
    
    # Find all audio files
    audio_extensions = {'.m4a', '.mp3', '.wav', '.flac', '.aac', '.ogg', '.mp4'}
    audio_files = [
        f for f in speech_path.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]
    
    if not audio_files:
        print(f"No audio files found in '{SPEECH_FOLDER}' folder.")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to process.\n")
    
    # Process each file
    success_count = 0
    for audio_file in audio_files:
        # Create output filename: remove "_raw" from end and replace with "_clean"
        stem = audio_file.stem
        if stem.endswith("_raw"):
            stem = stem[:-4] + "_clean"  # Remove "_raw" and add "_clean"
        else:
            stem = stem + "_clean"  # Add "_clean" if no "_raw" suffix
        output_file = output_path / f"{stem}.wav"
        if preprocess_audio(audio_file, output_file):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Done. Processed {success_count}/{len(audio_files)} file(s).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

