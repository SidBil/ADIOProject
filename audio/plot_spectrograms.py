#!/usr/bin/env python3
"""
Plot spectrograms for all audio files in the processed folder.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
AUDIO_FOLDER = "processed"
OUTPUT_FOLDER = "spectrograms"


def plot_spectrogram(audio_file: Path, output_file: Path):
    """Generate and save a spectrogram for an audio file."""
    # Load audio file
    y, sr = librosa.load(str(audio_file), sr=None)
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram: {audio_file.name}')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def main():
    """Plot spectrograms for all audio files."""
    audio_path = Path(AUDIO_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    if not audio_path.exists():
        print(f"Error: '{AUDIO_FOLDER}' folder not found.")
        return
    
    # Find all audio files
    audio_extensions = {'.wav', '.m4a', '.mp3', '.flac', '.aac', '.ogg', '.mp4'}
    audio_files = [
        f for f in audio_path.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]
    
    if not audio_files:
        print(f"No audio files found in '{AUDIO_FOLDER}' folder.")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to process.\n")
    
    # Process each file
    success_count = 0
    for audio_file in tqdm(audio_files, desc="Generating spectrograms"):
        output_file = output_path / f"{audio_file.stem}_spectrogram.png"
        try:
            plot_spectrogram(audio_file, output_file)
            print(f"✓ Generated: {output_file.name}")
            success_count += 1
        except Exception as e:
            print(f"✗ Error processing {audio_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Done. Generated {success_count}/{len(audio_files)} spectrogram(s).")
    print(f"Spectrograms saved to '{OUTPUT_FOLDER}' folder.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

