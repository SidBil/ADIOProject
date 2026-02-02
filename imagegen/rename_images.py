#!/usr/bin/env python3
"""
Simple script to rename images to img_001.png, img_002.png, etc.
"""

import os
import re
from pathlib import Path

IMAGES_FOLDER = "images"

def extract_number(filename):
    """Extract the leading number from filename for sorting."""
    match = re.match(r'^(\d+)', filename)
    return int(match.group(1)) if match else 0

def main():
    images_path = Path(IMAGES_FOLDER)
    
    if not images_path.exists():
        print(f"Error: '{IMAGES_FOLDER}' folder not found.")
        return
    
    # Get all PNG files
    png_files = list(images_path.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in '{IMAGES_FOLDER}' folder.")
        return
    
    # Sort by the numeric prefix
    png_files.sort(key=lambda f: extract_number(f.name))
    
    print(f"Found {len(png_files)} image(s) to rename.\n")
    
    # Rename files
    for idx, old_path in enumerate(png_files, start=1):
        new_name = f"img_{idx:03d}.png"
        new_path = images_path / new_name
        
        # Skip if already correctly named
        if old_path.name == new_name:
            print(f"⊙ Skipped (already named): {old_path.name}")
            continue
        
        # Check if target name already exists (shouldn't happen, but safety check)
        if new_path.exists() and new_path != old_path:
            print(f"⚠ Warning: {new_name} already exists. Skipping {old_path.name}")
            continue
        
        try:
            old_path.rename(new_path)
            print(f"✓ Renamed: {old_path.name} → {new_name}")
        except Exception as e:
            print(f"✗ Error renaming {old_path.name}: {e}")
    
    print(f"\nDone! Renamed {len(png_files)} image(s).")

if __name__ == "__main__":
    main()

