#!/usr/bin/env python3
"""
Validate that image metadata file has all required fields.
"""

import json
import sys
from pathlib import Path

# Required fields for each image
REQUIRED_IMAGE_FIELDS = {"image_id", "filename", "scene_type", "complexity"}

METADATA_FILE = "image_metadata.jsonl"


def fail(message: str):
    """Print error message and exit with failure code."""
    print(f"✗ Error: {message}")
    sys.exit(1)


def validate_metadata():
    """Validate image metadata file."""
    metadata_path = Path(METADATA_FILE)
    
    if not metadata_path.exists():
        fail(f"Metadata file '{METADATA_FILE}' not found.")
    
    # Read and parse JSONL file
    images = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                img = json.loads(line)
                images.append(img)
            except json.JSONDecodeError as e:
                fail(f"Invalid JSON on line {line_num}: {e}")
    
    if not images:
        fail(f"No images found in '{METADATA_FILE}'.")
    
    print(f"Validating {len(images)} image(s)...\n")
    
    # Validate each image
    errors = []
    for img in images:
        missing = REQUIRED_IMAGE_FIELDS - img.keys()
        if missing:
            image_id = img.get("image_id", "unknown")
            errors.append(f"Image {image_id} missing fields: {missing}")
    
    # Report results
    if errors:
        print("Validation failed:\n")
        for error in errors:
            print(f"  ✗ {error}")
        print(f"Found {len(errors)} error(s) in {len(images)} image(s).")
        sys.exit(1)
    else:
        print("✓ All images have required fields:")
        for field in sorted(REQUIRED_IMAGE_FIELDS):
            print(f"  - {field}")
        print(f"Validation passed: {len(images)} image(s) validated successfully.")

if __name__ == "__main__":
    validate_metadata()
