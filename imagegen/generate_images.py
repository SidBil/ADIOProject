#!/usr/bin/env python3
"""
Generate images from prompts in a CSV using OpenAI's gpt-image-1 model.

CSV format:
id,prompt
1,"/imagine prompt: Educational clip-art illustration in Lindamood-Bell Visualizing & Verbalizing® style — boy feeding a brown dog from a silver bowl. --ar 3:2 --v 6 --style educational-clipart --q 2 --s 250"
"""

import csv
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# -----------------------------
# Configuration
# -----------------------------
CSV_FILE = "image_prompts.csv"
IMAGES_FOLDER = "images"
IMAGE_SIZE = "1536x1024"      # For gpt-image-1: '1024x1024', '1024x1536', '1536x1024', 'auto'
MODEL = "gpt-image-1"         # Correct model for clean, educational clip-art
TEST_MODE = False             # Set False to process all prompts
TIMEOUT_SECS = 120
MAX_WORKERS = 10              # Number of parallel requests

# -----------------------------
# Core System Prompt
# -----------------------------
LINDAMOOD_BELL_SYSTEM_PROMPT = (
    "Produce clean, simple, educational illustrations in a clip-art style. Each image should have 1–2 clear subjects, a clear action, and a simple, recognizable setting or object. Avoid unnecessary details, realistic textures, or complex backgrounds. Use bright, flat colors, clear outlines, and minimal shading. Focus on making the scene easy to understand and visually clear, suitable for educational or classroom use. Avoid using bold lines."
)

# -----------------------------
# Helpers
# -----------------------------
def clean_prompt(prompt_text: str) -> str:
    """
    Clean prompt and combine with style instruction.
    Handles both Midjourney-style prompts and simple prompts.
    """
    # Remove /imagine prefix and MJ parameters if present
    prompt = re.sub(r'^\s*/imagine\s+prompt:\s*', '', prompt_text, flags=re.IGNORECASE)
    prompt = re.sub(r'\s*--[a-z]+\s+[^\s]+', '', prompt, flags=re.IGNORECASE)
    prompt = re.sub(r'\s+', ' ', prompt).strip()

    # Extract the main subject/action after an em dash if present
    main_content_match = re.search(r'—\s*([^.,;]+)', prompt)
    if main_content_match:
        main_content = main_content_match.group(1).strip()
    else:
        # Use the entire prompt if no em dash found
        main_content = prompt

    # Combine into the final cleaned prompt
    final_prompt = f"{LINDAMOOD_BELL_SYSTEM_PROMPT} Illustration: {main_content}."
    return final_prompt


def extract_descriptor(prompt_text: str) -> str:
    """
    Build safe, readable filenames.
    """
    # Try to extract content after em dash, otherwise use the prompt itself
    match = re.search(r'—\s*([^.,;]+)', prompt_text)
    if match:
        descriptor = match.group(1).strip()
    else:
        # Remove Midjourney syntax and use the prompt
        descriptor = re.sub(r'^\s*/imagine\s+prompt:\s*', '', prompt_text, flags=re.IGNORECASE)
        descriptor = re.sub(r'\s*--[a-z]+\s+[^\s]+', '', descriptor, flags=re.IGNORECASE)
        descriptor = descriptor[:60].strip()
    
    descriptor = re.sub(r'[^\w\s-]', '', descriptor)
    descriptor = re.sub(r'\s+', '_', descriptor).lower()
    return descriptor[:60]


# Thread-safe counter for progress tracking
progress_lock = threading.Lock()
success_count = 0
total_count = 0

def generate_image(api_key: str, prompt: str, image_id: str, descriptor: str) -> bool:
    """
    Generate image via GPT-Image-1 and save.
    Creates its own client for thread safety.
    Skips if image already exists.
    """
    global success_count
    Path(IMAGES_FOLDER).mkdir(parents=True, exist_ok=True)
    filename = f"{image_id}_{descriptor or 'image'}.png"
    filepath = os.path.join(IMAGES_FOLDER, filename)
    
    # Check if image already exists
    if os.path.exists(filepath):
        print(f"[{image_id}] ⊙ Skipped (already exists): {filename}")
        with progress_lock:
            success_count += 1
        return True
    
    client = OpenAI(api_key=api_key)
    
    try:
        print(f"[{image_id}] Generating: {descriptor[:50]}...")

        response = client.images.generate(
            model=MODEL,
            prompt=prompt,
            size=IMAGE_SIZE,
            n=1,
            quality="high",  # For gpt-image-1: 'high', 'medium', 'low', 'auto'
            timeout=TIMEOUT_SECS,
        )

        # Handle different response formats (URL or base64)
        import requests
        import base64
        
        image_data = None
        image_url = None
        
        if hasattr(response.data[0], 'url') and response.data[0].url:
            image_url = response.data[0].url
        elif hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
            image_data = base64.b64decode(response.data[0].b64_json)
        elif hasattr(response.data[0], 'revised_prompt'):
            # Some models return revised_prompt, check if there's a URL elsewhere
            if hasattr(response, 'data') and len(response.data) > 0:
                if hasattr(response.data[0], 'url'):
                    image_url = response.data[0].url
        
        if not image_url and not image_data:
            print(f"Response structure: {response.data[0] if response.data else 'No data'}")
            raise RuntimeError("No image URL or base64 data returned from API.")
        
        if image_data:
            # Save base64 decoded data
            with open(filepath, "wb") as f:
                f.write(image_data)
        else:
            # Download from URL
            r = requests.get(image_url, timeout=TIMEOUT_SECS)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(r.content)

        print(f"[{image_id}] ✓ Saved: {filename}")
        with progress_lock:
            success_count += 1
        return True

    except Exception as e:
        print(f"[{image_id}] ✗ Error: {e}")
        return False


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file '{CSV_FILE}' not found.")
        sys.exit(1)

    # Read all prompts first
    tasks = []
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            image_id = str(row.get("id", "")).strip() or str(idx)
            raw_prompt = (row.get("prompt") or "").strip()
            if not raw_prompt:
                print(f"Skipping empty prompt (id={image_id})")
                continue

            prompt = clean_prompt(raw_prompt)
            descriptor = extract_descriptor(raw_prompt)
            tasks.append((api_key, prompt, image_id, descriptor))

    global total_count, success_count
    total_count = len(tasks)
    success_count = 0

    if TEST_MODE:
        tasks = tasks[:1]
        total_count = 1
        print(f"Test mode: Processing 1 prompt...")
    else:
        print(f"Processing {total_count} prompts with {MAX_WORKERS} parallel workers...\n")

    # Process in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(generate_image, api_key, prompt, image_id, descriptor): (image_id, descriptor)
            for api_key, prompt, image_id, descriptor in tasks
        }

        for future in as_completed(futures):
            image_id, descriptor = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{image_id}] Unexpected error: {e}")

    print(f"\n{'='*60}")
    print(f"Done. Generated {success_count}/{total_count} image(s).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
