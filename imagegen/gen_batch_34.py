#!/usr/bin/env python3
"""
Regenerate images 1-20 at 3:4 portrait ratio.

Strategy:
  1. Generate at 1024x1536 (gpt-image-1 portrait size)
  2. Center-crop height 1536 → 1365 for exact 3:4 (1024x1365)
  3. Save as img_001.png … img_020.png, overwriting existing files

Run from imagegen/:
  source ../web/venv/bin/activate && python gen_batch_34.py
"""

import os, sys, csv, base64, threading
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("OPENAI_API_KEY not set in .env")

SYSTEM_PROMPT = (
    "Produce clean, simple, educational illustrations in a clip-art style. "
    "Each image should have 1–2 clear subjects, a clear action, and a simple, recognizable setting or object. "
    "Avoid unnecessary details, realistic textures, or complex backgrounds. "
    "Use bright, flat colors, clear outlines, and minimal shading. "
    "Focus on making the scene easy to understand and visually clear, suitable for educational or classroom use. "
    "Avoid using bold lines."
)

CSV_FILE   = Path(__file__).parent / "image_prompts.csv"
OUT_DIR    = Path(__file__).parent / "images"
MAX_IMAGES = 20
MAX_WORKERS = 4   # stay under 5/min rate limit
TARGET_SIZE = (1024, 1365)  # exact 3:4

lock = threading.Lock()
done = 0


def needs_regen(path: Path) -> bool:
    """Return True if file is missing or not already 3:4."""
    if not path.exists():
        return True
    import struct
    data = path.read_bytes()[:24]
    w = struct.unpack(">I", data[16:20])[0]
    h = struct.unpack(">I", data[20:24])[0]
    return (w, h) != TARGET_SIZE


def generate_one(idx: int, raw_prompt: str, total: int) -> bool:
    import time
    global done
    filename = OUT_DIR / f"img_{idx:03d}.png"
    full_prompt = f"{SYSTEM_PROMPT} Illustration: {raw_prompt}."

    client = OpenAI(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.images.generate(
                model="gpt-image-1",
                prompt=full_prompt,
                size="1024x1536",
                n=1,
                quality="high",
            )
            img_bytes = base64.b64decode(response.data[0].b64_json)
            img = Image.open(BytesIO(img_bytes))

            w, h = img.size
            target_h = int(w * 4 / 3)
            top = (h - target_h) // 2
            img = img.crop((0, top, w, top + target_h))

            img.save(filename, "PNG")
            with lock:
                done += 1
                print(f"[{done:2d}/{total}] ✓ img_{idx:03d}.png  ({img.size[0]}x{img.size[1]})")
            return True
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 20 * (attempt + 1)
                print(f"  ⏳ img_{idx:03d}.png rate-limited, retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"  ✗ img_{idx:03d}.png failed: {e}")
                return False
    return False


def main():
    tasks = []
    with open(CSV_FILE, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            if i > MAX_IMAGES:
                break
            path = OUT_DIR / f"img_{i:03d}.png"
            if needs_regen(path):
                tasks.append((i, row["prompt"].strip()))
            else:
                print(f"  ⊙ img_{i:03d}.png already 3:4, skipping")

    if not tasks:
        print("All images already at 3:4. Nothing to do.")
        return

    print(f"\nGenerating {len(tasks)} image(s) at 3:4 (1024×1365) with {MAX_WORKERS} workers…\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(generate_one, idx, prompt, len(tasks)): idx for idx, prompt in tasks}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Unexpected error: {e}")

    print(f"\nDone — {done}/{len(tasks)} images saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
