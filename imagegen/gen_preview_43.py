#!/usr/bin/env python3
"""
Generate img_001 at 4:3 ratio for quality preview.

gpt-image-1 doesn't support 4:3 natively, so we:
  1. Generate at 1536x1024 (3:2, landscape)
  2. Center-crop width from 1536 → 1365 to get exact 4:3

Output: imagegen/images/img_001_4x3_preview.png
"""

import os
import sys
import base64
import csv
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = (
    "Produce clean, simple, educational illustrations in a clip-art style. "
    "Each image should have 1–2 clear subjects, a clear action, and a simple, recognizable setting or object. "
    "Avoid unnecessary details, realistic textures, or complex backgrounds. "
    "Use bright, flat colors, clear outlines, and minimal shading. "
    "Focus on making the scene easy to understand and visually clear, suitable for educational or classroom use. "
    "Avoid using bold lines."
)

CSV_FILE = Path(__file__).parent / "image_prompts.csv"
OUT_DIR   = Path(__file__).parent / "images"
OUT_DIR.mkdir(exist_ok=True)

# Read the first prompt
with open(CSV_FILE, encoding="utf-8") as f:
    row = next(csv.DictReader(f))

raw_prompt = row["prompt"].strip()
full_prompt = f"{SYSTEM_PROMPT} Illustration: {raw_prompt}."
print(f"Prompt: {full_prompt[:120]}...")

print("Calling gpt-image-1 at 1024x1536 (portrait)...")
response = client.images.generate(
    model="gpt-image-1",
    prompt=full_prompt,
    size="1024x1536",
    n=1,
    quality="high",
)

# Decode image (gpt-image-1 returns b64_json)
b64 = response.data[0].b64_json
img_bytes = base64.b64decode(b64)
img = Image.open(BytesIO(img_bytes))
print(f"Generated: {img.size}")

# Center-crop to 3:4 → target height = floor(width * 4/3)
w, h = img.size
target_h = int(w * 4 / 3)
top = (h - target_h) // 2
img_34 = img.crop((0, top, w, top + target_h))
print(f"Cropped to 3:4: {img_34.size}")

out_path = OUT_DIR / "img_001_3x4_preview.png"
img_34.save(out_path, "PNG")
print(f"Saved → {out_path}")
