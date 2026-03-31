"""
One-time script: send each therapy image to GPT-4o vision and generate
the full image_metadata.csv with structure words and questions.

Usage:  python generate_metadata.py
"""

import base64
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_CSV = Path(__file__).resolve().parent / "image_metadata.csv"

load_dotenv(Path(__file__).resolve().parent / ".env")

FIELDS = [
    "image_id", "file_name", "complexity", "entities", "actions",
    "structure_who", "structure_what", "structure_where",
    "structure_color", "structure_shape", "structure_sound",
    "structure_size", "structure_number", "structure_movement",
    "structure_mood",
    "question_who", "question_what", "question_where",
    "question_color", "question_shape", "question_sound",
    "question_size", "question_number", "question_movement",
    "question_mood",
]

SYSTEM_PROMPT = """You are an expert speech-language pathologist designing therapy prompts
for children (ages 5-12) with Autism Spectrum Disorder, using the Visualization and
Verbalization (V/V) structure word framework.

Your job is twofold:
1) Analyze the image to extract structure-word ANSWERS (short factual phrases).
2) Write 10 QUESTIONS — one per structure word — that require the child to DESCRIBE,
   INFER, and ELABORATE in full sentences.  The questions progress from concrete
   observation to abstract reasoning.

──────────────────────────────────────────
WHAT MAKES A GOOD QUESTION (follow these)
──────────────────────────────────────────
• Require the child to produce at least a full sentence, not a single word.
• Mix description ("Tell me about…", "Describe…") with inference ("Why do you think…",
  "What might happen next…", "How can you tell…").
• Build on each other — earlier questions scaffold later ones.
• Use warm, encouraging language.  Never clinical.

BAD (do NOT write questions like these):
  "Who is in the picture?"          → child says "a cat" — done, no verbalization.
  "What color is the ball?"         → child says "red" — one word.
  "How many birds are there?"       → child says "two" — one word.
  "Is the dog big or small?"        → forced choice, not open-ended.

GOOD (write questions like these):
  "Tell me about the characters you see — what do they look like?"
  "Describe everything that is happening in this picture."
  "What do you think this place sounds like? Tell me all the sounds you imagine."
  "Why do you think the character feels that way? What clues do you see?"
  "If you could step into this picture, what is the first thing you would notice?"
  "Look at the sizes of things — describe what is big and what is small."
  "How can you tell that something is moving? Describe what the movement looks like."
  "What mood or feeling does this picture give you, and what in the picture makes you feel that way?"

──────────────────────────────────────────

Return ONLY valid JSON with these exact keys:

{
  "complexity": 1-3,
  "entities": "semicolon-separated list of main entities",
  "actions": "semicolon-separated list of actions happening",
  "structure_who": "short phrase: who/what characters are present",
  "structure_what": "short phrase: what is happening",
  "structure_where": "short phrase: the setting or location",
  "structure_color": "short phrase: prominent colors",
  "structure_shape": "short phrase: notable shapes",
  "structure_sound": "short phrase: associated sounds",
  "structure_size": "short phrase: size relationships",
  "structure_number": "short phrase: quantities of key elements",
  "structure_movement": "short phrase: movement or stillness",
  "structure_mood": "short phrase: mood or feeling of the scene",
  "question_who": "...",
  "question_what": "...",
  "question_where": "...",
  "question_color": "...",
  "question_shape": "...",
  "question_sound": "...",
  "question_size": "...",
  "question_number": "...",
  "question_movement": "...",
  "question_mood": "..."
}

Rules:
- Structure values: short factual phrases (2-8 words) — these are the EXPECTED answers.
- Questions: 1-2 warm sentences each.  Must require a descriptive or inferential answer
  of at least one full sentence.  Never answerable in one word.
- Complexity: 1 = simple (few elements), 2 = moderate, 3 = complex (many elements).
- Return ONLY the JSON object, no markdown."""


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_metadata_for_image(client: OpenAI, image_path: Path) -> dict:
    b64 = encode_image(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Analyze this therapy image ({image_path.name}) and return the metadata JSON."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                }},
            ]},
        ],
        temperature=0.4,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Check imagegen/.env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    image_files = sorted(IMAGE_DIR.glob("img_*.png"))

    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        sys.exit(1)

    print(f"Found {len(image_files)} images. Generating metadata...\n")

    rows = []
    for img_path in image_files:
        stem = img_path.stem
        idx = stem.replace("img_", "")

        print(f"  [{idx}] {img_path.name} ... ", end="", flush=True)
        try:
            meta = generate_metadata_for_image(client, img_path)
            row = {"image_id": idx, "file_name": img_path.name}
            for field in FIELDS[2:]:
                row[field] = str(meta.get(field, ""))
            rows.append(row)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            rows.append({"image_id": idx, "file_name": img_path.name})

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
