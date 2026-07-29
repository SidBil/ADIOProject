"""
Offline batch generation for the Adaptive Diagnostic's question-difficulty axis.

Expands image_metadata.csv from ONE question per structure-word into THREE
difficulty tiers per structure-word (easy / medium / hard), per
`PRDs/adaptive diagnostic/question_difficulty.md` §3.

Design:
  * The EXISTING `question_<word>` is kept as the MEDIUM tier verbatim — it is
    already SLP-tuned and shipping, so we don't regenerate it. We only ask the
    model for the EASY and HARD siblings around it.
  * One GPT-4o call per image (all 10 structure-words at once) — 20 calls for
    the current 20-image bank, not 600 single-variant calls.
  * Non-destructive: reads `image_metadata.csv`, writes a NEW
    `image_metadata_tiered.csv`, and KEEPS the original `question_<word>`
    columns. Nothing downstream (`SessionManager._load_metadata`) changes until
    someone deliberately adopts the tiered file.

Tier semantics (from question_difficulty.md §2, "color" example):
  easy   — single concrete referent, one expected answer
           ("What color is the cat?")
  medium — open enumeration, still concrete  [= the existing question]
           ("What colors do you see in the picture?")
  hard   — concrete + an abstract/interpretive layer
           ("What colors do you notice, and how do they make the scene feel?")

Usage:
  python generate_question_tiers.py                 # all images -> tiered CSV
  python generate_question_tiers.py --limit 2       # cheap smoke test (2 images)
  python generate_question_tiers.py --output foo.csv --model gpt-4o
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "image_metadata.csv"
DEFAULT_OUTPUT_CSV = HERE / "image_metadata_tiered.csv"

load_dotenv(HERE / ".env")

# Same 10 structure words, same order, as session_manager.STRUCTURE_WORDS.
STRUCTURE_WORDS = [
    "who", "what", "where", "color", "shape",
    "sound", "size", "number", "movement", "mood",
]

TIERS = ["easy", "medium", "hard"]


def tier_col(sw: str, tier: str) -> str:
    return f"question_{sw}_{tier}"


SYSTEM_PROMPT = """You are an expert speech-language pathologist designing therapy prompts
for children (ages 5-12) with Autism Spectrum Disorder, using the Visualization and
Verbalization (V/V) structure-word framework.

You are given, for ONE image, the 10 structure-word QUESTIONS already written for it
(one per structure word), each with its EXPECTED ANSWER, plus the scene's entities and
actions. The existing question is the MEDIUM difficulty tier and must NOT be changed.

Your job: for each structure word, write an EASY variant and a HARD variant of that
question, about the SAME image and SAME structure word.

──────────────────────────────────────────
WHAT EACH TIER MEANS
──────────────────────────────────────────
• EASY   — narrower and more concrete than the medium question: a single concrete
           referent with essentially one expected answer. Still open-ended (never
           yes/no or forced-choice), just easier to answer.
           e.g. medium "What colors do you see?"  -> easy "What color is the cat?"
• MEDIUM — the existing question (given to you; do not rewrite it).
• HARD   — the medium question PLUS an abstract/interpretive layer: it asks the
           child to observe AND infer/reason/connect (feeling, cause, comparison).
           e.g. medium "What colors do you see?"  ->
                hard "What colors do you notice, and how do they make the scene feel?"

──────────────────────────────────────────
STYLE (same as the existing questions)
──────────────────────────────────────────
• ONE short, direct question per tier (~6-14 words; hard may run slightly longer).
• No preambles ("Tell me about…", "Describe everything…", "If you could step into…").
• Warm, simple, child-friendly language. Never clinical.
• Open-ended: invites a descriptive/inferential answer, never yes/no or one word.
• CRITICAL: even the EASY tier must be open-ended — NEVER a yes/no question ("Is
  the bird moving?") and NEVER a forced choice ("Is it big or small?", "Which is
  bigger, X or Y?"). This applies to EVERY structure word. Every easy question MUST
  start with "What…" or "How…" — never "Is/Are/Does/Do/Which/Was…". Examples:
    - mood:     GOOD "How does the picture make you feel?"      BAD "Is it happy or sad?"
    - size:     GOOD "How big is the cat?" / "What size is the cat?"   BAD "Is the cat big or small?"
    - movement: GOOD "What is the bird doing?" / "How is the dog moving?"   BAD "Is the bird moving?"
• The easy variant must genuinely be easier than medium; the hard variant must
  genuinely add an inferential demand beyond medium.

Return ONLY valid JSON, no markdown, with exactly this shape — one object per
structure word, each with an "easy" and a "hard" string:

{
  "who":      {"easy": "...", "hard": "..."},
  "what":     {"easy": "...", "hard": "..."},
  "where":    {"easy": "...", "hard": "..."},
  "color":    {"easy": "...", "hard": "..."},
  "shape":    {"easy": "...", "hard": "..."},
  "sound":    {"easy": "...", "hard": "..."},
  "size":     {"easy": "...", "hard": "..."},
  "number":   {"easy": "...", "hard": "..."},
  "movement": {"easy": "...", "hard": "..."},
  "mood":     {"easy": "...", "hard": "..."}
}"""


def build_user_message(row: dict) -> str:
    """Compact, text-only context for one image (no vision call needed — we have
    the questions + expected answers already)."""
    lines = [
        f"Image: {row.get('file_name', '')}",
        f"Entities: {row.get('entities', '')}",
        f"Actions: {row.get('actions', '')}",
        "",
        "Structure words (medium question | expected answer):",
    ]
    for sw in STRUCTURE_WORDS:
        q = (row.get(f"question_{sw}", "") or "").strip()
        a = (row.get(f"structure_{sw}", "") or "").strip()
        if not q:
            continue
        lines.append(f"- {sw}: {q}  |  answer: {a}")
    lines.append("")
    lines.append("Return the easy and hard variant for each structure word as JSON.")
    return "\n".join(lines)


def generate_tiers_for_row(client: OpenAI, row: dict, model: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(row)},
        ],
        temperature=0.4,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def main():
    ap = argparse.ArgumentParser(description="Generate easy/medium/hard question tiers.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N images (cheap smoke test).")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV,
                    help="Output CSV path (default: image_metadata_tiered.csv).")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o).")
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Check imagegen/.env")
        sys.exit(1)

    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found.")
        sys.exit(1)

    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        in_fields = reader.fieldnames or []
        rows = list(reader)

    if args.limit is not None:
        rows = rows[: args.limit]

    # Output schema: every original column, then the 30 tiered columns
    # (skip any that somehow already exist so re-runs stay clean).
    tiered_fields = [
        tier_col(sw, t) for sw in STRUCTURE_WORDS for t in TIERS
    ]
    out_fields = in_fields + [c for c in tiered_fields if c not in in_fields]

    client = OpenAI(api_key=api_key)
    print(f"Generating easy/hard tiers for {len(rows)} image(s) with {args.model}...\n")

    ok = 0
    for row in rows:
        name = row.get("file_name") or row.get("image_id") or "?"
        print(f"  {name} ... ", end="", flush=True)

        # Medium tier is the existing question, verbatim, for every structure word.
        for sw in STRUCTURE_WORDS:
            row[tier_col(sw, "medium")] = (row.get(f"question_{sw}", "") or "").strip()

        try:
            tiers = generate_tiers_for_row(client, row, args.model)
            for sw in STRUCTURE_WORDS:
                variants = tiers.get(sw, {}) if isinstance(tiers, dict) else {}
                row[tier_col(sw, "easy")] = str(variants.get("easy", "")).strip()
                row[tier_col(sw, "hard")] = str(variants.get("hard", "")).strip()
            ok += 1
            print("done")
        except Exception as e:
            # Leave easy/hard blank on failure; medium is still populated so the
            # row remains usable (a session can fall back to medium).
            for sw in STRUCTURE_WORDS:
                row.setdefault(tier_col(sw, "easy"), "")
                row.setdefault(tier_col(sw, "hard"), "")
            print(f"FAILED: {e}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows ({ok} fully tiered) to {args.output}")
    print("Review it, then copy over both image_metadata.csv locations "
          "(imagegen/ and web/data/) once you're happy.")


if __name__ == "__main__":
    main()
