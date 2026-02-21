"""Preprocess ASR transcripts so they work better with CLIP's text encoder.

CLIP was trained on image-caption pairs ("a photo of a dog catching a ball"),
not conversational speech fragments ("uh the the dog catches ball").  These
helpers bridge that gap by removing ASR artefacts and optionally rephrasing
short fragments into caption-like descriptions.

Usage (from project root):
    python -m multimodal.transcript_normalization  # quick smoke-test
"""

import re

FILLER_WORDS = frozenset({
    "um", "uh", "uh-huh", "hmm", "hm", "ah", "er", "oh",
    "like", "you know", "i mean", "okay", "ok", "so", "well",
})


def normalize_transcript(text: str) -> str:
    """Clean an ASR transcript: lowercase, remove fillers, collapse stutters."""
    text = text.strip().lower()
    if not text:
        return text

    # Remove repeated words (stuttering / disfluency)
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)

    # Remove filler words (single-token and two-token phrases)
    words = text.split()
    cleaned: list[str] = []
    skip_next = False
    for i, w in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        bigram = f"{w} {words[i + 1]}" if i + 1 < len(words) else ""
        if bigram in FILLER_WORDS:
            skip_next = True
            continue
        if w not in FILLER_WORDS:
            cleaned.append(w)
    text = " ".join(cleaned)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_caption_style(text: str) -> str:
    """Rephrase a short transcript as a caption-like sentence for CLIP.

    For very short outputs (1-2 words) that CLIP struggles with, wrapping in
    "an image showing …" provides extra semantic context.
    """
    text = normalize_transcript(text)
    if not text:
        return text
    if len(text.split()) <= 2:
        return f"an image showing {text}"
    return text


def batch_normalize(
    texts: list[str], caption_style: bool = False
) -> list[str]:
    """Normalize a list of transcripts."""
    fn = to_caption_style if caption_style else normalize_transcript
    return [fn(t) for t in texts]


# ── smoke-test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    examples = [
        "Uh the the cat is is sleeping on the window",
        "um like a dog",
        "cat",
        "  A BOY riding a bicycle on a PATH  ",
        "",
    ]
    for raw in examples:
        norm = normalize_transcript(raw)
        cap = to_caption_style(raw)
        print(f"  raw:     {raw!r}")
        print(f"  norm:    {norm!r}")
        print(f"  caption: {cap!r}")
        print()
