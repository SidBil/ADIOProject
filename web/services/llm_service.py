"""
LLM Service — GPT-4o integration for therapeutic scaffolding.

Provides three capabilities:
1. Gold-standard description generation (V/V structure words)
2. Therapeutic question scaffolding
3. Response evaluation against gold standard
"""

import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

for env_path in [PROJECT_ROOT / ".env", PROJECT_ROOT / "imagegen" / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)
        break


class LLMService:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = None

    def load(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[LLM] WARNING: OPENAI_API_KEY not set. LLM features will be unavailable.")
            return
        self.client = OpenAI(api_key=api_key)
        print(f"[LLM] Ready (model={self.model})")

    def _call(self, system: str, user: str) -> str:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY in .env")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content

    def generate_gold_standard(self, image_prompt: str, image_id: str) -> dict:
        """Generate V/V structure-word description and scaffolded questions."""
        system = """You are a speech-language pathologist creating therapy materials for
children with Autism Spectrum Disorder. You use the Visualization and Verbalization (V/V)
structure word framework to guide children in describing images.

Return ONLY valid JSON with this exact schema:
{
  "description": "A detailed description of what the image shows",
  "structure_words": {
    "who": "...",
    "what": "...",
    "where": "...",
    "color": "...",
    "size": "...",
    "shape": "...",
    "mood": "...",
    "movement": "..."
  },
  "questions": [
    {
      "id": 1,
      "question": "...",
      "structure_word": "who|what|where|color|size|shape|mood|movement",
      "expected_answer": "...",
      "difficulty": 1
    }
  ]
}

Generate 6-8 questions ordered from easiest (difficulty 1) to hardest (difficulty 3).
Start with concrete, observable elements (who, what, where) and progress to abstract
qualities (mood, movement). Use simple, encouraging language appropriate for children.
Each expected_answer should be a short phrase (3-12 words)."""

        user = f"""Generate a gold-standard V/V description and scaffolded questions for this therapy image:

Image ID: {image_id}
Scene description: {image_prompt}

Remember: Return ONLY the JSON object, no markdown formatting."""

        raw = self._call(system, user)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw)

    def evaluate_response(self, question: str, expected_answer: str,
                          transcription: str, gold_description: str,
                          structure_word: str) -> dict:
        """Evaluate a child's response against the gold standard."""
        system = """You are a warm, encouraging speech-language pathologist evaluating
a child's verbal response during a V/V therapy session. The child was shown an image
and asked a question. Their speech was transcribed by ASR (may contain minor errors).

Evaluate their response and return ONLY valid JSON:
{
  "scores": {
    "accuracy": 0-5,
    "detail": 0-5,
    "clarity": 0-5,
    "relevance": 0-5
  },
  "feedback": "A warm, encouraging 1-2 sentence response to the child. Acknowledge what they said well. If they missed something, gently guide them.",
  "missed_elements": ["list of key details they omitted"],
  "extra_details": ["any additional correct observations they made"],
  "overall_score": 0-5
}

Be encouraging and supportive. Focus on what the child did right.
A score of 3 means adequate, 4 means good, 5 means excellent.
Even partial or unclear answers deserve acknowledgment."""

        user = f"""Question asked: "{question}"
Structure word focus: {structure_word}
Expected answer: "{expected_answer}"
Full image description: "{gold_description}"

Child's response (ASR transcription): "{transcription}"

Evaluate this response. Return ONLY the JSON object."""

        raw = self._call(system, user)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw)

    def generate_followup(self, question: str, transcription: str,
                          evaluation: dict, gold_description: str,
                          structure_word: str) -> dict:
        """Generate an adaptive follow-up based on the child's response.

        Returns a dict with ``comment`` (always) and, when the child struggled,
        ``suggested_question`` + ``structure_word`` so the session can insert a
        retry before moving on.
        """
        overall = evaluation.get("overall_score", 5)

        system = """You are a warm speech therapist working with a child. Based on their
response and evaluation, generate a JSON object with these fields:

{
  "comment": "A warm 1-2 sentence response to the child.",
  "suggested_question": "A follow-up question if the child struggled, or null if they did well.",
  "structure_word": "The V/V structure word the suggested question targets, or null."
}

Rules:
- If the child scored well (4-5): set suggested_question and structure_word to null. Praise them warmly.
- If the child scored poorly (1-3): write a gentle follow-up question that helps them try again
  or approach the same topic from a simpler angle. Keep it very simple and encouraging.
- Keep language simple, warm, and age-appropriate. No clinical jargon.
- Return ONLY valid JSON, no markdown."""

        missed = evaluation.get("missed_elements", [])
        feedback = evaluation.get("feedback", "")

        user = f"""Original question: "{question}"
Structure word: {structure_word}
Child said: "{transcription}"
Overall score: {overall}/5
Evaluation feedback: {feedback}
Missed elements: {missed}
Image description: "{gold_description}"

Return ONLY the JSON object."""

        raw = self._call(system, user)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]

        import json
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"comment": raw, "suggested_question": None, "structure_word": None}
