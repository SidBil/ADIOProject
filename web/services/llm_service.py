"""
LLM Service — GPT-4o integration for therapeutic evaluation and follow-ups.

Two capabilities:
1. Evaluate a child's spoken response against the expected answer
2. Generate a dynamic follow-up question when the child struggles (accuracy < 4)
"""

import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

WEB_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(WEB_ROOT / ".env")


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

    def _parse_json(self, raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            raw = raw.rsplit("```", 1)[0]
        return json.loads(raw)

    def evaluate_response(self, question: str, expected_answer: str,
                          transcription: str, structure_word: str,
                          entities: str = "", actions: str = "") -> dict:
        """Evaluate a child's response against the expected answer."""
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
  "overall_score": 0-5
}

Be encouraging and supportive. Focus on what the child did right.
A score of 3 means adequate, 4 means good, 5 means excellent.
Even partial or unclear answers deserve acknowledgment."""

        user = f"""Question asked: "{question}"
Structure word focus: {structure_word}
Expected answer: "{expected_answer}"
Image context: entities={entities}, actions={actions}

Child's response (ASR transcription): "{transcription}"

Evaluate this response. Return ONLY the JSON object."""

        raw = self._call(system, user)
        return self._parse_json(raw)

    def generate_followup(self, question: str, transcription: str,
                          evaluation: dict, structure_word: str,
                          entities: str = "", actions: str = "") -> dict:
        """Generate a dynamic follow-up when accuracy < 4.

        Returns dict with ``comment``, ``suggested_question``, ``structure_word``.
        """
        system = """You are a warm speech therapist working with a child. The child
struggled with a question about an image. Generate a JSON object:

{
  "comment": "A warm 1-2 sentence encouragement for the child.",
  "suggested_question": "A simpler follow-up question approaching the same topic from an easier angle.",
  "structure_word": "The V/V structure word this question targets."
}

Rules:
- The follow-up should be simpler and more specific than the original question.
- Keep language simple, warm, and age-appropriate. No clinical jargon.
- Return ONLY valid JSON, no markdown."""

        missed = evaluation.get("missed_elements", [])
        feedback = evaluation.get("feedback", "")

        user = f"""Original question: "{question}"
Structure word: {structure_word}
Child said: "{transcription}"
Overall score: {evaluation.get('overall_score', 0)}/5
Evaluation feedback: {feedback}
Missed elements: {missed}
Image context: entities={entities}, actions={actions}

Return ONLY the JSON object."""

        raw = self._call(system, user)
        try:
            return self._parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            return {
                "comment": raw,
                "suggested_question": question,
                "structure_word": structure_word,
            }
