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
and asked a question. Their speech was transcribed by ASR.

IMPORTANT CONSTRAINTS:
- This is a DIGITAL application. You can ONLY hear the child's speech. You CANNOT see
  the child, their gestures, or their surroundings.
- NEVER ask the child to point at something, show you something, gesture, or perform
  any physical action. You have no way to observe these.
- NEVER reference what the child might be "looking at" or "touching."
- Your feedback must be PURELY about what the child SAID.
- The ASR transcription will have NO capitalization and NO punctuation. It may also
  contain minor transcription errors (e.g. "bare" instead of "bear", "there" instead
  of "their"). Be forgiving of these — evaluate the meaning, not the surface text.
- Keep feedback to exactly 1-2 sentences. Be warm and encouraging.

Return ONLY valid JSON in this exact format:
{
  "scores": {
    "accuracy": 0-5,
    "detail": 0-5,
    "clarity": 0-5,
    "relevance": 0-5
  },
  "feedback": "your feedback here",
  "missed_elements": ["list of key details they omitted"],
  "overall_score": 0-5
}

Scoring guide: 1 = minimal effort, 2 = some attempt, 3 = adequate, 4 = good, 5 = excellent.
Even partial or unclear answers deserve acknowledgment for the attempt.

--- EXAMPLES ---

Example 1:
Question: "What color is the umbrella?"
Expected: "The umbrella is bright red."
Child said: "its red"
Result:
{"scores":{"accuracy":5,"detail":3,"clarity":4,"relevance":5},"feedback":"Great job! You noticed the red umbrella! Can you tell me more about it — is it big or small?","missed_elements":["bright"],"overall_score":4}

Example 2:
Question: "Who is in the picture?"
Expected: "A boy and his mother are walking together."
Child said: "um a boy and a lady they are walking"
Result:
{"scores":{"accuracy":4,"detail":4,"clarity":4,"relevance":5},"feedback":"Wonderful! You saw the boy and the lady walking together. What do you think they might be doing?","missed_elements":["mother specifically"],"overall_score":4}

Example 3:
Question: "What is the dog doing?"
Expected: "The brown dog is running through the grass."
Child said: "the dog is like running i think"
Result:
{"scores":{"accuracy":4,"detail":2,"clarity":3,"relevance":5},"feedback":"Yes, the dog is running! What color is the dog, and where is it running?","missed_elements":["brown","through the grass"],"overall_score":3}

--- END EXAMPLES ---

Do not include any text outside the JSON object."""

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
        system = """You are a warm speech therapist working with a child through a
DIGITAL application. The child struggled with a question about an image.

IMPORTANT CONSTRAINTS:
- This is a digital app. You can ONLY hear the child's speech.
- NEVER ask the child to point, gesture, show you something, or perform any physical
  action. You cannot see them.
- NEVER say things like "Can you point to..." or "Show me where..." — these are
  impossible in this context.
- Your follow-up question must be something the child can answer VERBALLY.
- Keep language simple, warm, and age-appropriate (roughly ages 5-10). No clinical jargon.

Generate a JSON object:
{
  "comment": "A warm 1-2 sentence encouragement for the child.",
  "suggested_question": "A simpler follow-up question the child can answer by speaking.",
  "structure_word": "The V/V structure word this question targets."
}

--- EXAMPLES ---

Example 1:
Original question: "Describe what is happening in the picture."
Child said: "um a dog"
Score: 2/5
Result:
{"comment":"Good start! You noticed the dog! Let's look a little closer at what the dog is doing.","suggested_question":"What is the dog doing — is it sitting, running, or sleeping?","structure_word":"what"}

Example 2:
Original question: "What color are the flowers?"
Child said: "i dont know"
Score: 1/5
Result:
{"comment":"That's okay! Let's try together. Take another look at the flowers in the picture.","suggested_question":"Do the flowers look more red, or more yellow?","structure_word":"color"}

--- END EXAMPLES ---

Return ONLY valid JSON, no markdown or extra text."""

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
