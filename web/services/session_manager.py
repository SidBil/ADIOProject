"""
Session Manager — tracks therapy session state.

Each session has an image, gold-standard description, a queue of questions,
and a history of responses with evaluations.
"""

import csv
import uuid
import random
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR = PROJECT_ROOT / "imagegen" / "images"
PROMPTS_CSV = PROJECT_ROOT / "imagegen" / "image_prompts.csv"


@dataclass
class QuestionState:
    id: int
    question: str
    structure_word: str
    expected_answer: str
    difficulty: int
    transcription: str | None = None
    evaluation: dict | None = None
    followup: str | None = None


@dataclass
class Session:
    session_id: str
    image_id: str
    image_filename: str
    image_prompt: str
    gold_standard: dict | None = None
    questions: list[QuestionState] = field(default_factory=list)
    current_question_idx: int = 0
    completed: bool = False

    @property
    def current_question(self) -> QuestionState | None:
        if self.current_question_idx < len(self.questions):
            return self.questions[self.current_question_idx]
        return None

    @property
    def progress(self) -> dict:
        answered = sum(1 for q in self.questions if q.transcription is not None)
        total = len(self.questions)
        scores = [q.evaluation.get("overall_score", 0)
                  for q in self.questions if q.evaluation]
        return {
            "answered": answered,
            "total": total,
            "average_score": sum(scores) / len(scores) if scores else 0,
            "completed": self.completed,
        }

    def summary(self) -> dict:
        category_scores = {"accuracy": [], "detail": [], "clarity": [], "relevance": []}
        qa_history = []

        for q in self.questions:
            entry = {
                "question": q.question,
                "structure_word": q.structure_word,
                "expected_answer": q.expected_answer,
                "transcription": q.transcription,
                "evaluation": q.evaluation,
                "followup": q.followup,
            }
            qa_history.append(entry)
            if q.evaluation and "scores" in q.evaluation:
                for cat in category_scores:
                    val = q.evaluation["scores"].get(cat, 0)
                    category_scores[cat].append(val)

        avg = {k: (sum(v) / len(v) if v else 0) for k, v in category_scores.items()}

        return {
            "session_id": self.session_id,
            "image_id": self.image_id,
            "image_filename": self.image_filename,
            "image_prompt": self.image_prompt,
            "progress": self.progress,
            "category_averages": avg,
            "qa_history": qa_history,
        }


def _load_image_prompts() -> dict[str, str]:
    """Map image IDs (e.g. 'img_001.png') to their scene prompts."""
    prompts = {}
    if not PROMPTS_CSV.exists():
        return prompts
    with open(PROMPTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["id"])
            filename = f"img_{idx:03d}.png"
            prompts[filename] = row["prompt"].strip().strip('"')
    return prompts


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.image_prompts = _load_image_prompts()
        self.available_images = sorted(
            [f.name for f in IMAGE_DIR.glob("img_*.png")]
        ) if IMAGE_DIR.exists() else []
        print(f"[Session] {len(self.available_images)} images, "
              f"{len(self.image_prompts)} prompts loaded")

    def create_session(self) -> Session:
        sid = str(uuid.uuid4())[:8]

        if not self.available_images:
            raise ValueError("No images available")
        candidates = [img for img in self.available_images
                      if img in self.image_prompts]
        chosen = random.choice(candidates) if candidates else random.choice(self.available_images)

        prompt = self.image_prompts.get(chosen, "A therapy image")
        session = Session(
            session_id=sid,
            image_id=chosen,
            image_filename=chosen,
            image_prompt=prompt,
        )
        self.sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def set_gold_standard(self, session_id: str, gold: dict):
        session = self.sessions[session_id]
        session.gold_standard = gold
        session.questions = [
            QuestionState(
                id=q["id"],
                question=q["question"],
                structure_word=q["structure_word"],
                expected_answer=q["expected_answer"],
                difficulty=q["difficulty"],
            )
            for q in gold["questions"]
        ]

    def record_answer(self, session_id: str, transcription: str,
                      evaluation: dict, followup: dict):
        session = self.sessions[session_id]
        q = session.current_question
        if q is None:
            return
        q.transcription = transcription
        q.evaluation = evaluation
        q.followup = followup.get("comment", "") if isinstance(followup, dict) else followup
        session.current_question_idx += 1

        suggested = followup.get("suggested_question") if isinstance(followup, dict) else None
        if suggested:
            sw = followup.get("structure_word") or q.structure_word
            retry = QuestionState(
                id=len(session.questions) + 1,
                question=suggested,
                structure_word=sw,
                expected_answer=q.expected_answer,
                difficulty=q.difficulty,
            )
            session.questions.insert(session.current_question_idx, retry)

        if session.current_question_idx >= len(session.questions):
            session.completed = True
