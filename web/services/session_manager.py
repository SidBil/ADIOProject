"""
Session Manager — tracks therapy session state.

Loads questions from the pre-generated image_metadata.csv rather than
calling the LLM at session start.  Each row in the CSV provides 10
structure-word answers and 10 corresponding questions.
"""

import csv
import uuid
import random
from pathlib import Path
from dataclasses import dataclass, field

WEB_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = WEB_ROOT / "data" / "images"
METADATA_CSV = WEB_ROOT / "data" / "image_metadata.csv"

STRUCTURE_WORDS = [
    "who", "what", "where", "color", "shape",
    "sound", "size", "number", "movement", "mood",
]


@dataclass
class QuestionState:
    id: int
    question: str
    structure_word: str
    expected_answer: str
    difficulty: int
    is_dynamic: bool = False
    transcription: str | None = None
    evaluation: dict | None = None
    followup: str | None = None


@dataclass
class ImageMetadata:
    image_id: str
    file_name: str
    complexity: int
    entities: str
    actions: str
    structures: dict[str, str]
    questions: dict[str, str]


@dataclass
class Session:
    session_id: str
    image_id: str
    image_filename: str
    metadata: ImageMetadata
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
        return {
            "answered": answered,
            "total": len(self.questions),
            "completed": self.completed,
        }

    def summary(self) -> dict:
        qa_history = []
        for q in self.questions:
            qa_history.append({
                "question": q.question,
                "structure_word": q.structure_word,
                "expected_answer": q.expected_answer,
                "transcription": q.transcription,
                "evaluation": q.evaluation,
                "followup": q.followup,
            })
        return {
            "session_id": self.session_id,
            "image_id": self.image_id,
            "image_filename": self.image_filename,
            "progress": self.progress,
            "qa_history": qa_history,
        }


def _load_metadata() -> dict[str, ImageMetadata]:
    """Load image_metadata.csv into a dict keyed by filename (img_XXX.png)."""
    metadata: dict[str, ImageMetadata] = {}
    if not METADATA_CSV.exists():
        return metadata

    with open(METADATA_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("file_name", "").strip()
            if not fname:
                continue

            structures = {}
            questions = {}
            for sw in STRUCTURE_WORDS:
                sv = row.get(f"structure_{sw}", "").strip()
                qv = row.get(f"question_{sw}", "").strip()
                if sv:
                    structures[sw] = sv
                if qv:
                    questions[sw] = qv

            try:
                complexity = int(row.get("complexity", "1").strip())
            except ValueError:
                complexity = 1

            metadata[fname] = ImageMetadata(
                image_id=row.get("image_id", "").strip(),
                file_name=fname,
                complexity=complexity,
                entities=row.get("entities", "").strip(),
                actions=row.get("actions", "").strip(),
                structures=structures,
                questions=questions,
            )
    return metadata


def _build_questions(meta: ImageMetadata) -> list[QuestionState]:
    """Build a question queue from metadata, ordered concrete -> abstract."""
    ordering = ["who", "what", "where", "color", "size",
                "number", "shape", "sound", "movement", "mood"]
    difficulty_map = {
        "who": 1, "what": 1, "where": 1,
        "color": 1, "size": 1, "number": 1,
        "shape": 2, "sound": 2,
        "movement": 3, "mood": 3,
    }

    questions: list[QuestionState] = []
    qid = 1
    for sw in ordering:
        q_text = meta.questions.get(sw)
        expected = meta.structures.get(sw)
        if q_text and expected:
            questions.append(QuestionState(
                id=qid,
                question=q_text,
                structure_word=sw,
                expected_answer=expected,
                difficulty=difficulty_map.get(sw, 1),
            ))
            qid += 1
    return questions


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.image_metadata = _load_metadata()
        self.available_images = sorted(self.image_metadata.keys())
        print(f"[Session] {len(self.available_images)} images with metadata loaded")

    def create_session(self) -> Session:
        sid = str(uuid.uuid4())[:8]

        if not self.available_images:
            raise ValueError("No images available")

        chosen = random.choice(self.available_images)
        meta = self.image_metadata[chosen]
        questions = _build_questions(meta)

        session = Session(
            session_id=sid,
            image_id=chosen,
            image_filename=chosen,
            metadata=meta,
            questions=questions,
        )
        self.sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    def record_answer(self, session_id: str, transcription: str,
                      evaluation: dict, followup: dict | None = None):
        session = self.sessions[session_id]
        q = session.current_question
        if q is None:
            return
        q.transcription = transcription
        q.evaluation = evaluation
        q.followup = followup.get("comment", "") if isinstance(followup, dict) else (followup or "")
        session.current_question_idx += 1

        if isinstance(followup, dict):
            suggested = followup.get("suggested_question")
            if suggested:
                sw = followup.get("structure_word") or q.structure_word
                retry = QuestionState(
                    id=len(session.questions) + 1,
                    question=suggested,
                    structure_word=sw,
                    expected_answer=q.expected_answer,
                    difficulty=q.difficulty,
                    is_dynamic=True,
                )
                session.questions.insert(session.current_question_idx, retry)

        if session.current_question_idx >= len(session.questions):
            session.completed = True
