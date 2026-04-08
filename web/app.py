"""
ADI/O Therapy Web Application

FastAPI backend that serves the therapy UI and exposes REST endpoints
for session management, ASR transcription, and LLM-powered evaluation.

Session start is instant (questions loaded from metadata CSV).
LLM is called only for evaluation and, when accuracy < 4, for
dynamic follow-up question generation.

Run: python app.py
"""

import os
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.asr_service import ASRService, MODAL_ASR_URL
from services.llm_service import LLMService
from services.session_manager import SessionManager, IMAGE_DIR

ACCURACY_THRESHOLD = 4

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
asr = ASRService()
llm = LLMService()
sessions = SessionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    asr.load()
    llm.load()
    yield

app = FastAPI(title="ADI/O Therapy", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

WEB_DIR = Path(os.environ.get("WEB_DIR", Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    pass

class EvaluateRequest(BaseModel):
    session_id: str
    transcription: str

# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/images/{filename}")
async def serve_image(filename: str):
    path = IMAGE_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Session — instant start from metadata, no LLM call needed
# ---------------------------------------------------------------------------

@app.post("/api/session/start")
async def start_session(req: StartRequest):
    try:
        session = sessions.create_session()
    except ValueError as e:
        raise HTTPException(400, str(e))

    q = session.current_question
    return {
        "session_id": session.session_id,
        "image_id": session.image_id,
        "image_url": f"/images/{session.image_filename}",
        "question": {
            "id": q.id,
            "text": q.question,
            "structure_word": q.structure_word,
            "difficulty": q.difficulty,
        } if q else None,
        "total_questions": len(session.questions),
        "progress": session.progress,
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session = sessions.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    q = session.current_question
    return {
        "session_id": session.session_id,
        "image_id": session.image_id,
        "image_url": f"/images/{session.image_filename}",
        "question": {
            "id": q.id,
            "text": q.question,
            "structure_word": q.structure_word,
            "difficulty": q.difficulty,
        } if q else None,
        "progress": session.progress,
        "completed": session.completed,
    }


# ---------------------------------------------------------------------------
# Transcribe
# ---------------------------------------------------------------------------

@app.post("/api/transcribe")
async def transcribe(session_id: str, audio: UploadFile = File(...)):
    session = sessions.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    content = await audio.read()

    if MODAL_ASR_URL:
        try:
            result = asr.transcribe_remote_bytes(
                content,
                content_type=audio.content_type or "audio/webm",
                image_id=session.image_id,
            )
        except Exception:
            traceback.print_exc()
            raise HTTPException(500, "Transcription failed")
        return result

    suffix = ".webm" if "webm" in (audio.content_type or "") else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix == ".webm":
            import subprocess
            wav_path = tmp_path.replace(".webm", ".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True, check=True,
            )
            tmp_path = wav_path

        result = asr.transcribe(tmp_path, image_id=session.image_id)
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Transcription failed")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# Evaluate — hybrid: static advance when good, dynamic followup when not
# ---------------------------------------------------------------------------

@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    session = sessions.get_session(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    q = session.current_question
    if not q:
        raise HTTPException(400, "No more questions")

    meta = session.metadata

    try:
        evaluation = llm.evaluate_response(
            question=q.question,
            expected_answer=q.expected_answer,
            transcription=req.transcription,
            structure_word=q.structure_word,
            entities=meta.entities,
            actions=meta.actions,
        )
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Evaluation failed. Check your OPENAI_API_KEY.")

    accuracy = evaluation.get("scores", {}).get("accuracy", 5)
    followup = None

    if accuracy < ACCURACY_THRESHOLD:
        try:
            followup = llm.generate_followup(
                question=q.question,
                transcription=req.transcription,
                evaluation=evaluation,
                structure_word=q.structure_word,
                entities=meta.entities,
                actions=meta.actions,
            )
        except Exception:
            traceback.print_exc()

    sessions.record_answer(session.session_id, req.transcription, evaluation, followup)

    comment = ""
    if followup and isinstance(followup, dict):
        comment = followup.get("comment", "")
    else:
        comment = evaluation.get("feedback", "")

    next_q = session.current_question
    return {
        "evaluation": evaluation,
        "followup": comment,
        "next_question": {
            "id": next_q.id,
            "text": next_q.question,
            "structure_word": next_q.structure_word,
            "difficulty": next_q.difficulty,
        } if next_q else None,
        "progress": session.progress,
        "completed": session.completed,
    }


# ---------------------------------------------------------------------------
# End session early
# ---------------------------------------------------------------------------

@app.post("/api/session/{session_id}/end")
async def end_session(session_id: str):
    session = sessions.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    session.completed = True
    return {"status": "ended"}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

@app.get("/api/session/{session_id}/summary")
async def session_summary(session_id: str):
    session = sessions.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session.summary()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
