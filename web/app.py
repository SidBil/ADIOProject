"""
ADI/O Therapy Web Application

FastAPI backend that serves the therapy UI and exposes REST endpoints
for session management, ASR transcription, and LLM-powered evaluation.

Run: python app.py
"""

import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.asr_service import ASRService
from services.llm_service import LLMService
from services.session_manager import SessionManager, IMAGE_DIR

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

WEB_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

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

@app.get("/", response_class=HTMLResponse)
async def index():
    return (WEB_DIR / "templates" / "index.html").read_text()


@app.get("/images/{filename}")
async def serve_image(filename: str):
    path = IMAGE_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@app.post("/api/session/start")
async def start_session(req: StartRequest):
    try:
        session = sessions.create_session()
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        gold = llm.generate_gold_standard(session.image_prompt, session.image_id)
        sessions.set_gold_standard(session.session_id, gold)
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Failed to generate therapy content. Check your OPENAI_API_KEY.")

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

    suffix = ".webm" if "webm" in (audio.content_type or "") else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await audio.read()
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
# Evaluate
# ---------------------------------------------------------------------------

@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    session = sessions.get_session(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    if not session.gold_standard:
        raise HTTPException(400, "Session has no gold standard")

    q = session.current_question
    if not q:
        raise HTTPException(400, "No more questions")

    try:
        evaluation = llm.evaluate_response(
            question=q.question,
            expected_answer=q.expected_answer,
            transcription=req.transcription,
            gold_description=session.gold_standard["description"],
            structure_word=q.structure_word,
        )

        followup = llm.generate_followup(
            question=q.question,
            transcription=req.transcription,
            evaluation=evaluation,
            gold_description=session.gold_standard["description"],
            structure_word=q.structure_word,
        )
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Evaluation failed. Check your OPENAI_API_KEY.")

    sessions.record_answer(session.session_id, req.transcription, evaluation, followup)

    comment = followup.get("comment", "") if isinstance(followup, dict) else followup
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
