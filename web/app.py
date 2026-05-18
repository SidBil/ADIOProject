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
import json as json_module
import tempfile
import traceback
import urllib.request
import urllib.error
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

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
    initiation_latency_ms: float | None = None

class TTSRequest(BaseModel):
    text: str

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

    sessions.record_answer(
        session.session_id,
        req.transcription,
        evaluation,
        followup,
        initiation_latency_ms=req.initiation_latency_ms,
    )

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

ENGAGEMENT_BASELINE_N = 5
ENGAGEMENT_MIN_SESSIONS = 3


def _fetch_past_latencies(user_id: str) -> list[float]:
    """Query Supabase for the user's past N sessions' avg_latency_ms."""
    sb_url = os.environ.get("EXPO_PUBLIC_SUPABASE_URL", "").rstrip("/")
    sb_key = os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY", "")
    if not sb_url or not sb_key:
        return []

    url = (
        f"{sb_url}/rest/v1/sessions"
        f"?user_id=eq.{user_id}"
        f"&avg_latency_ms=not.is.null"
        f"&select=avg_latency_ms"
        f"&order=created_at.desc"
        f"&limit={ENGAGEMENT_BASELINE_N}"
    )
    req = urllib.request.Request(url, headers={
        "apikey": sb_key,
        "Authorization": f"Bearer {sb_key}",
    })
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            rows = json_module.loads(resp.read().decode())
            return [r["avg_latency_ms"] for r in rows if r.get("avg_latency_ms") is not None]
    except Exception:
        traceback.print_exc()
        return []


def _count_past_sessions(user_id: str) -> int:
    """Count how many past sessions exist for this user."""
    sb_url = os.environ.get("EXPO_PUBLIC_SUPABASE_URL", "").rstrip("/")
    sb_key = os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY", "")
    if not sb_url or not sb_key:
        return 0

    url = (
        f"{sb_url}/rest/v1/sessions"
        f"?user_id=eq.{user_id}"
        f"&select=id"
        f"&limit=100"
    )
    req = urllib.request.Request(url, headers={
        "apikey": sb_key,
        "Authorization": f"Bearer {sb_key}",
        "Prefer": "count=exact",
    })
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            # The content-range header has the total count
            cr = resp.headers.get("content-range", "")
            if "/" in cr:
                total = cr.split("/")[-1]
                return int(total) if total != "*" else 0
            rows = json_module.loads(resp.read().decode())
            return len(rows)
    except Exception:
        traceback.print_exc()
        return 0


@app.get("/api/session/{session_id}/summary")
async def session_summary(session_id: str, user_id: str = ""):
    session = sessions.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    result = session.summary()
    scores = result.get("scores", {})

    # --- Compute Engagement from cross-session baseline ---
    if user_id and scores.get("avg_latency_ms") is not None:
        past_latencies = _fetch_past_latencies(user_id)
        past_count = _count_past_sessions(user_id)

        if len(past_latencies) >= ENGAGEMENT_MIN_SESSIONS:
            baseline = sum(past_latencies) / len(past_latencies)
            if baseline > 0:
                e = max(0.0, min(1.0, 1.0 - scores["avg_latency_ms"] / baseline))
                scores["engagement"] = round(e, 3)
            else:
                scores["engagement"] = 1.0
        else:
            scores["engagement"] = None

        scores["sessions_toward_baseline"] = past_count
        scores["baseline_min_sessions"] = ENGAGEMENT_MIN_SESSIONS
    else:
        scores["sessions_toward_baseline"] = 0
        scores["baseline_min_sessions"] = ENGAGEMENT_MIN_SESSIONS

    result["scores"] = scores
    return result


# ---------------------------------------------------------------------------
# Text-to-Speech (OpenAI TTS)
# ---------------------------------------------------------------------------

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    return await _generate_tts(req.text)


@app.get("/api/tts")
async def text_to_speech_get(text: str):
    return await _generate_tts(text)


async def _generate_tts(text: str):
    if not text.strip():
        raise HTTPException(400, "Empty text")
    try:
        client = OpenAI()
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
            response_format="mp3",
        )
        return StreamingResponse(
            response.iter_bytes(),
            media_type="audio/mpeg",
            headers={"Content-Type": "audio/mpeg"},
        )
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "TTS generation failed")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
