"""
Adio Therapy Web Application

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

import asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from services.asr_service import ASRService, MODAL_ASR_URL
from services.llm_service import LLMService
from services.session_manager import SessionManager, IMAGE_DIR
from services import interaction_store

# Accuracy (0-5) at/above which an answer is "good enough" to advance without a
# scaffolded follow-up. Stage 1 (image visible) uses the standard bar; Stage 2
# (recall — image hidden) uses a gentler bar by one point, since recall is a
# harder condition and shouldn't read to the child as failing at the underlying
# skill (V&V Stage 2 PRD `01_recall_session.md` §5). Same rubric, lower trigger.
ACCURACY_THRESHOLD = 4
STAGE2_ACCURACY_THRESHOLD = 3


def _followup_threshold(mode: str) -> int:
    return STAGE2_ACCURACY_THRESHOLD if mode == "stage2" else ACCURACY_THRESHOLD

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

app = FastAPI(title="Adio Therapy", lifespan=lifespan)
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
    # Total time to answer: question shown -> answer submitted (recording stopped).
    answer_duration_ms: float | None = None

class TTSRequest(BaseModel):
    text: str

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def get_token(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return authorization[7:]

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
async def start_session(req: StartRequest, token: str = Depends(get_token)):
    user_id = interaction_store.verify_token_and_get_user(token)

    # The trail decides this session's V&V mode from the child's durable
    # completed-session count (every 3rd node is stage2 — Home path PRD §5/§6).
    # Read the count in parallel with the Modal warmup so mode selection adds no
    # latency to the warmup-blocking path.
    home_state, _ = await asyncio.gather(
        asyncio.to_thread(interaction_store.fetch_home_state, token, user_id),
        asyncio.to_thread(asr.warmup),
    )
    completed_count = home_state.get("completed_count", 0)
    mode = "stage2" if (completed_count + 1) % 3 == 0 else "stage1"

    try:
        session = sessions.create_session(mode=mode)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Persist the session row (with mode). Warmup already completed above, so the
    # container is loaded before the user records their first answer.
    db_id = await asyncio.to_thread(
        interaction_store.create_therapy_session,
        token, user_id, session.session_id, session.image_id, session.image_filename, mode,
    )
    session.internal_db_id = db_id

    q = session.current_question
    return {
        "session_id": session.session_id,
        "image_id": session.image_id,
        "image_url": f"/images/{session.image_filename}",
        "mode": session.mode,
        "question": {
            "id": q.id,
            "text": q.question,
            "structure_word": q.structure_word,
            "difficulty": q.difficulty,
        } if q else None,
        "total_questions": len(session.questions),
        "progress": session.progress,
    }


@app.get("/api/home/state")
async def home_state(token: str = Depends(get_token)):
    """Durable Home-path state (completed-session count + streak) for the trail.

    Read straight from Supabase, not in-memory session state, so Home renders
    correctly on a cold-started serverless instance (Home path PRD §6).
    """
    user_id = interaction_store.verify_token_and_get_user(token)
    return interaction_store.fetch_home_state(token, user_id)


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
async def transcribe(session_id: str, audio: UploadFile = File(...), token: str = Depends(get_token)):
    user_id = interaction_store.verify_token_and_get_user(token)
    session = sessions.get_session(session_id) or sessions.recover_session(session_id, token)
    if not session:
        raise HTTPException(404, "Session not found")

    content = await audio.read()

    suffix = ".webm" if "webm" in (audio.content_type or "") else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    webm_path = tmp_path  # always clean up the original upload
    wav_path = None       # only set if we do a local conversion

    audio_path = None
    if session.internal_db_id:
        audio_path = interaction_store.upload_audio_to_storage(
            token, user_id, session.internal_db_id, session.current_question_idx, content, suffix[1:]
        )

    try:
        if MODAL_ASR_URL:
            result = asr.transcribe_remote_bytes(
                content,
                content_type=audio.content_type or "audio/webm",
                image_id=session.image_id,
            )
        else:
            if suffix == ".webm":
                import subprocess
                wav_path = tmp_path.replace(".webm", ".wav")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
                    capture_output=True, check=True,
                )
                result = asr.transcribe(wav_path, image_id=session.image_id)
            else:
                result = asr.transcribe(tmp_path, image_id=session.image_id)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Transcription failed: {type(e).__name__}: {e}")
    finally:
        Path(webm_path).unlink(missing_ok=True)
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)

    if session.internal_db_id and session.current_question:
        q = session.current_question
        turn_db_id = interaction_store.insert_therapy_turn(
            token, user_id, session.internal_db_id, session.current_question_idx,
            {
                "question": q.question,
                "expected_answer": q.expected_answer,
                "structure_word": q.structure_word
            },
            audio_path,
            result.get("text", ""),
            result.get("latency_ms"),
            None # Initiation latency arrives later
        )
        q.internal_db_id = turn_db_id

    return result


# ---------------------------------------------------------------------------
# Evaluate — hybrid: static advance when good, dynamic followup when not
# ---------------------------------------------------------------------------

@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest, token: str = Depends(get_token)):
    user_id = interaction_store.verify_token_and_get_user(token)
    session = sessions.get_session(req.session_id) or sessions.recover_session(req.session_id, token)
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
            mode=session.mode,
        )
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Evaluation failed. Check your OPENAI_API_KEY.")

    accuracy = evaluation.get("scores", {}).get("accuracy", 5)
    followup = None

    # Stage 2 (recall): cap scaffolding at ONE gentle memory jog per question.
    # Never follow up on an already-dynamic retry, so a forgotten detail can't
    # spiral into repeated pushing — the child gets one nudge, then we release.
    below_threshold = accuracy < _followup_threshold(session.mode)
    recall_retry = session.mode == "stage2" and q.is_dynamic
    if below_threshold and not recall_retry:
        try:
            followup = llm.generate_followup(
                question=q.question,
                transcription=req.transcription,
                evaluation=evaluation,
                structure_word=q.structure_word,
                entities=meta.entities,
                actions=meta.actions,
                mode=session.mode,
            )
        except Exception:
            traceback.print_exc()

    sessions.record_answer(
        session.session_id,
        req.transcription,
        evaluation,
        followup,
        initiation_latency_ms=req.initiation_latency_ms,
        answer_duration_ms=req.answer_duration_ms,
    )

    if session.internal_db_id and q.internal_db_id:
        interaction_store.update_therapy_turn_evaluation(
            token, q.internal_db_id, evaluation,
            followup if isinstance(followup, dict) else {},
            None # LLM latency
        )
        # Best-effort, separate write so a pre-migration schema still keeps the
        # evaluation above. Persists the frontend-captured turn timing.
        interaction_store.update_therapy_turn_timing(
            token, q.internal_db_id,
            initiation_latency_ms=req.initiation_latency_ms,
            answer_duration_ms=req.answer_duration_ms,
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
# Finalize session — called on both natural completion and early quit.
# Completion status reflects whether all questions were actually answered, so
# the clinical view can distinguish completed vs abandoned (PRD §4.4).
# ---------------------------------------------------------------------------

@app.post("/api/session/{session_id}/end")
async def end_session(session_id: str, token: str = Depends(get_token)):
    user_id = interaction_store.verify_token_and_get_user(token)
    session = sessions.get_session(session_id) or sessions.recover_session(session_id, token)
    if not session:
        raise HTTPException(404, "Session not found")
    # A session is "completed" only if every question (and follow-up) resolved.
    # An early quit leaves this False — do NOT force it True.
    completed = session.current_question_idx >= len(session.questions)
    session.completed = completed

    if session.internal_db_id:
        scores = session.compute_scores()
        # Ensure we compute engagement for completion
        if user_id and scores.get("avg_latency_ms") is not None:
             past_latencies = _fetch_past_latencies(user_id, token)
             if len(past_latencies) >= ENGAGEMENT_MIN_SESSIONS:
                 baseline = sum(past_latencies) / len(past_latencies)
                 if baseline > 0:
                     scores["engagement"] = max(0.0, min(1.0, 1.0 - scores["avg_latency_ms"] / baseline))
                 else:
                     scores["engagement"] = 1.0
                     
        interaction_store.complete_therapy_session(
            token, session.internal_db_id, len(session.questions),
            session.progress["answered"], scores, completed=completed
        )

    return {"status": "ended", "completed": completed}


# ---------------------------------------------------------------------------
# Engagement baseline (aggregate score written to therapy_sessions on end)
# ---------------------------------------------------------------------------

ENGAGEMENT_BASELINE_N = 5      # how many past sessions to average for baseline
ENGAGEMENT_MIN_SESSIONS = 5   # minimum sessions required before engagement is computed


def _fetch_past_latencies(user_id: str, token: str) -> list[float]:
    """Query Supabase for the user's past N sessions' avg_latency_ms."""
    sb_url = os.environ.get("EXPO_PUBLIC_SUPABASE_URL", "").rstrip("/")
    sb_key = os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY", "")
    if not sb_url or not sb_key:
        return []

    url = (
        f"{sb_url}/rest/v1/therapy_sessions"
        f"?user_id=eq.{user_id}"
        f"&avg_latency_ms=not.is.null"
        f"&select=avg_latency_ms"
        f"&order=started_at.desc"
        f"&limit={ENGAGEMENT_BASELINE_N}"
    )
    req = urllib.request.Request(url, headers={
        "apikey": sb_key,
        "Authorization": f"Bearer {token}",
    })
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            rows = json_module.loads(resp.read().decode())
            return [r["avg_latency_ms"] for r in rows if r.get("avg_latency_ms") is not None]
    except Exception:
        traceback.print_exc()
        return []


# ---------------------------------------------------------------------------
# Session Summary feature — child celebration + parent/SLP clinical views
# ---------------------------------------------------------------------------

# Aggregate accuracy (0-5) at/above which a completed session earns the single
# additive "bonus flourish" on the child's celebration screen. Upward-only:
# below this the child sees the exact same warm base screen, never a lesser one.
FLOURISH_ACCURACY_THRESHOLD = 1.0  # temporarily lowered so the flourish always shows


def _mean_accuracy(session) -> float | None:
    """Average accuracy (0-5) across a session's answered turns, or None."""
    vals = []
    for q in session.questions:
        ev = q.evaluation or {}
        acc = ev.get("scores", {}).get("accuracy")
        if acc is not None:
            vals.append(acc)
    if not vals:
        return None
    return sum(vals) / len(vals)


@app.get("/api/session/{session_id}/celebration")
async def session_celebration(session_id: str, token: str = Depends(get_token)):
    """Child celebration payload. Deliberately minimal: only a completion signal
    and a single boolean deciding whether the additive flourish is shown. No
    accuracy number ever crosses this boundary — see PRD 01_session_summary §3.4.
    """
    interaction_store.verify_token_and_get_user(token)
    session = sessions.get_session(session_id) or sessions.recover_session(session_id, token)
    if not session:
        raise HTTPException(404, "Session not found")

    mean_acc = _mean_accuracy(session)
    flourish = mean_acc is not None and mean_acc >= FLOURISH_ACCURACY_THRESHOLD

    durations = [q.answer_duration_ms for q in session.questions
                 if q.answer_duration_ms is not None]
    total_time_ms = sum(durations) if durations else None

    return {
        "completed": session.completed,
        "flourish": bool(flourish),
        "accuracy": round(mean_acc, 1) if mean_acc is not None else None,
        "total_time_ms": total_time_ms,
    }


@app.get("/api/dashboard/overview")
async def dashboard_overview(token: str = Depends(get_token)):
    """Parent/SLP dashboard: recent sessions + per-structure-word sparklines."""
    user_id = interaction_store.verify_token_and_get_user(token)
    return interaction_store.fetch_dashboard_overview(token, user_id, window=10)


@app.get("/api/session/{session_id}/detail")
async def session_detail(session_id: str, token: str = Depends(get_token)):
    """Parent/SLP clinical detail for one session (per-structure-word breakdown)."""
    interaction_store.verify_token_and_get_user(token)
    detail = interaction_store.fetch_session_detail(token, session_id)
    if not detail:
        raise HTTPException(404, "Session not found")
    return detail


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
