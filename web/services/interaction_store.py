import os
import json
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from supabase import create_client, Client
from fastapi import HTTPException

def get_supabase_client(token: Optional[str] = None) -> Client:
    """Returns a Supabase client, optionally acting as a specific user if a token is provided."""
    url: str = os.environ.get("EXPO_PUBLIC_SUPABASE_URL", "")
    key: str = os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY", "")
    
    if not url or not key:
        raise ValueError("Supabase URL or Key is missing from environment variables.")
        
    supabase: Client = create_client(url, key)

    if token:
        # Set postgrest auth so RLS policies apply correctly.
        supabase.postgrest.auth(token)
        # Inject the user JWT before supabase.storage is first accessed.
        # supabase.storage is lazy-initialized using supabase.options.headers,
        # so updating here ensures the storage client is built with the user token.
        # (storage3 stores a frozen header snapshot at init time, so we can't
        # patch it after the fact via supabase.storage._client.headers.)
        supabase.options.headers["Authorization"] = f"Bearer {token}"

    return supabase

def verify_token_and_get_user(token: str) -> str:
    """Extracts user ID from the JWT payload.

    Implicit-flow tokens don't have a server-side session entry, so calling
    supabase.auth.get_user() fails. Instead we decode the payload directly —
    Supabase RLS enforces real access control when the token is used in DB calls.
    """
    try:
        payload_b64 = token.split(".")[1]
        # Base64url → standard base64
        padding = (4 - len(payload_b64) % 4) % 4
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * padding))
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub claim")
        return user_id
    except HTTPException:
        raise
    except Exception as e:
        print(f"Auth verification failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

def create_therapy_session(token: str, user_id: str, app_session_id: str, image_id: Optional[str], image_filename: Optional[str], mode: str = "stage1") -> str:
    """Creates a new therapy session record and returns its internal UUID."""
    supabase = get_supabase_client(token)

    data = {
        "user_id": user_id,
        "app_session_id": app_session_id,
        "image_id": image_id,
        "image_filename": image_filename,
        "mode": mode,
    }
    
    try:
        response = supabase.table("therapy_sessions").insert(data).execute()
        if not response.data:
            raise Exception("No data returned from insert")
        return response.data[0]["id"]
    except Exception as e:
        print(f"Failed to create therapy session: {e}")
        # Return a fallback or raise depending on how strict we want to be.
        # For now, we'll return None and handle it in the caller if we want graceful degradation.
        return None

def upload_audio_to_storage(token: str, user_id: str, session_internal_id: str, turn_index: int, audio_bytes: bytes, extension: str = "webm") -> Optional[str]:
    """Uploads raw audio to the therapy-audio bucket."""
    supabase = get_supabase_client(token)
    
    # E.g., users/123/sessions/456/turns/0/audio.webm
    path = f"users/{user_id}/sessions/{session_internal_id}/turns/{turn_index}/audio.{extension}"
    
    try:
        res = supabase.storage.from_("therapy-audio").upload(
            path,
            audio_bytes,
            file_options={"content-type": f"audio/{extension}"}
        )
        return path
    except Exception as e:
        print(f"Failed to upload audio: {e}")
        return None

def insert_therapy_turn(
    token: str, 
    user_id: str, 
    session_internal_id: str, 
    turn_index: int, 
    question_data: Dict[str, Any],
    audio_path: Optional[str],
    transcription: str,
    asr_latency_ms: Optional[float],
    initiation_latency_ms: Optional[float]
) -> Optional[str]:
    """Inserts a new therapy turn after audio is transcribed."""
    if not session_internal_id:
        return None
        
    supabase = get_supabase_client(token)
    
    data = {
        "session_id": session_internal_id,
        "user_id": user_id,
        "turn_index": turn_index,
        "question_text": question_data.get("question", ""),
        "expected_answer": question_data.get("expected_answer", ""),
        "structure_word": question_data.get("structure_word", ""),
        "audio_storage_path": audio_path,
        "transcription": transcription,
        "initiation_latency_ms": initiation_latency_ms,
        "asr_latency_ms": asr_latency_ms
    }
    
    try:
        # Upsert in case it already exists (e.g., retries)
        response = supabase.table("therapy_turns").upsert(data, on_conflict="session_id, turn_index").execute()
        if response.data:
            return response.data[0]["id"]
        return None
    except Exception as e:
        print(f"Failed to insert therapy turn: {e}")
        return None

def update_therapy_turn_evaluation(
    token: str,
    turn_internal_id: str,
    evaluation: Dict[str, Any],
    followup: Dict[str, Any],
    llm_latency_ms: Optional[float]
):
    """Updates an existing therapy turn with LLM evaluation results."""
    if not turn_internal_id:
        return
        
    supabase = get_supabase_client(token)
    
    data = {
        "llm_evaluation": evaluation,
        "llm_followup": followup,
        "llm_latency_ms": llm_latency_ms,
        "scores": evaluation.get("scores", {})
    }
    
    try:
        supabase.table("therapy_turns").update(data).eq("id", turn_internal_id).execute()
    except Exception as e:
        print(f"Failed to update therapy turn with evaluation: {e}")

def update_therapy_turn_timing(
    token: str,
    turn_internal_id: str,
    initiation_latency_ms: Optional[float] = None,
    answer_duration_ms: Optional[float] = None,
):
    """Best-effort write of per-turn timing captured on the frontend.

    Kept as a separate update from the evaluation write so that a schema that
    hasn't applied migration 003 (answer_duration_ms) yet can still persist the
    LLM evaluation — only the timing write fails, and it fails silently.
    """
    if not turn_internal_id:
        return

    data = {}
    if initiation_latency_ms is not None:
        data["initiation_latency_ms"] = initiation_latency_ms
    if answer_duration_ms is not None:
        data["answer_duration_ms"] = answer_duration_ms
    if not data:
        return

    try:
        supabase = get_supabase_client(token)
        supabase.table("therapy_turns").update(data).eq("id", turn_internal_id).execute()
    except Exception as e:
        print(f"Failed to update therapy turn timing: {e}")


# ---------------------------------------------------------------------------
# Read helpers for the parent/SLP clinical views (Session Summary feature)
# ---------------------------------------------------------------------------

# The session's fixed question contract (single source of truth in
# session_manager). The clinical views render exactly these structure words, in
# this order, so detail entries and sparklines always match what a session asks.
from services.session_manager import SESSION_STRUCTURE_WORDS as STRUCTURE_WORDS


def _duration_ms(started_at: Optional[str], ended_at: Optional[str]) -> Optional[float]:
    """Milliseconds between two ISO timestamps, or None if either is missing."""
    if not started_at or not ended_at:
        return None
    try:
        s = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        e = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
        return max(0.0, (e - s).total_seconds() * 1000.0)
    except Exception:
        return None


def _accuracy(turn: Dict[str, Any]) -> Optional[float]:
    """Pull the accuracy score (0-5) out of a turn's evaluation, if present."""
    scores = (turn.get("scores") or {})
    if "accuracy" in scores and scores["accuracy"] is not None:
        return scores["accuracy"]
    ev = turn.get("llm_evaluation") or {}
    ev_scores = ev.get("scores") or {}
    return ev_scores.get("accuracy")


def _turn_has_followup(turn: Dict[str, Any]) -> bool:
    fu = turn.get("llm_followup") or {}
    return isinstance(fu, dict) and bool(fu.get("comment") or fu.get("suggested_question"))


def fetch_home_state(token: str, user_id: str) -> Dict[str, Any]:
    """Durable Home-path state for a child (Home session path, PRD §5/§6).

    The trail is a pure function of ONE quantity: the count of *completed*
    sessions. This reads that count (and a forgiving day-streak) straight from
    Supabase rather than the in-memory session dict, so Home renders correctly on
    a cold-started serverless instance.
    """
    from datetime import date, timedelta, datetime as _dt

    supabase = get_supabase_client(token)
    res = supabase.table("therapy_sessions") \
        .select("started_at, image_filename, image_id") \
        .eq("user_id", user_id) \
        .eq("completed", True) \
        .order("started_at", desc=True) \
        .execute()
    rows = res.data or []

    completed_count = len(rows)
    last_session_at = rows[0]["started_at"] if rows else None

    # Image filename per completed session, oldest-first — so trail node N (1-based,
    # == the child's Nth session) maps to completed_images[N-1] for its thumbnail.
    completed_images = [
        (r.get("image_filename") or r.get("image_id"))
        for r in reversed(rows)
    ]

    # Child's nickname for the Home greeting (user_profiles is keyed by auth id).
    child_name = None
    try:
        prof = supabase.table("user_profiles") \
            .select("child_nickname") \
            .eq("id", user_id) \
            .maybe_single() \
            .execute()
        if prof and prof.data:
            child_name = (prof.data.get("child_nickname") or "").strip() or None
    except Exception:
        child_name = None

    # Forgiving day-streak: distinct calendar days (UTC) with a completed
    # session, counted back from today/yesterday. Never zeroed hard mid-run —
    # a single missed day simply doesn't extend it. (Silent freeze/protection
    # nuance from the streak PRD is out of scope for this pass.)
    days: set = set()
    for r in rows:
        ts = r.get("started_at")
        if not ts:
            continue
        try:
            days.add(_dt.fromisoformat(ts.replace("Z", "+00:00")).date())
        except Exception:
            continue

    streak_days = 0
    if days:
        today = date.today()
        cursor = today if today in days else today - timedelta(days=1)
        while cursor in days:
            streak_days += 1
            cursor -= timedelta(days=1)

    return {
        "completed_count": completed_count,
        "streak_days": streak_days,
        "last_session_at": last_session_at,
        "completed_images": completed_images,
        "child_name": child_name,
    }


def fetch_dashboard_overview(token: str, user_id: str, window: int = 10) -> Dict[str, Any]:
    """Recent sessions + per-structure-word accuracy sparklines.

    Returns the last `window` sessions of ANY kind (completed or abandoned),
    newest first, plus one sparkline series per structure word. Each series has
    `window` points, oldest -> newest, where the value is that session's
    accuracy for the structure word (follow-up answer's accuracy if a follow-up
    occurred, else the original answer's accuracy) or None when the structure
    word was never answered in that session (rendered as a gap, not a zero).
    """
    supabase = get_supabase_client(token)

    sessions_res = supabase.table("therapy_sessions") \
        .select("id, app_session_id, image_id, image_filename, started_at, ended_at, "
                "completed, questions_answered, total_questions") \
        .eq("user_id", user_id) \
        .order("started_at", desc=True) \
        .limit(window) \
        .execute()
    sessions = sessions_res.data or []

    session_ids = [s["id"] for s in sessions]
    turns_by_session: Dict[str, list] = {sid: [] for sid in session_ids}
    if session_ids:
        turns_res = supabase.table("therapy_turns") \
            .select("session_id, turn_index, structure_word, scores, llm_evaluation, llm_followup") \
            .in_("session_id", session_ids) \
            .order("turn_index") \
            .execute()
        for t in (turns_res.data or []):
            turns_by_session.setdefault(t["session_id"], []).append(t)

    session_summaries = []
    for s in sessions:
        turns = turns_by_session.get(s["id"], [])
        # A follow-up is any turn whose structure word already appeared earlier
        # in the session (the first occurrence is the original question).
        seen: set = set()
        followups = 0
        for t in turns:
            sw = t.get("structure_word")
            if sw in seen:
                followups += 1
            else:
                seen.add(sw)
        session_summaries.append({
            "session_id": s.get("app_session_id"),
            "image_id": s.get("image_id"),
            "image_filename": s.get("image_filename") or s.get("image_id"),
            "started_at": s.get("started_at"),
            "ended_at": s.get("ended_at"),
            "completed": s.get("completed", False),
            "questions_answered": s.get("questions_answered", 0),
            "total_questions": s.get("total_questions"),
            "duration_ms": _duration_ms(s.get("started_at"), s.get("ended_at")),
            "followup_count": followups,
        })

    # Sparklines: iterate sessions oldest -> newest so the line reads left->right.
    ordered = list(reversed(sessions))
    sparklines: Dict[str, list] = {sw: [] for sw in STRUCTURE_WORDS}
    for s in ordered:
        turns = turns_by_session.get(s["id"], [])
        # Last turn per structure word = the follow-up when one occurred, else
        # the original — matches the PRD's "plot follow-up accuracy if any".
        latest_for_word: Dict[str, Dict[str, Any]] = {}
        for t in turns:
            sw = t.get("structure_word")
            if sw in STRUCTURE_WORDS:
                latest_for_word[sw] = t
        for sw in STRUCTURE_WORDS:
            t = latest_for_word.get(sw)
            sparklines[sw].append(_accuracy(t) if t else None)

    return {
        "sessions": session_summaries,
        "sparklines": sparklines,
        "structure_words": STRUCTURE_WORDS,
        "window": window,
    }


def fetch_session_detail(token: str, app_session_id: str) -> Optional[Dict[str, Any]]:
    """Full per-structure-word clinical breakdown for one session.

    Returns one entry per structure word (all 10, in canonical order). Each
    entry has an `original` turn and, when a follow-up was triggered for that
    word, a nested `followup` turn. Structure words never reached (e.g. an
    abandoned session) have `original: null`.
    """
    supabase = get_supabase_client(token)

    ses_res = supabase.table("therapy_sessions") \
        .select("id, app_session_id, image_id, image_filename, started_at, ended_at, "
                "completed, questions_answered, total_questions") \
        .eq("app_session_id", app_session_id) \
        .maybe_single() \
        .execute()
    if not ses_res or not ses_res.data:
        return None
    ses = ses_res.data

    # answer_duration_ms is added by migration 003; a deployment that hasn't
    # applied it must not 500 the whole detail view. Select it when present, fall
    # back without it otherwise (shape() reads it via .get, so None is fine).
    _base_cols = ("turn_index, structure_word, question_text, expected_answer, "
                  "transcription, scores, llm_evaluation, llm_followup, "
                  "initiation_latency_ms")

    def _load_turns(cols: str):
        return supabase.table("therapy_turns") \
            .select(cols) \
            .eq("session_id", ses["id"]) \
            .order("turn_index") \
            .execute()

    try:
        turns_res = _load_turns(_base_cols + ", answer_duration_ms")
    except Exception:
        turns_res = _load_turns(_base_cols)
    turns = turns_res.data or []

    def shape(turn: Dict[str, Any]) -> Dict[str, Any]:
        ev = turn.get("llm_evaluation") or {}
        return {
            "question": turn.get("question_text", ""),
            "expected_answer": turn.get("expected_answer", ""),
            "transcription": turn.get("transcription", ""),
            "accuracy": _accuracy(turn),
            "feedback": ev.get("feedback", ""),
            "identified_elements": ev.get("identified_elements", []) or [],
            "missed_elements": ev.get("missed_elements", []) or [],
            "answer_duration_ms": turn.get("answer_duration_ms"),
        }

    # Group turns by structure word in ask order: first = original, rest = follow-ups.
    grouped: Dict[str, list] = {}
    followup_count = 0
    for t in turns:
        sw = t.get("structure_word")
        grouped.setdefault(sw, []).append(t)
        if len(grouped[sw]) > 1:
            followup_count += 1

    entries = []
    for sw in STRUCTURE_WORDS:
        group = grouped.get(sw, [])
        original = shape(group[0]) if group else None
        followup = shape(group[1]) if len(group) > 1 else None
        entries.append({
            "structure_word": sw,
            "original": original,
            "followup": followup,
        })

    return {
        "session_id": ses.get("app_session_id"),
        "image_id": ses.get("image_id"),
        "image_filename": ses.get("image_filename") or ses.get("image_id"),
        "started_at": ses.get("started_at"),
        "ended_at": ses.get("ended_at"),
        "completed": ses.get("completed", False),
        "questions_answered": ses.get("questions_answered", 0),
        "total_questions": ses.get("total_questions"),
        "duration_ms": _duration_ms(ses.get("started_at"), ses.get("ended_at")),
        "followup_count": followup_count,
        "entries": entries,
    }


def complete_therapy_session(
    token: str,
    session_internal_id: str,
    total_questions: int,
    questions_answered: int,
    scores: Dict[str, Any],
    completed: bool = True,
):
    """Finalizes a session: sets ended_at, aggregate scores, and whether it was
    actually completed (all questions answered) vs abandoned early."""
    if not session_internal_id:
        return

    supabase = get_supabase_client(token)

    data = {
        "completed": completed,
        "ended_at": datetime.utcnow().isoformat(),
        "total_questions": total_questions,
        "questions_answered": questions_answered,
        "observation_score": scores.get("observation"),
        "understanding_score": scores.get("understanding"),
        "engagement_score": scores.get("engagement"),
        "avg_latency_ms": scores.get("avg_latency_ms")
    }
    
    try:
        supabase.table("therapy_sessions").update(data).eq("id", session_internal_id).execute()
    except Exception as e:
        print(f"Failed to complete therapy session: {e}")

def log_analytics_error(
    token: str,
    user_id: str,
    area: str,
    error_message: str,
    session_id: Optional[str] = None
):
    """Logs an error to the analytics_errors table."""
    try:
         supabase = get_supabase_client(token)
         supabase.table("analytics_errors").insert({
             "user_id": user_id,
             "area": area,
             "error_message": error_message,
             "session_id": session_id
         }).execute()
    except Exception:
         pass # Don't crash on analytics logging failures
