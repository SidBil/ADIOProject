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
        # storage3 (used by supabase-py 2.x) has no public .auth() method;
        # set the Authorization header directly instead.
        try:
            if hasattr(supabase.storage, "auth"):
                supabase.storage.auth(token)
            else:
                supabase.storage._client.headers["Authorization"] = f"Bearer {token}"
        except Exception as e:
            print(f"Warning: could not set storage auth token: {e}")
            
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

def create_therapy_session(token: str, user_id: str, app_session_id: str, image_id: Optional[str], image_filename: Optional[str]) -> str:
    """Creates a new therapy session record and returns its internal UUID."""
    supabase = get_supabase_client(token)
    
    data = {
        "user_id": user_id,
        "app_session_id": app_session_id,
        "image_id": image_id,
        "image_filename": image_filename
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

def complete_therapy_session(
    token: str,
    session_internal_id: str,
    total_questions: int,
    questions_answered: int,
    scores: Dict[str, Any]
):
    """Marks a session as completed and updates aggregate scores."""
    if not session_internal_id:
        return
        
    supabase = get_supabase_client(token)
    
    data = {
        "completed": True,
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
