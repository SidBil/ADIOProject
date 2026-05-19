import os
import json
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
        # Important: To act on behalf of the user, we set the session.
        # This ensures RLS policies apply correctly.
        try:
            # We don't have the refresh token, but setting the access token is often enough
            # for the PostgREST client to include it in the Authorization header.
            # In the Python SDK v2.x, setting the header directly on the postgrest client
            # is the most reliable way to perform authenticated requests without a full sign-in.
            supabase.postgrest.auth(token)
            supabase.storage.auth(token)
        except Exception as e:
            print(f"Error setting Supabase auth: {e}")
            raise HTTPException(status_code=401, detail="Invalid auth token")
            
    return supabase

def verify_token_and_get_user(token: str) -> str:
    """Verifies the JWT token and returns the user ID."""
    try:
        supabase = get_supabase_client()
        # The most secure way to verify a token is to ask Supabase to get the user for it
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
             raise HTTPException(status_code=401, detail="Invalid token")
        return user_response.user.id
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
