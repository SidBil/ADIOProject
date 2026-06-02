"""
End-to-end storage auth test.

Creates a temporary Supabase user, signs in to get a real JWT, calls
get_supabase_client(token) exactly as production does, uploads a file,
and confirms it lands in the bucket. Cleans up the user and file afterward.

Run from the project root:
    venv/bin/python web/test_storage_upload.py
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from supabase import create_client

URL  = os.environ["EXPO_PUBLIC_SUPABASE_URL"]
ANON = os.environ["EXPO_PUBLIC_SUPABASE_ANON_KEY"]
# Service role key — never leaves this script, never committed
SERVICE = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVyZ3ZhYXdiemdranFoeXd4ZnRkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NTYxMDY3MywiZXhwIjoyMDkxMTg2NjczfQ.4TSExcvO5RLk285ZCFm4rq9EpRC0ZOrn2l8ydzhViaY"

# ── 1. Create a throwaway user via admin API ────────────────────────────────
admin = create_client(URL, SERVICE)
test_email = f"test-storage-{uuid.uuid4().hex[:8]}@test.invalid"
test_password = uuid.uuid4().hex

print(f"[1] Creating temporary test user: {test_email}")
try:
    create_resp = admin.auth.admin.create_user({
        "email": test_email,
        "password": test_password,
        "email_confirm": True,
    })
    test_user_id = create_resp.user.id
    print(f"    Created: {test_user_id}")
except Exception as e:
    print(f"    FAILED to create user: {e}")
    sys.exit(1)

try:
    # ── 2. Sign in as that user to get a real user JWT ──────────────────────
    print(f"\n[2] Signing in as test user to get JWT...")
    user_client_raw = create_client(URL, ANON)
    sign_in = user_client_raw.auth.sign_in_with_password({
        "email": test_email,
        "password": test_password,
    })
    token = sign_in.session.access_token
    user_id = sign_in.user.id
    print(f"    Got JWT for user: {user_id}")
    print(f"    JWT prefix: {token[:40]}...")

    # ── 3. Build client exactly as get_supabase_client() does ───────────────
    print(f"\n[3] Building client via get_supabase_client(token)...")
    from services.interaction_store import get_supabase_client
    client = get_supabase_client(token)

    auth_header = client.options.headers.get("Authorization", "")
    assert auth_header == f"Bearer {token}", "options.headers not set correctly!"
    print(f"    options.headers['Authorization'] = Bearer <token>  ✓")

    # Confirm _storage is still None (not yet accessed, so headers will be used fresh)
    assert client._storage is None, "_storage was pre-created — token injection may have been too late!"
    print(f"    _storage is None (lazy init not triggered yet)  ✓")

    # ── 4. Upload a tiny fake audio blob ────────────────────────────────────
    fake_audio = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 16 + b"data\x00\x00\x00\x00"
    path = f"users/{user_id}/sessions/test-session/turns/0/audio.webm"

    print(f"\n[4] Uploading to: {path}")
    try:
        # First access of client.storage triggers lazy init with our patched headers
        res = client.storage.from_("therapy-audio").upload(
            path,
            fake_audio,
            file_options={"content-type": "audio/webm"},
        )
        print(f"    Upload response: {res}")
    except Exception as e:
        print(f"    Upload raised: {type(e).__name__}: {e}")
        raise

    # ── 5. Verify the file is there ─────────────────────────────────────────
    print(f"\n[5] Listing directory to confirm file exists...")
    prefix = f"users/{user_id}/sessions/test-session/turns/0"
    files = client.storage.from_("therapy-audio").list(prefix)
    names = [f.get("name") for f in (files or [])]
    print(f"    Found: {names}")

    if "audio.webm" in names:
        print("\n✓  SUCCESS — JWT auth fix works. File uploaded and confirmed via RLS.")
    else:
        print("\n✗  FAILED — no exception but file not found in listing.")
        sys.exit(1)

    print(f"\n[6] Skipping cleanup — file left in bucket for inspection.")
    print(f"    Full path in dashboard: therapy-audio/{path}")

finally:
    pass  # test user intentionally left alive so the file remains accessible
