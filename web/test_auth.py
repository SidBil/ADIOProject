import os
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client

url = os.environ.get("EXPO_PUBLIC_SUPABASE_URL")
key = os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY")

supabase = create_client(url, key)

print("Has auth method?", hasattr(supabase.storage, "auth"))

# Let's also see what headers look like on the client
print("Client headers before:", dict(supabase.storage._client.headers))

token = "fake_jwt_token"
if hasattr(supabase.storage, "auth"):
    print("Calling auth()")
    supabase.storage.auth(token)
else:
    print("Setting header manually")
    supabase.storage._client.headers["Authorization"] = f"Bearer {token}"

print("Client headers after:", dict(supabase.storage._client.headers))
