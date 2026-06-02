import os
from dotenv import load_dotenv
load_dotenv()

from supabase import create_client

url = os.environ.get("EXPO_PUBLIC_SUPABASE_URL")
key = os.environ.get("EXPO_PUBLIC_SUPABASE_ANON_KEY")

supabase = create_client(url, key)

try:
    print("Getting 'therapy-audio' bucket details...")
    bucket = supabase.storage.get_bucket("therapy-audio")
    print(f"Bucket found! ID: {bucket.id}, Name: {bucket.name}, Public: {bucket.public}, Created at: {bucket.created_at}")
except Exception as e:
    print(f"Error getting bucket: {type(e).__name__}: {e}")
