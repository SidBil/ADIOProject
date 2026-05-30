-- Migration: 002_storage_policies.sql
-- Description: Creates the 'therapy-audio' storage bucket and configures Row Level Security (RLS) policies
--              to allow authenticated users to upload, read, and delete their own audio recordings.

-- 1. Create the therapy-audio bucket if it doesn't already exist
INSERT INTO storage.buckets (id, name, public)
VALUES ('therapy-audio', 'therapy-audio', false)
ON CONFLICT (id) DO NOTHING;

-- 2. Ensure Row Level Security is enabled on storage.objects
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;

-- 3. Policy to allow authenticated users to upload (insert) files to their own directory in the bucket
-- Path format: users/{user_id}/sessions/{session_id}/turns/{turn_index}/audio.{ext}
-- split_part(name, '/', 1) = 'users'
-- split_part(name, '/', 2) = {user_id}
CREATE POLICY "Allow authenticated uploads to own folder"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (
  bucket_id = 'therapy-audio' AND
  (split_part(name, '/', 2)) = auth.uid()::text
);

-- 4. Policy to allow authenticated users to read (select) files in their own directory
CREATE POLICY "Allow authenticated select from own folder"
ON storage.objects FOR SELECT
TO authenticated
USING (
  bucket_id = 'therapy-audio' AND
  (split_part(name, '/', 2)) = auth.uid()::text
);

-- 5. Policy to allow authenticated users to delete files from their own directory
CREATE POLICY "Allow authenticated delete from own folder"
ON storage.objects FOR DELETE
TO authenticated
USING (
  bucket_id = 'therapy-audio' AND
  (split_part(name, '/', 2)) = auth.uid()::text
);
