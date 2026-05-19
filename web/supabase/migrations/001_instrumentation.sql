-- Migration: 001_instrumentation.sql
-- Description: Creates the normalized tables for tracking Adio therapy sessions, turns, and user profiles.

-- 1. User Profiles (Onboarding Data)
create table public.user_profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  guardian_first_name text not null,
  guardian_last_name text not null,
  child_nickname text not null,
  grade_level text not null,
  speech_data_opt_in boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 2. Therapy Sessions
create table public.therapy_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  image_id text,
  image_filename text,
  image_complexity int,
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  completed boolean not null default false,
  total_questions int,
  questions_answered int not null default 0,
  observation_score numeric,
  understanding_score numeric,
  engagement_score numeric,
  avg_latency_ms numeric,
  metadata jsonb not null default '{}'::jsonb
);

-- 3. Therapy Turns (Question/Answer pairs)
create table public.therapy_turns (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.therapy_sessions(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  turn_index int not null,
  question_id text,
  structure_word text,
  difficulty int,
  question_text text not null,
  expected_answer text,
  image_id text,
  audio_storage_path text,
  audio_content_type text,
  audio_size_bytes bigint,
  audio_duration_ms numeric,
  recording_started_at timestamptz,
  recording_stopped_at timestamptz,
  initiation_latency_ms numeric,
  transcription text,
  asr_mode text,
  asr_latency_ms numeric,
  asr_result jsonb not null default '{}'::jsonb,
  llm_model text,
  llm_latency_ms numeric,
  llm_evaluation jsonb not null default '{}'::jsonb,
  llm_followup jsonb not null default '{}'::jsonb,
  scores jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(session_id, turn_index)
);

-- 4. Analytics Errors
create table public.analytics_errors (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete set null,
  session_id uuid references public.therapy_sessions(id) on delete set null,
  turn_id uuid references public.therapy_turns(id) on delete set null,
  area text not null,
  error_code text,
  error_message text,
  context jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

-- 5. User Consents
create table public.user_consents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  consent_type text not null,
  granted boolean not null,
  version text not null,
  created_at timestamptz not null default now()
);

-- RLS Policies

alter table public.user_profiles enable row level security;
alter table public.therapy_sessions enable row level security;
alter table public.therapy_turns enable row level security;
alter table public.analytics_errors enable row level security;
alter table public.user_consents enable row level security;

-- Users can read/write their own profile
create policy "users can read their own profile"
on public.user_profiles for select to authenticated using (auth.uid() = id);

create policy "users can insert their own profile"
on public.user_profiles for insert to authenticated with check (auth.uid() = id);

create policy "users can update their own profile"
on public.user_profiles for update to authenticated using (auth.uid() = id);

-- Users can read their own sessions and turns
create policy "users can read their own sessions"
on public.therapy_sessions for select to authenticated using (auth.uid() = user_id);

create policy "users can read their own turns"
on public.therapy_turns for select to authenticated using (auth.uid() = user_id);

-- Note: In this architecture, we prefer the backend (using a service role key or 
-- acting on behalf of the user after verifying the JWT) to handle the actual inserts
-- for sessions, turns, and errors to prevent client-side manipulation of therapy data.
-- If the client needs to insert directly, add INSERT policies here.
