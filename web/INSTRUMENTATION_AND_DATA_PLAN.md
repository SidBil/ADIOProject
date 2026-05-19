# Adio Instrumentation and Interaction Data Plan

This document explains how to instrument Adio in two layers:

1. **Product analytics events**: lightweight behavioral events in Firebase / Google Analytics for Firebase.
2. **Interaction records**: sensitive therapy-session data stored in Supabase Postgres + Supabase Storage.

Do **not** store audio, transcripts, LLM responses, email addresses, names, or child-identifying data in Firebase Analytics. Firebase Analytics is for aggregate usage behavior. Supabase is the system of record for therapy interactions.

---

## 1. One-Page Instructions for the Coding Agent

### Goal

Add instrumentation to the Adio app without changing the therapy flow.

Use:

- **Firebase Analytics** for anonymous product events.
- **Supabase Postgres** for structured session/turn records.
- **Supabase Storage** for uploaded audio files.

### Files to Touch

Frontend:

- `web/src/api.ts`
- `web/App.tsx`
- `web/src/screens/LoginScreen.tsx`
- `web/src/screens/WelcomeScreen.tsx`
- `web/src/screens/SessionScreen.tsx`
- `web/src/screens/SummaryScreen.tsx`
- New file: `web/src/lib/analytics.ts`

Backend:

- `web/app.py`
- `web/services/session_manager.py`
- New file: `web/services/interaction_store.py`

Database:

- Add a Supabase SQL migration or a checked-in SQL file under `web/supabase/`.

### Firebase Analytics Tasks

Create `web/src/lib/analytics.ts` with a small wrapper:

```ts
export function track(eventName: string, params?: Record<string, any>) {
  // No PII, no transcripts, no audio filenames, no free text.
}
```

Track these events:

- `auth_started`
- `auth_completed`
- `session_started`
- `question_viewed`
- `recording_started`
- `recording_stopped`
- `transcription_completed`
- `evaluation_completed`
- `session_completed`
- `session_abandoned`
- `app_error`

Allowed event params:

- `screen`
- `question_index`
- `structure_word`
- `image_complexity`
- `duration_ms`
- `latency_ms`
- `success`
- `error_code`
- `asr_mode`

Never send:

- Child name
- Parent email
- Audio URL/path
- Transcript text
- Expected answer
- LLM feedback
- Session ID if it can identify a user

### Supabase Persistence Tasks

Create persistent records for:

- One `therapy_sessions` row per session.
- One `therapy_turns` row per question/answer turn.
- One private Storage object per audio recording.

Flow:

1. Frontend sends Supabase JWT to backend on every API request:

   ```ts
   const { data } = await supabase.auth.getSession();
   const token = data.session?.access_token;
   headers.Authorization = `Bearer ${token}`;
   ```

2. Backend verifies the JWT and derives `user_id`.
3. `/api/session/start` inserts `therapy_sessions`.
4. `/api/transcribe` uploads raw audio to private Supabase Storage and inserts/updates `therapy_turns`.
5. `/api/evaluate` updates the same `therapy_turns` row with LLM output.
6. `/api/session/{id}/summary` reads from Postgres, not only in-memory state.

### Storage Path

Use this path format:

```text
users/{user_id}/sessions/{session_id}/turns/{turn_id}/audio.webm
```

Bucket:

```text
therapy-audio
```

The bucket must be private.

### Privacy Requirements

- Raw child audio is sensitive personal data.
- Keep Firebase Analytics free of PII and free-text content.
- Use Supabase RLS for tables.
- Use private Supabase Storage for audio.
- Add a retention policy: either delete audio after transcription or keep it only with explicit parent consent.

---

## 2. Recommended Metrics

The metrics should answer five questions:

1. Are people able to complete sessions?
2. Where do they get stuck?
3. Is the ASR pipeline fast and accurate enough?
4. Are LLM evaluations useful and stable?
5. Is the therapy interaction improving over time?

### Product Usage Metrics

| Metric | Definition | Source |
|---|---:|---|
| Daily active users | Distinct users who open the app per day | Firebase |
| Session starts | Count of `session_started` | Firebase + Supabase |
| Session completions | Count of `session_completed` | Firebase + Supabase |
| Completion rate | Completed sessions / started sessions | Firebase or Supabase |
| Abandonment rate | Started but not completed | Firebase or Supabase |
| Questions answered per session | Average completed turns per session | Supabase |
| Time per session | `ended_at - started_at` | Supabase |
| Recording attempts per question | Count recording starts per turn | Firebase + Supabase |
| Error rate | Errors / sessions or errors / turns | Firebase |
| Mic permission denial rate | Permission denied / recording starts | Firebase |

### Therapy Interaction Metrics

| Metric | Definition | Source |
|---|---:|---|
| Observation score | Current app formula based on omissions + relevance | Supabase |
| Understanding score | Current app formula based on relevance + accuracy | Supabase |
| Engagement proxy | Response initiation latency relative to baseline | Supabase |
| Average initiation latency | Time from question shown to recording started | Supabase |
| Average response duration | Recording duration per turn | Supabase |
| Follow-up rate | Turns where LLM generated a follow-up / total turns | Supabase |
| Structure-word coverage | Turns by `who`, `what`, `where`, `color`, etc. | Supabase |
| Difficulty progression | Accuracy/detail by difficulty level | Supabase |
| Improvement over time | Score trend across sessions per user | Supabase |

### ASR Metrics

| Metric | Definition | Source |
|---|---:|---|
| ASR latency | Audio upload received to transcript returned | Supabase |
| ASR mode | `asr_only` vs `multimodal` | Supabase |
| Hypothesis count | Number of ASR candidates | Supabase |
| CLIP rerank changed winner | Top fused hypothesis differs from top ASR hypothesis | Supabase |
| Average confidence/fused score | Mean top score per turn | Supabase |
| Transcription failure rate | Failed transcriptions / transcription attempts | Firebase + Supabase |
| Manual WER sample | Human-labeled sample audit, not automatic | Offline eval table |

### LLM Metrics

| Metric | Definition | Source |
|---|---:|---|
| Evaluation latency | LLM call start to response parsed | Supabase |
| JSON parse failure rate | Failed JSON parses / LLM calls | Supabase |
| Average accuracy/detail/clarity/relevance | Mean LLM score dimensions | Supabase |
| Follow-up generation rate | Accuracy below threshold and follow-up created | Supabase |
| Model version distribution | Which model evaluated each turn | Supabase |
| Token usage/cost | Prompt + completion tokens if available | Supabase |

### Safety and Compliance Metrics

| Metric | Definition | Source |
|---|---:|---|
| Audio retained count | Audio files currently stored | Supabase Storage |
| Audio deleted count | Files deleted by retention job | Supabase |
| Consent status coverage | Users/sessions with consent recorded | Supabase |
| Access denied count | Unauthorized reads/writes blocked | Supabase logs |
| Data export/delete requests | Parent/user privacy operations | Supabase |

---

## 3. Backend Data Store Recommendation

### Recommendation

Use **Supabase Postgres + Supabase Storage** for interaction data.

Reason:

- The app already uses Supabase Auth and writes session summary data to Supabase.
- Postgres is better than Firebase Analytics for structured therapy records, JSON output, reporting joins, retention, and auditability.
- Supabase Storage is designed for files and supports private access with Row Level Security.
- Keeping interaction data in Supabase avoids having two identity systems.

Use **Firebase Analytics only for product-event analytics**.

### Alternatives Considered

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| Firebase Analytics only | Easy charts | Cannot store audio/transcripts; PII restrictions | Not suitable |
| Firebase Firestore + Cloud Storage | Good all-Google stack | Duplicates existing Supabase auth/data | Reasonable, but unnecessary here |
| Supabase Postgres + Storage | Fits current app, SQL reporting, private storage | Requires backend persistence work | Recommended |
| S3 + Postgres | Flexible and scalable | More infra to operate | Overkill for this project |

---

## 4. Proposed Supabase Data Model

### Tables

```sql
create table public.therapy_sessions (
  id uuid primary key default gen_random_uuid(),
  app_session_id text unique not null,
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

create table public.user_consents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  consent_type text not null,
  granted boolean not null,
  granted_by_user_id uuid references auth.users(id),
  version text not null,
  created_at timestamptz not null default now()
);
```

### Storage Bucket

Create a private bucket:

```text
therapy-audio
```

Store recordings at:

```text
users/{user_id}/sessions/{session_id}/turns/{turn_id}/audio.{ext}
```

### RLS Policy Shape

Enable RLS on the tables:

```sql
alter table public.therapy_sessions enable row level security;
alter table public.therapy_turns enable row level security;
alter table public.analytics_errors enable row level security;
alter table public.user_consents enable row level security;
```

For authenticated users:

```sql
create policy "users can read their own sessions"
on public.therapy_sessions
for select
to authenticated
using (auth.uid() = user_id);

create policy "users can read their own turns"
on public.therapy_turns
for select
to authenticated
using (auth.uid() = user_id);
```

For writes, prefer server-side writes using the Supabase service role key after the backend verifies the user JWT. Do **not** expose the service role key to the frontend.

### Why Store Both Raw JSON and Computed Columns?

Keep raw model output in `jsonb`:

- `asr_result`
- `llm_evaluation`
- `llm_followup`

Also store common fields as columns:

- `asr_mode`
- `llm_model`
- `transcription`
- `observation_score`
- `understanding_score`
- `avg_latency_ms`

This gives both auditability and easy dashboard queries.

---

## 5. Firebase Analytics Plan

### Event Taxonomy

Use a small number of stable events. Put variation in parameters instead of creating many event names.

| Event | When | Important Params |
|---|---|---|
| `auth_started` | User starts sign-in/sign-up | `method`, `screen` |
| `auth_completed` | Auth succeeds | `method`, `success` |
| `session_started` | Backend creates a session | `image_complexity`, `total_questions` |
| `question_viewed` | Question card appears | `question_index`, `structure_word`, `difficulty` |
| `recording_started` | Mic recording begins | `question_index`, `structure_word` |
| `recording_stopped` | Mic recording stops | `duration_ms` |
| `transcription_completed` | ASR returns | `latency_ms`, `asr_mode`, `success` |
| `evaluation_completed` | LLM evaluation returns | `latency_ms`, `followup_created`, `success` |
| `session_completed` | Summary reached | `duration_ms`, `questions_answered` |
| `session_abandoned` | User exits early | `questions_answered`, `screen` |
| `app_error` | Recoverable app/backend error | `area`, `error_code` |

### Firebase Implementation Shape

Create a wrapper so analytics calls do not leak sensitive content:

```ts
// web/src/lib/analytics.ts
import { initializeApp } from "firebase/app";
import { getAnalytics, logEvent, isSupported } from "firebase/analytics";

const firebaseConfig = {
  apiKey: process.env.EXPO_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.EXPO_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.EXPO_PUBLIC_FIREBASE_PROJECT_ID,
  appId: process.env.EXPO_PUBLIC_FIREBASE_APP_ID,
  measurementId: process.env.EXPO_PUBLIC_FIREBASE_MEASUREMENT_ID,
};

let analyticsPromise: Promise<ReturnType<typeof getAnalytics> | null> | null = null;

function getAnalyticsClient() {
  if (!analyticsPromise) {
    analyticsPromise = isSupported().then((supported) => {
      if (!supported) return null;
      const app = initializeApp(firebaseConfig);
      return getAnalytics(app);
    });
  }
  return analyticsPromise;
}

export async function track(eventName: string, params: Record<string, any> = {}) {
  const analytics = await getAnalyticsClient();
  if (!analytics) return;

  const safeParams = Object.fromEntries(
    Object.entries(params).filter(([key]) => ![
      "email",
      "name",
      "transcription",
      "expected_answer",
      "audio_path",
      "llm_feedback",
    ].includes(key))
  );

  logEvent(analytics, eventName, safeParams);
}
```

### How to See Firebase Analytics

1. Open the Firebase console.
2. Select the Adio Firebase project.
3. Go to **Analytics > Dashboard** for aggregate active users, engagement, and event summaries.
4. Go to **Analytics > Events** to inspect event counts for `session_started`, `recording_started`, `session_completed`, etc.
5. Use **DebugView** during implementation to verify events in near real time.
6. For deeper analysis, enable Firebase export to BigQuery and query raw Analytics events there.

Suggested Firebase console checks:

- Does `session_started` fire once per session?
- Does `question_viewed` fire once per question card?
- Is `session_completed / session_started` reasonable?
- Are `app_error` spikes tied to ASR or LLM?
- Are test events excluded by using DebugView during development?

---

## 6. Supabase Dashboard Plan

Firebase will show behavioral analytics. Supabase should show therapy and model interaction analytics.

### Dashboard Options

| Option | Best For | Effort |
|---|---|---:|
| Supabase SQL Editor saved queries | Fast inspection | Low |
| Postgres views + custom React dashboard | Parent/research dashboard inside Adio | Medium |
| Metabase / Grafana / Evidence / Retool | Rich internal analytics | Medium |
| BigQuery + Looker Studio | Combined Firebase + Supabase warehouse | Higher |

Recommended first version:

1. Create Postgres views in Supabase.
2. Add an internal dashboard screen in the existing Expo app or a simple admin web page.
3. Later, connect Metabase or another BI tool if needed.

### Useful SQL Views

#### Session Overview

```sql
create or replace view public.v_session_overview as
select
  s.id,
  s.user_id,
  s.started_at,
  s.ended_at,
  s.completed,
  s.image_id,
  s.total_questions,
  s.questions_answered,
  s.observation_score,
  s.understanding_score,
  s.engagement_score,
  s.avg_latency_ms,
  extract(epoch from (s.ended_at - s.started_at)) * 1000 as session_duration_ms
from public.therapy_sessions s;
```

#### Turn Quality

```sql
create or replace view public.v_turn_quality as
select
  t.id,
  t.session_id,
  t.user_id,
  t.created_at,
  t.turn_index,
  t.structure_word,
  t.difficulty,
  t.asr_mode,
  t.asr_latency_ms,
  t.llm_latency_ms,
  t.initiation_latency_ms,
  nullif(t.scores->>'accuracy', '')::numeric as accuracy,
  nullif(t.scores->>'detail', '')::numeric as detail,
  nullif(t.scores->>'clarity', '')::numeric as clarity,
  nullif(t.scores->>'relevance', '')::numeric as relevance,
  (nullif(t.llm_followup->>'suggested_question', '') is not null) as followup_created
from public.therapy_turns t;
```

#### User Progress

```sql
create or replace view public.v_user_progress as
select
  user_id,
  date_trunc('week', started_at) as week,
  count(*) as sessions,
  avg(observation_score) as avg_observation,
  avg(understanding_score) as avg_understanding,
  avg(engagement_score) as avg_engagement,
  avg(avg_latency_ms) as avg_latency_ms
from public.therapy_sessions
where completed = true
group by user_id, date_trunc('week', started_at);
```

### Dashboard Cards to Build

Build these dashboard cards from Supabase views:

| Card | Query Source |
|---|---|
| Sessions completed this week | `v_session_overview` |
| Completion rate | `therapy_sessions` |
| Average questions answered | `v_session_overview` |
| Observation trend | `v_user_progress` |
| Understanding trend | `v_user_progress` |
| Engagement / latency trend | `v_user_progress` |
| ASR latency p50/p95 | `v_turn_quality` |
| LLM latency p50/p95 | `v_turn_quality` |
| Accuracy by structure word | `v_turn_quality` |
| Follow-up rate by difficulty | `v_turn_quality` |
| Error log | `analytics_errors` |

### Example Supabase Queries

Completion rate:

```sql
select
  count(*) filter (where completed) * 1.0 / nullif(count(*), 0) as completion_rate
from public.therapy_sessions
where started_at >= now() - interval '30 days';
```

Accuracy by structure word:

```sql
select
  structure_word,
  avg(accuracy) as avg_accuracy,
  count(*) as turns
from public.v_turn_quality
where created_at >= now() - interval '30 days'
group by structure_word
order by structure_word;
```

Latency percentiles:

```sql
select
  percentile_cont(0.5) within group (order by asr_latency_ms) as asr_p50,
  percentile_cont(0.95) within group (order by asr_latency_ms) as asr_p95,
  percentile_cont(0.5) within group (order by llm_latency_ms) as llm_p50,
  percentile_cont(0.95) within group (order by llm_latency_ms) as llm_p95
from public.v_turn_quality
where created_at >= now() - interval '7 days';
```

User progress over time:

```sql
select
  week,
  avg_observation,
  avg_understanding,
  avg_engagement,
  avg_latency_ms
from public.v_user_progress
where user_id = auth.uid()
order by week;
```

---

## 7. Backend Implementation Notes

### Authentication

The backend currently trusts `session_id` passed from the client. That is not enough for persisted sensitive data.

Add:

- `Authorization: Bearer <supabase_jwt>` from frontend to backend.
- Backend verification of Supabase JWT.
- Backend-derived `user_id`.

All inserts/updates should use that `user_id`.

### Turn Lifecycle

Recommended turn state progression:

```text
question_viewed
recording_started
recording_stopped
audio_uploaded
asr_completed
llm_evaluated
turn_completed
```

In Supabase, it is okay for a turn row to start partially filled and be updated as the backend completes ASR and LLM work.

### Retention

Pick one:

1. **Privacy-first**: keep audio only long enough to transcribe, then delete it.
2. **Research mode**: keep audio only if parent/guardian consent allows it.
3. **Hybrid**: keep transcript/evaluation by default, keep raw audio only for opted-in users.

For a child-facing therapy project, default to the privacy-first or hybrid option.

---

## 8. Source Notes

Relevant official docs:

- Firebase Analytics overview: <https://firebase.google.com/docs/analytics>
- Firebase event logging: <https://firebase.google.com/docs/analytics/events?platform=web>
- Firebase DebugView: <https://firebase.google.com/docs/analytics/debugview>
- Firebase BigQuery export: <https://firebase.google.com/docs/projects/bigquery-export>
- Google Analytics PII guidance: <https://support.google.com/analytics/answer/6366371>
- Supabase Storage: <https://supabase.com/docs/guides/storage>
- Supabase Storage access control: <https://supabase.com/docs/guides/storage/security/access-control>
- Supabase Storage schema: <https://supabase.com/docs/guides/storage/schema/design>
- FTC COPPA voice recording guidance: <https://www.ftc.gov/node/45451>
- FTC COPPA compliance plan: <https://www.ftc.gov/business-guidance/resources/childrens-online-privacy-protection-rule-six-step-compliance-plan-your-business>

This is an engineering plan, not legal advice. Because Adio may collect audio from children, review consent, retention, and deletion requirements before storing raw recordings in production.
