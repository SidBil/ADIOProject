# Adio Session Summary — Feature Spec

**Parent doc:** `00_north_star.md` (inherits all principles)
**Covers:** The child celebration screen and the parent/SLP clinical detail view
**Last updated:** July 26, 2026

---

## 1. Purpose

Define exactly what happens right after a session ends, for both the child (celebration, in the moment) and the parent/SLP (clinical detail, on the Dashboard, whenever they look).

## 2. Current state (for contrast)

- `web/App.tsx`'s screen state machine already has a `summary` state, entered via `handleEnd` when a session finishes.
- `web/src/screens/SummaryScreen.tsx` (591 lines) currently renders three aggregate score cards (Understanding / Observation / Engagement) plus a full per-question history — all on one screen, shown to whoever is holding the device after the session, child or parent.
- `web/src/screens/DashboardScreen.tsx` currently lists past sessions (completion counts, date, image) with no drill-in to per-session detail and no scoring information.
- Per-turn data already includes everything needed for the clinical view: `services/session_manager.py` tags every question and answer with its `structure_word` (`_build_questions`, `qa_history`, and the Supabase turn fetch all carry `structure_word`), and `services/llm_service.py`'s evaluation output already includes `scores.accuracy`, `feedback`, `identified_elements`, and `missed_elements` per turn.
- This feature **replaces `SummaryScreen.tsx` and its `/api/session/{id}/summary` endpoint outright**, and **enriches `DashboardScreen.tsx`** into the parent/SLP entry point.

## 3. The child view: celebration

### 3.1 What it shows

- **An encouragement message** — short, warm text, consistent with Adio's calm, non-judgmental tone. There is a **single fixed base message** — no pool, no randomization, no rotation. The same base text appears every session.
- **A completion visual** — a simple, non-numeric marker that the session is done (e.g. a stamp, checkmark, or stars). Exact treatment is an open question (§5).
- **Nothing else.** No score, no per-question detail, no structure-word breakdown, no numbers of any kind. If a piece of content requires session accuracy data to render, it does not belong on this screen.

### 3.2 Trigger and placement

- Shown **only when a session is fully completed** — i.e. all 10 questions (and any triggered follow-ups) have resolved. It *is* the `summary` state in the existing screen flow, entered the moment the last question (or its follow-up) resolves.
- **Abandoned / early-quit sessions do not show the celebration.** If the child leaves before completing all 10 questions, the app exits directly to `landing` — no celebration screen, no partial version of it. The celebration is a completion moment, so an incomplete session simply does not reach it.
- For a completed session, the celebration is not optional and not skippable to a different destination.

### 3.2.1 Dismissal and navigation

- The celebration screen is **manually dismissed** by a single "done" control that returns the app to `landing`. It does **not** auto-advance on a timer.
- The celebration screen contains **no path into the parent/SLP clinical view**. A parent/SLP reaches the Dashboard the same way they do today (from `landing`); there is no direct child → clinical link on this screen. This keeps the child from ever landing on scored content by default.

### 3.3 Positive-only variation

- The base message/visual is always warm and always shown, regardless of performance — a struggling session gets the exact same floor as a strong one.
- A **strong session may additionally get a single bonus flourish** on top of the base (e.g. an extra visual detail, a slightly more enthusiastic message variant) — but this is purely additive. There is exactly **one** bonus flourish (not a pool), and it is either shown or not; there is no reduced or lesser version for a weaker session, and the worst case is identical to the median case.
- This mirrors the gamification layer's completion-gated principle in spirit (reward is never taken away or diminished by performance) without formally depending on that PRD.

### 3.4 Data used

- Only a completion signal (session finished, all/most questions answered) and, for the bonus flourish, an aggregate accuracy signal used purely to decide *whether* to add the bonus — never surfaced as a number or shown to the child in any form.

## 4. The parent/SLP view: clinical detail

### 4.1 Access

- Reached through the existing `DashboardScreen.tsx`, which is enriched (not replaced) to support two things: a per-structure-word trend sparkline, and drill-in from a list row into a single session's full detail. No new gating/authentication mechanism is introduced — access follows however the Dashboard is already reached today.

### 4.2 Dashboard list: trend view

- Each past session appears as a row (date/time, completion status, duration).
- Alongside the list, show a **per-structure-word sparkline** — one small line chart per structure-word (who/what/where/color/shape/sound/size/number/movement/mood), plotting accuracy across a **fixed recent window of the last 10 sessions**.
  - **Window:** the last 10 sessions of *any* kind (completed or abandoned), not just completed ones.
  - **Point value:** each session's point is the turn's `scores.accuracy` (0–5). For a structure-word that triggered a follow-up in that session, plot the **follow-up answer's** accuracy; if no follow-up occurred, plot the original answer's accuracy.
  - **Missing data:** a structure-word that was never answered in a given session (e.g. an abandoned session that never reached it) renders as a **gap / break in the line** for that session — it is *not* plotted as zero.
  - Fewer than 10 sessions in history simply produces a shorter line — no separate cold-start/placeholder state.
- Tapping a session row opens that session's full detail view (§4.3).

### 4.3 Session detail view: per-structure-word breakdown

For the selected session, show one entry per structure-word (all 10, always, matching the session's fixed question contract). Each entry includes:

- **Score** — the accuracy value from that turn's LLM evaluation (`scores.accuracy`, on its native 0–5 scale). **Only accuracy is surfaced** — the other components the evaluation also produces (`detail`, `clarity`, `relevance`, `overall_score`) are intentionally not shown in this view.
- **Transcript** — the raw ASR transcription for that turn, plus the expected/reference answer for comparison.
- **LLM reasoning** — the evaluation's `feedback` text and `identified_elements` / `missed_elements`, so a parent/SLP can see *why* the score landed where it did, not just the number.
- **Follow-up, if triggered** — shown as a **nested sub-entry under the same structure-word**, with its own transcript, score, and reasoning. This preserves the full exchange (original answer → LLM feedback → follow-up question → follow-up answer) without flattening it into a second top-level row.

### 4.4 Session-level metadata

Shown alongside the per-structure-word breakdown, at the top of the detail view:

- **Timing** — session date/time, total duration, and per-question response latency. "Response latency" here means the **total time to answer** each question (question shown → answer submitted). This value is *not* captured today — the existing per-turn fields are `initiation_latency_ms` (question shown → recording started), `asr_latency_ms`, and `llm_latency_ms` (system processing) — so this feature must **add new instrumentation** to record answer duration per turn. See §5.
- **Session context** — which image was shown for the session.
- **Completion status** — whether all 10 questions were answered or the session was abandoned partway, and how many follow-ups were triggered in total. (Note: an abandoned session is still recorded and appears here in the clinical view even though the child never saw a celebration for it — see §3.2.)

## 5. Open questions / tunables

Still open:

- **Completion visual treatment:** stamp vs. checkmark vs. stars vs. other — undecided, needs a design pass.
- **Bonus flourish criteria:** exact accuracy threshold (or other signal) that triggers the single upward-only flourish described in §3.3, and what the one flourish actually looks like. (There is exactly one flourish; only its trigger and appearance are open.)
- **Sparkline interaction:** whether tapping a point on a structure-word's sparkline jumps directly to that session's detail view, or the sparkline is view-only.
- **Detail view layout:** whether all 10 structure-word entries are visible at once (scrollable list) or collapsed/expandable by default given the amount of content (score + transcript + reasoning + possible nested follow-up) per entry.

### 5.1 Resolved (previously ambiguous)

- **Abandoned sessions & the child screen:** the celebration is shown *only* on full completion of all 10 questions; an early quit exits directly to `landing` with no celebration. (§3.2)
- **Post-celebration navigation:** manual "done" dismiss back to `landing`, no auto-advance, and no child → clinical path on the celebration screen. (§3.2.1)
- **Child message selection:** a single fixed base message (no pool/randomization) plus exactly one additive bonus flourish. (§3.1, §3.3)
- **Sparkline window / point / gaps:** last 10 sessions of any kind; plot the follow-up accuracy when a follow-up occurred else the original; unanswered structure-words are gaps in the line, not zeros. (§4.2)
- **Detail-view score fields:** accuracy only; `detail`/`clarity`/`relevance`/`overall_score` are not surfaced. (§4.3)

### 5.2 Implementation dependency

- **Answer-duration instrumentation:** the per-question "response latency" in §4.4 (question shown → answer submitted) is not captured today and requires a new per-turn field plus wiring in the session/turn flow (`session_manager.py`, `interaction_store.py`, and the turn-timing capture on the frontend). This must ship with the feature for §4.4 to be complete.
