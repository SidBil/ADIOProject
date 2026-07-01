# Adio V&V Stage 2 — Feature 1: The Recall Session

**Parent doc:** `00_north_star.md` (inherits all principles)
**Covers:** The mechanics of a single Stage 2 session
**Last updated:** June 30, 2026

---

## 1. Purpose

Define exactly how a Stage 2 session differs from a Stage 1 session: the image is shown, then hidden, and the child answers the same 10 structure-word questions from memory instead of by looking at the picture.

## 2. Current state (Stage 1, for contrast)

Per investigation of the existing codebase:

- The session image is bound once at session start (`POST /api/session/start` returns `image_url`, `app.py`) and rendered unconditionally for the full session duration via a single `<Image>` element in `SessionScreen.tsx` — no hide/show state exists today.
- All 10 structure-word questions are built once per session by `session_manager.py::_build_questions` and answered against that one image.
- There is no timer or pacing mechanism; a child advances by manually recording an answer and tapping "Next."

Stage 2 needs new infrastructure: an image visibility state, a "ready" control, a peek allowance, and a session-level mode flag — the question/scoring pipeline itself is reused unchanged.

## 3. Session flow

1. Session starts exactly as today: one image is chosen, all 10 structure-word questions are built (`_build_questions`), image is shown at full visibility.
2. The child views the image for as long as they want — **no minimum viewing time is enforced.** The "ready" control is tappable immediately.
3. The child taps a **"ready" control** to hide the image. This is a deliberate action, not a timer — the child controls the pace.
4. Once hidden, the image is **replaced with a blurred version of itself** (not a blank placeholder, not a layout collapse) — a calm visual cue that something was there, consistent with Adio's sensory-calm design language, without giving away recallable detail.
5. The image stays hidden (blurred) for **all 10 questions** — this is a single hide event per session, not a per-question toggle. The child is being asked to hold and recall one mental image across the whole question set, not to re-image before each question.
6. The child answers all 10 structure-word questions exactly as in Stage 1: record → transcribe → evaluate → feedback → next.

## 4. The peek allowance

- Each Stage 2 session grants **one free peek** — the child can reveal the actual (unblurred) image one time during the session.
- After the peek is used, the image returns to blurred/hidden for the remainder of the session. There is no cost or penalty attached to using it — it's a scaffold, not a resource to manage strategically.
- **Peek usage is logged** (which turn it was used on, if any) for instrumentation/data purposes, but **does not affect the accuracy score** for that answer. This keeps peek data available for future analysis (e.g. "does peek usage correlate with true recall difficulty on certain images") without penalizing the child in the moment.

## 5. Scoring: gentler threshold, same pipeline

- Stage 2 reuses `LLMService` and the existing 0-5 `accuracy` scale exactly as Stage 1 does — no new evaluation model.
- The **follow-up trigger threshold is gentler** for Stage 2 than Stage 1's `ACCURACY_THRESHOLD = 4`, reflecting that recall is a harder condition than description and shouldn't read to the child as failing at the underlying skill.
  - Proposed default: lower the effective threshold by one point (e.g. `3` instead of `4`) as a flat adjustment to the existing rubric — not a rewritten LLM prompt. See open questions for the alternative (a recall-aware prompt) if a flat threshold proves insufficient.

## 6. Question count and shape

- Stage 2 asks the same **all 10 structure-words**, same ordering, same underlying question bank as Stage 1 — no reduction, no separate question set. This preserves the north star's "same contract, harder condition" principle (`00_north_star.md` §2).

## 7. Relationship to session progression

- A session's mode (Stage 1 vs Stage 2) is decided before the session starts, by the cadence defined in `02_session_progression.md` — the recall session itself has no say in when it's triggered; it only defines what happens once triggered.

## 8. Open questions / tunables

- **Exact gentler threshold value:** flat `3` vs. a recall-aware LLM prompt rewrite — flagged as undecided; start with the flat adjustment as the simpler MVP path.
- **Peek instrumentation detail:** exact event schema for logging peek usage (which turn, timestamp) — needs to slot into the existing `interaction_store.py` / `therapy_turns` schema.
- **Blur implementation:** exact blur radius/technique (CSS filter on web vs. native blur) and whether it's precomputed or applied client-side at hide time.
- **Parent/SLP visibility into peek usage:** whether peek usage per session surfaces anywhere in the parent dashboard or summary screen, or stays purely backend instrumentation for now.
