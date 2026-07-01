# Adio V&V Stage 2 — Feature 2: Session Progression (The Stone Path)

**Parent doc:** `00_north_star.md` (inherits all principles)
**Covers:** How children move between Stage 1 and Stage 2 sessions over time
**Last updated:** June 30, 2026

---

## 1. Purpose

Define the mechanism by which sessions alternate between Stage 1 (image visible) and Stage 2 (recall — see `01_recall_session.md`), and the screen a child uses to start each session.

## 2. The core concept: a path of stones

Session selection moves from a generic "Start" button to a **linear path of stones**, where each stone represents one session. Tapping the next stone in the path starts that session.

- The path is a **straight linear sequence** — one path, one direction, no branches or loops. Simplest to build and reason about, like a road.
- **Completed stones stay visible** on the path — it functions as a persistent trail/history, not just a forward-looking queue. A child (or parent) can look back and see session history alongside what's coming.
- The path is **visible in advance**: the child can see upcoming stones, including which ones are Stage 2 ("special") sessions, before reaching them. This is a deliberate departure from the surprise-based reveals in the gamification layer (`PRDs/gamification/`) — the cadence here is meant to be predictable, not a variable reward.

## 3. Replaces the current start flow

- The path **replaces** the existing Landing/Welcome → generic "Start" entry point. Every session, of either stage, begins by tapping the appropriate stone on the path — there is no separate generic start button once this ships.

## 4. Cadence: fixed pattern

- Stage 2 ("special") sessions occur on a **fixed pattern**: two Stage 1 sessions, then one Stage 2 session, repeating (i.e. every 3rd session is Stage 2).
- This cadence is **not configurable** at launch — same fixed pattern for every child, no SLP/parent tuning knob (consistent with Stage 2 being on-by-default for everyone, per `00_north_star.md` §4).
- Example sequence: **Stage 1, Stage 1, Stage 2, Stage 1, Stage 1, Stage 2...**

## 5. Stone visual distinction

- Stage 1 and Stage 2 stones must be **visually distinguishable** on the path (e.g. a distinct color, icon, or shape for the "special"/recall stone) so the advance-visibility principle (§2) actually functions — a child glancing at the path should be able to tell a recall session is coming without opening it.
- Exact visual treatment is not yet specified — see open questions.

## 6. Relationship to other systems

- **Not merged with gamification.** The path is a separate concept/screen from the collection wall, streak flame, and milestone bar (`PRDs/gamification/`) — session sequencing and reward mechanics remain distinct systems, per the north star's non-goals (`00_north_star.md` §4).
- **Not tied to the adaptive diagnostic.** Which sessions are Stage 2 is decided purely by the fixed 2:1 cadence, not by diagnostic-measured readiness or the difficulty axes in `PRDs/adaptive diagnostic/`.
- **Feeds `01_recall_session.md`.** The path's only responsibility is deciding *when* a Stage 2 session happens; what happens *inside* that session is fully specified in the recall session doc.

## 7. Data model implication

- Today, `session_manager.py::create_session` picks a random image and builds a Stage-1-shaped session with no mode concept. This needs a session-level mode flag (e.g. `mode: "stage1" | "stage2"`) determined by the child's position in their session sequence (count of sessions completed so far, mod 3) before `create_session` runs, so the correct mode is threaded through session start, the frontend's `SessionScreen`, and stored alongside the session record (`therapy_sessions` in Supabase) for history/path rendering.

## 8. Open questions / tunables

- **Exact stone visual treatment:** color/icon/shape distinction between Stage 1 and Stage 2 stones; whether rarity-style visual language from the gamification collection (`PRDs/gamification/01_collection.md` §5) is reused or deliberately kept separate.
- **Path scale/pagination:** whether the path scrolls infinitely as history grows, or windows to a recent stretch + upcoming — undecided, default to simple infinite scroll for MVP.
- **Where the path lives in the navigation/screen state machine** (`App.tsx`'s `landing → welcome → session → summary` flow) — likely replaces `landing`/`welcome`, exact integration not yet mapped.
- **First-run behavior:** what a brand-new child's path looks like before any sessions are completed (e.g. does it show the first 5-10 stones with the cadence pre-rendered, or reveal one stone at a time).
- **Interaction with session-end/summary flow:** whether completing a session auto-returns the child to the path (showing the newly-completed stone) or to the existing summary screen first, then the path.
