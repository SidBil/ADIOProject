# Adio V&V Stage 2 — North Star

**Status:** Draft for review
**Owner:** Sidharth
**Last updated:** June 30, 2026

> This is the guiding document for Adio's second V&V stage: recall-based sessions. The two feature docs (recall session, session progression) inherit every principle here. If a feature decision conflicts with this document, this document wins.

---

## 1. What we're building

Adio is a Visualization & Verbalization (V&V) app for children with communication differences. Today, every session is what V&V calls **Stage 1**: an image is shown and stays visible the whole time, while the child answers 10 structure-word questions (who/what/where/color/shape/sound/size/number/movement/mood) about it.

**Stage 2** is the next V&V stage: the image is shown, then removed, and the child answers the same structure-word questions **from memory** rather than by looking at the picture. This is a genuine escalation in difficulty — the child must build and hold a mental image, not just describe what's in front of them — and it is the first time Adio asks a child to work from a visualized image rather than a physical one.

This effort has two parts, covered by the two feature docs:

| Doc | Covers |
|---|---|
| `01_recall_session.md` | The mechanics of a single Stage 2 (recall) session: hiding the image, the peek allowance, scoring |
| `02_session_progression.md` | The "stone path" — how children move between Stage 1 and Stage 2 sessions over time |

## 2. The central commitment: same contract, harder condition

Stage 2 does not change *what* is asked. It changes *how much support the child has while answering*.

- Every Stage 2 session still asks all 10 structure-words, for one image, exactly as a Stage 1 session does today (per `CLAUDE.md` and the existing `session_manager.py::_build_questions` contract).
- The evaluation pipeline (`LLMService`, the 0-5 `accuracy` scale) is reused, not replaced.
- What changes is a single condition: the image is present for Stage 1, removed (with a limited peek) for Stage 2.

Keeping the question/scoring contract identical means Stage 2 is a **condition change**, not a new feature surface — it slots into the existing session model rather than requiring a parallel one.

## 3. Design principles

1. **Reuse, don't rebuild.** Stage 2 reuses the existing question bank, the existing 10-structure-word contract, and the existing LLM evaluation pipeline. Only image visibility and its knock-on effects (peek allowance, gentler threshold) are new.
2. **Child controls pace.** No forced minimum viewing time before the child can hide the image and move to questions — trust the child (and, implicitly, the parent/SLP watching) to view long enough.
3. **A safety net, not a cliff.** Removing the image is a real difficulty jump. A single free peek per session exists so a child who's stuck isn't simply failed by the mechanic — but it's capped, not unlimited, so recall is still the point.
4. **Gentler grading for a harder task.** Recall is inherently harder than description. Stage 2 sessions use a gentler accuracy threshold for triggering follow-ups than Stage 1, so the added difficulty of the condition doesn't read to the child as being worse at the underlying skill.
5. **Predictable rhythm, not child choice.** Which sessions are Stage 2 is decided by a fixed, visible cadence (see `02_session_progression.md`), not left for the child to pick each time or hidden behind a parent setting at launch. Every child gets it, on by default.
6. **Calm visual language, consistent with the rest of Adio.** Hiding the image uses a blur of the original (not a jarring blank/collapse), keeping continuity with Adio's sensory-calm design language.

## 4. Goals & non-goals

**Goals**
- Introduce a second, harder V&V condition (recall) without disrupting the existing Stage 1 session shape or the underlying question/scoring pipeline.
- Give every child periodic exposure to Stage 2 via a predictable, visible progression rhythm.
- Keep Stage 2 forgiving enough (one peek, gentler threshold) that it doesn't become a drop-off point.

**Non-goals**
- No change to the 10-structure-word-per-session contract.
- No new LLM scoring model — Stage 2 reuses `LLMService`'s existing 0-5 `accuracy` scale, only the follow-up trigger threshold is adjusted for the recall condition.
- No parent/SLP opt-in gate at launch — on by default for everyone (may become configurable later; see open questions).
- No tie-in to the adaptive diagnostic's difficulty axes (`PRDs/adaptive diagnostic/`) — kept as a separate, orthogonal system.
- No merging into the gamification surfaces (`PRDs/gamification/`) — the session path is its own screen, not a 4th gamification mechanic.

## 5. How this relates to other in-flight PRDs

- **Adaptive diagnostic** (`PRDs/adaptive diagnostic/`): explicitly out of scope for interaction. Stage 2 readiness is not calibrated by the diagnostic's image-complexity or question-difficulty axes at launch.
- **Gamification** (`PRDs/gamification/`): explicitly a separate concept/screen. The session path (`02_session_progression.md`) is not the same surface as the collection wall, streak flame, or milestone bar, and does not currently feed into or draw from them.

## 6. Open questions (cross-cutting)

- Whether Stage 2 should eventually become parent/SLP-configurable (per-child opt-in, adjustable cadence) rather than a fixed default for everyone.
- Whether Stage 2 and the adaptive diagnostic's difficulty axes should eventually be connected (e.g. does diagnostic-measured recall ability affect Stage 2 cadence).
- Whether Stage 2 and gamification should eventually share a surface, now that both are between/around-session concepts.

Feature-specific open questions live in each feature doc.

## 7. Success criteria

- Children reach and complete Stage 2 sessions without a measurable spike in drop-off compared to Stage 1 sessions.
- Peek usage data shows the single-peek allowance is being used as a scaffold (occasional use), not as a routine crutch (near-100% usage every session).
- Stage 2 accuracy, even under the gentler threshold, provides a meaningful, distinct signal from Stage 1 accuracy — confirming the condition is actually harder and actually measuring something new.
