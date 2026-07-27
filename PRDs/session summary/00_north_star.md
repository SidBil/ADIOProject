# Adio Session Summary — North Star

**Status:** Draft for review
**Owner:** Sidharth
**Last updated:** July 12, 2026

> This is the guiding document for Adio's session summary feature. The feature doc (`01_session_summary.md`) inherits every principle here. If a feature decision conflicts with this document, this document wins.

---

## 1. What we're building

Adio is a Visualization & Verbalization (V&V) app for children with communication differences: it shows an image and asks the child 10 structure-word questions about it, transcribes and scores each answer, and occasionally follows up on a weak answer.

The session summary is what happens **the moment a session ends**. It is not one screen — it is two different views of the same underlying session data, built for two different audiences who need fundamentally different things from it:

| View | Audience | Shows | Detail doc |
|---|---|---|---|
| Celebration | The child | Nothing but completion | `01_session_summary.md` §2 |
| Clinical detail | Parent / SLP | Everything, per structure-word | `01_session_summary.md` §3 |

This spec **replaces the existing `SummaryScreen.tsx` and its `/api/session/{id}/summary` endpoint outright** — that implementation (three aggregate score cards: Understanding/Observation/Engagement, shown to whoever is holding the device) is being torn out, not extended. It also enriches the existing `DashboardScreen.tsx` (currently a thin, undifferentiated list of past sessions) into the parent/SLP entry point for clinical detail.

## 2. The central commitment: two audiences, two views, zero overlap

A single shared "summary" screen — the old model — has to serve both a child who should never feel graded and a parent/SLP who needs a real clinical picture. Those two needs are in direct tension: enough score detail to be clinically useful is exactly the kind of detail that can make a child feel evaluated.

Adio resolves this by **splitting into two screens with no shared content**, not by finding a compromise:

- The **child never sees a score, a structure-word breakdown, or anything else that could read as a grade.** The child's screen is a celebration of completion, full stop.
- The **parent/SLP sees everything** — full per-structure-word scores, transcripts, and LLM reasoning — on a separate surface the child doesn't land on by default.

Neither view is a watered-down version of the other. They are built for different jobs.

## 3. Design principles

1. **The child's screen never grades.** No numbers, no per-question detail, no structure-word breakdown — ever. (See `01_session_summary.md` §2.)
2. **Positive variation only, never negative.** The child's completion message/visual can occasionally get a little extra flourish for a strong session, but a weaker session never gets a lesser version. The floor is always warm; only the ceiling moves.
3. **The parent/SLP view holds nothing back.** Full per-structure-word score, transcript, and LLM reasoning (feedback + identified/missed elements) for every question, including nested follow-up exchanges. This is a clinical tool, not a simplified digest.
4. **One data source, two renderings.** Both views read from the same session/turn data (`therapy_sessions`, `therapy_turns`, each turn's `structure_word` field) — nothing new is computed for the child view; it simply omits everything except the completion signal.
5. **Reuse existing scoring, don't invent new metrics.** Structure-word-level accuracy is already produced per turn by the existing `LLMService` evaluation (`services/llm_service.py`) — this feature is about surfacing and organizing that data, not scoring differently.
6. **Self-contained.** This feature does not depend on, reference, or resolve open questions from the other in-flight PRDs (`PRDs/V&V stage 2/`, `PRDs/adaptive diagnostic/`, `PRDs/gamification/`). It is scoped purely to what happens with the data a session already produces today.

## 4. Goals & non-goals

**Goals**
- Give the child a consistent, warm, completion-only moment at the end of every session — never a grade.
- Give the parent/SLP a genuinely useful, per-structure-word clinical breakdown of any past session.
- Let the parent/SLP see trends across recent sessions per structure-word, not just isolated snapshots.
- Fully replace the old combined summary screen, which served neither audience well.

**Non-goals**
- No change to the V&V session loop itself (question set, scoring pipeline, follow-up logic stay as-is).
- No score, number, or per-question detail ever shown to the child.
- No new LLM evaluation model — reuses the existing accuracy/feedback output already produced per turn.
- No dependency on or integration with Stage 2 recall, the adaptive diagnostic, or the gamification layer.
- No new parent-mode gating/authentication mechanism — accessed through the existing Dashboard surface as-is.

## 5. Open questions (cross-cutting)

- Exact visual treatment of the child's completion moment (stamp / checkmark / stars / other) — undecided, see feature doc.
- Exact criteria for when the "bonus flourish" upward variation triggers.

Feature-specific open questions live in the feature doc.

## 6. Success criteria

- Every session ends with the child seeing the same celebration screen, regardless of how they performed — verified by the fact that no accuracy data ever reaches the child-facing component.
- A parent/SLP can, for any past session, reconstruct exactly what was asked, what was said, how it was scored, and why — without needing to look anywhere else.
- The old `SummaryScreen.tsx` and its endpoint are fully removed, with no remaining references.
