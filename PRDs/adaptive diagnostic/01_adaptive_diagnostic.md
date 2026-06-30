# Adio Adaptive Diagnostic — Feature Spec

**Status:** Draft for review
**Owner:** Sidharth
**Last updated:** June 29, 2026

> **Parent doc:** `00_north_star.md` (inherits all principles). Two companion docs go deep on each difficulty axis: `image_complexity.md` and `question_difficulty.md`. If a detail conflicts between this doc and a companion doc, the companion doc wins for its own axis. If a detail conflicts with the north star, the north star wins.

---

## 1. What we're building

A short, adaptive placement test that runs **once, at signup**, before a child's first real V&V session. Its sole job is to estimate where a new child should start on two independent difficulty axes, so their very first real session is neither too easy (boring, no signal) nor too hard (frustrating, no signal).

This is the **first time "difficulty" becomes a formal concept** in Adio. Prior to this feature, difficulty exists only informally: a per-image `complexity` int in `image_metadata.csv` that nothing currently consumes, and a `difficulty_map` in `session_manager.py::_build_questions` that orders the 10 structure-words concrete → abstract but is not otherwise used for selection (every session asks all 10 structure-words for one image, unconditionally).

## 2. Why this is needed

Today, every child gets the exact same experience shape on session 1: the same structure-word ordering, against whatever image the session-start logic hands them, with no calibration to the individual child's current ability. There's no signal-driven way to start a struggling child gently or a strong child meaningfully. The diagnostic exists to produce that starting signal, once, cheaply, before therapy begins.

## 3. The two-axis difficulty model

Difficulty in Adio is **not a single number**. It is two independent axes that get consumed differently downstream:

| Axis | What it controls | Cardinality | Detail doc |
|---|---|---|---|
| **Image complexity** | Which image (by complexity tier) a session is built around | 1 scalar, 3 tiers | `image_complexity.md` |
| **Question-type difficulty** | The phrasing/scope tier used for each of the 10 structure-word questions | A 10-slot profile, 3 tiers per slot | `question_difficulty.md` |

### Why two axes, not one composite score

Early in scoping this, we considered collapsing both axes into a single composite difficulty score. We rejected that:

- **Image complexity** picks *which image* — a single scalar makes sense because a session is built around exactly one image.
- **Question-type difficulty** doesn't pick anything — every regular session already asks **all 10 structure-words**, unconditionally, every time. There's nothing to select. What varies is the *phrasing tier* used for each structure-word's question independently. A child might be ready for hard "color" questions but still need easy "mood" questions. Collapsing that into one number would destroy exactly the information a regular session needs to assemble itself, and you'd have to decompose the composite back into per-structure-word tiers anyway.

So the diagnostic's output is two side-by-side artifacts, not one number:

```
{
  "image_complexity": 2,                 // single value, 1-3
  "question_tier_profile": {             // one tier per structure-word
    "who": "easy", "what": "medium", "where": "easy",
    "color": "hard", "shape": "medium", "sound": "easy",
    "size": "medium", "number": "easy", "movement": "hard", "mood": "hard"
  }
}
```

### Relationship to the existing `difficulty_map`

`session_manager.py::_build_questions` already contains a structure-word-level difficulty ranking (`who/what/where/color/size/number = 1`, `shape/sound = 2`, `movement/mood = 3`) used only to order questions concrete → abstract within a session. We explicitly rejected extending *that* ranking into a difficulty axis (i.e. "easy-mode children get fewer/easier structure-words") because every session is contractually required to ask all 10 structure-words — reworking that would mean either dropping structure-words for some children or generating entirely new per-child question sets, both of which break the current session model and the `image_metadata.csv` authoring pipeline.

The diagnostic's question-difficulty axis is **perpendicular** to that existing ranking: it doesn't change *which* structure-words get asked (always all 10), it changes the *difficulty tier of the phrasing used for each one*. The existing `difficulty_map` ordering (concrete → abstract sequencing within a session) is unaffected and can continue to be used as-is, or could later inform the *order* in which the diagnostic visits structure-words (not yet decided — see open questions).

## 4. Trigger

- Runs exactly once, automatically, immediately after signup, before the child's first real V&V session.
- Framed to the child as "getting to know you," not as a test — consistent with the gamification north star's calm/non-anxiety design principles (see `PRDs/gamification/00_north_star.md` §3).

## 5. Adaptive mechanism: interleaved staircase

The diagnostic is a **staircase**, not a fixed battery and not full IRT/CAT:

- Starts at a medium difficulty on both axes.
- Each answer steps the relevant axis/axes up or down based on the existing 0-5 `accuracy` score already produced by `LLMService` for every answer (the same scoring used in regular sessions today — no new scoring model needed).
- **Interleaved across both axes**: each step, the diagnostic picks the next image+structure-word+tier combination that best narrows whichever axis (image complexity, or a specific structure-word's tier) currently has the least certainty — rather than calibrating one axis fully before starting the other.

This was chosen over:
- **Sequential** (calibrate image complexity fully, then question-type difficulty fully) — rejected because it front-loads one axis and risks running out of attention budget before the second axis gets enough signal.
- **Single combined ladder** — rejected for the same reason the composite-score idea was rejected (§3): the axes are consumed differently downstream, so calibrating them as one ladder loses information a session needs.
- **Full IRT/CAT** — rejected as more accurate than needed and meaningfully more complex to build/tune for an MVP, given the population's attention constraints already cap session length low.

## 6. Length & structure

- **~10-12 questions total**, spanning **multiple images** (not the single-image-per-session shape of a regular session) — needed because image complexity can only be calibrated by varying images.
- This is longer than a regular session's question count specifically because two axes need independent signal; it is still kept short relative to a full IRT battery given the population's attention constraints.
- **No follow-up questions during the diagnostic.** Regular sessions trigger an LLM-generated follow-up when accuracy < 4 (`LLMService`, `ACCURACY_THRESHOLD`). The diagnostic explicitly skips this: follow-ups don't produce clean calibration data points, and they make the question budget unpredictable. Every diagnostic question is single-shot.

## 7. Scoring signal

- Reuses the existing 0-5 `accuracy` score from `LLMService` per answer — no new scoring model needed for the diagnostic itself.
- A step-up/step-down threshold (e.g. accuracy ≥ 4 steps up, < 4 steps down) mirrors the existing `ACCURACY_THRESHOLD = 4` already used to gate follow-ups in regular sessions, for consistency — exact thresholds are tunable (see open questions).

## 8. How a regular session consumes diagnostic output

Once the diagnostic produces `image_complexity` and `question_tier_profile`:

- Session image selection picks from the image bank at (or near) the converged `image_complexity` tier.
- Each of the 10 structure-word questions in the session is rendered at its tier from `question_tier_profile`, using the offline-generated tiered question variants (see `question_difficulty.md`).
- Both values are expected to **continue adapting** after the diagnostic, based on ongoing regular-session accuracy — the diagnostic sets the *starting point*, not a permanent fixed value. (Exact post-diagnostic adaptation mechanism is out of scope for this doc — see open questions.)

## 9. Non-goals

- Not a replacement for, or a redesign of, the regular V&V session loop.
- Not an accuracy-gated or pass/fail test — there is no "failing" the diagnostic; it only produces a starting placement.
- Not a one-number difficulty score (see §3).
- Not full IRT/CAT.
- Does not currently specify interaction with the gamification layer (streak/collection/milestones) — open question below.

## 10. Open questions

- **Gamification interaction:** does completing the diagnostic count as session 1 for collection/streak/milestone purposes, or is it explicitly outside that system? Unresolved — flagged but deliberately deferred during scoping.
- **Post-diagnostic adaptation:** exact mechanism by which `image_complexity` and `question_tier_profile` continue to evolve after the diagnostic, during ongoing regular sessions.
- **Step thresholds:** exact accuracy thresholds and step sizes for the staircase (placeholder: mirror `ACCURACY_THRESHOLD = 4`).
- **Convergence criteria:** fixed-length (~10-12 questions, always) vs. early-stop on high confidence.
- **Structure-word visit order within the diagnostic:** whether to seed the interleaving with the existing concrete → abstract `difficulty_map` ordering, or treat order as fully adaptive.
- **Cold-start image bank:** the diagnostic needs images across all 3 complexity tiers to calibrate the complexity axis, but the bank currently only has tiers 1-2 populated (see `image_complexity.md` §3) — diagnostic behavior while tier 3 is empty needs a fallback.
