# Adio Adaptive Diagnostic — Axis 2: Question-Type Difficulty

**Parent doc:** `01_adaptive_diagnostic.md` (which inherits from `00_north_star.md`)
**Last updated:** June 29, 2026

---

## 1. Purpose

Question-type difficulty is the second of the two independent difficulty axes the diagnostic calibrates (see parent doc §3). Unlike image complexity, it is not a single scalar — it is a **profile of 10 values**, one tier per structure-word (`who, what, where, color, shape, sound, size, number, movement, mood`), because every regular session asks all 10 structure-words unconditionally and each one needs its own placement.

## 2. The core design decision: perpendicular to structure-words, not a ranking of them

Two approaches were considered:

### Rejected: rank the structure-words themselves
Treat some structure-words (e.g. "mood," "sound") as inherently harder than others (e.g. "color," "where"), and vary difficulty by which structure-words a child gets asked.

**Rejected because:** every regular session is contractually built around asking **all 10 structure-words**, every time, for the one image in that session (`session_manager.py::_build_questions` queues all 10 unconditionally). Making structure-word selection difficulty-dependent would require either dropping structure-words for some children (breaking the all-10 contract) or generating entirely new per-difficulty question sets (breaking the current single-question-per-structure-word-per-image authoring model in `image_metadata.csv`). Note the codebase already contains a structure-word-level ranking (`difficulty_map` in `session_manager.py`: `who/what/where/color/size/number=1, shape/sound=2, movement/mood=3`), but it is only used to **order** questions concrete → abstract within a session, never to select or omit them. This feature does not change that usage.

### Chosen: tier each structure-word's own question independently
Every structure-word gets **3 difficulty tiers of its own phrasing** — easy, medium, hard — and the diagnostic's job is to find the right tier *per structure-word*, not to decide which structure-words to ask. All 10 structure-words are still asked in every regular session; only the phrasing/scope of each one's question varies by tier.

**Example — "color," across tiers, for a given image:**

| Tier | Question |
|---|---|
| Easy | "What color is the cat?" — single concrete referent, one expected answer |
| Medium | "What colors do you see in the picture?" — open enumeration, still concrete |
| Hard | "What colors do you notice, and how do they make the scene feel?" — concrete + abstract/interpretive layer |

This keeps the 10-structure-word contract fully intact for every child, at every difficulty level, while still letting question difficulty vary per child and per structure-word.

## 3. Authoring & generation

- Tiered question variants are **batch-generated offline**, not generated live/on-the-fly during a session — keeps runtime simple and predictable, and allows review (e.g. by an SLP) before anything ships to children.
- A generation script will take the existing single `question_<structure_word>` per image (and its `structure_<structure_word>` expected-answer text) from `image_metadata.csv` and produce easy/medium/hard variants, written back into the CSV.
- **CSV schema expands 3x**: today there is one `question_<word>` column per structure-word per image (10 columns). This becomes 3 columns per structure-word (`question_<word>_easy`, `question_<word>_medium`, `question_<word>_hard`, or similar), i.e. 30 question columns per image instead of 10. At 20 images today, that's 600 question variants (vs. 200 today); the number scales linearly as the image bank grows.
- `SessionManager._load_metadata()` and `_build_questions()` will need to be updated to read the tiered columns and select the right tier per structure-word per child, per `00_adaptive_diagnostic.md` CLAUDE.md guidance that adding fields to `image_metadata.csv` requires updating `_load_metadata()`.

## 4. How the diagnostic uses this axis

- The interleaved staircase (parent doc §5) maintains a separate tier estimate per structure-word, each starting at medium.
- A given structure-word's tier estimate only gets signal when that specific structure-word is asked during the diagnostic — with ~10-12 total diagnostic questions across 10 structure-words, most structure-words will get only one, maybe two, data points each. This is a real constraint on how confidently each of the 10 tiers can be converged within the diagnostic's question budget (see open questions).
- Uses the same 0-5 `accuracy` score from `LLMService` as the image-complexity axis and as regular sessions today — no new scoring model.

## 5. How a regular session uses this axis

- For each of the 10 structure-words asked in a session, the session pulls the question text from the tier in the child's `question_tier_profile` for that structure-word (falling back to "medium" for any structure-word the diagnostic didn't get a confident read on).
- Post-diagnostic adaptation of individual structure-word tiers over time (e.g. a child consistently nailing "color" at hard but struggling with "mood" at easy) is not yet specified — see parent doc open questions.

## 6. Open questions / tunables

- **Per-structure-word convergence confidence:** with only ~1-2 diagnostic data points per structure-word, how much confidence can the diagnostic actually have per tier? May need a fallback/default (e.g. "medium" for any structure-word not directly tested) and let regular-session data refine it post-diagnostic.
- **Generation script design:** prompt/approach for deriving easy/medium/hard variants from the existing single question + expected-answer text per structure-word per image; review/approval workflow before new variants ship.
- **CSV column naming convention** for the tiered schema.
- **Validation:** how to confirm a generated "hard" variant for a given structure-word is actually harder than its "medium"/"easy" siblings (e.g. SLP review, or empirical accuracy data once enough sessions exist) — mirrors the same empirical-validation idea considered (and deferred) for ranking structure-words themselves.
- **Interaction with existing `difficulty_map` ordering:** whether the diagnostic's question-visit order should be seeded by the existing concrete → abstract structure-word ordering, purely adaptive, or some hybrid.
