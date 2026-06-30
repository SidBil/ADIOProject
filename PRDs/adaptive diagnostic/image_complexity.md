# Adio Adaptive Diagnostic — Axis 1: Image Complexity

**Parent doc:** `01_adaptive_diagnostic.md` (which inherits from `00_north_star.md`)
**Last updated:** June 29, 2026

---

## 1. Purpose

Image complexity is one of the two independent difficulty axes the diagnostic calibrates (see parent doc §3). It controls **which image** (by complexity tier) a session is built around. It is a single scalar per child, not a profile — a session is built around exactly one image, so there's nothing to decompose.

## 2. Definition

Image complexity describes how visually/conceptually busy a scene is — number of entities, number of actions, how much is happening, how cluttered the composition is — independent of which structure-word questions get asked about it.

This is **not** the same axis as question-type difficulty (`question_difficulty.md`). A simple image can still be asked a hard "mood" question about it, and a complex image can still be asked an easy "color" question. The two axes are orthogonal by design.

## 3. Current state of the image bank

- The bank currently has **20 images** (`image_metadata.csv`), not the 100 originally assumed during scoping — corrected during PRD discussion.
- A `complexity` field already exists in the CSV and is populated, but only across **2 of the planned 3 tiers**: 11 images at complexity `1`, 9 images at complexity `2`, **0 images at complexity `3`**.
- The bank is expected to **grow** to fill out complexity `3` and to generally expand. Until it does, the diagnostic (and any session logic that depends on complexity `3`) has a cold-start gap — see Open Questions and the parent doc's open question on this.

## 4. Tiers

Three tiers, matching the question-difficulty axis's tier count for consistency:

| Tier | Label | Notes |
|---|---|---|
| 1 | Easy/simple | Few entities, single clear action, uncluttered composition |
| 2 | Medium | Moderate entity/action count |
| 3 | Hard/complex | Many entities/actions, busier composition — **not yet populated in the bank** |

Exact rubric for what separates tier 1 from tier 2 from tier 3 (entity count thresholds, action count thresholds, etc.) is not yet formally defined — currently `complexity` appears to be assigned by judgment at image-authoring time, not by a measured rubric. Formalizing this rubric is recommended before generating new tier-3 images, so new images are scored consistently with the existing 20.

## 5. How the diagnostic uses this axis

- The interleaved staircase (parent doc §5) steps this single value up/down based on accuracy on whichever image the child was just shown, using the same 0-5 `accuracy` score as the question-difficulty axis.
- Because this axis only has signal from *which image was shown*, not from *which structure-word was asked*, every diagnostic question contributes a (possibly weak) data point to this axis regardless of which structure-word it targets — the interleaving logic should account for this asymmetry (image complexity gets a signal on every question; a given structure-word's tier only gets a signal when that structure-word is asked).

## 6. How a regular session uses this axis

- Once converged, a regular session's image-selection step should prefer images at (or near) the child's converged `image_complexity` tier from the image bank.
- Post-diagnostic adaptation of this value over time (e.g. does a string of high-accuracy sessions nudge it up) is not yet specified — see parent doc open questions.

## 7. Open questions / tunables

- **Formal complexity rubric:** entity count / action count / composition thresholds that define tier boundaries, so new images can be scored consistently and the existing 20 can be audited.
- **Tier-3 backfill plan:** how many new tier-3 images are needed before the diagnostic can reliably calibrate the top of this axis, and what the diagnostic does in the meantime (e.g. cap effective range at tier 2 until tier 3 exists).
- **Who scores complexity for new images:** manual (SLP/author judgment) vs. a scripted heuristic (e.g. derived from the `entities`/`actions` CSV columns) vs. hybrid.
- **Relationship to image authoring pipeline:** `imagegen/generate_images.py` and `generate_metadata.py` currently produce images/metadata without a target complexity tier in mind — does the pipeline need to be tier-aware so new images are generated *to* a target complexity rather than scored after the fact?
