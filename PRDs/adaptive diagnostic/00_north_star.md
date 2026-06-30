# Adio Adaptive Diagnostic — North Star

**Status:** Draft for review
**Owner:** Sidharth
**Last updated:** June 29, 2026

> This is the guiding document for Adio's adaptive diagnostic feature. The feature spec and both axis docs (`01_adaptive_diagnostic.md`, `image_complexity.md`, `question_difficulty.md`) inherit every principle here. If a feature decision conflicts with this document, this document wins.

---

## 1. What we're building

Adio is a Visualization & Verbalization (V&V) app for children with communication differences: it shows an image and asks the child questions about it to build comprehension and expressive language.

The adaptive diagnostic is a short, one-time placement step that runs automatically at signup, before a child's first real V&V session. Its sole job is to estimate where that child should start on two difficulty axes — image complexity and question-type difficulty — so the first real session is calibrated to the child rather than identical for everyone.

This is the **first time "difficulty" becomes a formal, consumed concept** in Adio. Until now it has existed only as inert metadata: a per-image `complexity` field nothing reads, and a structure-word ordering nothing varies by child.

## 2. The central commitment: placement, not evaluation

The diagnostic exists to find a **starting point**, not to judge, score, or label the child.

This distinction matters as much here as completion-gating mattered for the gamification layer (`PRDs/gamification/00_north_star.md` §2). A diagnostic that *feels* like a test the child can fail — or that produces an output the child or parent experiences as a verdict on the child's ability — undermines the trust the rest of Adio depends on. The diagnostic adapts its questions based on performance, but that adaptation is invisible machinery in service of placement, never a visible judgment delivered to the child. There is no passing or failing the diagnostic; there is only "where do we start."

## 3. Design principles

1. **One-time, automatic, at signup.** The diagnostic runs once, before the first real session, with no parent/child action required to trigger it.
2. **Placement, not evaluation.** (See §2.) Its output sets a starting point; it does not produce a clinical judgment, a label, or a score shown as a verdict.
3. **Calm framing.** Presented to the child as "getting to know you," not as a test — consistent with the rest of Adio's sensory-calm, non-anxiety design language.
4. **A starting point, not a permanent fixed value.** Both difficulty axes continue to adapt from regular-session performance after the diagnostic; the diagnostic seeds the system, it doesn't lock it.
5. **Two independent axes, never collapsed into one score.** Image complexity and question-type difficulty are consumed differently by a session, so they are calibrated and reported separately (see `01_adaptive_diagnostic.md` §3).
6. **Reuse, don't reinvent, scoring.** The diagnostic uses the same 0-5 `accuracy` score that regular sessions already produce — no new evaluation model or rubric for the diagnostic specifically.
7. **Bounded and brief.** Given the population's attention constraints, the diagnostic runs to a short, predictable question budget (no open-ended follow-ups, no unbounded adaptive testing).
8. **Never breaks the regular session contract.** Whatever the diagnostic outputs, every regular session still asks all 10 structure-words, for one image, exactly as it does today — the diagnostic changes *which tier* of image and question phrasing get used, never the shape of the session itself.

## 4. Goals & non-goals

**Goals**
- Estimate a sensible starting image-complexity level and a starting per-structure-word question-difficulty profile for every new child.
- Do so quickly enough to not fatigue the child before their first real session even begins.
- Stay consistent with Adio's calm, non-judgmental tone — no part of the diagnostic should read as a test the child can fail.
- Produce output that a regular session can consume directly, with no further interpretation step.

**Non-goals**
- No changes to the regular V&V session loop's shape (still 1 image, all 10 structure-words, per session).
- No clinical diagnosis, labeling, or scoring presented as a verdict to the child or parent.
- No pass/fail outcome.
- No single collapsed difficulty score (see principle 5).
- No new LLM scoring model — diagnostic accuracy reuses the existing 0-5 scale.
- No follow-up questions during the diagnostic (kept single-shot and predictable).

## 5. How this relates to difficulty as a concept going forward

The two axes this feature introduces — image complexity and question-type difficulty — aren't diagnostic-only constructs. They become the durable difficulty model the rest of the app (regular sessions, and potentially future features) reasons about going forward. The diagnostic is simply the first and fastest way to seed values for a brand-new child; the axes themselves outlive the diagnostic and continue to be shaped by ordinary session performance afterward.

## 6. Open questions (cross-cutting)

- **Interaction with the gamification layer:** does completing the diagnostic count toward the collection/streak/milestone system (`PRDs/gamification/`), or is it explicitly outside it? Deliberately unresolved during initial scoping — needs a decision before build.
- **Post-diagnostic adaptation mechanism:** exact rule by which ongoing regular-session performance continues to move the two difficulty axes after the diagnostic sets the starting point.
- Feature-specific tunables (step thresholds, convergence criteria, image-bank cold-start handling) live in `01_adaptive_diagnostic.md` and the two axis docs.

## 7. Success criteria

- New children's first real session accuracy clusters in a healthy middle range (neither near-ceiling "too easy" nor near-floor "too hard") more often than if every child started at the same fixed difficulty.
- The diagnostic does not measurably increase distress/drop-off relative to going straight into a regular first session (validated with beta families/SLPs, mirroring the gamification layer's validation approach).
- Diagnostic completion rate is high — the diagnostic itself doesn't become a drop-off point before therapy even starts.
