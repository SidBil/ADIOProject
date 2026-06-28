# Adio Gamification — Feature 2: The Forgiving Streak

**Parent doc:** `00_north_star.md` (inherits all principles; completion-gating is mandatory)
**Time horizon:** This week — gentle daily rhythm
**Last updated:** June 27, 2026

---

## 1. Purpose

The streak adds a light daily rhythm — a gentle reason to return tomorrow — without importing the anxiety that conventional loss-aversion streaks create. For Adio's population, a streak that can catastrophically vanish is a churn trap and an anxiety generator, so this streak is **forgiving by design**: it is the higher-performing version, not a watered-down one (Duolingo's streak freeze cut churn ~21% precisely by relieving anxiety).

## 2. Trigger

- Advances on session **completion**, never accuracy (per north star §2).
- One completed session = the streak continues for that day.

## 3. Visual

- A familiar **flame icon** with a day count.
- **Present but secondary.** The flame must not be the prominent emotional centerpiece — keeping it quiet avoids the daily-pressure anxiety that prominent loss-aversion streaks produce. Show progress; de-emphasize the cliff.

## 4. Silent, automatic, retroactive protection

This is the single most important rule for the population.

- If a day is missed, the streak is protected **silently and automatically**.
- The child simply opens Adio next time and sees it **continued**.
- There is **never** a "use your freeze?" prompt, confirmation, or any required recovery action.
- Rationale: any forgiveness that requires the child to act at the moment of failure leaks exactly the users it's meant to retain. Protection happens on the child's behalf, while they're away.

## 5. Hold-then-wind-down (bounded, gentle)

Protection is **bounded** so the streak stays meaningful — but the bound is a gentle slope, never a cliff.

1. **Protected window:** after a missed day, a short window of missed days is fully invisible / protected.
2. **Hold:** beyond that window, the flame **holds in place** for a while. A child returning after a short break finds it exactly where they left it.
3. **Wind-down:** only after an extended absence does the flame **slowly wind down**, stepping down gradually.

The flame **never snaps to zero.** A child away two weeks returns to a *smaller* flame, not a dead one.

## 6. Parameters (tunable — defaults proposed)

| Parameter | Proposed default | Notes |
|-----------|------------------|-------|
| Protected window | ~2–3 days | A bit more generous than Duolingo's ~2, given the population |
| Hold duration | ~7 days | Flame frozen in place before wind-down begins |
| Wind-down rate | Gentle step-down | e.g. one day's worth every day or two; never a reset |

## 7. Relationship to other mechanics

- The streak is the only **time-sensitive** mechanic, so it provides the daily rhythm the collection and milestones don't.
- It is deliberately kept **quiet and secondary** so it doesn't introduce daily-pressure anxiety that the rest of the design avoids. When the streak slips, the collection and milestones still pull the child back.

## 8. Open questions / tunables

- Final numbers for protected window / hold / wind-down rate.
- Whether a gentle, non-required visual marker (e.g. a quiet "rest day" on the gap) is desirable, or whether protection should be entirely invisible. *(Current decision: fully silent — child just sees it continued.)*
- Reveal/animation intensity config for the flame, consistent with sensory settings.
