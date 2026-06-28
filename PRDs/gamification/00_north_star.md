# Adio Gamification — North Star

**Status:** Draft for review
**Owner:** Sidharth
**Last updated:** June 27, 2026

> This is the guiding document for Adio's gamification layer. The three feature docs (collection, streak, milestones) inherit every principle here. If a feature decision conflicts with this document, this document wins.

---

## 1. What we're building

Adio is a Visualization & Verbalization (V&V) app for children with communication differences: it shows an image and asks the child questions about it to build comprehension and expressive language.

This gamification layer operates **between** sessions. It does **not** change the V&V session itself in any way. Its sole job is to encourage the child to come back and start the next session.

It is one system made of three interlocking mechanics, each covering a different time horizon:

| Mechanic | Rewards | Time horizon | Detail doc |
|----------|---------|--------------|------------|
| Collection | A surprise creature every completed session | This session | `01_collection.md` |
| Forgiving streak | A gentle daily rhythm (flame) | This week | `02_streak.md` |
| Milestones | Special creatures at cumulative totals | The long arc | `03_milestones.md` |

## 2. The central commitment: completion-gated, never accuracy-gated

Every reward in every mechanic is earned by **completing** a session — never by performing well in it.

This is non-negotiable and it is the most important line in this document. The research on gamified apps for children with disabilities is clear that extrinsic rewards which gate on performance erode the autonomy and competence that intrinsic motivation depends on. For Adio specifically, accuracy-gating would teach the struggling children the app exists for that effort without correctness goes unrewarded — the exact opposite of the goal. A struggling child must be rewarded exactly as much as a fluent one.

## 3. Design principles

1. **Operate between sessions, not within them.** Rewards accrue and are viewed outside the session; the session stays focused on V&V.
2. **Completion-gated, never accuracy-gated.** (See §2.)
3. **Additive, never subtractive.** The child gains things over time; nothing is taken away as a penalty.
4. **Forgiving by default.** Inconsistent days are normal for this population. The layer never creates anxiety around breaks, and **never requires the child to act at the moment of failure**.
5. **Calm wrapper, surprising contents.** Reveal moments are consistent and gentle in form even when content varies.
6. **Non-comparative.** Progress is always against the child's own history, never another child's.

## 4. Goals & non-goals

**Goals**
- Encourage children to return for the next V&V session.
- Reward returning without altering or distracting from the in-session activity.
- Stay calm and predictable for sensory-sensitive users.
- Never punish missed days or poor performance.

**Non-goals**
- No changes to the V&V session loop (images, questions, feedback stay as-is).
- No leaderboards or child-to-child comparison.
- No streak that resets to zero on a missed day.
- No accuracy-gated rewards anywhere.
- No companion character.

## 5. How the three mechanics reinforce each other

- **Collection** rewards *each* session (immediate, every time).
- **Streak** rewards *consecutive* days (gentle daily rhythm).
- **Milestones** reward *cumulative* totals (the long arc).

They share one surface — milestone creatures and per-session creatures live on the same wall — so the three feel like one world, not three bolted-on systems. Covering three time horizons means that when one thread slips (a broken streak, a distant milestone), another still pulls the child back. Because all three are completion-gated and additive, an inconsistent or struggling child still grows their collection, keeps a held (never zeroed) flame, and reaches every milestone.

## 6. Anti-patterns (explicitly out)

- Accuracy-gated rewards anywhere.
- Streaks that reset to zero, or that require the child to act to be protected.
- Leaderboards or child-to-child comparison.
- "You got nothing" outcomes on a surprise reward.
- Any mechanic that changes the in-session V&V activity.
- High-stimulation, unpredictable celebrations as a non-configurable default.

## 7. Research basis (summary)

- Collectible albums are near-universal in top mobile games and provide a proven "fill the set" pull; the surprise/curiosity layer is a recognized engagement driver.
- For children with disabilities specifically, evidence is mixed and the risk is identified: extrinsic or controlling rewards can diminish intrinsic motivation by impeding autonomy, and reward-chasing can degrade the core activity. Self-Determination Theory (autonomy, competence, relatedness) is the recommended design frame. → drives completion-gating and the "child builds/owns the collection" framing.
- Streak forgiveness *increases* engagement (Duolingo's freeze cut churn ~21% by relieving anxiety). Forgiveness must be automatic/retroactive (no action at failure) and bounded. → drives silent protection + hold-then-wind-down.
- Milestones complement streaks (long-arc engagement when daily novelty fades; mutual backstop when one slips). Early milestones must be near; progress should be visible. → drives session-5 first milestone, capped gaps, visible bar.

## 8. Open questions (cross-cutting)

- **Sensory configuration:** parent/SLP control over reveal intensity (animation, sound) — almost certainly yes, consistent with the rest of Adio. Confirm scope.
- **Between-session home:** one surface holding wall + flame + milestone bar, or separate places?
- Feature-specific tunables live in each feature doc.

## 9. Success criteria

- Children return for subsequent sessions at a higher rate than without the layer.
- Missed days and weaker-performing sessions do not lead to drop-off.
- No increase in distress events attributable to reward moments (validated with beta families/SLPs).
- The between-session surface is actually visited (evidence the rewards pull children back).
