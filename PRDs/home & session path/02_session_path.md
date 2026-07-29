# Home & Session Path — Feature 2: The Session Path

**Parent doc:** `00_north_star.md` (inherits all principles)
**Covers:** The winding trail mechanic — node shape, states/locking, Stage 1/Stage 2 cadence, milestone goal nodes, on-path reveal, path horizon
**Supersedes:** the "stone path" concept in `PRDs/V&V stage 2/02_session_progression.md`
**Last updated:** July 27, 2026

---

## 1. Purpose

Define the trail itself: what a node is, how nodes lock and unlock, how Stage 1 and Stage 2 sessions are laid out along it, what the child climbs toward, and what happens when they reach it. The Home *surface* that hosts this trail is `01_home_screen.md`; the *navigation shell and data model* is `03_navigation_shell.md`.

## 2. Shape: a single linear winding trail

- The path is **one continuous, winding, vertical trail** of session nodes — the classic single-column Duolingo "snake." It reads bottom-to-top or top-scrolls-down (implementation detail; the child climbs forward).
- **Linear only.** One path, one direction, **no branches, no loops, no forks.** The child never chooses *which* session to do next — there is exactly one next node (`00_north_star.md` §3 principle 1). This carries forward the old stone-path's "one path, one direction, like a road" principle.
- **Completed nodes stay visible** on the trail behind the current position — the path doubles as a visible history the child (or an adult) can scroll back through. It is a persistent trail, not a disappearing queue.

## 3. Node states and locking

Each node is one V&V session. Nodes have three states:

| State | Appearance | Interaction |
|---|---|---|
| **Completed** | Filled/"done" clay stone, calm, checked | Not tappable (see below) — history only |
| **Current** | Lit, largest visual weight, the hero node | Tappable — starts the session |
| **Locked** | Muted/greyed clay stone, visibly inactive | Not tappable |

- **Locked until reached.** Only the current node is tappable. Future nodes are visibly locked; the child cannot skip ahead. This is classic Duolingo gating and it enforces the "one obvious next action" principle — there is never ambiguity about what to tap.
- **Completed nodes are not replayable in v1.** Re-tapping a done node does nothing (or, at most, shows a small read-only recap). We explicitly chose gated single-forward progression over replayable past nodes to keep the child's choice space at exactly one. (Replay of past sessions is a possible later enhancement — open question.)
- Because progression is **completion-gated** (`00_north_star.md` §2), the current node advances to completed the instant its session is *finished*, regardless of accuracy. A low-scoring session advances the trail exactly like a high-scoring one.

## 4. What the child climbs toward: the next milestone creature

The trail is not an undifferentiated infinite scroll — it climbs toward a **visible, near goal**, and that goal is the **next milestone creature** (`PRDs/gamification/03_milestones.md`).

- **Goal node = milestone node.** At the end of the current stretch sits a distinct **milestone node** (visually special — the destination, not just another session stone). Reaching and completing the session that lands on it triggers the milestone reveal (§5).
- **Stretch length = milestone spacing.** We do not invent a separate "how many nodes per goal" number; the stretch is defined by the existing milestone cadence: **first milestone at session 5, then gaps that widen but are capped ~8–10** (milestones doc §5: `5, 11, 18, 26, 34, 42…`). So the first goal is 5 nodes up; later goals are 6–10 nodes up. This keeps the goal always near enough to pull, by construction (the milestone doc caps gaps precisely so the bar — and now the trail — never looks stuck).
- **After a goal, the next stretch renders** toward the following milestone. The trail keeps extending milestone-to-milestone.
- **Finite goal set.** Milestones are a completable set (~8–10, milestones doc §6). What the trail climbs toward *after the final milestone* is an open question (`00_north_star.md` §6) — deferred until the milestone endpoint is designed.

> **Coupling note:** the milestone *reward moment* is pulled onto the path (§5), but the milestone *progress bar and archive* remain on the **Progress tab** (`03_navigation_shell.md`). The trail's climb toward a milestone node is the path-native expression of the same progress the Progress-tab bar shows; they must stay in sync (both driven by cumulative completed-session count).

## 5. Milestone reveal — on the path

When the child completes the session that lands on a milestone node:

1. The normal per-session completion celebration plays first (`01_home_screen.md` §6).
2. Then the **milestone reveal plays on the path**, at the milestone node itself — the special milestone creature is revealed right there on Home, where the child is. Home owns the moment (`00_north_star.md` §3 principle 6).
3. The revealed creature is then **permanently archived** on the **Collection** wall (and counted on the **Progress** milestone bar). Home owns the reveal; the other tabs own the archive.
4. The trail then advances: the milestone node becomes completed, and the next stretch toward the following milestone becomes visible with a new current node.

This is the "reveal on the path" decision: we do **not** route the child away to the Progress/Collection tab for the reveal, and we do **not** use a tab-agnostic modal divorced from the trail. The reveal happens at the node.

## 6. Stage 1 vs Stage 2 nodes on the trail

The trail schedules the two V&V conditions defined in `PRDs/V&V stage 2/` — Stage 1 (image visible) and Stage 2 (recall, image removed).

- **Cadence is preserved from the V&V Stage 2 spec:** a fixed **2:1 pattern** — two Stage 1 sessions, then one Stage 2 ("special"/recall) session, repeating (every 3rd node is Stage 2). Example: `S1, S1, S2, S1, S1, S2…`. Not configurable at launch.
- **Stage 2 nodes are visually distinct** from Stage 1 nodes on the trail (distinct color/icon/shape for the recall stone), so a child glancing at the path can see a recall session coming before they reach it — the "advance-visibility" principle from the V&V progression doc. This distinction is *within* the single winding trail; Stage 2 nodes are not a separate structural chapter, just a differently-styled node in the same line.
- **Two independent visual axes exist on the trail** and must not be confused:
  - *Stage axis:* Stage 1 node vs Stage 2 (recall) node — the every-3rd-node rhythm.
  - *Goal axis:* ordinary node vs milestone (goal) node — the 5/11/18… rhythm.
  - A single node can be both a Stage 2 node *and* a milestone node when the cadences coincide; the visual system must gracefully compose the two treatments (open question §9).
- **What's inside each node is unchanged:** all 10 structure-words, one image, existing LLM 0–5 scoring. The trail only decides `mode` (stage1/stage2) and which image tier (from the adaptive-diagnostic-seeded difficulty), then threads it into session start. It never touches the session contract.

## 7. Path horizon: fixed goal ahead

- The child always sees a **fixed goal ahead** — the next milestone node — plus the run of nodes leading to it, and completed nodes behind.
- The trail is **not** presented as an infinite goal-less scroll; every visible stretch terminates in a concrete milestone destination (§4). Reaching it opens the next stretch.
- Practically, the rendered window is: completed nodes behind (scrollable history), the current lit node, and the upcoming nodes **through the next milestone node**. Nodes beyond the next milestone need not be rendered until that milestone is reached.

## 8. Relationship to other systems (summary)

- **Supersedes** `PRDs/V&V stage 2/02_session_progression.md` — carries forward its linear-path, persistent-history, Stage-2-distinct-stone, and `mode`-flag ideas; overrides where they differ.
- **Feeds `PRDs/V&V stage 2/01_recall_session.md`** — the trail decides *when* a Stage 2 session happens (every 3rd node); what happens *inside* it (blur, single peek, gentler threshold) is fully specified there.
- **Consumes `PRDs/gamification/03_milestones.md`** — milestone spacing defines stretch length; milestone creatures are the goal; the reveal is pulled onto the path while the bar/archive stay on the Progress tab.
- **Orthogonal to `PRDs/adaptive diagnostic/`** — difficulty (image tier, question profile) is decided by the diagnostic-seeded axes and consumed *inside* a node's session; it never changes the trail's shape or scheduling.

## 9. Open questions / tunables

- **Composed visual treatment** when a node is simultaneously a Stage 2 (recall) node and a milestone (goal) node — how the two distinct visual languages combine without reading as a third, confusing state.
- **Exact Stage 2 node visual** (color/icon/shape) and whether it borrows rarity-style visual language from the collection or stays deliberately separate (inherited open question from the V&V progression doc).
- **Replayable completed nodes** — kept out of v1; revisit as an enhancement (repeat a past session with a fresh image).
- **Post-final-milestone trail** — what the trail climbs toward once the finite milestone set is exhausted (`00_north_star.md` §6).
- **Trail rendering/pagination** as history grows very long — how much completed history stays inline on Home vs offloaded to the Progress tab's history view.
- **First-node-of-a-new-stretch animation** — whether extending the trail after a milestone is an on-screen "the path grows" beat or simply present on the next Home render.
