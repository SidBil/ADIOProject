# Home & Session Path — Feature 1: The Home Screen

**Parent doc:** `00_north_star.md` (inherits all principles)
**Covers:** The Home tab surface — layout, the streak badge, first-run, post-session return, visual language
**Last updated:** July 27, 2026

---

## 1. Purpose

Define the **Home tab** as a surface: everything the child sees on the default screen *except* the trail's node-by-node mechanics (which live in `02_session_path.md`) and the tab bar / navigation shell (which lives in `03_navigation_shell.md`). Home is the screen the child lands on when they open Adio and the screen they return to after every session.

Home replaces the current `WelcomeScreen` "Begin a Session" entry point (`web/src/screens/WelcomeScreen.tsx`).

## 2. What lives on Home

| Element | Role | Detailed in |
|---|---|---|
| The session path | The winding vertical trail of nodes; the dominant element, fills most of the surface | `02_session_path.md` |
| Streak badge | A compact, secondary flame + day count in a corner | §3 below |
| Bottom tab bar | Persistent Home / Collection / Progress / Adult nav | `03_navigation_shell.md` |

Explicitly **not** on Home: the milestone progress bar (Progress tab), the creature collection wall (Collection tab), parent/SLP controls (Adult tab). Home is intentionally uncluttered so the path is the unmistakable focus.

## 3. The streak badge

The streak (`PRDs/gamification/02_streak.md`) surfaces on Home as a **compact corner badge**, not a header bar.

- **Placement:** a small badge in a top corner of Home, above the trail — present but not dominating. It must not become the emotional centerpiece (streak doc §3: "present but secondary," de-emphasize the cliff).
- **Content:** the familiar flame icon + current day count. Nothing else.
- **Tap behavior:** tapping the badge may open a small, calm detail (current streak, gentle explanation) — but Home never shows a "use your freeze?" prompt or any required recovery action. All streak protection is silent and automatic (streak doc §4). If a day was missed, the child simply opens Home and sees the flame continued or gently smaller — never zeroed, never a modal about it.
- **Why a corner badge and not a Duolingo-style top bar:** a prominent, always-in-your-face streak counter manufactures the daily-pressure anxiety the streak design explicitly avoids for this population. The badge shows progress and de-emphasizes the cliff.

## 4. First-run (brand-new child)

A brand-new child reaches Home **after signup, onboarding, and the one-time placement diagnostic** (`PRDs/adaptive diagnostic/`). Home is the first "real" surface they see.

- **Pre-rendered path to the first goal.** On first open, the full stretch of nodes from node 1 up to the **first milestone (session 5**, per `PRDs/gamification/03_milestones.md` §5) is drawn on the trail, with the milestone node visible as the destination at the top of the stretch.
- **Node 1 is the obvious start target** — lit, largest visual weight, unmistakably tappable. Every other node in the stretch is drawn but locked.
- Seeing the whole first stretch (not just a single node) is deliberate: it immediately communicates "this is a journey with a near goal," which is the core pull (`00_north_star.md` §3 principle 2). It trades a slightly busier first screen for an immediately legible sense of direction.
- **No coach-mark required at launch** — the lit-vs-locked contrast and a single obvious target should carry the first tap. A gentle first-run pointer is a possible enhancement (open question), not a v1 requirement.

## 5. Returning child (the common case)

- On open, Home restores to the child's **current position** on the trail — the current lit node in view, recently completed nodes scrolled just below/behind it, the next goal above.
- The trail's scroll position defaults to keeping the current node comfortably in the tappable center of the screen, not at the very edge.
- The streak badge reflects whatever the silent protection logic resolved to while the child was away.

## 6. Post-session return: celebration, then Home

When a session completes, the flow is **celebration → Home** (not straight back to Home, and not auto-advance into the next session):

1. The existing completion/celebration moment plays first — the session summary / creature reveal (`web/src/screens/CelebrationScreen.tsx`, and `PRDs/session summary/`, `PRDs/gamification/01_collection.md`).
2. The child is then dropped back onto the **Home path**, with the just-finished node now shown as **completed** and the **current position advanced** to the next node.
3. If the completed session was the one that reached a **milestone goal node**, the milestone reveal plays *on the path* at that node before the position advances — see `02_session_path.md` §5. The per-session celebration and the milestone reveal are distinct moments; on a milestone session the child sees the normal completion celebration, then the milestone reveal on the trail.

Home is always the surface the child returns to — the hub (`00_north_star.md` §3 principle 6).

## 7. Visual language

Home reuses Adio's existing claymorphism and theme tokens — **no new visual system**.

- **Tokens:** `web/src/theme.ts` — background `colors.bg` (#FFF6EF), dark-blue text (`colors.darkBlueText`), the yellow/blue/pink/green card+border pairs, `fonts.heading` (League Spartan) for headings, `fonts.body` (Inter) for text.
- **Nodes as clay stones:** trail nodes reuse the chunky claymorphism recipe already in `WelcomeScreen` (layered outer drop shadow + inset sculpt + top highlight; see the `beginClayShadow` / `historyClayShadow` construction in that file). The current/lit node is the "Begin"-equivalent hero treatment.
- **Calm motion:** transitions are soft and slow (the existing `180ms ease` box-shadow/transform language), never high-stimulation. Any celebratory motion respects sensory settings.
- **Background:** the calm `colors.bg` field. A lightly themed environment behind the trail is explicitly *out* for v1 (the "reuse clay style," not "themed world," decision) — the trail sits on Adio's existing calm background.

## 8. Responsive / platform behavior

- **Mobile-first.** The vertical winding trail is designed for a tall phone viewport. The current node sits in the vertical center of the scrollable trail; the streak badge pins to a top corner; the tab bar pins to the bottom.
- **Web target.** On the wider Expo web canvas (what Vercel serves), the trail is centered in a phone-width column with the calm background filling the margins, rather than stretching nodes across the full width. It must remain usable, not pixel-perfect, on web.
- Reuse the responsive scaling approach already in `WelcomeScreen` (`useWindowDimensions` + clamped `Math.max/Math.min` sizing) rather than fixed pixel sizes.

## 9. Open questions / tunables

- Exact corner and size of the streak badge; whether tapping it opens an inline popover or routes to a streak detail (possibly on the Progress tab).
- Whether a one-time first-run pointer/coach-mark at node 1 is worth adding, or whether the lit/locked contrast is sufficient (`00_north_star.md` §6, `01` §4).
- How far "behind" (completed nodes) the trail keeps in view on return before it stops rendering old history inline — ties to the Progress tab's history view.
- Whether the celebration→Home transition should visually animate the node flipping to "completed" and the position advancing (a small satisfying beat) or simply present the updated state on arrival.
