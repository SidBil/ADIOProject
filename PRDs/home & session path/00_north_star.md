# Adio Home & Session Path — North Star

**Status:** Draft for review
**Owner:** Sidharth
**Last updated:** July 27, 2026

> This is the guiding document for Adio's home screen and its Duolingo-like session progression. The three feature docs (`01_home_screen.md`, `02_session_path.md`, `03_navigation_shell.md`) inherit every principle here. If a feature decision conflicts with this document, this document wins.

---

## 1. What we're building

Adio is a Visualization & Verbalization (V&V) app for children with communication differences: it shows an image and asks the child questions about it to build comprehension and expressive language.

Today the child enters a session from a generic `WelcomeScreen` "Begin a Session" button (`web/src/screens/WelcomeScreen.tsx`). We are replacing that entry point with a **home screen built around a Duolingo-like session path**: a winding vertical trail of session nodes the child climbs, one session at a time, toward a visible goal.

This does two things at once:

1. It gives every session a **place in a sequence** the child can see — where they've been, where they are, and what they're climbing toward — instead of an identity-less "start" button.
2. It establishes the app's **persistent home surface**: the screen the child lands on when they open Adio and the screen they return to after every session.

## 2. The central commitment: the path is the home, and it is calm

The path is the emotional and navigational center of the app, but it is a **calm** center. We are deliberately building the *forgiving, low-pressure* version of a Duolingo path — the same way the streak (`PRDs/gamification/02_streak.md`) is the forgiving version of a Duolingo streak.

Two non-negotiables follow from this and from the gamification north star (`PRDs/gamification/00_north_star.md` §2):

- **Completion-gated, never accuracy-gated.** A node is completed by finishing its session, never by scoring well in it. The path advances identically for a struggling child and a fluent one. Progress on the trail is never a visible verdict on ability (consistent with the adaptive diagnostic's "placement, not evaluation" stance, `PRDs/adaptive diagnostic/00_north_star.md` §2).
- **Additive and forgiving.** The trail only ever grows. Missed days never cost the child a node, never move them backward, and never present a "you lost progress" moment.

## 3. Design principles

1. **One obvious next action.** At any moment the child has exactly one lit, tappable node — the current session. Past nodes are done; future nodes are visibly locked. The child is never asked to choose *which* session to do.
2. **A visible goal, close ahead.** The path always climbs toward a concrete, near destination (the next milestone creature — see `02_session_path.md` §4). The child can see where they're headed before they get there. Predictable, not a variable-reward surprise.
3. **Child-first, adult-reachable.** The home surface is designed for the child to navigate alone: large targets, minimal text, calm motion. Parent/SLP information is present but lives one deliberate step away (the Adult tab — `03_navigation_shell.md`).
4. **Mobile-first, responsive.** The primary target is a tall phone screen (the vertical trail is native to that shape). It must still render sanely on the Expo web target that Vercel serves.
5. **Reuse Adio's visual language.** The path reuses the existing claymorphism and theme tokens (`web/src/theme.ts`) — chunky clay nodes, the established yellow/blue/dark-blue palette, calm sensory motion. It does not introduce a new visual system.
6. **Home owns the moment; other tabs own the archive.** Reveals and celebrations (milestone creatures, session completion) happen *on the path*, where the child is. The permanent record of those rewards lives on the Collection and Progress tabs. The path is where things *happen*; the other tabs are where things are *kept*.
7. **Reuse, don't rebuild, the session.** Nothing inside a V&V session changes. The path decides *which* session runs next (Stage 1 vs Stage 2, which image tier) and threads that in; it never alters the 10-structure-word contract or the LLM evaluation pipeline.

## 4. Goals & non-goals

**Goals**
- Replace the generic start flow with a persistent, child-navigable home screen centered on a winding session path.
- Give every session a visible position in a sequence, climbing toward a near, concrete goal.
- Establish the four-tab navigation shell (Home / Collection / Progress / Adult) that the previously-separate gamification and progress surfaces slot into.
- Keep the whole surface calm, forgiving, and completion-gated.

**Non-goals**
- No change to the V&V session loop (images, 10 structure-words, LLM scoring stay exactly as they are).
- No change to *what* the gamification mechanics reward — the path consumes the existing milestone/streak/collection systems, it does not redefine them.
- No accuracy-gated progression, no backward movement, no "you lost your place" states.
- No branching, no loops, no child choice of which session to play — a single linear trail (see `02_session_path.md` §2).
- No leaderboards or child-to-child comparison (inherited from gamification north star §4).

## 5. How this supersedes and relates to other PRDs

- **Supersedes `PRDs/V&V stage 2/02_session_progression.md`.** That doc introduced the original "stone path" concept. This PRD folder is now the source of truth for session progression and the home surface. The old doc's core ideas (linear path, completed stones stay visible, Stage 2 stones visually distinct, session-level `mode` flag) are **carried forward and expanded here**; where the two differ, this folder wins. The two other V&V Stage 2 docs (`00_north_star.md`, `01_recall_session.md`) are unaffected — Stage 2 recall mechanics still live there and the path simply schedules them.
- **Consumes `PRDs/gamification/`.** The gamification north star left an open question (§8): "Between-session home — one surface holding wall + flame + milestone bar, or separate places?" This PRD answers it: the **streak** surfaces on Home (as a compact badge); the **collection** and **milestones** get their own tabs (Collection, Progress) rather than being embedded on Home. The milestone *reward moment* is pulled onto the path (§ principle 6), but the milestone *archive/bar* stays on the Progress tab.
- **Orthogonal to `PRDs/adaptive diagnostic/`.** The diagnostic runs once at signup, before the child ever sees the path, and seeds difficulty. The path renders the same regardless of diagnostic output; it schedules sessions, the diagnostic-seeded difficulty axes decide what's *inside* them. No coupling at launch.

## 6. Open questions (cross-cutting)

- Whether the Adult surface is a true peer tab in the bottom bar or a gated area reachable from it (see `03_navigation_shell.md` §3) — leaning gated.
- Whether the path's milestone-goal coupling should ever become configurable, or whether milestone spacing (`PRDs/gamification/03_milestones.md` §5) permanently defines stretch length.
- Whether, after the finite milestone set is completed (~8–10 milestones, per milestones doc §6), the path keeps climbing toward a new kind of goal or becomes an open-ended trail with no terminal goal. Deferred until the milestone endpoint is designed.
- Sensory/motion intensity configuration for path animations and the on-path celebration — almost certainly parent-controllable, consistent with the rest of Adio; exact scope TBD.

Feature-specific open questions live in each feature doc.

## 7. Success criteria

- The home path becomes the actual entry point: children start sessions from the trail without a measurable increase in "how do I start" confusion or drop-off vs the old Begin button.
- The one-lit-node model works: children reliably tap the correct next node without needing an adult to point (validated with beta families/SLPs).
- The near-goal pull is real: children who can see an approaching milestone node return at a higher rate than a flat, goal-less trail would produce.
- No increase in distress attributable to progression (a locked node, a missed day, a reveal moment) — validated the same way the gamification layer is.
