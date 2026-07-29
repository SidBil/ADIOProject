# Home & Session Path — Feature 3: Navigation Shell & Data Model

**Parent doc:** `00_north_star.md` (inherits all principles)
**Covers:** The bottom tab bar, its four tabs, integration with `App.tsx`'s screen state machine, and the data model / backend implications of a positioned path
**Last updated:** July 27, 2026

---

## 1. Purpose

Home is not a standalone screen — it is the default tab of a **persistent bottom tab bar** that becomes the app's top-level navigation shell. This doc defines that shell, how it replaces the current `landing → welcome` flow in `App.tsx`, and the data the path needs to render a child's position.

## 2. The bottom tab bar

A persistent bottom navigation bar (Duolingo-style), visible on all child-facing top-level surfaces:

| Tab | Surface | Source PRD |
|---|---|---|
| **Home** | The winding session path + streak badge — the default landing tab | `01_home_screen.md`, `02_session_path.md` |
| **Collection** | The creature collection wall (per-session drops + archived milestone creatures) | `PRDs/gamification/01_collection.md` |
| **Progress** | The milestone progress bar + session history / stats (the long-arc view) | `PRDs/gamification/03_milestones.md`, `PRDs/session summary/` |
| **Adult / Settings** | Parent/SLP information and controls | see §3 |

- **Home is the default tab** — the app opens here, and every session returns here after its celebration (`01_home_screen.md` §6).
- The tab bar is **hidden during an active V&V session** and during the placement diagnostic — those are focused, full-screen flows, not tabbed surfaces. It reappears on return to Home.
- Tabs are **peer surfaces the child can move between freely** (unlike the gated single-forward *path*). Moving to Collection or Progress and back never affects trail position.

This resolves the gamification north star's open "between-session home" question (`PRDs/gamification/00_north_star.md` §8): the three gamification mechanics are distributed across the shell — **streak on Home, collection on Collection, milestones on Progress** — rather than crammed onto one wall. They still form one world because the shell binds them and the milestone reveal is pulled onto Home (`02_session_path.md` §5).

## 3. The Adult / Settings surface

- The Adult surface holds parent/SLP-facing information (progress detail, sensory/intensity settings, consent, feedback/bug report) and is **child-first-safe** — the child landing here by accident should find nothing alarming or breakable.
- **Open question (leaning gated):** whether Adult is a true fourth peer tab always visible in the bar, or a **discreetly gated** entry (e.g. a small gear that requires a deliberate long-press / simple hold-to-enter gesture so a child doesn't wander into settings). The north star (`00_north_star.md` §3 principle 3) calls for adult info to be "one deliberate step away," which biases toward a gated entry rather than an equal-prominence tab. To be decided before build.
- Whichever form it takes, it must not gate the child out of anything the child needs, and it must not become a place accuracy or judgment is surfaced (consistent with the diagnostic's placement-not-evaluation stance).

## 4. Integration with `App.tsx`'s screen state machine

Today `App.tsx` is a flat string state machine: `landing → welcome → session → summary`, plus `onboarding`, `dashboard`, `about` (`type Screen = ...` in `web/App.tsx`).

This feature restructures the top level:

- **`landing` and `welcome` are removed** as the session entry point (superseding the old stone-path doc's "replaces the current start flow"). `WelcomeScreen.tsx` is retired as the start surface. On `localhost` the existing dev auto-sign-in as `dev@adio.local` still applies; after auth the child lands on the **Home tab**, not `welcome`.
- **A persistent tab shell wraps the child-facing surfaces.** Rather than `home`/`collection`/`progress` being sibling string states that fully replace each other, they become tabs under a shell that keeps the bar mounted. `session` and the diagnostic remain full-screen states that take over (tab bar hidden), then return to Home.
- **`dashboard` and the summary/detail screens fold into tabs:** the existing `DashboardScreen`/`SessionDetailScreen` content becomes the **Progress** tab; `CelebrationScreen` remains the post-session celebration that plays before returning to Home. `about` can move under Adult/Settings.
- Exact refactor (introduce a tab navigator vs. extend the existing string machine with a mounted bar component) is an implementation decision for the build, but the **externally-visible behavior** is fixed: four tabs, Home default, bar hidden in-session, celebration-then-Home on completion.

## 5. Data model: a positioned path

The path must render each child's exact position: which node is current, which are done, where the next Stage 2 node falls, and where the next milestone node falls. All of this derives from **one durable quantity: the count of sessions the child has completed.**

- **Session mode flag.** Carried forward from the V&V progression doc: `create_session` (`web/services/session_manager.py`) must stamp a session-level `mode: "stage1" | "stage2"`, determined by the child's position (completed-count `mod 3` → every 3rd is stage2) *before* the session is built, and stored on the `therapy_sessions` row in Supabase so the trail can render historical node types.
- **Derived, not stored, path geometry.** Node states (completed/current/locked), Stage-2 positions (every 3rd), and milestone positions (`5, 11, 18, 26…` from `PRDs/gamification/03_milestones.md` §5) are all **derived from the completed-session count** — they should not be redundantly persisted per-node. The trail is a pure function of "how many sessions has this child completed."
- **Completion-gated advance.** Position advances on session *completion* only (not accuracy, not start). The completed-count increments when a session finishes; a low-accuracy session still increments it.

## 6. Backend / serverless implications

Per `CLAUDE.md`, backend sessions live in an in-memory dict on `SessionManager` that is empty on Vercel's cold-started serverless instances, with `recover_session()` reconstructing from Supabase.

- **The path's completed-count must come from Supabase, not in-memory state.** Rendering Home on a cold instance requires reading the child's completed-session count (and recent session `mode`s for history) from `therapy_sessions` via `interaction_store.py` — it cannot rely on the in-memory session dict.
- Any **new endpoint** that returns path/home state must follow the established fallback pattern (`sessions.get_session(...) or sessions.recover_session(...)`), or better, read the durable count directly from the DB rather than session memory, since path state is a persistent per-user fact, not a per-session-instance one.
- The existing **Modal warmup on session start** is unaffected: the child still taps a node → `start_session` runs the DB insert and `asr.warmup()` in parallel. The tap target moved from a Begin button to a trail node; the start latency contract behind it is unchanged.

## 7. Open questions / tunables

- **Adult surface: gated entry vs. peer tab** (§3) — decision needed before build; leaning gated.
- **Refactor strategy for `App.tsx`** — adopt a real tab navigator vs. mount a persistent bar over the existing string state machine.
- **Where session history/detail lives** — fully under the Progress tab, or partially inline as the trail's completed-node scrollback (ties to `02_session_path.md` §9 rendering/pagination question).
- **Whether the completed-session count should be denormalized** (a cached counter on `user_profiles`) for fast Home rendering, or always computed from `therapy_sessions` rows.
- **Cross-device consistency** — the path is derived from a server-side count, so it should be identical across devices for the same child; confirm no local-only position state drifts from the DB.
