-- Migration: 004_session_mode.sql
-- Description: Adds the session-level V&V "mode" flag used by the Home session
-- path (PRD `home & session path/03_navigation_shell.md` §5).
--
-- "mode" records which V&V condition a session ran under:
--   * stage1 — image visible (the default)
--   * stage2 — recall (image removed), every 3rd node on the trail
--
-- It is stamped at session creation from the child's completed-session count
-- (every 3rd session is stage2) and stored so the trail can render historical
-- node types. Nullable + defaulted so pre-migration rows stay valid and read as
-- ordinary Stage 1 sessions.

alter table public.therapy_sessions
  add column if not exists mode text not null default 'stage1';
