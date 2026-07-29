-- Migration: 003_answer_duration.sql
-- Description: Adds per-turn "answer duration" instrumentation used by the
-- parent/SLP clinical detail view (Session Summary feature, PRD 01_session_summary §4.4).
--
-- "answer_duration_ms" is the total time to answer a question — from the moment
-- the question is shown to the moment the answer is submitted (recording stopped).
-- This is distinct from the existing latencies:
--   * initiation_latency_ms — question shown -> recording started
--   * asr_latency_ms / llm_latency_ms — system processing time
--
-- Nullable so historical turns (and turns that never reach evaluation) stay valid.

alter table public.therapy_turns
  add column if not exists answer_duration_ms numeric;
