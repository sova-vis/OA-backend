-- =============================================================================
-- 005 · Student modelling in Postgres  (Architecture report · Phase 2)
--
-- Moves the per-student modelling data OUT of Supabase Storage JSON blobs and
-- into relational tables:
--
--   attempts/<clerkId>.json            ->  public.attempts
--   s/<clerkId>/<paperKey>.json.report ->  public.graded_reports
--   s/<clerkId>/<paperKey>.json        ->  public.practice_sessions
--
-- KEPT in Storage (unchanged): handwritten uploads (files/…), annotated PDFs
-- (checks/…), and the Grok pattern cache (patterns/<clerkId>.json).
--
-- SAFETY / ROLLOUT
--   This migration only CREATES tables — it drops nothing and touches no
--   existing data. The backend dual-writes to Storage AND these tables, and
--   keeps reading from Storage until STUDENT_DATA_SOURCE=postgres is set. So:
--     1. Run this migration.
--     2. Deploy the backend (it starts dual-writing new data here).
--     3. Run scripts/backfill-student-data.js (copies existing blobs in).
--     4. Set STUDENT_DATA_SOURCE=postgres to cut reads over. Roll back any time
--        by removing that variable — Storage is still written and authoritative.
--
-- Run in: Supabase Dashboard -> SQL Editor -> New query -> Run
-- =============================================================================
create extension if not exists pgcrypto;

-- ----------------------------------------------------------------------------
-- attempts — the Phase-1 backbone. One row per graded question, per student.
-- Powers Weakness Map, Predicted Grade, Spaced Repetition, Daily Plan, Momentum.
-- The id is the same stable id the app already assigns (questionId + timestamp),
-- so re-running the backfill is idempotent and per-attempt writes can never lose
-- each other the way the read-modify-write blob append did.
-- ----------------------------------------------------------------------------
create table if not exists public.attempts (
  id           text primary key,               -- app-assigned stable id
  clerk_id     text not null,
  question_id  text not null,
  subject      text,
  topic        text,
  theme        text,
  type         text,                            -- 'mcq' | 'structured'
  verdict      text,                            -- correct|partial|weak|unanswered|incorrect
  earned       numeric,
  max          numeric,
  reason       text,
  year         text,
  session      text,
  paper        text,
  variant      text,
  created_at   timestamptz not null default now()
);

create index if not exists idx_attempts_clerk_created on public.attempts (clerk_id, created_at desc);
create index if not exists idx_attempts_clerk_topic   on public.attempts (clerk_id, subject, topic);
-- Cohort analytics (the obvious next teacher feature): "which topics does this
-- group fail most" becomes a single grouped query instead of reading every blob.
create index if not exists idx_attempts_subject_topic on public.attempts (subject, topic);

-- ----------------------------------------------------------------------------
-- graded_reports — the marked-paper report. One current report per (student,
-- paper), matching today's behaviour where a re-grade overwrites the previous
-- report embedded in the session document.
-- ----------------------------------------------------------------------------
create table if not exists public.graded_reports (
  id         uuid primary key default gen_random_uuid(),
  clerk_id   text not null,
  paper_key  text not null,
  earned     numeric,
  total      numeric,
  percent    numeric,
  grade      text,
  model      text,
  graded_at  timestamptz not null default now(),
  report     jsonb not null,
  unique (clerk_id, paper_key)
);

create index if not exists idx_graded_reports_clerk on public.graded_reports (clerk_id, graded_at desc);

-- ----------------------------------------------------------------------------
-- practice_sessions — autosave / resume state for a paper in progress.
-- ----------------------------------------------------------------------------
create table if not exists public.practice_sessions (
  clerk_id                text not null,
  paper_key               text not null,
  subject                 text,
  year                    text,
  session                 text,
  paper                   text,
  variant                 text,
  is_mcq                  boolean,
  solve_mode              text,                 -- 'digital' | 'handwritten'
  status                  text,                 -- 'in_progress' | 'completed'
  answers                 jsonb,
  uploads                 jsonb,
  answered_count          int,
  total_count             int,
  timer_duration_seconds  int,
  timer_elapsed_seconds   int,
  report                  jsonb,                -- embedded report (round-trip parity)
  started_at              timestamptz,
  updated_at              timestamptz not null default now(),
  primary key (clerk_id, paper_key)
);

create index if not exists idx_practice_sessions_clerk on public.practice_sessions (clerk_id, updated_at desc);

-- ----------------------------------------------------------------------------
-- Row Level Security: DENY ALL. These are private per-student rows reached only
-- through the Express backend with the Supabase service-role key, which bypasses
-- RLS. With no permissive policy, the anon/public key can read nothing here —
-- deliberately stricter than the world-readable question bank.
-- ----------------------------------------------------------------------------
alter table public.attempts          enable row level security;
alter table public.graded_reports    enable row level security;
alter table public.practice_sessions enable row level security;
-- (no CREATE POLICY on purpose — service-role only)
