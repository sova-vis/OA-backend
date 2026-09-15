-- 024_ask_ai_embeddings.sql
--
-- Ask-AI semantic search index (pgvector), reached the SAME way the app reaches
-- everything else: over the Supabase REST API (PostgREST), not a direct Postgres
-- connection. This file is the one-time DDL — run it once against the Oracle
-- self-hosted Supabase (Studio SQL editor, or your normal migration runner
-- pointed at that DB). After it exists, the build job and the live service both
-- read/write it purely over HTTPS.
--
-- ADDITIVE AND NON-DESTRUCTIVE: creates one table + one function + enables two
-- extensions. It does NOT read, alter, or touch public.questions /
-- public.question_parts.

create extension if not exists vector;   -- pgvector: embedding column + ANN search
create extension if not exists pg_trgm;  -- fast substring / "exact term" matching

create table if not exists public.ask_ai_chunks (
  question_id     text primary key,      -- mirrors public.questions.question_id
  level           text,                  -- 'olevel' | 'alevel'
  subject         text,
  type            text,                  -- 'mcq' | 'structured'
  exam_year       int,
  session         text,
  paper           text,
  variant         text,
  question_number text,
  topic           text,
  content         text not null,         -- exact text that was embedded
  embedding       vector(768) not null,  -- BAAI/bge-base-en-v1.5 dimensionality
  updated_at      timestamptz not null default now()
);

create index if not exists ask_ai_chunks_meta_idx
  on public.ask_ai_chunks (level, subject, exam_year desc);
create index if not exists ask_ai_chunks_content_trgm_idx
  on public.ask_ai_chunks using gin (content gin_trgm_ops);

-- Approximate-nearest-neighbour index (cosine). HNSW needs pgvector >= 0.5
-- (standard on recent Supabase) and works on an empty table, so it can live in
-- the migration. If your pgvector is older, drop this and instead create an
-- ivfflat index AFTER the build job has loaded the rows.
create index if not exists ask_ai_chunks_embedding_hnsw
  on public.ask_ai_chunks using hnsw (embedding vector_cosine_ops);

-- RLS: readable (service_role bypasses it anyway; anon can read).
alter table public.ask_ai_chunks enable row level security;
drop policy if exists "Read ask_ai_chunks" on public.ask_ai_chunks;
create policy "Read ask_ai_chunks" on public.ask_ai_chunks for select using (true);

-- Vector search exposed as a PostgREST RPC, so the service can search over HTTPS
-- (no direct Postgres access needed). Always scoped by level so O and A never
-- mix; subject is optional.
create or replace function public.match_ask_ai_chunks(
  query_embedding vector(768),
  match_level     text default null,
  match_subject   text default null,
  match_count     int  default 10
)
returns table (
  question_id text, level text, subject text, type text, exam_year int,
  session text, paper text, variant text, question_number text, topic text,
  content text, distance double precision
)
language sql stable
-- pgvector lives in the `extensions` schema on Supabase; without this the
-- `<=>` operator isn't found at function runtime ("operator does not exist").
set search_path = public, extensions
as $$
  select c.question_id, c.level, c.subject, c.type, c.exam_year, c.session,
         c.paper, c.variant, c.question_number, c.topic, c.content,
         (c.embedding <=> query_embedding) as distance
  from public.ask_ai_chunks c
  where (match_level   is null or c.level   = match_level)
    and (match_subject is null or c.subject = match_subject)
  order by c.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;
