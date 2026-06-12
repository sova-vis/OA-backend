create extension if not exists pgcrypto;

create table if not exists public.o_level_subjects (
  id uuid primary key default gen_random_uuid(),
  name text not null unique,
  slug text not null unique,
  syllabus_code text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.o_level_questions (
  id text primary key,
  subject_id uuid not null references public.o_level_subjects(id) on delete cascade,
  subject text not null,
  subject_slug text not null,
  year int not null,
  session text,
  paper text,
  variant text,
  question_number text not null,
  sub_question text,
  marks int,
  topic_syllabus text,
  topic_general text,
  question_text text not null,
  stem text,
  question_kind text not null default 'structured',
  source_type text not null default 'batch',
  batch_file text,
  options jsonb not null default '[]'::jsonb,
  marking_scheme text,
  correct_option text check (correct_option is null or correct_option in ('A', 'B', 'C', 'D')),
  requires_diagram boolean not null default false,
  images jsonb not null default '[]'::jsonb,
  syllabus_ref jsonb not null default '{}'::jsonb,
  reference jsonb not null default '{}'::jsonb,
  source jsonb not null default '{}'::jsonb,
  raw_question jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.o_level_questions
  add column if not exists question_kind text not null default 'structured',
  add column if not exists source_type text not null default 'batch',
  add column if not exists batch_file text;

create index if not exists o_level_questions_subject_year_idx
  on public.o_level_questions(subject_slug, year desc);

create index if not exists o_level_questions_topic_idx
  on public.o_level_questions(subject_slug, topic_syllabus);

create index if not exists o_level_questions_general_topic_idx
  on public.o_level_questions(subject_slug, topic_general);

create index if not exists o_level_questions_paper_idx
  on public.o_level_questions(subject_slug, year desc, session, paper, variant);

create index if not exists o_level_questions_kind_idx
  on public.o_level_questions(subject_slug, question_kind);

create index if not exists o_level_questions_images_gin_idx
  on public.o_level_questions using gin(images);

create index if not exists o_level_questions_syllabus_ref_gin_idx
  on public.o_level_questions using gin(syllabus_ref);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists set_o_level_subjects_updated_at on public.o_level_subjects;
create trigger set_o_level_subjects_updated_at
before update on public.o_level_subjects
for each row execute function public.set_updated_at();

drop trigger if exists set_o_level_questions_updated_at on public.o_level_questions;
create trigger set_o_level_questions_updated_at
before update on public.o_level_questions
for each row execute function public.set_updated_at();

alter table public.o_level_subjects enable row level security;
alter table public.o_level_questions enable row level security;

drop policy if exists "Read O Level subjects" on public.o_level_subjects;
create policy "Read O Level subjects"
on public.o_level_subjects
for select
using (true);

drop policy if exists "Read O Level questions" on public.o_level_questions;
create policy "Read O Level questions"
on public.o_level_questions
for select
using (true);

insert into storage.buckets (id, name, public)
values ('question-assets', 'question-assets', true)
on conflict (id) do update set public = excluded.public;

drop policy if exists "Read question assets" on storage.objects;
create policy "Read question assets"
on storage.objects
for select
using (bucket_id = 'question-assets');
