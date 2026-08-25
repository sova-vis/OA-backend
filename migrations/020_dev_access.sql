-- Dev mode: a single shared password (bcrypt hash) that unlocks in-place editing
-- of the live question bank from the practice UI. One row only. Edits go straight
-- to public.questions / public.question_parts (no drafts, no history) so every
-- normal user sees the correction on their next load.

create table if not exists public.dev_access (
  id              smallint primary key default 1,
  password_hash   text not null,
  set_by_clerk_id text,
  updated_at      timestamptz not null default now(),
  constraint dev_access_single_row check (id = 1)
);
