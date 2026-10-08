-- 035_ask_chats.sql
-- Account-synced Ask-AI chat history, so a student's Ask / Find conversations
-- follow them across devices (localStorage stays an offline cache that merges on
-- load). One row per user per mode holds the capped list of sessions as JSONB.
create table if not exists public.ask_chats (
  clerk_id text not null,
  mode text not null,
  sessions jsonb not null default '[]'::jsonb,
  updated_at timestamptz not null default now(),
  primary key (clerk_id, mode)
);

notify pgrst, 'reload schema';
