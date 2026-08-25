-- Per-part images for structured questions.
-- Lets dev mode attach figures to an individual sub-part (a, b(i), …), the same
-- way whole-question figures live in questions.images. Same jsonb shape:
--   [{ role, caption, width, height, data_url }]
alter table public.question_parts
  add column if not exists images jsonb not null default '[]'::jsonb;

-- refresh PostgREST's cached schema so the new column is selectable immediately
notify pgrst, 'reload schema';
