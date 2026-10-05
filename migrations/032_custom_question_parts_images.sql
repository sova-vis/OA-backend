-- 032: Rich custom questions — multi-part + figure images.
--
-- Teachers can design a full question (a stem plus labelled parts, each with its
-- own figures) instead of a single text box, matching how bank questions render.
--
--   parts : jsonb array of
--           { "label": "(a)", "body": "...", "marks": 3,
--             "images": [ { "data_url": "...", "alt": "...", "caption": null } ] }
--   images: jsonb array of  { "data_url": "...", "alt": "...", "caption": null }
--           (top-level question figures)
--
-- Both columns are additive and default to an empty JSON array, so every existing
-- custom question and all read paths keep working unchanged.

alter table public.custom_questions
  add column if not exists parts  jsonb not null default '[]'::jsonb,
  add column if not exists images jsonb not null default '[]'::jsonb;

-- Let PostgREST pick up the new columns without a container restart.
notify pgrst, 'reload schema';
