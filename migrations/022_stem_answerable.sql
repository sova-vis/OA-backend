-- Whether a structured question's STEM itself needs a student answer box.
-- Nullable tri-state so existing data keeps its current behaviour:
--   null  → auto: a box shows only when the question has no parts (old default)
--   true  → force a stem answer box even when the question has parts
--   false → context-only: never show a stem answer box
alter table public.questions
  add column if not exists stem_answerable boolean;

notify pgrst, 'reload schema';
