-- 034_submission_mark_annotations.sql
-- Teacher "red pen" annotations on a student's answer: ticks, crosses and note
-- pins placed on the script, stored per question (per submission_mark) as a JSONB
-- array of { id, kind: 'note'|'tick'|'cross', x, y, text? } where x/y are percent
-- positions over the answer box. Released to the student with their results so
-- they see the marked-up answer exactly as the teacher left it.
alter table public.submission_marks
  add column if not exists annotations jsonb not null default '[]'::jsonb;

notify pgrst, 'reload schema';
