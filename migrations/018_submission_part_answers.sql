-- =============================================================================
-- Propel — per-part answers for classroom submissions
--
-- Students answer a structured question part-by-part (a, b, c…) just like the
-- Practice page, instead of one combined box. We keep `answer_text` as the
-- combined text (so AI marking + released results are unchanged) and store the
-- per-part answers alongside so the UI can rehydrate each box on reload.
--
--   part_answers : { "0": "answer to (a)", "1": "answer to (b)", ... }  (by index)
--
-- Idempotent (re-run safe).
-- =============================================================================

ALTER TABLE submission_answers ADD COLUMN IF NOT EXISTS part_answers JSONB NOT NULL DEFAULT '{}'::jsonb;
