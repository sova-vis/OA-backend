-- =============================================================================
-- Teacher review — voice note per question (user request).
-- A teacher can attach a short voice note to any marked answer; the student
-- hears it with their released result. Stored as a base64 data URL.
-- Idempotent (re-run safe).
-- =============================================================================

ALTER TABLE submission_marks ADD COLUMN IF NOT EXISTS voice_note TEXT;
