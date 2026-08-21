-- =============================================================================
-- Propel — server-side personalization (per-level subjects + active level)
--
-- Selected subjects and the O/A toggle used to live only in the browser's
-- localStorage, so the same account showed different subjects on a different
-- device / domain. Move the source of truth to the profile so it follows the
-- account everywhere.
--
--   subjects_by_level : { "olevel": string[], "alevel": string[] }
--   active_level      : 'olevel' | 'alevel'
--
-- The flat `selected_subjects` column is kept in sync (as the union of both
-- levels) because the datesheet + mentoring routes still read it.
--
-- Idempotent (re-run safe).
-- =============================================================================

ALTER TABLE profiles ADD COLUMN IF NOT EXISTS subjects_by_level JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS active_level      TEXT;  -- 'olevel' | 'alevel'
