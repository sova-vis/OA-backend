-- =============================================================================
-- Teacher Portal — grading review (feature group 9)
-- Adds: moderator flagging on marks (§9.1), per-teacher auto-approve threshold
-- (§9.6) with an institution floor, and sampling record on assignments (§9.7).
-- The AI-vs-final score separation (§9.9) already exists in submission_marks;
-- override before/after is written to activity_log (migration 010).
--
-- Idempotent (re-run safe).
-- =============================================================================

-- Moderator flag for re-marking (§9.1 filter "flagged by a moderator", §4.2).
ALTER TABLE submission_marks ADD COLUMN IF NOT EXISTS flagged     BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE submission_marks ADD COLUMN IF NOT EXISTS flagged_by  TEXT;
ALTER TABLE submission_marks ADD COLUMN IF NOT EXISTS flagged_at  TIMESTAMPTZ;
ALTER TABLE submission_marks ADD COLUMN IF NOT EXISTS flag_reason TEXT;

-- Auto-approve threshold per teacher (§9.6, default 0.85), applied across their
-- classes with a per-assignment override handled at review time.
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS auto_approve_threshold NUMERIC NOT NULL DEFAULT 0.85;

-- Institution floor: teachers can tighten above it, never loosen below (§9.6).
ALTER TABLE institutions ADD COLUMN IF NOT EXISTS auto_approve_floor NUMERIC;

-- Sampling record (§9.7) so moderators know how an assignment was marked.
ALTER TABLE assignments ADD COLUMN IF NOT EXISTS marked_via           TEXT;      -- 'full' | 'sample'
ALTER TABLE assignments ADD COLUMN IF NOT EXISTS sample_rate          NUMERIC;   -- e.g. 0.10
ALTER TABLE assignments ADD COLUMN IF NOT EXISTS sample_override_rate NUMERIC;   -- observed override rate in the sample
