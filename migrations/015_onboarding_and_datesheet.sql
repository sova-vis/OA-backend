-- =============================================================================
-- Propel — student/teacher onboarding survey + exam datesheet
-- (propel-onboarding-datesheet-spec.md)
--
-- Idempotent (re-run safe).
-- =============================================================================

-- The original profiles table pre-existed without these (migration 000's CREATE
-- TABLE IF NOT EXISTS silently skipped them). Ensure they exist.
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS selected_subjects TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];
CREATE UNIQUE INDEX IF NOT EXISTS profiles_clerk_id_key ON profiles(clerk_id);

-- Onboarding survey fields (profiles already has: role, level, selected_subjects,
-- school_name, photo_url, onboarding_complete, full_name).
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS exam_session       TEXT;    -- 'Oct/Nov 2026' | 'May/June 2027' | ... | 'Not sure'
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS target_grade       TEXT;    -- 'A*' | 'A' | 'B' | 'C' | 'pass'
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS study_days         TEXT;    -- '2-3' | '4-5' | 'every'
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS subject_confidence JSONB NOT NULL DEFAULT '{}'::jsonb;  -- {subject: 'strong'|'okay'|'weak'}
-- Teacher onboarding.
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS experience_years   INT;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS school_type        TEXT;    -- 'private' | 'public' | 'none'

-- ----------------------------------------------------------------------------
-- Exam timetable — SESSION-LEVEL (one parse serves every student on a session).
-- Tier 1 (published) rows are status='confirmed'; Tier 2 (predicted) rows are
-- status='predicted' with a date_range and are rendered distinctly (spec §3.2).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS exam_timetable (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  exam_session  TEXT NOT NULL,            -- 'Oct/Nov 2026'
  level         TEXT,                     -- 'O' | 'A'
  syllabus_code TEXT,                     -- '1123'
  subject       TEXT NOT NULL,            -- 'English Language'  (matched to student's selected_subjects)
  component     TEXT NOT NULL,            -- 'Paper 1'
  exam_date     DATE,                     -- confirmed single date (null when predicted-range only)
  date_start    DATE,                     -- predicted range start
  date_end      DATE,                     -- predicted range end
  session_slot  TEXT,                     -- 'Morning' | 'Afternoon'
  duration      TEXT,                     -- '1h 30m'
  status        TEXT NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed', 'predicted')),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_exam_timetable_lookup ON exam_timetable(exam_session, subject);
