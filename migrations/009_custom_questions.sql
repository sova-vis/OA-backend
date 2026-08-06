-- =============================================================================
-- Teacher Portal — custom questions with discrete mark-scheme criteria (§5.4)
-- Teachers upload their own question + a mark scheme entered as DISCRETE
-- CRITERIA (not a text blob). This is the structured-criteria model that
-- per-criterion AI marking (§8.4) and teacher override (§9.4) depend on — the
-- bank's text `marking_scheme` cannot support that until content ingestion
-- structures it, so custom questions are where the criteria model lands first.
--
-- Idempotent (re-run safe).
-- =============================================================================

CREATE TABLE IF NOT EXISTS custom_questions (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_clerk_id        TEXT NOT NULL,
  institution_id        UUID REFERENCES institutions(id),
  subject               TEXT NOT NULL,
  syllabus_code         TEXT,
  topic                 TEXT,                 -- feeds mastery tracking (§5.4)
  sub_topic             TEXT,
  question_text         TEXT NOT NULL,
  question_type         TEXT NOT NULL DEFAULT 'structured'
                          CHECK (question_type IN ('mcq', 'structured', 'extended')),
  marks                 INT NOT NULL DEFAULT 0,  -- sum of criteria marks
  shared_to_institution BOOLEAN NOT NULL DEFAULT FALSE,  -- private by default
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_custom_questions_owner   ON custom_questions(owner_clerk_id);
CREATE INDEX IF NOT EXISTS idx_custom_questions_subject ON custom_questions(subject);
CREATE INDEX IF NOT EXISTS idx_custom_questions_inst    ON custom_questions(institution_id) WHERE shared_to_institution;

CREATE TABLE IF NOT EXISTS custom_question_criteria (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  custom_question_id UUID NOT NULL REFERENCES custom_questions(id) ON DELETE CASCADE,
  order_index        INT NOT NULL DEFAULT 0,
  criterion_text     TEXT NOT NULL,
  marks              INT NOT NULL DEFAULT 1,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_custom_criteria_question ON custom_question_criteria(custom_question_id, order_index);

-- Make assignment questions polymorphic so custom questions can be assigned
-- alongside bank questions (§5.4 — AI marking works identically on both).
ALTER TABLE assignment_questions ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'bank'
  CHECK (source IN ('bank', 'custom'));
ALTER TABLE assignment_questions ADD COLUMN IF NOT EXISTS custom_question_id UUID REFERENCES custom_questions(id) ON DELETE SET NULL;
