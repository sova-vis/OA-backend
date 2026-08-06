-- =============================================================================
-- Teacher Portal — feedback & result release (feature groups 10, 11)
-- Per-criterion comments (§10.1), AI-drafted overall feedback (§10.2), a
-- topic-filtered comment bank (§10.3), and release control/content (§11).
--
-- Idempotent (re-run safe).
-- =============================================================================

-- §10.1 — a comment attaches to a specific criterion, not the question. Stored
-- as [{index, text}] aligned to the mark's criteria.
ALTER TABLE submission_marks ADD COLUMN IF NOT EXISTS criterion_comments JSONB NOT NULL DEFAULT '[]'::jsonb;

-- §10.2 — overall script feedback, AI-drafted then teacher-edited/accepted.
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS overall_feedback  TEXT;
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS feedback_is_draft BOOLEAN NOT NULL DEFAULT FALSE;

-- §11 — release control. Automatic release is the default (§11.1). release_content
-- (§11.3) chooses what students receive; examiner notes excluded by default.
ALTER TABLE assignments ADD COLUMN IF NOT EXISTS auto_release    BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE assignments ADD COLUMN IF NOT EXISTS release_content JSONB NOT NULL DEFAULT
  '{"marks":true,"breakdown":true,"comments":true,"scheme_missed":true,"ai_reasoning":false,"examiner_notes":false}'::jsonb;

-- §10.3 — comment bank, per teacher, filtered by topic, optionally shared.
CREATE TABLE IF NOT EXISTS comment_bank (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_clerk_id        TEXT NOT NULL,
  institution_id        UUID REFERENCES institutions(id),
  topic                 TEXT,
  text                  TEXT NOT NULL,
  shared_to_institution BOOLEAN NOT NULL DEFAULT FALSE,
  use_count             INT NOT NULL DEFAULT 0,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comment_bank_owner ON comment_bank(owner_clerk_id, topic);
CREATE INDEX IF NOT EXISTS idx_comment_bank_inst  ON comment_bank(institution_id) WHERE shared_to_institution;
