-- =============================================================================
-- Teacher Portal — submissions & marking (feature groups 7, 8; feeds 9, 11)
--
-- Contract with the marking engine (§13): the portal does NOT mark theory; it
-- stores per-criterion outputs (§8.4) and a calibrated confidence per criterion
-- (§8.6 — currently a blocking dependency, stored as-is when the engine can't
-- yet provide it). MCQ is auto-marked by the portal against the answer key
-- (§8.1, deterministic, bypasses review). The AI score is kept separate from
-- the teacher's final score and never overwritten (§9.9 audit trail).
--
-- Idempotent (re-run safe).
-- =============================================================================

CREATE TABLE IF NOT EXISTS submissions (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  assignment_id      UUID NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
  student_clerk_id   TEXT NOT NULL,
  attempt_number     INT NOT NULL DEFAULT 1,
  status             TEXT NOT NULL DEFAULT 'in_progress'
                       CHECK (status IN ('in_progress', 'submitted', 'late', 'returned')),
  started_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  submitted_at       TIMESTAMPTZ,
  time_spent_seconds INT NOT NULL DEFAULT 0,
  -- Per-student deadline extension (§7.3); reopening (§7.4).
  extension_until    TIMESTAMPTZ,
  reopened_at        TIMESTAMPTZ,
  reopen_reason      TEXT,
  -- Release gate (§11) — results invisible to students until set (Phase 6).
  released_at        TIMESTAMPTZ,
  -- Computed after marking.
  total_score        NUMERIC,
  total_marks        INT,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (assignment_id, student_clerk_id, attempt_number)
);

CREATE INDEX IF NOT EXISTS idx_submissions_assignment ON submissions(assignment_id, status);
CREATE INDEX IF NOT EXISTS idx_submissions_student    ON submissions(student_clerk_id);

CREATE TABLE IF NOT EXISTS submission_answers (
  id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  submission_id          UUID NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
  assignment_question_id UUID NOT NULL REFERENCES assignment_questions(id) ON DELETE CASCADE,
  answer_text            TEXT,
  selected_option        TEXT,               -- MCQ choice
  images                 JSONB NOT NULL DEFAULT '[]'::jsonb,  -- handwritten scans, retained (§8.2)
  ocr_text               TEXT,               -- extracted text
  ocr_confidence         NUMERIC,            -- per-question OCR confidence (§8.2)
  ocr_status             TEXT NOT NULL DEFAULT 'none'
                           CHECK (ocr_status IN ('none', 'pending', 'ok', 'failed')),
  answered               BOOLEAN NOT NULL DEFAULT FALSE,
  updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (submission_id, assignment_question_id)
);

CREATE INDEX IF NOT EXISTS idx_submission_answers ON submission_answers(submission_id);

CREATE TABLE IF NOT EXISTS submission_marks (
  id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  submission_id          UUID NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
  assignment_question_id UUID NOT NULL REFERENCES assignment_questions(id) ON DELETE CASCADE,
  question_type          TEXT NOT NULL DEFAULT 'theory' CHECK (question_type IN ('mcq', 'theory')),

  -- §8.4 per-criterion breakdown: array of
  -- {criterion_text, marks_available, marks_awarded, awarded, reasoning, confidence}
  ai_criteria            JSONB NOT NULL DEFAULT '[]'::jsonb,
  ai_score               NUMERIC,            -- original AI total — never overwritten (§9.9)
  ai_marks_available     INT,
  -- §8.6 question-level confidence = MIN of criteria confidences.
  confidence             NUMERIC,

  -- Teacher's final (starts equal to AI; edited during review, §9.4).
  final_criteria         JSONB,
  final_score            NUMERIC,

  status                 TEXT NOT NULL DEFAULT 'pending'
                           CHECK (status IN ('pending', 'auto_approved', 'needs_review', 'ocr_failed', 'approved', 'overridden')),
  reviewed_by            TEXT,
  reviewed_at            TIMESTAMPTZ,
  created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (submission_id, assignment_question_id)
);

CREATE INDEX IF NOT EXISTS idx_submission_marks_sub    ON submission_marks(submission_id);
CREATE INDEX IF NOT EXISTS idx_submission_marks_status ON submission_marks(status);

-- Activity log (§18.3) — immutable audit of overrides, releases, reopens, etc.
-- Introduced here because reopen/extension events (§7.4) are logged; grows in later phases.
CREATE TABLE IF NOT EXISTS activity_log (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  actor_clerk_id TEXT,
  institution_id UUID,
  event_type     TEXT NOT NULL,     -- 'reopen','extension','override','bulk_approve','release','withdraw',...
  target_type    TEXT,              -- 'submission','assignment','mark',...
  target_id      UUID,
  detail         JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_activity_log_target ON activity_log(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_activity_log_inst   ON activity_log(institution_id, created_at DESC);
