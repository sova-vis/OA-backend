-- =============================================================================
-- Teacher Portal — assignments (spec §11 / feature group 6)
-- An assignment is a set of question references (into the schema-v2 questions
-- bank) plus rules and recipients. Marking/submissions come in Phase 4; this is
-- the create → track scaffold.
--
-- Idempotent (re-run safe).
-- =============================================================================

CREATE TABLE IF NOT EXISTS assignments (
  id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  class_id              UUID NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
  owner_clerk_id        TEXT NOT NULL,
  title                 TEXT NOT NULL,
  -- How the question set was assembled (§6.1/6.2/6.3).
  source_mode           TEXT NOT NULL DEFAULT 'selected'
                          CHECK (source_mode IN ('full_paper', 'selected', 'topic')),
  -- full_paper: {subject,year,session,paper,variant}; topic: {topics:[],types:[]}
  source_meta           JSONB NOT NULL DEFAULT '{}'::jsonb,

  status                TEXT NOT NULL DEFAULT 'draft'
                          CHECK (status IN ('draft', 'scheduled', 'published', 'closed')),

  -- Rules (§6.4). Deadlines are stored UTC; rendered in the viewer's timezone.
  deadline_at           TIMESTAMPTZ,
  timed                 BOOLEAN NOT NULL DEFAULT FALSE,
  duration_minutes      INT,
  attempt_limit         INT,                 -- null = unlimited
  -- Default 'after_release' — releasing the scheme earlier invalidates AI marking.
  mark_scheme_visibility TEXT NOT NULL DEFAULT 'after_release'
                          CHECK (mark_scheme_visibility IN ('never', 'after_submission', 'after_deadline', 'after_release')),

  total_marks           INT NOT NULL DEFAULT 0,

  -- Recipients (§6.5). target_all=true → whole class; else assignment_recipients.
  target_all            BOOLEAN NOT NULL DEFAULT TRUE,

  scheduled_publish_at  TIMESTAMPTZ,         -- §6.6 schedule
  published_at          TIMESTAMPTZ,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_assignments_class  ON assignments(class_id, status);
CREATE INDEX IF NOT EXISTS idx_assignments_owner  ON assignments(owner_clerk_id);

-- One row per question in the assignment. References the questions bank by uid;
-- a snapshot is kept so display survives a re-ingest of the bank.
CREATE TABLE IF NOT EXISTS assignment_questions (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  assignment_id UUID NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
  question_uid  UUID REFERENCES public.questions(id) ON DELETE SET NULL,
  question_ref  TEXT,                         -- questions.question_id (human id)
  order_index   INT NOT NULL DEFAULT 0,
  marks         INT,
  excluded      BOOLEAN NOT NULL DEFAULT FALSE, -- §6.1 exclude a question in-flow
  snapshot      JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (assignment_id, question_uid)
);

CREATE INDEX IF NOT EXISTS idx_assignment_questions ON assignment_questions(assignment_id, order_index);

-- Individually-targeted recipients (only used when target_all = false).
CREATE TABLE IF NOT EXISTS assignment_recipients (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  assignment_id    UUID NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
  student_clerk_id TEXT NOT NULL,
  UNIQUE (assignment_id, student_clerk_id)
);

CREATE INDEX IF NOT EXISTS idx_assignment_recipients ON assignment_recipients(assignment_id);
