-- =============================================================================
-- Teacher Portal — institution layer, permissions, settings, compliance
-- (feature groups 14–18)
--
-- Idempotent (re-run safe).
-- =============================================================================

-- §4.2 / §14.6 / §15.2 — oversight roles are scope GRANTS, not roles. A grant
-- filters by subject and level and carries capabilities; overlapping grants
-- resolve as a union. `label` is the school's own terminology, shown in the UI.
CREATE TABLE IF NOT EXISTS scope_grants (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_clerk_id  TEXT NOT NULL,
  institution_id UUID NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
  filter_subjects TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],  -- empty = all subjects
  filter_levels   TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],  -- empty = all levels (O/A)
  capabilities    TEXT[] NOT NULL DEFAULT ARRAY['view']::TEXT[],  -- view | moderate | assign
  label           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scope_grants_user ON scope_grants(user_clerk_id);
CREATE INDEX IF NOT EXISTS idx_scope_grants_inst ON scope_grants(institution_id);

-- §17 — teacher settings: grading defaults, notification prefs, timezone.
-- (auto_approve_threshold added in migration 011; timezone in 006.)
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS default_mark_scheme_visibility TEXT NOT NULL DEFAULT 'after_release';
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS default_release_content JSONB NOT NULL DEFAULT
  '{"marks":true,"breakdown":true,"comments":true,"scheme_missed":true,"ai_reasoning":false,"examiner_notes":false}'::jsonb;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS notif_prefs JSONB NOT NULL DEFAULT
  '{"new_submissions":true,"deadline_reached":true,"review_backlog":true,"moderator_flag":true,"enrolment_request":true}'::jsonb;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS digest_frequency TEXT NOT NULL DEFAULT 'daily';  -- daily | weekly | off
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMPTZ;  -- §14.3 deactivate teacher

-- §18.1 — retention policy per institution (config; automated purge is ops).
ALTER TABLE institutions ADD COLUMN IF NOT EXISTS retention_policy JSONB NOT NULL DEFAULT
  '{"submissions_days":1095,"ocr_images_days":90,"activity_log_days":2555}'::jsonb;

-- §16.1 — in-app notifications, batched by assignment/class. Read on view.
CREATE TABLE IF NOT EXISTS notifications (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  recipient_clerk_id TEXT NOT NULL,
  type           TEXT NOT NULL,       -- new_submissions | deadline_reached | review_backlog | moderator_flag | enrolment_request
  class_id       UUID,
  assignment_id  UUID,
  body           TEXT NOT NULL,
  read_at        TIMESTAMPTZ,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_notifications_recipient ON notifications(recipient_clerk_id, read_at);

-- §18.2 — account deletion requests (fulfilment is an institution-admin action).
CREATE TABLE IF NOT EXISTS deletion_requests (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  target_clerk_id TEXT NOT NULL,
  requested_by   TEXT NOT NULL,
  institution_id UUID,
  reason         TEXT,
  status         TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'fulfilled')),
  requested_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  fulfilled_at   TIMESTAMPTZ
);
