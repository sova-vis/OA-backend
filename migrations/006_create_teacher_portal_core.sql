-- =============================================================================
-- Teacher Portal — Phase 0 core data model
-- Spec: teacher-portal-specification.md §1.2 (institution stub), §3.1 (classes),
--       §3.2 (join code), §3.5 (co-teaching), §4.4 (enrolment requests)
--
-- Idempotent: this file is re-run by scripts/run-migrations.js on every deploy,
-- so every statement is guarded with IF NOT EXISTS / DO-block existence checks.
-- =============================================================================

-- ----------------------------------------------------------------------------
-- Institutions. A school always exists in data before it signs up: on teacher
-- registration we mint an `unclaimed` stub (§1.2) so every class can carry a
-- non-null institution_id and later "roll in" with no migration when claimed.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS institutions (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name                TEXT NOT NULL,
  -- unclaimed: stub created by a teacher, no admin, no seat limit (§1.2)
  -- claimed:   a school has claimed it and verification is pending/complete
  -- active:    fully onboarded paying institution
  status              TEXT NOT NULL DEFAULT 'unclaimed'
                        CHECK (status IN ('unclaimed', 'claimed', 'active')),
  -- Normalised lowercase name used to detect "school already exists" on
  -- registration (§1.1 edge case) without fuzzy matching at write time.
  name_normalized     TEXT,
  domain              TEXT,               -- email domain for claim verification (§1.3)
  country             TEXT,
  address             TEXT,
  contact_email       TEXT,
  logo_url            TEXT,
  admin_clerk_id      TEXT,               -- null until claimed (§1.2)
  seat_limit_teachers INT,               -- null = unlimited (unclaimed stubs)
  seat_limit_students INT,
  academic_year_start DATE,               -- drives report presets (§17.3)
  academic_year_end   DATE,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_institutions_status     ON institutions(status);
CREATE INDEX IF NOT EXISTS idx_institutions_name_norm  ON institutions(name_normalized);

-- ----------------------------------------------------------------------------
-- Profile extensions for teacher onboarding (§1.1, §1.5). The base `profiles`
-- table (migration 000) already has clerk_id, role, selected_subjects, level.
-- ----------------------------------------------------------------------------
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS institution_id UUID REFERENCES institutions(id);
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS levels         TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];  -- ['O','A']
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS syllabus_codes TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];  -- controlled list e.g. '0620'
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS school_name    TEXT;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS timezone       TEXT;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS photo_url      TEXT;

CREATE INDEX IF NOT EXISTS idx_profiles_institution ON profiles(institution_id);

-- ----------------------------------------------------------------------------
-- Classes (§3.1). A class belongs to exactly one syllabus code and one
-- institution (never null — §1.2/§14.1). Owner is a teacher (clerk_id).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS classes (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  institution_id    UUID NOT NULL REFERENCES institutions(id),
  owner_clerk_id    TEXT NOT NULL,
  name              TEXT NOT NULL,
  subject           TEXT NOT NULL,
  syllabus_code     TEXT NOT NULL,        -- controlled list, drives available questions (§3.1)
  level             TEXT NOT NULL CHECK (level IN ('O', 'A')),
  year_group        TEXT,
  description       TEXT,
  -- Enrolment (§3.2): 6-char alphanumeric, stored uppercase, case-insensitive
  -- on entry. Unique while enabled; regeneration issues a new code.
  join_code         TEXT,
  join_enabled      BOOLEAN NOT NULL DEFAULT TRUE,
  auto_approve_joins BOOLEAN NOT NULL DEFAULT FALSE,   -- §4.4 optional setting
  exam_series       TEXT,                 -- March / June / November, drives grade boundaries (§17.3)
  archived_at       TIMESTAMPTZ,          -- §3.4 archive (soft); hard delete not available to teachers
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_classes_owner       ON classes(owner_clerk_id);
CREATE INDEX IF NOT EXISTS idx_classes_institution ON classes(institution_id);
-- A join code is only unique among classes where it is currently enabled; a
-- disabled/regenerated code may be reused elsewhere. Partial unique index.
CREATE UNIQUE INDEX IF NOT EXISTS idx_classes_join_code_active
  ON classes(join_code) WHERE join_enabled AND join_code IS NOT NULL;

-- ----------------------------------------------------------------------------
-- Co-teaching (§3.5). A second grantee gets can_grade=true; an observer false.
-- Independent of institution admin.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS class_co_teachers (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  class_id          UUID NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
  teacher_clerk_id  TEXT NOT NULL,
  can_grade         BOOLEAN NOT NULL DEFAULT TRUE,
  invited_email     TEXT,                 -- email used to grant, for display before first login
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (class_id, teacher_clerk_id)
);

CREATE INDEX IF NOT EXISTS idx_co_teachers_teacher ON class_co_teachers(teacher_clerk_id);

-- ----------------------------------------------------------------------------
-- Class enrolments (§4.4). Students join by code (pending → approved) or are
-- teacher-provisioned/added directly (active). Removal preserves attempts
-- (§4.5) so it is a soft status change, never a delete.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS class_enrollments (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  class_id          UUID NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
  student_clerk_id  TEXT NOT NULL,
  status            TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'active', 'rejected', 'removed')),
  joined_via        TEXT NOT NULL DEFAULT 'code'
                      CHECK (joined_via IN ('code', 'link', 'qr', 'provision', 'manual', 'transfer')),
  requested_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  approved_at       TIMESTAMPTZ,
  removed_at        TIMESTAMPTZ,
  -- §4.4: a rejected student may not retry until this time (24h after rejection)
  retry_blocked_until TIMESTAMPTZ,
  UNIQUE (class_id, student_clerk_id)
);

CREATE INDEX IF NOT EXISTS idx_enrollments_class   ON class_enrollments(class_id, status);
CREATE INDEX IF NOT EXISTS idx_enrollments_student ON class_enrollments(student_clerk_id, status);
