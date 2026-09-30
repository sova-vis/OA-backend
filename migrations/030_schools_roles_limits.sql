-- ============================================================================
-- Teacher Portal v1 (Sept 2026 spec) — Phase 0 foundation.
--
-- This is the NEW top layer that sits ABOVE the teacher: platform-owner schools,
-- per-school limits/quota, the school->consumer funnel, and a platform audit log.
-- It is deliberately ADDITIVE: none of these table names collide with the legacy
-- teacher-portal tables (classes/assignments/submissions/...), so it is safe to
-- apply before any teardown.
--
-- IMPORTANT: run-migrations.js REPLAYS every .sql on every run and has no ledger,
-- so everything here is written to be idempotent (CREATE ... IF NOT EXISTS, and
-- constraint DROP-then-ADD). Never put a standing DROP of a table that a later
-- migration recreates in a replayed file.
--
-- Locked decisions (spec §7): #1 discount is per-school; #3 priced per student
-- seat with a teacher-count cap; #5 Ask AI has its own allowance (not marking
-- quota); #6 school admin sees aggregates only unless opted in; #8 marking quota
-- is counted in question-parts.
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 0. Widen the profiles.role ENUM to the new 4-tier hierarchy.
--    The LIVE DB stores role as the Postgres ENUM `user_role` (values: student,
--    teacher, admin) — NOT text+CHECK as the reconstructed migration 000 implies.
--    Add the two new roles as enum values; 'admin' stays as the platform owner.
--    IF NOT EXISTS keeps this idempotent.
--    IMPORTANT: `ALTER TYPE ... ADD VALUE` cannot be used in the same transaction
--    it is added in, so the apply step runs these two lines FIRST (auto-committed)
--    before the transactional block below. Re-running them here is a harmless
--    no-op once the values exist.
-- ---------------------------------------------------------------------------
ALTER TYPE user_role ADD VALUE IF NOT EXISTS 'owner';
ALTER TYPE user_role ADD VALUE IF NOT EXISTS 'school_admin';

-- ---------------------------------------------------------------------------
-- 1. schools — one row per licensed school. Created by the platform owner.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schools (
  id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name                    TEXT NOT NULL,
  logo_url                TEXT,
  status                  TEXT NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active', 'suspended', 'pending')),
  licence_start           DATE,
  licence_expiry          DATE,
  -- §7#1: discounted personal rate for this school's students, applied
  -- automatically to the 6.2 upgrade prompt. 0 = no discount.
  discount_pct            NUMERIC(5,2) NOT NULL DEFAULT 0
                            CHECK (discount_pct >= 0 AND discount_pct <= 100),
  -- §7#6: school admins see aggregates only; individual scripts require this opt-in.
  allow_admin_script_view BOOLEAN NOT NULL DEFAULT FALSE,
  -- §3.4: per-school feature flags for piloting.
  feature_flags           JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_by              TEXT,          -- owner clerk_id (Supabase UUID)
  created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_schools_status ON schools(status);

-- Link a profile to its school (school_admin, teacher, or classroom student).
-- Distinct from the legacy profiles.institution_id (removed during teardown).
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS school_id UUID REFERENCES schools(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_profiles_school ON profiles(school_id);

-- ---------------------------------------------------------------------------
-- 2. school_limits — the platform-owner-set ceilings, enforced server-side.
--    One row per school (school_id is the PK).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS school_limits (
  school_id                UUID PRIMARY KEY REFERENCES schools(id) ON DELETE CASCADE,
  max_teachers             INTEGER NOT NULL DEFAULT 10,   -- §7#3 guardrail cap
  max_students_per_teacher INTEGER NOT NULL DEFAULT 60,
  max_classes_per_teacher  INTEGER NOT NULL DEFAULT 15,
  max_students_total       INTEGER NOT NULL DEFAULT 300,  -- §7#3 the billed seat ceiling
  -- §7#8: marking quota is counted in QUESTION-PARTS per period.
  marking_quota_units      INTEGER NOT NULL DEFAULT 5000,
  quota_period             TEXT NOT NULL DEFAULT 'month'
                             CHECK (quota_period IN ('month', 'term')),
  -- §7#5: Ask AI gets its OWN allowance, separate from marking quota.
  askai_allowance          INTEGER NOT NULL DEFAULT 5000,
  -- §3.2: which syllabuses this school may access (empty = all entitled).
  subject_entitlements     TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  -- §5.9: per-school storage cap for teacher notes/resources, in megabytes.
  storage_cap_mb           INTEGER NOT NULL DEFAULT 2048,
  updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- 3. marking_ledger — one row per marking call (§3.3 "logged against the
--    school for reconciliation"). Units are question-parts (§7#8). submission_id
--    is loose (no FK) because the submissions table is rebuilt in Phase 1.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS marking_ledger (
  id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  school_id          UUID REFERENCES schools(id) ON DELETE CASCADE,
  teacher_clerk_id   TEXT,
  student_clerk_id   TEXT,
  submission_id      UUID,
  kind               TEXT NOT NULL DEFAULT 'marking'
                       CHECK (kind IN ('marking', 'askai')),  -- meter both, separately
  units              INTEGER NOT NULL DEFAULT 1,   -- question-parts (marking) / calls (askai)
  cost_estimate      NUMERIC(10,4),
  model              TEXT,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_marking_ledger_school_time ON marking_ledger(school_id, created_at);
CREATE INDEX IF NOT EXISTS idx_marking_ledger_kind ON marking_ledger(school_id, kind, created_at);

-- ---------------------------------------------------------------------------
-- 4. quota_events — §3.3 threshold notifications (80% warn, 100% queue).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS quota_events (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  school_id     UUID REFERENCES schools(id) ON DELETE CASCADE,
  kind          TEXT NOT NULL CHECK (kind IN ('warn_80', 'hit_100', 'queued')),
  period_start  DATE NOT NULL,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_quota_events_school ON quota_events(school_id, created_at);

-- ---------------------------------------------------------------------------
-- 5. funnel_events — §6.2 school->consumer funnel. Every time the practice
--    paywall prompt is shown, and every time it converts.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS funnel_events (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  school_id         UUID REFERENCES schools(id) ON DELETE SET NULL,
  student_clerk_id  TEXT NOT NULL,
  kind              TEXT NOT NULL CHECK (kind IN ('prompt_shown', 'converted')),
  context           TEXT,            -- where in the app the prompt fired
  created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_funnel_events_school ON funnel_events(school_id, kind, created_at);

-- ---------------------------------------------------------------------------
-- 6. audit_log — §3.4 / §5.5 platform audit trail: account changes, support
--    impersonation, and (later) teacher mark overrides. Replaces the legacy
--    activity_log (dropped in teardown).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  actor_clerk_id TEXT,
  actor_role   TEXT,
  action       TEXT NOT NULL,
  target_type  TEXT,
  target_id    TEXT,
  before       JSONB,
  after        JSONB,
  school_id    UUID REFERENCES schools(id) ON DELETE SET NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_log_school ON audit_log(school_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_actor ON audit_log(actor_clerk_id, created_at);

-- ---------------------------------------------------------------------------
-- 7. Lock the new tables to the backend. Enabling RLS with NO policies denies
--    the public anon/authenticated keys entirely, while the backend (which uses
--    the service_role key) bypasses RLS and keeps full access. These tables are
--    never queried directly from the browser — only through the API.
-- ---------------------------------------------------------------------------
ALTER TABLE schools        ENABLE ROW LEVEL SECURITY;
ALTER TABLE school_limits  ENABLE ROW LEVEL SECURITY;
ALTER TABLE marking_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE quota_events   ENABLE ROW LEVEL SECURITY;
ALTER TABLE funnel_events  ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log      ENABLE ROW LEVEL SECURITY;

-- PostgREST caches the schema; reload it so the new tables/columns are visible.
NOTIFY pgrst, 'reload schema';
