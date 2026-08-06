-- =============================================================================
-- Teacher Portal — teacher-provisioned student accounts (spec §4.3)
-- Many schools don't allow students to self-register with personal email, so a
-- teacher provisions accounts: system generates a username + one-time password,
-- the student sets a new password on first login. Provisioned accounts may have
-- no email.
--
-- These students have no Clerk identity, so they get a synthetic clerk_id of the
-- form 'prov_<token>' — this keeps enrolments and (future) attempts keyed
-- uniformly on clerk_id across self-registered and provisioned students.
--
-- Idempotent (re-run safe).
-- =============================================================================

ALTER TABLE profiles ADD COLUMN IF NOT EXISTS username             TEXT;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS password_hash        TEXT;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS must_change_password BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS is_provisioned       BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE profiles ADD COLUMN IF NOT EXISTS provisioned_by       TEXT;  -- teacher clerk_id who created it

-- Usernames are unique where present (self-registered students have none).
CREATE UNIQUE INDEX IF NOT EXISTS idx_profiles_username
  ON profiles(lower(username)) WHERE username IS NOT NULL;
