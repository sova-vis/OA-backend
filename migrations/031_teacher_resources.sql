-- Teacher Portal §5.9 — notes & resources.
--
-- A teacher uploads a small file (base64 data URL, ≤ the 2 MB request limit) OR
-- links an external resource (Drive/Dropbox/etc. for anything larger), tags it to
-- a topic, and publishes it to one of their classes. Students enrolled in that
-- class see the published resources. Total storage is capped per school
-- (school_limits.storage_cap_mb; a default applies to standalone teachers).
--
-- Backend-only: RLS enabled with NO policies, so the public anon key is denied and
-- only the service_role backend can read/write (students fetch via the API).

CREATE TABLE IF NOT EXISTS teacher_resources (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  owner_clerk_id   TEXT NOT NULL,
  class_id         UUID NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
  title            TEXT NOT NULL,
  topic            TEXT,
  kind             TEXT NOT NULL DEFAULT 'file' CHECK (kind IN ('file', 'link')),
  url              TEXT,                 -- external link when kind = 'link'
  data             TEXT,                 -- base64 data URL when kind = 'file'
  mime             TEXT,
  size_bytes       INTEGER NOT NULL DEFAULT 0,
  published        BOOLEAN NOT NULL DEFAULT TRUE,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_teacher_resources_class ON teacher_resources(class_id, published);
CREATE INDEX IF NOT EXISTS idx_teacher_resources_owner ON teacher_resources(owner_clerk_id);

ALTER TABLE teacher_resources ENABLE ROW LEVEL SECURITY;
