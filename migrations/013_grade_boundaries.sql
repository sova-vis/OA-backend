-- =============================================================================
-- Teacher Portal — grade boundaries (spec §12.6)
-- Boundaries are stored as DATA per syllabus, series, and variant — never
-- hardcoded. The UI states which session's boundaries it used. Populated by
-- content ingestion (a blocking dependency); until then predicted grades fall
-- back to "boundaries not configured".
--
-- Idempotent (re-run safe).
-- =============================================================================

CREATE TABLE IF NOT EXISTS grade_boundaries (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  syllabus_code TEXT NOT NULL,
  series        TEXT,                    -- March / June / November
  year          INT,
  variant       TEXT,
  grade         TEXT NOT NULL,           -- 'A*','A','B','C','D','E','U'
  min_percent   NUMERIC NOT NULL,        -- rolling-average % at/above which this grade is reached
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (syllabus_code, series, year, variant, grade)
);

CREATE INDEX IF NOT EXISTS idx_grade_boundaries_lookup ON grade_boundaries(syllabus_code, series, year);
