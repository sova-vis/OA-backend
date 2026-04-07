CREATE TABLE IF NOT EXISTS teacher_profiles (
  clerk_id TEXT PRIMARY KEY,
  headline TEXT,
  bio TEXT,
  subjects TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  availability JSONB NOT NULL DEFAULT '[]'::JSONB,
  meeting_provider TEXT NOT NULL DEFAULT 'google_meet',
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mentoring_conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  student_clerk_id TEXT NOT NULL,
  teacher_clerk_id TEXT NOT NULL,
  is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (student_clerk_id, teacher_clerk_id)
);

CREATE TABLE IF NOT EXISTS mentoring_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES mentoring_conversations(id) ON DELETE CASCADE,
  sender_clerk_id TEXT NOT NULL,
  sender_role TEXT NOT NULL CHECK (sender_role IN ('student', 'teacher')),
  body TEXT NOT NULL,
  is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mentoring_meetings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  student_clerk_id TEXT NOT NULL,
  teacher_clerk_id TEXT NOT NULL,
  requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  start_time TIMESTAMPTZ,
  end_time TIMESTAMPTZ,
  status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'scheduled', 'completed', 'cancelled', 'declined')) DEFAULT 'pending',
  agenda TEXT NOT NULL,
  note_from_student TEXT,
  teacher_notes TEXT,
  meeting_link TEXT,
  provider TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_mentoring_conversations_student ON mentoring_conversations(student_clerk_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_mentoring_conversations_teacher ON mentoring_conversations(teacher_clerk_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_mentoring_messages_conversation ON mentoring_messages(conversation_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_mentoring_meetings_teacher_time ON mentoring_meetings(teacher_clerk_id, start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_mentoring_meetings_student ON mentoring_meetings(student_clerk_id, requested_at DESC);

ALTER TABLE profiles
  ADD COLUMN IF NOT EXISTS onboarding_complete BOOLEAN NOT NULL DEFAULT FALSE;
