CREATE TABLE IF NOT EXISTS user_paper_tracking (
  clerk_id TEXT NOT NULL,
  paper_id TEXT NOT NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  view_url TEXT,
  download_url TEXT,
  embed_url TEXT,
  saved_at TIMESTAMPTZ NOT NULL,
  statuses TEXT[] NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (clerk_id, paper_id)
);

CREATE INDEX IF NOT EXISTS idx_user_paper_tracking_clerk_updated
  ON user_paper_tracking(clerk_id, updated_at DESC);
