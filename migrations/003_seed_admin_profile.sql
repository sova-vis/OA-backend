-- Seed and normalize admin profile row in Supabase.
-- Note: password should remain in Clerk auth, not in this profile table.

UPDATE profiles
SET role = 'admin',
    full_name = COALESCE(NULLIF(full_name, ''), 'Admin')
WHERE email = 'sovavis2025@gmail.com';

INSERT INTO profiles (clerk_id, email, full_name, role)
SELECT 'admin-seed-placeholder', 'sovavis2025@gmail.com', 'Admin', 'admin'
WHERE NOT EXISTS (
  SELECT 1 FROM profiles WHERE email = 'sovavis2025@gmail.com'
);
