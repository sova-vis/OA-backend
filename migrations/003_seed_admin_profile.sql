-- Admin seeding intentionally DISABLED.
-- Previously this seeded/normalised an admin profile for sovavis2025@gmail.com,
-- which re-created that admin account on every migration run. Removed so no
-- admin account exists by default. To create an admin deliberately, use
-- scripts/promote-user-admin.js <email> after that user has signed in once.
SELECT 1;
