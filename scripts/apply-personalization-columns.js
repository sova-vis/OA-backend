// Applies migration 017: per-level subjects + active level on profiles, so
// personalization is server-backed (follows the account to any device).
// Idempotent. Usage: node ./scripts/apply-personalization-columns.js
require('dotenv').config();
const { Client } = require('pg');

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
  await client.connect();
  await client.query(`ALTER TABLE profiles ADD COLUMN IF NOT EXISTS subjects_by_level JSONB NOT NULL DEFAULT '{}'::jsonb`);
  await client.query(`ALTER TABLE profiles ADD COLUMN IF NOT EXISTS active_level TEXT`);
  const { rows } = await client.query(
    `SELECT column_name, data_type FROM information_schema.columns
     WHERE table_name = 'profiles' AND column_name IN ('subjects_by_level','active_level')
     ORDER BY column_name`
  );
  console.log('profiles now has:', rows.map((r) => `${r.column_name}:${r.data_type}`).join(', ') || '(none — check!)');
  // Make PostgREST aware of the new columns immediately.
  try { await client.query(`NOTIFY pgrst, 'reload schema'`); } catch {}
  await client.end();
  console.log('Done.');
}
main().catch((e) => { console.error('Failed:', e.message); process.exit(1); });
