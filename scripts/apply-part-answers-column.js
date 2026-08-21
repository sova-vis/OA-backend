// Applies migration 018: per-part answers on submission_answers. Idempotent.
// Usage: node ./scripts/apply-part-answers-column.js
require('dotenv').config();
const { Client } = require('pg');

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
  await client.connect();
  await client.query(`ALTER TABLE submission_answers ADD COLUMN IF NOT EXISTS part_answers JSONB NOT NULL DEFAULT '{}'::jsonb`);
  const { rows } = await client.query(
    `SELECT column_name, data_type FROM information_schema.columns WHERE table_name='submission_answers' AND column_name='part_answers'`
  );
  console.log('submission_answers.part_answers:', rows.map((r) => r.data_type).join(', ') || '(missing — check!)');
  try { await client.query(`NOTIFY pgrst, 'reload schema'`); } catch {}
  await client.end();
  console.log('Done.');
}
main().catch((e) => { console.error('Failed:', e.message); process.exit(1); });
