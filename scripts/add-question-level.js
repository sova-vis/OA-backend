// Adds a `level` column to the question bank so O-Level and A-Level content stay
// separate. Existing rows default to 'olevel'. Idempotent. Usage:
//   node ./scripts/add-question-level.js
require('dotenv').config();
const { Client } = require('pg');

async function main() {
  const client = new Client({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
  await client.connect();
  for (const table of ['questions', 'topics']) {
    try {
      await client.query(`ALTER TABLE public.${table} ADD COLUMN IF NOT EXISTS level text NOT NULL DEFAULT 'olevel'`);
      await client.query(`UPDATE public.${table} SET level = 'olevel' WHERE level IS NULL`);
      await client.query(`CREATE INDEX IF NOT EXISTS ${table}_level_idx ON public.${table} (level)`);
      const { rows } = await client.query(`SELECT level, count(*)::int c FROM public.${table} GROUP BY level ORDER BY level`);
      console.log(`${table}.level →`, rows.map((r) => `${r.level}:${r.c}`).join(', ') || '(no rows)');
    } catch (e) {
      console.log(`${table}: ${e.message}`);
    }
  }
  // Make PostgREST aware of the new column immediately.
  try { await client.query(`NOTIFY pgrst, 'reload schema'`); } catch {}
  await client.end();
  console.log('Done. All existing questions are tagged olevel; import A-level with --level=alevel.');
}
main().catch((e) => { console.error('Failed:', e.message); process.exit(1); });
