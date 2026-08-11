// DESTRUCTIVE (test reset): clears all assignments, submissions, answers and
// marks/reviews so the assign → submit → review flow can be tested fresh.
// KEEPS accounts, classes, enrolments and custom questions. Usage:
//   node ./scripts/clear-assignments.js --yes
require('dotenv').config();
const { Client } = require('pg');

// Assignment + submission domain only. Order is child → parent; CASCADE mops up.
const TABLES = [
  'submission_marks',
  'submission_answers',
  'submissions',
  'assignment_recipients',
  'assignment_questions',
  'assignments',
];

async function main() {
  if (!process.argv.includes('--yes')) {
    console.error('Refusing to run without --yes. This DELETES all assignments, submissions and reviews.');
    process.exit(1);
  }
  const client = new Client({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
  await client.connect();

  const { rows } = await client.query(
    `select table_name from information_schema.tables where table_schema='public' and table_name = any($1)`,
    [TABLES]
  );
  const existing = TABLES.filter((t) => rows.some((r) => r.table_name === t));

  const count = async (t) => { try { return (await client.query(`select count(*)::int c from public.${t}`)).rows[0].c; } catch { return 0; } };
  const before = {};
  for (const t of existing) before[t] = await count(t);

  const list = existing.map((t) => `public.${t}`).join(', ');
  console.log(`Truncating: ${existing.join(', ')}`);
  await client.query(`TRUNCATE ${list} RESTART IDENTITY CASCADE`);

  console.log('Cleared (rows removed):');
  for (const t of existing) console.log(`  ${t}: ${before[t]}`);
  console.log('Kept: profiles, classes, class_enrollments, custom_questions.');
  await client.end();
}

main().catch((e) => { console.error('Clear failed:', e.message); process.exit(1); });
