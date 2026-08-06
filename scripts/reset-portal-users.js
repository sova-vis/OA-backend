// DESTRUCTIVE: wipes all user accounts + portal/practice data so every account
// re-onboards from scratch. KEEPS the question-bank content (questions, topics,
// papers, grade_boundaries) and does NOT touch Clerk accounts (they simply
// re-onboard because their profile row is gone). Usage:
//   node ./scripts/reset-portal-users.js --yes
require('dotenv').config();
const { Client } = require('pg');

// User/portal data only — never the content bank.
const CANDIDATE_TABLES = [
  'submission_marks', 'submission_answers', 'submissions',
  'assignment_recipients', 'assignment_questions', 'assignments',
  'class_enrollments', 'class_co_teachers', 'classes',
  'custom_question_criteria', 'custom_questions',
  'comment_bank', 'scope_grants', 'notifications', 'activity_log', 'deletion_requests',
  'user_paper_tracking',
  'mentoring_messages', 'mentoring_meetings', 'mentoring_conversations',
  'attempts', 'graded_reports', 'practice_sessions',
  'institutions',
  'teacher_profiles', 'profiles',
];

async function main() {
  if (!process.argv.includes('--yes')) {
    console.error('Refusing to run without --yes. This DELETES all user accounts and portal data.');
    process.exit(1);
  }
  const client = new Client({ connectionString: process.env.DATABASE_URL, ssl: { rejectUnauthorized: false } });
  await client.connect();

  // Only truncate tables that actually exist.
  const { rows } = await client.query(
    `select table_name from information_schema.tables where table_schema='public' and table_name = any($1)`,
    [CANDIDATE_TABLES]
  );
  const existing = CANDIDATE_TABLES.filter((t) => rows.some((r) => r.table_name === t));

  // Report counts before.
  let profilesBefore = 0;
  try {
    const r = await client.query('select count(*)::int as c from profiles');
    profilesBefore = r.rows[0].c;
  } catch {}

  const list = existing.map((t) => `public.${t}`).join(', ');
  console.log(`Truncating ${existing.length} tables (keeping question bank + grade boundaries)...`);
  await client.query(`TRUNCATE ${list} RESTART IDENTITY CASCADE`);

  console.log(`Done. Removed ${profilesBefore} profile(s) and all portal/practice data.`);
  console.log('Clerk accounts are untouched — signing in again will show the onboarding role picker.');
  await client.end();
}

main().catch((e) => { console.error('Reset failed:', e.message); process.exit(1); });
