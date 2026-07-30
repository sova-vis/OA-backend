/**
 * Backfill student modelling from Supabase Storage blobs into Postgres
 * (Architecture report · Phase 2 · migration 005).
 *
 *   attempts/<clerkId>.json            -> public.attempts
 *   s/<clerkId>/<paperKey>.json        -> public.practice_sessions
 *   s/<clerkId>/<paperKey>.json.report -> public.graded_reports
 *
 * Safe to run repeatedly: every write is an upsert keyed on a stable id, so a
 * second run changes nothing. It only READS Storage — no blob is modified or
 * deleted, so Storage remains a complete fallback. Run AFTER applying
 * migrations/005_student_modelling.sql and BEFORE setting
 * STUDENT_DATA_SOURCE=postgres.
 *
 *   node scripts/backfill-student-data.js
 */
const path = require('path');
const dotenv = require('dotenv');
const { createClient } = require('@supabase/supabase-js');

dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_ROLE_KEY =
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY || '';
const BUCKET = 'practice-progress';

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in OA-backend/.env');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { autoRefreshToken: false, persistSession: false },
});

const num = (v) => {
  const n = typeof v === 'number' ? v : Number.parseFloat(String(v));
  return Number.isFinite(n) ? n : 0;
};

/** List every file directly under a prefix, following pagination. Folders (id null) are returned too. */
async function listAll(prefix) {
  const out = [];
  const pageSize = 1000;
  let offset = 0;
  for (;;) {
    const { data, error } = await supabase.storage
      .from(BUCKET)
      .list(prefix, { limit: pageSize, offset, sortBy: { column: 'name', order: 'asc' } });
    if (error) throw error;
    if (!data || data.length === 0) break;
    out.push(...data);
    if (data.length < pageSize) break;
    offset += pageSize;
  }
  return out;
}

async function downloadJson(fullPath) {
  const { data, error } = await supabase.storage.from(BUCKET).download(fullPath);
  if (error || !data) return null;
  try {
    return JSON.parse(Buffer.from(await data.arrayBuffer()).toString('utf8'));
  } catch {
    return null;
  }
}

async function backfillAttempts() {
  const files = (await listAll('attempts')).filter((e) => e.id && e.name.endsWith('.json'));
  let students = 0;
  let rows = 0;
  for (const file of files) {
    const clerkId = file.name.replace(/\.json$/, '');
    const doc = await downloadJson(`attempts/${file.name}`);
    const items = doc && Array.isArray(doc.items) ? doc.items : [];
    if (!items.length) continue;
    const mapped = items
      .filter((a) => a && a.questionId)
      .map((a) => ({
        id: String(a.id || `${a.questionId}_${a.at}`),
        clerk_id: clerkId,
        question_id: String(a.questionId),
        subject: a.subject || null,
        topic: a.topic || null,
        theme: a.theme || null,
        type: a.type === 'mcq' ? 'mcq' : 'structured',
        verdict: a.verdict || 'weak',
        earned: num(a.earned),
        max: num(a.max),
        reason: a.reason || null,
        year: a.year || null,
        session: a.session || null,
        paper: a.paper || null,
        variant: a.variant || null,
        created_at: a.at || new Date().toISOString(),
      }));
    // Upsert in chunks to stay under payload limits.
    for (let i = 0; i < mapped.length; i += 500) {
      const chunk = mapped.slice(i, i + 500);
      const { error } = await supabase.from('attempts').upsert(chunk, { onConflict: 'id' });
      if (error) throw error;
      rows += chunk.length;
    }
    students += 1;
  }
  return { students, rows };
}

async function backfillSessions() {
  const clerkFolders = (await listAll('s')).filter((e) => !e.id); // folders have id null
  let sessions = 0;
  let reports = 0;
  for (const folder of clerkFolders) {
    const clerkId = folder.name;
    const files = (await listAll(`s/${clerkId}`)).filter((e) => e.id && e.name.endsWith('.json'));
    for (const file of files) {
      const doc = await downloadJson(`s/${clerkId}/${file.name}`);
      if (!doc || !doc.paperKey) continue;
      const { error } = await supabase.from('practice_sessions').upsert(
        {
          clerk_id: clerkId,
          paper_key: doc.paperKey,
          subject: doc.subject || null,
          year: doc.year || null,
          session: doc.session || null,
          paper: doc.paper || null,
          variant: doc.variant || null,
          is_mcq: Boolean(doc.isMcq),
          solve_mode: doc.solveMode || null,
          status: doc.status || null,
          answers: doc.answers || null,
          uploads: doc.uploads || null,
          answered_count: doc.answeredCount ?? null,
          total_count: doc.totalCount ?? null,
          timer_duration_seconds: doc.timerDurationSeconds ?? null,
          timer_elapsed_seconds: doc.timerElapsedSeconds ?? null,
          report: doc.report || null,
          started_at: doc.startedAt || null,
          updated_at: doc.updatedAt || new Date().toISOString(),
        },
        { onConflict: 'clerk_id,paper_key' },
      );
      if (error) throw error;
      sessions += 1;

      if (doc.report) {
        const r = doc.report;
        const { error: reportError } = await supabase.from('graded_reports').upsert(
          {
            clerk_id: clerkId,
            paper_key: doc.paperKey,
            earned: num(r.earned),
            total: num(r.total),
            percent: num(r.percent),
            grade: r.grade || null,
            model: r.model || null,
            graded_at: r.gradedAt || new Date().toISOString(),
            report: r,
          },
          { onConflict: 'clerk_id,paper_key' },
        );
        if (reportError) throw reportError;
        reports += 1;
      }
    }
  }
  return { sessions, reports };
}

async function main() {
  console.log(`Backfilling student modelling from Storage bucket "${BUCKET}" into Postgres...`);
  const a = await backfillAttempts();
  console.log(`  attempts:         ${a.rows} rows from ${a.students} students`);
  const s = await backfillSessions();
  console.log(`  practice_sessions:${s.sessions}`);
  console.log(`  graded_reports:   ${s.reports}`);
  console.log('Backfill complete. Verify counts, then set STUDENT_DATA_SOURCE=postgres to cut reads over.');
}

main().catch((error) => {
  console.error('Backfill failed:', error.message || error);
  process.exit(1);
});
