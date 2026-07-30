import { supabase } from './supabase';
import type { AttemptRecord, AttemptVerdict, PracticeProgressDoc, PracticeReport } from './practiceStore';

/**
 * Postgres persistence for student modelling (Architecture report · Phase 2).
 *
 * The blob store in practiceStore.ts remains the source of truth. This module
 * adds a parallel relational copy in the `attempts`, `graded_reports` and
 * `practice_sessions` tables (migration 005) so the aggregate queries behind
 * Predicted Grade / Weakness Map / SRS / cohort analytics become single indexed
 * queries instead of parsing every student's JSON on every request — and so the
 * attempts append stops losing data under concurrency.
 *
 * Rollout is controlled by two env flags so it can never break the live app:
 *   STUDENT_DATA_DUAL_WRITE (default "true")  — mirror writes into Postgres.
 *   STUDENT_DATA_SOURCE     (default "storage")— where reads come from. Flip to
 *                                                "postgres" only after backfill.
 *
 * Every write here is BEST-EFFORT: callers wrap it so a Postgres failure (e.g.
 * the migration hasn't been run yet) logs and is swallowed, never surfacing to
 * the student.
 */

export function studentDataSource(): 'storage' | 'postgres' {
  return (process.env.STUDENT_DATA_SOURCE || 'storage').trim().toLowerCase() === 'postgres'
    ? 'postgres'
    : 'storage';
}

export function pgDualWriteEnabled(): boolean {
  return (process.env.STUDENT_DATA_DUAL_WRITE || 'true').trim().toLowerCase() !== 'false';
}

/**
 * Throttled warning for best-effort Postgres failures. Before migration 005 is
 * run these fire on every write; throttling to once per 60s per context keeps a
 * mis-ordered deploy from flooding the logs while still surfacing a persistent
 * problem. Never throws.
 */
const lastWarnedAt = new Map<string, number>();
export function warnPgOnce(context: string, error: unknown): void {
  const now = Date.now();
  const last = lastWarnedAt.get(context) ?? 0;
  if (now - last < 60_000) return;
  lastWarnedAt.set(context, now);
  console.warn(`[studentData] ${context} (non-fatal; Storage remains source of truth):`, (error as Error)?.message || error);
}

/** Upper bound on rows returned by a single attempts read (mirrors blob cap intent). */
const PG_ATTEMPTS_LIMIT = 5000;

const VALID_VERDICTS: AttemptVerdict[] = ['correct', 'partial', 'weak', 'unanswered', 'incorrect'];

// ---------------------------------------------------------------------------
// attempts
// ---------------------------------------------------------------------------
export async function pgUpsertAttempts(clerkId: string, items: AttemptRecord[]): Promise<void> {
  if (!items.length) return;
  const rows = items.map((a) => ({
    id: a.id,
    clerk_id: clerkId,
    question_id: a.questionId,
    subject: a.subject || null,
    topic: a.topic || null,
    theme: a.theme ?? null,
    type: a.type,
    verdict: a.verdict,
    earned: a.earned,
    max: a.max,
    reason: a.reason || null,
    year: a.year ?? null,
    session: a.session ?? null,
    paper: a.paper ?? null,
    variant: a.variant ?? null,
    created_at: a.at,
  }));
  // Independent per-row upsert on the app-assigned id → concurrent writers can
  // never clobber each other's attempts (the blob append could).
  const { error } = await supabase.from('attempts').upsert(rows, { onConflict: 'id' });
  if (error) throw error;
}

export async function pgReadAttempts(clerkId: string): Promise<AttemptRecord[]> {
  const { data, error } = await supabase
    .from('attempts')
    .select('id,question_id,subject,topic,theme,type,verdict,earned,max,reason,year,session,paper,variant,created_at')
    .eq('clerk_id', clerkId)
    .order('created_at', { ascending: false })
    .limit(PG_ATTEMPTS_LIMIT);
  if (error) throw error;
  return (data ?? []).map((r) => {
    const verdict = VALID_VERDICTS.includes(r.verdict as AttemptVerdict)
      ? (r.verdict as AttemptVerdict)
      : 'weak';
    const num = (v: unknown) => {
      const n = typeof v === 'number' ? v : Number.parseFloat(String(v));
      return Number.isFinite(n) ? n : 0;
    };
    const rec: AttemptRecord = {
      id: String(r.id),
      questionId: String(r.question_id),
      subject: r.subject ?? '',
      topic: r.topic ?? 'Uncategorised',
      theme: r.theme ?? undefined,
      type: r.type === 'mcq' ? 'mcq' : 'structured',
      verdict,
      earned: num(r.earned),
      max: num(r.max),
      reason: r.reason ?? '',
      year: r.year ?? undefined,
      session: r.session ?? undefined,
      paper: r.paper ?? undefined,
      variant: r.variant ?? undefined,
      at: typeof r.created_at === 'string' ? r.created_at : new Date(r.created_at as string).toISOString(),
    };
    return rec;
  });
}

// ---------------------------------------------------------------------------
// practice_sessions (+ graded_reports when a report is present)
// ---------------------------------------------------------------------------
export async function pgUpsertSession(clerkId: string, doc: PracticeProgressDoc): Promise<void> {
  const { error } = await supabase.from('practice_sessions').upsert(
    {
      clerk_id: clerkId,
      paper_key: doc.paperKey,
      subject: doc.subject || null,
      year: doc.year || null,
      session: doc.session || null,
      paper: doc.paper || null,
      variant: doc.variant || null,
      is_mcq: doc.isMcq,
      solve_mode: doc.solveMode,
      status: doc.status,
      answers: doc.answers,
      uploads: doc.uploads,
      answered_count: doc.answeredCount,
      total_count: doc.totalCount,
      timer_duration_seconds: doc.timerDurationSeconds,
      timer_elapsed_seconds: doc.timerElapsedSeconds,
      report: doc.report,
      started_at: doc.startedAt || null,
      updated_at: doc.updatedAt || new Date().toISOString(),
    },
    { onConflict: 'clerk_id,paper_key' },
  );
  if (error) throw error;

  if (doc.report) {
    const r = doc.report;
    const { error: reportError } = await supabase.from('graded_reports').upsert(
      {
        clerk_id: clerkId,
        paper_key: doc.paperKey,
        earned: r.earned,
        total: r.total,
        percent: r.percent,
        grade: r.grade || null,
        model: r.model || null,
        graded_at: r.gradedAt || new Date().toISOString(),
        report: r,
      },
      { onConflict: 'clerk_id,paper_key' },
    );
    if (reportError) throw reportError;
  }
}

export async function pgReadSession(clerkId: string, paperKey: string): Promise<PracticeProgressDoc | null> {
  const { data, error } = await supabase
    .from('practice_sessions')
    .select('*')
    .eq('clerk_id', clerkId)
    .eq('paper_key', paperKey)
    .maybeSingle();
  if (error) throw error;
  if (!data) return null;
  const doc: PracticeProgressDoc = {
    paperKey: data.paper_key,
    subject: data.subject ?? '',
    year: data.year ?? '',
    session: data.session ?? '',
    paper: data.paper ?? '',
    variant: data.variant ?? '',
    isMcq: Boolean(data.is_mcq),
    solveMode: data.solve_mode === 'handwritten' ? 'handwritten' : 'digital',
    status: data.status === 'completed' ? 'completed' : 'in_progress',
    answers:
      data.answers && typeof data.answers === 'object'
        ? { mcq: data.answers.mcq ?? {}, parts: data.answers.parts ?? {} }
        : { mcq: {}, parts: {} },
    uploads: Array.isArray(data.uploads) ? data.uploads : [],
    answeredCount: Number(data.answered_count) || 0,
    totalCount: Number(data.total_count) || 0,
    timerDurationSeconds: Number(data.timer_duration_seconds) || 0,
    timerElapsedSeconds: Number(data.timer_elapsed_seconds) || 0,
    report: (data.report as PracticeReport | null) ?? null,
    startedAt: data.started_at ?? '',
    updatedAt: data.updated_at ?? new Date().toISOString(),
  };
  return doc;
}
