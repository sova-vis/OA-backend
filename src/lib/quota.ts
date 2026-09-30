/**
 * Per-school limits + marking/Ask-AI quota accounting (spec §3.2 limits, §3.3
 * quota behaviour). Everything here is enforced SERVER-SIDE — the spec is explicit
 * that limits protect margin and must not be UI-only.
 *
 * Quota rule (§3.3): warn at 80%, and at 100% QUEUE further marking rather than
 * failing it — never hard-stop a class mid-lesson. So callers check the returned
 * state and decide to queue; they must not throw on 'full'.
 */
import { supabase } from './supabase';

export interface SchoolLimits {
  school_id: string;
  max_teachers: number;
  max_students_per_teacher: number;
  max_classes_per_teacher: number;
  max_students_total: number;
  marking_quota_units: number;
  quota_period: 'month' | 'term';
  askai_allowance: number;
  subject_entitlements: string[];
  storage_cap_mb: number;
}

export async function getLimits(schoolId: string): Promise<SchoolLimits | null> {
  const { data } = await supabase.from('school_limits').select('*').eq('school_id', schoolId).maybeSingle();
  return (data as SchoolLimits | null) ?? null;
}

/**
 * Start of the current quota period. 'month' = 1st of this month (UTC). 'term' =
 * 4-month terms anchored to Jan/May/Sep (Cambridge-ish) — good enough to meter,
 * refined when term dates are modelled.
 */
export function periodStart(period: 'month' | 'term', now = new Date()): Date {
  const y = now.getUTCFullYear();
  if (period === 'term') {
    const m = now.getUTCMonth();
    const termStartMonth = m < 4 ? 0 : m < 8 ? 4 : 8;
    return new Date(Date.UTC(y, termStartMonth, 1));
  }
  return new Date(Date.UTC(y, now.getUTCMonth(), 1));
}

async function sumLedger(schoolId: string, kind: 'marking' | 'askai', since: Date): Promise<number> {
  const { data } = await supabase
    .from('marking_ledger')
    .select('units')
    .eq('school_id', schoolId)
    .eq('kind', kind)
    .gte('created_at', since.toISOString());
  return ((data as { units: number }[] | null) ?? []).reduce((a, r) => a + (r.units || 0), 0);
}

export interface QuotaStatus {
  used: number;
  quota: number;
  pct: number;
  state: 'ok' | 'warn' | 'full';
  periodStart: string;
}

export async function markingQuotaStatus(schoolId: string): Promise<QuotaStatus | null> {
  const limits = await getLimits(schoolId);
  if (!limits) return null;
  const since = periodStart(limits.quota_period);
  const used = await sumLedger(schoolId, 'marking', since);
  const quota = limits.marking_quota_units;
  const pct = quota > 0 ? Math.round((used / quota) * 100) : 0;
  const state: QuotaStatus['state'] = pct >= 100 ? 'full' : pct >= 80 ? 'warn' : 'ok';
  return { used, quota, pct, state, periodStart: since.toISOString() };
}

export async function askAiUsage(schoolId: string): Promise<{ used: number; allowance: number }> {
  const limits = await getLimits(schoolId);
  const since = periodStart(limits?.quota_period ?? 'month');
  const used = await sumLedger(schoolId, 'askai', since);
  return { used, allowance: limits?.askai_allowance ?? 0 };
}

/** §6.3 record one Ask-AI generation against the school's Ask-AI allowance pool. */
export async function recordAskAi(opts: { schoolId: string; studentClerkId?: string; units?: number; model?: string }): Promise<void> {
  await supabase.from('marking_ledger').insert({
    school_id: opts.schoolId,
    student_clerk_id: opts.studentClerkId ?? null,
    kind: 'askai',
    units: opts.units ?? 1,
    model: opts.model ?? null,
  });
}

export async function countActiveTeachers(schoolId: string): Promise<number> {
  const { count } = await supabase
    .from('profiles')
    .select('id', { count: 'exact', head: true })
    .eq('school_id', schoolId)
    .eq('role', 'teacher')
    .is('deactivated_at', null);
  return count ?? 0;
}

export async function countStudents(schoolId: string): Promise<number> {
  // Students aren't stamped with school_id — count distinct active enrolments in
  // classes owned by the school's teachers (the real, seat-relevant number).
  const { data: teachers } = await supabase.from('profiles').select('clerk_id').eq('school_id', schoolId).eq('role', 'teacher');
  const teacherIds = ((teachers ?? []) as { clerk_id: string }[]).map((t) => t.clerk_id);
  if (!teacherIds.length) return 0;
  const { data: classes } = await supabase.from('classes').select('id').in('owner_clerk_id', teacherIds);
  const classIds = ((classes ?? []) as { id: string }[]).map((c) => c.id);
  if (!classIds.length) return 0;
  const { data: enr } = await supabase.from('class_enrollments').select('student_clerk_id').in('class_id', classIds).eq('status', 'active');
  return new Set(((enr ?? []) as { student_clerk_id: string }[]).map((e) => e.student_clerk_id)).size;
}

export interface CapacityError { status: number; message: string; }

/** §3.2 teacher cap, server-side. Throws {status,message} when it would exceed. */
export async function assertCanAddTeachers(schoolId: string, adding = 1): Promise<void> {
  const limits = await getLimits(schoolId);
  if (!limits) throw { status: 400, message: 'School has no limits configured' } as CapacityError;
  const current = await countActiveTeachers(schoolId);
  if (current + adding > limits.max_teachers) {
    throw {
      status: 403,
      message: `Teacher limit reached (${current}/${limits.max_teachers}). Ask the platform admin to raise it.`,
    } as CapacityError;
  }
}

/**
 * §3.3 record a marking (or Ask-AI) call against the school and return the new
 * quota status. Emits a quota_event once per period per threshold crossing.
 * NEVER hard-stops — the caller queues at 'full'.
 */
export async function recordMarking(opts: {
  schoolId: string;
  teacherClerkId?: string;
  studentClerkId?: string;
  submissionId?: string;
  units: number;
  costEstimate?: number;
  model?: string;
  kind?: 'marking' | 'askai';
}): Promise<QuotaStatus | null> {
  await supabase.from('marking_ledger').insert({
    school_id: opts.schoolId,
    teacher_clerk_id: opts.teacherClerkId ?? null,
    student_clerk_id: opts.studentClerkId ?? null,
    submission_id: opts.submissionId ?? null,
    kind: opts.kind ?? 'marking',
    units: opts.units,
    cost_estimate: opts.costEstimate ?? null,
    model: opts.model ?? null,
  });

  const status = await markingQuotaStatus(opts.schoolId);
  if (status && (status.state === 'warn' || status.state === 'full')) {
    const kind = status.state === 'full' ? 'hit_100' : 'warn_80';
    const period = status.periodStart.slice(0, 10);
    const { data: existing } = await supabase
      .from('quota_events')
      .select('id')
      .eq('school_id', opts.schoolId)
      .eq('kind', kind)
      .eq('period_start', period)
      .limit(1)
      .maybeSingle();
    if (!existing) {
      await supabase.from('quota_events').insert({ school_id: opts.schoolId, kind, period_start: period });
    }
  }
  return status;
}
