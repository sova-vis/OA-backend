/**
 * School→consumer paywall funnel (spec §6.2).
 *
 * A student enrolled in a school's class browses the library free but their own
 * self-practice/marking is a Pro upsell — offered at the school's discount. There
 * is no student->school column yet (profiles.school_id is set only for staff, and
 * classes has institution_id, not school_id), so we DERIVE the link at request
 * time via the reliable chain:
 *
 *   class_enrollments(active) -> classes.owner_clerk_id (the teacher)
 *     -> teacher profiles.school_id -> schools.discount_pct
 *
 * This needs no schema change and is naturally safe: a student not in a
 * school-linked class resolves to null and sees no change at all.
 */
import { supabase } from './supabase';

export interface SchoolOffer {
  school_id: string;
  school_name: string;
  discount_pct: number;
}

/** The set of school ids a student is linked to via their active class enrolments. */
async function studentSchoolIds(clerkId: string): Promise<string[]> {
  const { data: enr } = await supabase
    .from('class_enrollments')
    .select('class_id')
    .eq('student_clerk_id', clerkId)
    .eq('status', 'active');
  const classIds = Array.from(new Set(((enr ?? []) as { class_id: string }[]).map((e) => e.class_id)));
  if (!classIds.length) return [];

  const { data: cls } = await supabase.from('classes').select('owner_clerk_id').in('id', classIds);
  const ownerIds = Array.from(new Set(((cls ?? []) as { owner_clerk_id: string }[]).map((c) => c.owner_clerk_id).filter(Boolean)));
  if (!ownerIds.length) return [];

  const { data: owners } = await supabase.from('profiles').select('school_id').in('clerk_id', ownerIds);
  return Array.from(new Set(
    ((owners ?? []) as { school_id: string | null }[]).map((o) => o.school_id).filter((x): x is string => !!x),
  ));
}

/**
 * The best discount offer for a student, or null. "Best" = the active school with
 * the highest discount_pct (> 0) among the schools they're linked to. Fail-safe:
 * any error returns null (no offer, no price change).
 */
export async function resolveStudentSchoolOffer(clerkId: string): Promise<SchoolOffer | null> {
  try {
    const ids = await studentSchoolIds(clerkId);
    if (!ids.length) return null;
    const { data: schools } = await supabase
      .from('schools')
      .select('id, name, discount_pct, status')
      .in('id', ids);
    const best = ((schools ?? []) as { id: string; name: string; discount_pct: number | string; status: string }[])
      .filter((s) => s.status === 'active' && Number(s.discount_pct) > 0)
      .sort((a, b) => Number(b.discount_pct) - Number(a.discount_pct))[0];
    if (!best) return null;
    return { school_id: best.id, school_name: best.name, discount_pct: Number(best.discount_pct) };
  } catch (err) {
    console.error('[schoolOffer] resolve failed:', err);
    return null;
  }
}

/** The primary school a student is linked to (for §6.3 metering), or null. */
export async function primaryStudentSchoolId(clerkId: string): Promise<string | null> {
  const ids = await studentSchoolIds(clerkId);
  return ids[0] ?? null;
}

/** Apply a school discount %, clamped and rounded to a whole rupee. */
export function applyDiscount(basePkr: number, pct: number): number {
  const p = Math.max(0, Math.min(100, Number(pct) || 0));
  return Math.max(0, Math.round(basePkr * (1 - p / 100)));
}

/**
 * Record a funnel event (§6.2), best-effort. Resolves the student's primary school
 * for attribution; never throws into the caller.
 */
export async function logFunnel(clerkId: string, kind: 'prompt_shown' | 'converted', context?: string): Promise<void> {
  try {
    const ids = await studentSchoolIds(clerkId);
    await supabase.from('funnel_events').insert({
      school_id: ids[0] ?? null,
      student_clerk_id: clerkId,
      kind,
      context: context ?? null,
    });
  } catch (err) {
    console.error('[funnel] write failed:', err);
  }
}
