import { supabase } from './supabase';

/**
 * Shared class-access resolution for teacher-portal routes. Access is owner or
 * co-teacher; write capability is gated by `canGrade` for co-teachers.
 * (Mirrors the in-file helper in classes.routes.ts so assignment routes enforce
 * the same scope — hidden UI is not a permission, §4.3.)
 */

export interface PortalClass {
  id: string;
  institution_id: string;
  owner_clerk_id: string;
  name: string;
  subject: string;
  syllabus_code: string;
  level: 'O' | 'A';
  archived_at: string | null;
}

export interface ClassAccess {
  klass: PortalClass;
  isOwner: boolean;
  isCoTeacher: boolean;
  canGrade: boolean;
  // Oversight via a scope grant (§4.2): read-only across classes, plus the
  // `moderate` capability (flag for re-marking). Grant holders can NEVER grade
  // a class they don't teach — that is what makes overlapping oversight safe.
  viaGrant?: boolean;
  canModerate?: boolean;
}

/** Does any of the caller's scope grants cover this class? Union resolution (§15.2). */
async function grantAccess(klass: PortalClass, clerkId: string): Promise<{ view: boolean; moderate: boolean }> {
  const { data: grants } = await supabase
    .from('scope_grants')
    .select('filter_subjects, filter_levels, capabilities')
    .eq('user_clerk_id', clerkId)
    .eq('institution_id', klass.institution_id);
  let view = false;
  let moderate = false;
  const subject = (klass.subject || '').toLowerCase();
  for (const g of (grants ?? []) as { filter_subjects: string[]; filter_levels: string[]; capabilities: string[] }[]) {
    const subjOk = !g.filter_subjects?.length || g.filter_subjects.some((s) => s.toLowerCase() === subject || subject.includes(s.toLowerCase()));
    const levelOk = !g.filter_levels?.length || g.filter_levels.includes(klass.level);
    if (subjOk && levelOk) {
      if (g.capabilities?.includes('view') || g.capabilities?.length) view = true;
      if (g.capabilities?.includes('moderate')) moderate = true;
    }
  }
  return { view, moderate };
}

export async function resolveClassAccess(classId: string, clerkId: string): Promise<ClassAccess | null> {
  const { data: klass, error } = await supabase
    .from('classes')
    .select('id, institution_id, owner_clerk_id, name, subject, syllabus_code, level, archived_at')
    .eq('id', classId)
    .maybeSingle();
  if (error) throw error;
  if (!klass) return null;

  const row = klass as PortalClass;
  if (row.owner_clerk_id === clerkId) {
    return { klass: row, isOwner: true, isCoTeacher: false, canGrade: true };
  }

  const { data: coTeacher, error: coError } = await supabase
    .from('class_co_teachers')
    .select('can_grade')
    .eq('class_id', classId)
    .eq('teacher_clerk_id', clerkId)
    .maybeSingle();
  if (coError) throw coError;

  if (coTeacher) {
    return {
      klass: row,
      isOwner: false,
      isCoTeacher: true,
      canGrade: Boolean((coTeacher as { can_grade: boolean }).can_grade),
    };
  }

  // Oversight via scope grant — read-only, optionally moderate.
  const grant = await grantAccess(row, clerkId);
  if (grant.view || grant.moderate) {
    return { klass: row, isOwner: false, isCoTeacher: false, canGrade: false, viaGrant: true, canModerate: grant.moderate };
  }

  return null;
}
