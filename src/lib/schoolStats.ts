/**
 * School / teacher aggregate stats (spec §3.4 owner monitoring, §4.2 school-admin
 * oversight). Students are NOT stamped with a school_id, so everything about
 * students is derived from active class enrolments in classes owned by the
 * school's teachers. All helpers use a small, fixed number of queries regardless
 * of how many teachers/classes there are.
 */
import { supabase } from './supabase';

export interface TeacherStat { classes: number; students: number; assignments: number }

/** Clerk ids of the (active) teachers in a school. */
export async function schoolTeacherIds(schoolId: string): Promise<string[]> {
  const { data } = await supabase
    .from('profiles').select('clerk_id')
    .eq('school_id', schoolId).eq('role', 'teacher').is('deactivated_at', null);
  return ((data ?? []) as { clerk_id: string }[]).map((t) => t.clerk_id);
}

/** Per-teacher {classes, students, assignments} for a set of teachers. */
export async function teacherStats(teacherIds: string[]): Promise<Map<string, TeacherStat>> {
  const out = new Map<string, TeacherStat>();
  for (const id of teacherIds) out.set(id, { classes: 0, students: 0, assignments: 0 });
  if (!teacherIds.length) return out;

  const { data: classesData } = await supabase.from('classes').select('id, owner_clerk_id').in('owner_clerk_id', teacherIds);
  const classes = (classesData ?? []) as { id: string; owner_clerk_id: string }[];
  const classOwner = new Map(classes.map((c) => [c.id, c.owner_clerk_id]));
  for (const c of classes) { const s = out.get(c.owner_clerk_id); if (s) s.classes++; }

  const classIds = classes.map((c) => c.id);
  if (classIds.length) {
    const { data: enr } = await supabase.from('class_enrollments').select('class_id, student_clerk_id').in('class_id', classIds).eq('status', 'active');
    const byTeacher = new Map<string, Set<string>>();
    for (const e of ((enr ?? []) as { class_id: string; student_clerk_id: string }[])) {
      const owner = classOwner.get(e.class_id);
      if (!owner) continue;
      if (!byTeacher.has(owner)) byTeacher.set(owner, new Set());
      byTeacher.get(owner)!.add(e.student_clerk_id);
    }
    for (const [owner, set] of byTeacher) { const s = out.get(owner); if (s) s.students = set.size; }
  }

  const { data: asg } = await supabase.from('assignments').select('owner_clerk_id').in('owner_clerk_id', teacherIds);
  for (const a of ((asg ?? []) as { owner_clerk_id: string }[])) { const s = out.get(a.owner_clerk_id); if (s) s.assignments++; }

  return out;
}

/** School-wide totals (distinct students via enrolments, classes, assignments). */
export async function schoolTotals(schoolId: string): Promise<{ teachers: number; students: number; classes: number; assignments: number }> {
  const teacherIds = await schoolTeacherIds(schoolId);
  if (!teacherIds.length) return { teachers: 0, students: 0, classes: 0, assignments: 0 };

  const { data: classesData } = await supabase.from('classes').select('id').in('owner_clerk_id', teacherIds);
  const classIds = ((classesData ?? []) as { id: string }[]).map((c) => c.id);

  const students = new Set<string>();
  if (classIds.length) {
    const { data: enr } = await supabase.from('class_enrollments').select('student_clerk_id').in('class_id', classIds).eq('status', 'active');
    for (const e of ((enr ?? []) as { student_clerk_id: string }[])) students.add(e.student_clerk_id);
  }
  const { count: assignments } = await supabase
    .from('assignments').select('id', { count: 'exact', head: true }).in('owner_clerk_id', teacherIds);

  return { teachers: teacherIds.length, students: students.size, classes: classIds.length, assignments: assignments ?? 0 };
}
