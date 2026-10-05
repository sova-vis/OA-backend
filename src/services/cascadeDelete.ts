/**
 * App-layer cascade deletes for the hard-delete features (owner deletes a school,
 * school-admin deletes a teacher, teacher removes a student / assignment).
 *
 * Why app-layer: most *_clerk_id columns are plain TEXT with no FK, and
 * profiles.school_id is ON DELETE SET NULL — so a raw `schools` row delete would
 * orphan staff and leave every class/assignment/submission intact, and never
 * touch Supabase Auth. We mirror the proven manual cleanup in
 * `/classes/:id/delete` and `/auth/delete-account`, and delete the Auth user with
 * the service-role admin client. The live schema has drifted from the migrations,
 * so we delete children explicitly rather than trust DB cascades.
 */
import { createClient } from '@supabase/supabase-js';
import { supabase } from '../lib/supabase';

const admin = createClient(
  process.env.SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY || '',
  { auth: { autoRefreshToken: false, persistSession: false } },
);

/** Remove submissions (+ their answers + marks) for a set of assignment ids. */
async function deleteSubmissionsForAssignments(assignmentIds: string[]): Promise<void> {
  if (!assignmentIds.length) return;
  const { data: subs } = await supabase.from('submissions').select('id').in('assignment_id', assignmentIds);
  const sIds = ((subs ?? []) as { id: string }[]).map((s) => s.id);
  if (sIds.length) {
    await supabase.from('submission_marks').delete().in('submission_id', sIds);
    await supabase.from('submission_answers').delete().in('submission_id', sIds);
  }
  await supabase.from('submissions').delete().in('assignment_id', assignmentIds);
}

/** Delete a whole assignment's data (submissions, answers, marks, questions, recipients, row). */
export async function deleteAssignmentCascade(assignmentId: string): Promise<void> {
  await deleteSubmissionsForAssignments([assignmentId]);
  await supabase.from('assignment_questions').delete().eq('assignment_id', assignmentId);
  await supabase.from('assignment_recipients').delete().eq('assignment_id', assignmentId);
  await supabase.from('assignments').delete().eq('id', assignmentId);
}

/** Delete classes and everything beneath them (assignments, submissions, enrolments). */
export async function deleteClassesCascade(classIds: string[]): Promise<void> {
  if (!classIds.length) return;
  const { data: aqs } = await supabase.from('assignments').select('id').in('class_id', classIds);
  const aIds = ((aqs ?? []) as { id: string }[]).map((a) => a.id);
  if (aIds.length) {
    await deleteSubmissionsForAssignments(aIds);
    await supabase.from('assignment_questions').delete().in('assignment_id', aIds);
    await supabase.from('assignment_recipients').delete().in('assignment_id', aIds);
    await supabase.from('assignments').delete().in('class_id', classIds);
  }
  await supabase.from('class_enrollments').delete().in('class_id', classIds);
  await supabase.from('class_co_teachers').delete().in('class_id', classIds);
  // teacher_resources is newer (migration 031); tolerate it not existing on drift.
  try { await supabase.from('teacher_resources').delete().in('class_id', classIds); } catch { /* best-effort */ }
  await supabase.from('classes').delete().in('id', classIds);
}

/** Hard-delete a teacher: their classes (+ all data), clerk-keyed rows, profile, Auth user. */
export async function deleteTeacherCascade(clerkId: string): Promise<void> {
  const { data: cls } = await supabase.from('classes').select('id').eq('owner_clerk_id', clerkId);
  await deleteClassesCascade(((cls ?? []) as { id: string }[]).map((c) => c.id));
  await supabase.from('class_co_teachers').delete().eq('teacher_clerk_id', clerkId);
  await supabase.from('custom_questions').delete().eq('owner_clerk_id', clerkId);
  await supabase.from('comment_bank').delete().eq('owner_clerk_id', clerkId);
  await supabase.from('scope_grants').delete().eq('user_clerk_id', clerkId);
  await supabase.from('notifications').delete().eq('recipient_clerk_id', clerkId);
  await supabase.from('profiles').delete().eq('clerk_id', clerkId);
  try { await admin.auth.admin.deleteUser(clerkId); } catch { /* profile + data already gone */ }
}

/** Hard-delete a school-admin (no owned classes): clerk-keyed rows, profile, Auth user. */
export async function deleteStaffAccount(clerkId: string): Promise<void> {
  await supabase.from('scope_grants').delete().eq('user_clerk_id', clerkId);
  await supabase.from('notifications').delete().eq('recipient_clerk_id', clerkId);
  await supabase.from('profiles').delete().eq('clerk_id', clerkId);
  try { await admin.auth.admin.deleteUser(clerkId); } catch { /* best-effort */ }
}

/** Hard-delete a STUDENT: their submissions (+answers+marks), enrolments, clerk-keyed
 * rows, profile, Auth user. Mirrors the student path of /auth/delete-account. */
export async function deleteStudentAccount(clerkId: string): Promise<void> {
  const { data: subs } = await supabase.from('submissions').select('id').eq('student_clerk_id', clerkId);
  const sIds = ((subs ?? []) as { id: string }[]).map((s) => s.id);
  if (sIds.length) {
    await supabase.from('submission_marks').delete().in('submission_id', sIds);
    await supabase.from('submission_answers').delete().in('submission_id', sIds);
    await supabase.from('submissions').delete().in('id', sIds);
  }
  await supabase.from('class_enrollments').delete().eq('student_clerk_id', clerkId);
  await supabase.from('scope_grants').delete().eq('user_clerk_id', clerkId);
  await supabase.from('notifications').delete().eq('recipient_clerk_id', clerkId);
  await supabase.from('profiles').delete().eq('clerk_id', clerkId);
  try { await admin.auth.admin.deleteUser(clerkId); } catch { /* best-effort */ }
}

/** Route an account to the right cascade by role (owner/admin are not deletable here). */
export async function deleteAccountByRole(clerkId: string, role: string): Promise<void> {
  if (role === 'teacher') return deleteTeacherCascade(clerkId);
  if (role === 'school_admin') return deleteStaffAccount(clerkId);
  return deleteStudentAccount(clerkId);
}

/** Clear ONE student's submissions (+answers+marks) for a single class's assignments. */
export async function clearStudentSubmissionsForClass(classId: string, studentClerkId: string): Promise<void> {
  const { data: aqs } = await supabase.from('assignments').select('id').eq('class_id', classId);
  const aIds = ((aqs ?? []) as { id: string }[]).map((a) => a.id);
  if (!aIds.length) return;
  const { data: subs } = await supabase
    .from('submissions').select('id').in('assignment_id', aIds).eq('student_clerk_id', studentClerkId);
  const sIds = ((subs ?? []) as { id: string }[]).map((s) => s.id);
  if (!sIds.length) return;
  await supabase.from('submission_marks').delete().in('submission_id', sIds);
  await supabase.from('submission_answers').delete().in('submission_id', sIds);
  await supabase.from('submissions').delete().in('id', sIds);
}

export interface SchoolDeleteSummary {
  teachers: number; admins: number; classes: number; assignments: number; studentsUnenrolled: number;
}

/**
 * Delete a school and everything it owns. Staff (teachers + school-admins) accounts
 * are hard-deleted; ENROLLED STUDENTS KEEP THEIR ACCOUNTS — they are only unenrolled
 * (their classes are deleted) and their school-tied answers are cleared with those
 * classes. Returns counts for the confirmation UI.
 */
export async function deleteSchoolCascade(schoolId: string): Promise<SchoolDeleteSummary> {
  const { data: staff } = await supabase.from('profiles').select('clerk_id, role').eq('school_id', schoolId);
  const rows = (staff ?? []) as { clerk_id: string; role: string }[];
  const teachers = rows.filter((s) => s.role === 'teacher');
  const admins = rows.filter((s) => s.role === 'school_admin');

  // Gather counts up front (before deletion) for the summary.
  let classes = 0, assignments = 0, studentsUnenrolled = 0;
  const teacherIds = teachers.map((t) => t.clerk_id);
  if (teacherIds.length) {
    const { data: cls } = await supabase.from('classes').select('id').in('owner_clerk_id', teacherIds);
    const classIds = ((cls ?? []) as { id: string }[]).map((c) => c.id);
    classes = classIds.length;
    if (classIds.length) {
      const { data: aqs } = await supabase.from('assignments').select('id').in('class_id', classIds);
      assignments = ((aqs ?? []) as { id: string }[]).length;
      const { data: enr } = await supabase
        .from('class_enrollments').select('student_clerk_id').in('class_id', classIds).eq('status', 'active');
      studentsUnenrolled = new Set(((enr ?? []) as { student_clerk_id: string }[]).map((e) => e.student_clerk_id)).size;
    }
  }

  // Delete staff accounts (teachers take their classes + students' enrolments with them).
  for (const t of teachers) await deleteTeacherCascade(t.clerk_id);
  for (const a of admins) await deleteStaffAccount(a.clerk_id);

  // Finally the school row (cascades school_limits / marking_ledger / quota_events).
  await supabase.from('schools').delete().eq('id', schoolId);

  return { teachers: teachers.length, admins: admins.length, classes, assignments, studentsUnenrolled };
}
