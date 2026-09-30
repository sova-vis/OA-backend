/**
 * School-admin console (spec §4). An account manager for ONE school — creates
 * teachers (single or CSV), assigns their subjects/year groups, deactivates them,
 * and sees seat usage. NOT a super-teacher: no grading, no viewing scripts (§2).
 *
 * Guarded by requireSchoolScope('school_admin') — the caller must be a school_admin
 * WITH a school_id, which scopes every query to their own school. Mounted at
 * /school-admin.
 */
import { Router, Response } from 'express';
import { clerkAuth } from './lib/clerkAuth';
import { ActorRequest, requireSchoolScope } from './lib/roles';
import { supabase } from './lib/supabase';
import { logAudit } from './lib/audit';
import { getLimits, countActiveTeachers, countStudents, markingQuotaStatus, assertCanAddTeachers, CapacityError } from './lib/quota';
import { createStaffAccount } from './services/staffAccounts';

const router = Router();
router.use(clerkAuth, requireSchoolScope('school_admin'));

function schoolId(req: ActorRequest): string {
  return req.actor!.schoolId as string; // guaranteed by requireSchoolScope
}

interface TeacherRow { email?: string; name?: string; subjects?: string[]; levels?: string[]; }

/** Minimal CSV parser for "name,subjects,levels" (subjects/levels use `;`). Logins
 *  are generated from the name, so no email column is needed. */
function parseTeacherCsv(csv: string): TeacherRow[] {
  const lines = csv.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (!lines.length) return [];
  const header = lines[0].toLowerCase().split(',').map((h) => h.trim());
  const hasHeader = header.includes('name');
  const start = hasHeader ? 1 : 0;
  const iName = hasHeader ? header.indexOf('name') : 0;
  const iSub = hasHeader ? header.indexOf('subjects') : 1;
  const iLvl = hasHeader ? header.indexOf('levels') : 2;
  const rows: TeacherRow[] = [];
  for (let i = start; i < lines.length; i++) {
    const c = lines[i].split(',').map((x) => x.trim());
    const name = c[iName];
    if (!name) continue;
    rows.push({
      name,
      subjects: iSub >= 0 && c[iSub] ? c[iSub].split(';').map((s) => s.trim()).filter(Boolean) : undefined,
      levels: iLvl >= 0 && c[iLvl] ? c[iLvl].split(';').map((s) => s.trim()).filter(Boolean) : undefined,
    });
  }
  return rows;
}

// §4.1 School info + seat usage (seats used vs available).
router.get('/', async (req: ActorRequest, res: Response) => {
  try {
    const sid = schoolId(req);
    const [{ data: school }, limits, teachers, students, marking] = await Promise.all([
      supabase.from('schools').select('*').eq('id', sid).maybeSingle(),
      getLimits(sid), countActiveTeachers(sid), countStudents(sid), markingQuotaStatus(sid),
    ]);
    return res.json({
      school,
      limits,
      usage: {
        teachers_used: teachers, teachers_max: limits?.max_teachers ?? null,
        students_used: students, students_max: limits?.max_students_total ?? null,
        marking,
      },
    });
  } catch (err: unknown) {
    console.error('GET /school-admin', err);
    return res.status(500).json({ error: 'Failed to load school' });
  }
});

// §4.1 List teachers in this school.
router.get('/teachers', async (req: ActorRequest, res: Response) => {
  try {
    const { data } = await supabase
      .from('profiles')
      .select('clerk_id, full_name, email, syllabus_codes, levels, deactivated_at, must_change_password, created_at')
      .eq('school_id', schoolId(req))
      .eq('role', 'teacher')
      .order('full_name', { ascending: true });
    return res.json({ teachers: (data as unknown[]) ?? [] });
  } catch (err: unknown) {
    console.error('GET /school-admin/teachers', err);
    return res.status(500).json({ error: 'Failed to list teachers' });
  }
});

// §4.1 Create one teacher (subjects + year groups), enforcing the teacher cap.
router.post('/teachers', async (req: ActorRequest, res: Response) => {
  try {
    const sid = schoolId(req);
    const { name, subjects, levels } = req.body || {};
    if (!name) return res.status(400).json({ error: 'name is required' });

    await assertCanAddTeachers(sid, 1);

    const result = await createStaffAccount({
      name, role: 'teacher', schoolId: sid,
      subjects: Array.isArray(subjects) ? subjects : undefined,
      levels: Array.isArray(levels) ? levels : undefined,
      createdBy: req.actor?.clerkId,
    });
    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: 'teacher.create', targetType: 'profile', targetId: result.clerkId, schoolId: sid,
    });
    return res.status(201).json({
      teacher: { email: result.email, tempPassword: result.tempPassword, existed: result.existed, clerkId: result.clerkId },
    });
  } catch (err: unknown) {
    const e = err as CapacityError & { statusCode?: number };
    console.error('POST /school-admin/teachers', err);
    return res.status(e.status || e.statusCode || 500).json({ error: e.message || 'Failed to create teacher' });
  }
});

// §4.1 Bulk create teachers (CSV string or JSON array), enforcing the cap first.
router.post('/teachers/bulk', async (req: ActorRequest, res: Response) => {
  try {
    const sid = schoolId(req);
    let rows: TeacherRow[] = [];
    if (typeof req.body?.csv === 'string') rows = parseTeacherCsv(req.body.csv);
    else if (Array.isArray(req.body?.teachers)) rows = req.body.teachers;
    rows = rows.filter((r) => r.name);
    if (!rows.length) return res.status(400).json({ error: 'No valid teacher rows (need a name)' });

    await assertCanAddTeachers(sid, rows.length);

    const results: Array<{ email: string; ok: boolean; tempPassword?: string | null; existed?: boolean; error?: string }> = [];
    for (const r of rows) {
      try {
        const out = await createStaffAccount({
          name: r.name!, role: 'teacher', schoolId: sid,
          subjects: r.subjects, levels: r.levels, createdBy: req.actor?.clerkId,
        });
        results.push({ email: out.email, ok: true, tempPassword: out.tempPassword, existed: out.existed });
      } catch (e: unknown) {
        results.push({ email: r.name!, ok: false, error: (e as Error).message });
      }
    }
    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: 'teacher.bulk_create', targetType: 'school', targetId: sid,
      after: { attempted: rows.length, created: results.filter((r) => r.ok).length }, schoolId: sid,
    });
    return res.status(201).json({ results });
  } catch (err: unknown) {
    const e = err as CapacityError & { statusCode?: number };
    console.error('POST /school-admin/teachers/bulk', err);
    return res.status(e.status || e.statusCode || 500).json({ error: e.message || 'Bulk create failed' });
  }
});

// §4.1 Update a teacher's subjects/levels, or deactivate / reactivate them.
router.patch('/teachers/:clerkId', async (req: ActorRequest, res: Response) => {
  try {
    const sid = schoolId(req);
    // Ensure the teacher belongs to THIS school (no cross-school edits).
    const { data: t } = await supabase
      .from('profiles').select('clerk_id, school_id, role, deactivated_at')
      .eq('clerk_id', req.params.clerkId).maybeSingle();
    const teacher = t as { school_id?: string; role?: string } | null;
    if (!teacher || teacher.school_id !== sid || teacher.role !== 'teacher') {
      return res.status(404).json({ error: 'Teacher not found in your school' });
    }

    const patch: Record<string, unknown> = { updated_at: new Date().toISOString() };
    if (Array.isArray(req.body?.subjects)) patch.syllabus_codes = req.body.subjects;
    if (Array.isArray(req.body?.levels)) patch.levels = req.body.levels;
    if (typeof req.body?.full_name === 'string') patch.full_name = req.body.full_name;
    if (req.body?.active === false) patch.deactivated_at = new Date().toISOString();
    if (req.body?.active === true) patch.deactivated_at = null;

    const { data: after, error } = await supabase
      .from('profiles').update(patch).eq('clerk_id', req.params.clerkId)
      .select('clerk_id, full_name, email, syllabus_codes, levels, deactivated_at').single();
    if (error) throw error;

    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: req.body?.active === false ? 'teacher.deactivate' : 'teacher.update',
      targetType: 'profile', targetId: req.params.clerkId, schoolId: sid,
    });
    return res.json({ teacher: after });
  } catch (err: unknown) {
    console.error('PATCH /school-admin/teachers/:clerkId', err);
    return res.status(500).json({ error: 'Failed to update teacher' });
  }
});

export default router;
