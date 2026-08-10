import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import {
  ensureInstitutionForTeacher,
  generateUniqueJoinCode,
  normalizeJoinCode,
} from './lib/institutions';
import {
  generatePassword,
  generateUniqueUsername,
  hashPassword,
  syntheticClerkId,
} from './lib/provisioning';
import { displayName } from './lib/names';
import { ensureStudentProfile } from './lib/clerkUser';

/**
 * Teacher Portal — class & enrolment management (spec §3, §4.4, §3.5).
 * All routes require authentication; write access is scoped per-class in-code
 * (hidden UI is not a permission — §4.3).
 */

const router = Router();
router.use(clerkAuth);

type Role = 'student' | 'teacher' | 'admin';

interface ProfileRow {
  clerk_id: string;
  full_name: string | null;
  email: string | null;
  role: Role;
  institution_id: string | null;
  school_name: string | null;
}

interface ClassRow {
  id: string;
  institution_id: string;
  owner_clerk_id: string;
  name: string;
  subject: string;
  syllabus_code: string;
  level: 'O' | 'A';
  year_group: string | null;
  description: string | null;
  join_code: string | null;
  join_enabled: boolean;
  auto_approve_joins: boolean;
  exam_series: string | null;
  archived_at: string | null;
  created_at: string;
  updated_at: string;
}

const PROFILE_COLUMNS = 'clerk_id, full_name, email, role, institution_id, school_name';

async function getProfile(clerkId: string): Promise<ProfileRow | null> {
  const { data, error } = await supabase
    .from('profiles')
    .select(PROFILE_COLUMNS)
    .eq('clerk_id', clerkId)
    .maybeSingle();
  if (error) throw error;
  return (data as ProfileRow | null) ?? null;
}

interface ClassAccess {
  klass: ClassRow;
  isOwner: boolean;
  isCoTeacher: boolean;
  canGrade: boolean;
}

/**
 * Resolve a caller's access to a class: owner, co-teacher, or none.
 * Returns null when the caller has no relationship to the class (→ 404/403).
 */
async function resolveClassAccess(classId: string, clerkId: string): Promise<ClassAccess | null> {
  const { data: klass, error } = await supabase
    .from('classes')
    .select('*')
    .eq('id', classId)
    .maybeSingle();
  if (error) throw error;
  if (!klass) return null;

  const row = klass as ClassRow;
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

  return null;
}

function isTeacherRole(role: Role | undefined): boolean {
  return role === 'teacher' || role === 'admin';
}

function serializeClass(row: ClassRow, counts?: { students: number; pending: number }) {
  return {
    id: row.id,
    institution_id: row.institution_id,
    owner_clerk_id: row.owner_clerk_id,
    name: row.name,
    subject: row.subject,
    syllabus_code: row.syllabus_code,
    level: row.level,
    year_group: row.year_group,
    description: row.description,
    join_code: row.join_code,
    join_enabled: row.join_enabled,
    auto_approve_joins: row.auto_approve_joins,
    exam_series: row.exam_series,
    archived: Boolean(row.archived_at),
    archived_at: row.archived_at,
    created_at: row.created_at,
    updated_at: row.updated_at,
    student_count: counts?.students ?? undefined,
    pending_count: counts?.pending ?? undefined,
  };
}

// ---------------------------------------------------------------------------
// POST /classes — create a class (§3.1). Teacher/admin only.
// ---------------------------------------------------------------------------
router.post('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const profile = await getProfile(clerkId);
    if (!profile || !isTeacherRole(profile.role)) {
      return res.status(403).json({ error: 'Only teachers can create classes' });
    }

    const { name, subject, syllabus_code, level, year_group, description } = req.body ?? {};
    if (!name?.trim() || !subject?.trim() || !syllabus_code?.trim()) {
      return res.status(400).json({ error: 'name, subject and syllabus_code are required' });
    }
    if (level !== 'O' && level !== 'A') {
      return res.status(400).json({ error: "level must be 'O' or 'A'" });
    }

    const institutionId = await ensureInstitutionForTeacher(profile);
    const joinCode = await generateUniqueJoinCode();

    const { data, error } = await supabase
      .from('classes')
      .insert({
        institution_id: institutionId,
        owner_clerk_id: clerkId,
        name: name.trim(),
        subject: subject.trim(),
        syllabus_code: syllabus_code.trim(),
        level,
        year_group: year_group?.trim() || null,
        description: description?.trim() || null,
        join_code: joinCode,
      })
      .select('*')
      .single();
    if (error) throw error;

    return res.status(201).json(serializeClass(data as ClassRow, { students: 0, pending: 0 }));
  } catch (err) {
    console.error('Create class error:', err);
    return res.status(500).json({ error: 'Failed to create class' });
  }
});

// ---------------------------------------------------------------------------
// GET /classes — list classes the teacher owns or co-teaches (§3.3).
// ---------------------------------------------------------------------------
router.get('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const includeArchived = req.query.includeArchived === 'true';

    const [{ data: owned, error: ownedErr }, { data: coRows, error: coErr }] = await Promise.all([
      supabase.from('classes').select('*').eq('owner_clerk_id', clerkId),
      supabase.from('class_co_teachers').select('class_id, can_grade').eq('teacher_clerk_id', clerkId),
    ]);
    if (ownedErr) throw ownedErr;
    if (coErr) throw coErr;

    const coClassIds = (coRows ?? []).map((r) => (r as { class_id: string }).class_id);
    let coClasses: ClassRow[] = [];
    if (coClassIds.length > 0) {
      const { data, error } = await supabase.from('classes').select('*').in('id', coClassIds);
      if (error) throw error;
      coClasses = (data as ClassRow[]) ?? [];
    }

    const ownedIds = new Set((owned ?? []).map((c) => (c as ClassRow).id));
    const all: ClassRow[] = [
      ...((owned as ClassRow[]) ?? []),
      ...coClasses.filter((c) => !ownedIds.has(c.id)),
    ].filter((c) => includeArchived || !c.archived_at);

    // Tally student (active) and pending counts in one enrolments read.
    const classIds = all.map((c) => c.id);
    const counts = new Map<string, { students: number; pending: number }>();
    if (classIds.length > 0) {
      const { data: enrollments, error: enrErr } = await supabase
        .from('class_enrollments')
        .select('class_id, status')
        .in('class_id', classIds)
        .in('status', ['active', 'pending']);
      if (enrErr) throw enrErr;
      for (const row of (enrollments ?? []) as { class_id: string; status: string }[]) {
        const entry = counts.get(row.class_id) ?? { students: 0, pending: 0 };
        if (row.status === 'active') entry.students += 1;
        else if (row.status === 'pending') entry.pending += 1;
        counts.set(row.class_id, entry);
      }
    }

    const result = all
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((c) => ({
        ...serializeClass(c, counts.get(c.id) ?? { students: 0, pending: 0 }),
        is_owner: c.owner_clerk_id === clerkId,
      }));

    return res.json(result);
  } catch (err) {
    console.error('List classes error:', err);
    return res.status(500).json({ error: 'Failed to list classes' });
  }
});

// ---------------------------------------------------------------------------
// GET /classes/:id — class detail with counts (owner or co-teacher).
// ---------------------------------------------------------------------------
router.get('/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });

    const { data: enrollments, error } = await supabase
      .from('class_enrollments')
      .select('status')
      .eq('class_id', access.klass.id);
    if (error) throw error;

    let students = 0;
    let pending = 0;
    for (const row of (enrollments ?? []) as { status: string }[]) {
      if (row.status === 'active') students += 1;
      else if (row.status === 'pending') pending += 1;
    }

    return res.json({
      ...serializeClass(access.klass, { students, pending }),
      is_owner: access.isOwner,
      can_grade: access.canGrade,
    });
  } catch (err) {
    console.error('Get class error:', err);
    return res.status(500).json({ error: 'Failed to load class' });
  }
});

// ---------------------------------------------------------------------------
// PATCH /classes/:id — edit class fields (owner only).
// ---------------------------------------------------------------------------
router.patch('/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    if (!access.isOwner) return res.status(403).json({ error: 'Only the class owner can edit it' });

    const body = req.body ?? {};
    const update: Record<string, unknown> = { updated_at: new Date().toISOString() };
    if (typeof body.name === 'string' && body.name.trim()) update.name = body.name.trim();
    if (typeof body.subject === 'string' && body.subject.trim()) update.subject = body.subject.trim();
    if (typeof body.syllabus_code === 'string' && body.syllabus_code.trim()) {
      update.syllabus_code = body.syllabus_code.trim();
    }
    if (body.level === 'O' || body.level === 'A') update.level = body.level;
    if ('year_group' in body) update.year_group = body.year_group?.trim() || null;
    if ('description' in body) update.description = body.description?.trim() || null;
    if ('exam_series' in body) update.exam_series = body.exam_series?.trim() || null;
    if (typeof body.auto_approve_joins === 'boolean') update.auto_approve_joins = body.auto_approve_joins;

    const { data, error } = await supabase
      .from('classes')
      .update(update)
      .eq('id', access.klass.id)
      .select('*')
      .single();
    if (error) throw error;

    return res.json(serializeClass(data as ClassRow));
  } catch (err) {
    console.error('Update class error:', err);
    return res.status(500).json({ error: 'Failed to update class' });
  }
});

// ---------------------------------------------------------------------------
// POST /classes/:id/archive & /restore (§3.4). Owner only; soft, reversible.
// ---------------------------------------------------------------------------
async function setArchived(req: AuthenticatedRequest, res: Response, archived: boolean) {
  const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
  if (!access) return res.status(404).json({ error: 'Class not found' });
  if (!access.isOwner) return res.status(403).json({ error: 'Only the class owner can archive it' });

  const { data, error } = await supabase
    .from('classes')
    .update({ archived_at: archived ? new Date().toISOString() : null, updated_at: new Date().toISOString() })
    .eq('id', access.klass.id)
    .select('*')
    .single();
  if (error) throw error;
  return res.json(serializeClass(data as ClassRow));
}

router.post('/:id/archive', async (req: AuthenticatedRequest, res: Response) => {
  try {
    return await setArchived(req, res, true);
  } catch (err) {
    console.error('Archive class error:', err);
    return res.status(500).json({ error: 'Failed to archive class' });
  }
});

router.post('/:id/restore', async (req: AuthenticatedRequest, res: Response) => {
  try {
    return await setArchived(req, res, false);
  } catch (err) {
    console.error('Restore class error:', err);
    return res.status(500).json({ error: 'Failed to restore class' });
  }
});

// ---------------------------------------------------------------------------
// Join-code management (§3.2). Regenerate invalidates the previous code but
// does not affect enrolled students; disabling stops new joins once full.
// ---------------------------------------------------------------------------
router.post('/:id/join-code/regenerate', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    if (!access.isOwner) return res.status(403).json({ error: 'Only the class owner can regenerate the code' });

    const joinCode = await generateUniqueJoinCode();
    const { data, error } = await supabase
      .from('classes')
      .update({ join_code: joinCode, join_enabled: true, updated_at: new Date().toISOString() })
      .eq('id', access.klass.id)
      .select('*')
      .single();
    if (error) throw error;
    return res.json(serializeClass(data as ClassRow));
  } catch (err) {
    console.error('Regenerate join code error:', err);
    return res.status(500).json({ error: 'Failed to regenerate join code' });
  }
});

router.patch('/:id/join-code', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    if (!access.isOwner) return res.status(403).json({ error: 'Only the class owner can change the code' });
    if (typeof req.body?.enabled !== 'boolean') {
      return res.status(400).json({ error: 'enabled (boolean) is required' });
    }

    const { data, error } = await supabase
      .from('classes')
      .update({ join_enabled: req.body.enabled, updated_at: new Date().toISOString() })
      .eq('id', access.klass.id)
      .select('*')
      .single();
    if (error) throw error;
    return res.json(serializeClass(data as ClassRow));
  } catch (err) {
    console.error('Toggle join code error:', err);
    return res.status(500).json({ error: 'Failed to update join code' });
  }
});

// ---------------------------------------------------------------------------
// Co-teaching (§3.5). Grant an existing teacher access by email.
// ---------------------------------------------------------------------------
router.get('/:id/co-teachers', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });

    const { data, error } = await supabase
      .from('class_co_teachers')
      .select('id, teacher_clerk_id, can_grade, invited_email, created_at')
      .eq('class_id', access.klass.id);
    if (error) throw error;

    // Attach display names.
    const ids = (data ?? []).map((r) => (r as { teacher_clerk_id: string }).teacher_clerk_id);
    const names = new Map<string, ProfileRow>();
    if (ids.length > 0) {
      const { data: profiles } = await supabase.from('profiles').select(PROFILE_COLUMNS).in('clerk_id', ids);
      for (const p of (profiles ?? []) as ProfileRow[]) names.set(p.clerk_id, p);
    }

    return res.json(
      (data ?? []).map((r) => {
        const row = r as { id: string; teacher_clerk_id: string; can_grade: boolean; invited_email: string | null; created_at: string };
        const p = names.get(row.teacher_clerk_id);
        return { ...row, full_name: p?.full_name ?? null, email: p?.email ?? row.invited_email };
      })
    );
  } catch (err) {
    console.error('List co-teachers error:', err);
    return res.status(500).json({ error: 'Failed to list co-teachers' });
  }
});

router.post('/:id/co-teachers', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    if (!access.isOwner) return res.status(403).json({ error: 'Only the class owner can add co-teachers' });

    const email = String(req.body?.email ?? '').trim().toLowerCase();
    const canGrade = req.body?.can_grade !== false; // default true (second marker)
    if (!email) return res.status(400).json({ error: 'email is required' });

    const { data: target, error: targetErr } = await supabase
      .from('profiles')
      .select(PROFILE_COLUMNS)
      .ilike('email', email)
      .maybeSingle();
    if (targetErr) throw targetErr;

    if (!target) {
      return res.status(404).json({ error: 'No registered teacher with that email. Ask them to sign up first.' });
    }
    const targetProfile = target as ProfileRow;
    if (!isTeacherRole(targetProfile.role)) {
      return res.status(400).json({ error: 'That account is not a teacher' });
    }
    if (targetProfile.clerk_id === access.klass.owner_clerk_id) {
      return res.status(400).json({ error: 'That teacher already owns this class' });
    }

    const { data, error } = await supabase
      .from('class_co_teachers')
      .upsert(
        {
          class_id: access.klass.id,
          teacher_clerk_id: targetProfile.clerk_id,
          can_grade: canGrade,
          invited_email: email,
        },
        { onConflict: 'class_id,teacher_clerk_id' }
      )
      .select('id, teacher_clerk_id, can_grade, invited_email, created_at')
      .single();
    if (error) throw error;

    return res.status(201).json({ ...data, full_name: targetProfile.full_name, email: targetProfile.email });
  } catch (err) {
    console.error('Add co-teacher error:', err);
    return res.status(500).json({ error: 'Failed to add co-teacher' });
  }
});

router.delete('/:id/co-teachers/:teacherClerkId', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    if (!access.isOwner) return res.status(403).json({ error: 'Only the class owner can revoke co-teachers' });

    // Revocation does not remove marks the co-teacher already awarded (§3.5).
    const { error } = await supabase
      .from('class_co_teachers')
      .delete()
      .eq('class_id', access.klass.id)
      .eq('teacher_clerk_id', req.params.teacherClerkId);
    if (error) throw error;
    return res.json({ ok: true });
  } catch (err) {
    console.error('Revoke co-teacher error:', err);
    return res.status(500).json({ error: 'Failed to revoke co-teacher' });
  }
});

// ---------------------------------------------------------------------------
// Enrolment (§4.4). Teacher lists / approves / rejects; student joins by code.
// ---------------------------------------------------------------------------
router.get('/:id/enrollments', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });

    const statusFilter = typeof req.query.status === 'string' ? req.query.status : null;
    let query = supabase
      .from('class_enrollments')
      .select('id, student_clerk_id, status, joined_via, requested_at, approved_at, removed_at')
      .eq('class_id', access.klass.id);
    if (statusFilter) query = query.eq('status', statusFilter);

    const { data, error } = await query;
    if (error) throw error;

    const ids = (data ?? []).map((r) => (r as { student_clerk_id: string }).student_clerk_id);
    const names = new Map<string, ProfileRow>();
    if (ids.length > 0) {
      const { data: profiles } = await supabase.from('profiles').select(PROFILE_COLUMNS).in('clerk_id', ids);
      for (const p of (profiles ?? []) as ProfileRow[]) names.set(p.clerk_id, p);
    }

    return res.json(
      (data ?? []).map((r) => {
        const row = r as Record<string, unknown> & { student_clerk_id: string };
        const p = names.get(row.student_clerk_id);
        return { ...row, full_name: p?.full_name ?? null, email: p?.email ?? null };
      })
    );
  } catch (err) {
    console.error('List enrollments error:', err);
    return res.status(500).json({ error: 'Failed to list enrollments' });
  }
});

async function decideEnrollments(
  req: AuthenticatedRequest,
  res: Response,
  decision: 'active' | 'rejected'
) {
  const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
  if (!access) return res.status(404).json({ error: 'Class not found' });

  const rawIds = req.body?.student_clerk_ids;
  const ids: string[] = Array.isArray(rawIds)
    ? rawIds.filter((x) => typeof x === 'string')
    : typeof req.body?.student_clerk_id === 'string'
      ? [req.body.student_clerk_id]
      : [];
  if (ids.length === 0) {
    return res.status(400).json({ error: 'student_clerk_ids is required' });
  }

  const now = new Date();
  const update: Record<string, unknown> =
    decision === 'active'
      ? { status: 'active', approved_at: now.toISOString() }
      : {
          status: 'rejected',
          // §4.4: rejected students may not retry for 24 hours.
          retry_blocked_until: new Date(now.getTime() + 24 * 60 * 60 * 1000).toISOString(),
        };

  const { data, error } = await supabase
    .from('class_enrollments')
    .update(update)
    .eq('class_id', access.klass.id)
    .in('student_clerk_id', ids)
    .eq('status', 'pending')
    .select('id, student_clerk_id, status');
  if (error) throw error;

  return res.json({ updated: (data ?? []).length, rows: data ?? [] });
}

router.post('/:id/enrollments/approve', async (req: AuthenticatedRequest, res: Response) => {
  try {
    return await decideEnrollments(req, res, 'active');
  } catch (err) {
    console.error('Approve enrollment error:', err);
    return res.status(500).json({ error: 'Failed to approve enrollments' });
  }
});

router.post('/:id/enrollments/reject', async (req: AuthenticatedRequest, res: Response) => {
  try {
    return await decideEnrollments(req, res, 'rejected');
  } catch (err) {
    console.error('Reject enrollment error:', err);
    return res.status(500).json({ error: 'Failed to reject enrollments' });
  }
});

// ---------------------------------------------------------------------------
// GET /classes/student/mine — the classrooms a student has joined, with the
// teacher's name (for the student Classroom hub).
// ---------------------------------------------------------------------------
router.get('/student/mine', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { data: enrollments, error } = await supabase
      .from('class_enrollments')
      .select('class_id, status')
      .eq('student_clerk_id', clerkId)
      .in('status', ['active', 'pending']);
    if (error) throw error;

    const classIds = ((enrollments ?? []) as { class_id: string; status: string }[]).map((e) => e.class_id);
    if (classIds.length === 0) return res.json([]);
    const statusByClass = new Map(((enrollments ?? []) as { class_id: string; status: string }[]).map((e) => [e.class_id, e.status]));

    const { data: klasses } = await supabase
      .from('classes')
      .select('id, name, subject, syllabus_code, level, owner_clerk_id')
      .in('id', classIds);
    const rows = (klasses ?? []) as ClassRow[];

    const ownerIds = Array.from(new Set(rows.map((c) => c.owner_clerk_id)));
    const teacherMap = new Map<string, string>();
    if (ownerIds.length > 0) {
      const { data: teachers } = await supabase.from('profiles').select('clerk_id, full_name, email').in('clerk_id', ownerIds);
      for (const t of (teachers ?? []) as ProfileRow[]) teacherMap.set(t.clerk_id, displayName(t.full_name, t.email, 'Teacher'));
    }

    return res.json(
      rows.map((c) => ({
        class_id: c.id,
        class_name: c.name,
        subject: c.subject,
        syllabus_code: c.syllabus_code,
        level: c.level,
        teacher_clerk_id: c.owner_clerk_id,
        teacher_name: teacherMap.get(c.owner_clerk_id) || 'Teacher',
        enrollment_status: statusByClass.get(c.id) || 'active',
      }))
    );
  } catch (err) {
    console.error('Student classrooms error:', err);
    return res.status(500).json({ error: 'Failed to load classrooms' });
  }
});

// ---------------------------------------------------------------------------
// POST /classes/join — a student joins by code (§3.2, §4.4). Case-insensitive.
// ---------------------------------------------------------------------------
router.post('/join', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const code = normalizeJoinCode(String(req.body?.code ?? ''));
    if (!code) return res.status(400).json({ error: 'code is required' });

    const { data: klass, error } = await supabase
      .from('classes')
      .select('*')
      .eq('join_code', code)
      .eq('join_enabled', true)
      .maybeSingle();
    if (error) throw error;
    if (!klass) return res.status(404).json({ error: 'No open class with that code' });

    const row = klass as ClassRow;
    if (row.archived_at) return res.status(400).json({ error: 'That class is archived' });

    // Capture the student's identity now so the teacher's Requests tab shows a
    // name/email instead of an anonymous "Student" (they join before onboarding).
    await ensureStudentProfile(clerkId);

    const { data: existing, error: existErr } = await supabase
      .from('class_enrollments')
      .select('id, status, retry_blocked_until')
      .eq('class_id', row.id)
      .eq('student_clerk_id', clerkId)
      .maybeSingle();
    if (existErr) throw existErr;

    // Class context the join UI uses to pre-fill onboarding and route to Classroom.
    const classMeta = { class_id: row.id, class_name: row.name, level: row.level, subject: row.subject };

    if (existing) {
      const ex = existing as { id: string; status: string; retry_blocked_until: string | null };
      if (ex.status === 'active') return res.status(200).json({ status: 'active', already: true, ...classMeta });
      if (ex.status === 'pending') return res.status(200).json({ status: 'pending', already: true, ...classMeta });
      if (ex.status === 'rejected' && ex.retry_blocked_until && new Date(ex.retry_blocked_until) > new Date()) {
        return res.status(429).json({ error: 'You were recently declined for this class. Try again later.' });
      }
      // removed or retry-window elapsed → re-request
      const nextStatus = row.auto_approve_joins ? 'active' : 'pending';
      const { error: updErr } = await supabase
        .from('class_enrollments')
        .update({
          status: nextStatus,
          joined_via: 'code',
          requested_at: new Date().toISOString(),
          approved_at: nextStatus === 'active' ? new Date().toISOString() : null,
          removed_at: null,
          retry_blocked_until: null,
        })
        .eq('id', ex.id);
      if (updErr) throw updErr;
      return res.status(200).json({ status: nextStatus, ...classMeta });
    }

    const nextStatus = row.auto_approve_joins ? 'active' : 'pending';
    const { error: insErr } = await supabase.from('class_enrollments').insert({
      class_id: row.id,
      student_clerk_id: clerkId,
      status: nextStatus,
      joined_via: 'code',
      approved_at: nextStatus === 'active' ? new Date().toISOString() : null,
    });
    if (insErr) throw insErr;

    return res.status(201).json({ status: nextStatus, ...classMeta });
  } catch (err) {
    console.error('Join class error:', err);
    return res.status(500).json({ error: 'Failed to join class' });
  }
});

// ===========================================================================
// Student management (spec §4.2 profile, §4.3 provisioning, §4.5 add/remove/transfer)
// Routes live under /classes/:id/students. `provision` is registered before the
// parameterised :studentClerkId routes so it is never captured as an id.
// ===========================================================================

// POST /classes/:id/students/provision — create accounts for a list of names.
router.post('/:id/students/provision', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });

    const raw = req.body?.students;
    const students: { name: string; email?: string }[] = Array.isArray(raw)
      ? raw
          .map((s: unknown) => {
            if (typeof s === 'string') return { name: s.trim() };
            const obj = s as { name?: unknown; email?: unknown };
            return {
              name: typeof obj?.name === 'string' ? obj.name.trim() : '',
              email: typeof obj?.email === 'string' && obj.email.trim() ? obj.email.trim() : undefined,
            };
          })
          .filter((s) => s.name)
      : [];

    if (students.length === 0) return res.status(400).json({ error: 'students (array of names) is required' });
    if (students.length > 200) return res.status(400).json({ error: 'Provision at most 200 students at a time' });

    const created: { name: string; username: string; password: string }[] = [];
    for (const student of students) {
      const clerkId = syntheticClerkId();
      const username = await generateUniqueUsername(student.name);
      const password = generatePassword();
      const passwordHash = await hashPassword(password);

      const { error: profileErr } = await supabase.from('profiles').insert({
        clerk_id: clerkId,
        full_name: student.name,
        email: student.email ?? null,
        role: 'student',
        username,
        password_hash: passwordHash,
        must_change_password: true,
        is_provisioned: true,
        provisioned_by: req.auth!.clerkId,
        institution_id: access.klass.institution_id,
      });
      if (profileErr) throw profileErr;

      const { error: enrollErr } = await supabase.from('class_enrollments').insert({
        class_id: access.klass.id,
        student_clerk_id: clerkId,
        status: 'active',
        joined_via: 'provision',
        approved_at: new Date().toISOString(),
      });
      if (enrollErr) throw enrollErr;

      created.push({ name: student.name, username, password });
    }

    // Plaintext passwords are returned exactly once, for the credential sheet.
    return res.status(201).json({ created });
  } catch (err) {
    console.error('Provision students error:', err);
    return res.status(500).json({ error: 'Failed to provision students' });
  }
});

// POST /classes/:id/students — add an existing student by email (§4.5).
router.post('/:id/students', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });

    const email = String(req.body?.email ?? '').trim().toLowerCase();
    if (!email) return res.status(400).json({ error: 'email is required' });

    const { data: target, error: targetErr } = await supabase
      .from('profiles')
      .select('clerk_id, role')
      .ilike('email', email)
      .maybeSingle();
    if (targetErr) throw targetErr;
    if (!target) return res.status(404).json({ error: 'No student account with that email' });

    const targetRow = target as { clerk_id: string; role: Role };
    if (targetRow.role !== 'student') return res.status(400).json({ error: 'That account is not a student' });

    const { error } = await supabase.from('class_enrollments').upsert(
      {
        class_id: access.klass.id,
        student_clerk_id: targetRow.clerk_id,
        status: 'active',
        joined_via: 'manual',
        approved_at: new Date().toISOString(),
        removed_at: null,
        retry_blocked_until: null,
      },
      { onConflict: 'class_id,student_clerk_id' }
    );
    if (error) throw error;

    return res.status(201).json({ ok: true, student_clerk_id: targetRow.clerk_id });
  } catch (err) {
    console.error('Add student error:', err);
    return res.status(500).json({ error: 'Failed to add student' });
  }
});

// GET /classes/:id/students/:studentClerkId — student profile (§4.2).
router.get('/:id/students/:studentClerkId', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    const studentClerkId = req.params.studentClerkId;

    const { data: enrollment, error: enrErr } = await supabase
      .from('class_enrollments')
      .select('id, status, joined_via, requested_at, approved_at, removed_at')
      .eq('class_id', access.klass.id)
      .eq('student_clerk_id', studentClerkId)
      .maybeSingle();
    if (enrErr) throw enrErr;
    if (!enrollment) return res.status(404).json({ error: 'Student not in this class' });

    const { data: profile, error: profErr } = await supabase
      .from('profiles')
      .select('clerk_id, full_name, email, username, is_provisioned')
      .eq('clerk_id', studentClerkId)
      .maybeSingle();
    if (profErr) throw profErr;

    // Subject switcher (§4.2): other classes this teacher runs that hold the
    // same student. Profile metrics are always scoped to one subject/class.
    const { data: myClasses } = await supabase
      .from('classes')
      .select('id, name, subject, syllabus_code')
      .eq('owner_clerk_id', req.auth!.clerkId);
    const myClassIds = ((myClasses ?? []) as { id: string }[]).map((c) => c.id);
    let siblingClasses: { id: string; name: string; subject: string; syllabus_code: string }[] = [];
    if (myClassIds.length > 0) {
      const { data: siblings } = await supabase
        .from('class_enrollments')
        .select('class_id')
        .eq('student_clerk_id', studentClerkId)
        .eq('status', 'active')
        .in('class_id', myClassIds);
      const siblingIds = new Set(((siblings ?? []) as { class_id: string }[]).map((s) => s.class_id));
      siblingClasses = ((myClasses ?? []) as typeof siblingClasses).filter((c) => siblingIds.has(c.id));
    }

    return res.json({
      profile: profile ?? { clerk_id: studentClerkId, full_name: null, email: null, username: null, is_provisioned: false },
      enrollment,
      class: {
        id: access.klass.id,
        name: access.klass.name,
        subject: access.klass.subject,
        syllabus_code: access.klass.syllabus_code,
        level: access.klass.level,
      },
      sibling_classes: siblingClasses,
      // Metrics require the attempts/marks data built in Phase 4; honest empty
      // state until then rather than fabricated numbers.
      metrics: null,
      topic_mastery: [],
      attempts: [],
    });
  } catch (err) {
    console.error('Get student profile error:', err);
    return res.status(500).json({ error: 'Failed to load student' });
  }
});

// POST /classes/:id/students/:studentClerkId/remove — unlink, preserve attempts.
router.post('/:id/students/:studentClerkId/remove', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });

    const { error } = await supabase
      .from('class_enrollments')
      .update({ status: 'removed', removed_at: new Date().toISOString() })
      .eq('class_id', access.klass.id)
      .eq('student_clerk_id', req.params.studentClerkId);
    if (error) throw error;
    return res.json({ ok: true });
  } catch (err) {
    console.error('Remove student error:', err);
    return res.status(500).json({ error: 'Failed to remove student' });
  }
});

// POST /classes/:id/students/:studentClerkId/transfer — move to another class
// owned by the same teacher, carrying attempt history (§4.5).
router.post('/:id/students/:studentClerkId/transfer', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const sourceAccess = await resolveClassAccess(req.params.id, clerkId);
    if (!sourceAccess) return res.status(404).json({ error: 'Class not found' });

    const targetClassId = String(req.body?.target_class_id ?? '');
    if (!targetClassId) return res.status(400).json({ error: 'target_class_id is required' });
    if (targetClassId === sourceAccess.klass.id) return res.status(400).json({ error: 'Target class must differ from the source' });

    const targetAccess = await resolveClassAccess(targetClassId, clerkId);
    if (!targetAccess || !targetAccess.isOwner) {
      return res.status(403).json({ error: 'You can only transfer into a class you own' });
    }

    const studentClerkId = req.params.studentClerkId;

    // Remove from source (attempts stay keyed on student_clerk_id).
    const { error: srcErr } = await supabase
      .from('class_enrollments')
      .update({ status: 'removed', removed_at: new Date().toISOString() })
      .eq('class_id', sourceAccess.klass.id)
      .eq('student_clerk_id', studentClerkId);
    if (srcErr) throw srcErr;

    // Activate in target.
    const { error: tgtErr } = await supabase.from('class_enrollments').upsert(
      {
        class_id: targetClassId,
        student_clerk_id: studentClerkId,
        status: 'active',
        joined_via: 'transfer',
        approved_at: new Date().toISOString(),
        removed_at: null,
        retry_blocked_until: null,
      },
      { onConflict: 'class_id,student_clerk_id' }
    );
    if (tgtErr) throw tgtErr;

    return res.json({ ok: true, target_class_id: targetClassId });
  } catch (err) {
    console.error('Transfer student error:', err);
    return res.status(500).json({ error: 'Failed to transfer student' });
  }
});

export default router;
