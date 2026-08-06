import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';

/**
 * Teacher Portal — institution layer (spec feature group 14) + permissions
 * (§15) + compliance views (§18). The institution admin is an account manager,
 * NOT a super-teacher: no grading, no viewing individual student scripts (§14.3,
 * §14.5). Aggregate performance only.
 */

const router = Router();
router.use(clerkAuth);

// Resolve the caller's admin institution: platform 'admin' role, or the user
// recorded as a claimed institution's admin_clerk_id.
async function adminInstitution(clerkId: string): Promise<string | null> {
  const { data: profile } = await supabase.from('profiles').select('role, institution_id').eq('clerk_id', clerkId).maybeSingle();
  const p = profile as { role?: string; institution_id?: string } | null;
  if (!p) return null;
  const { data: inst } = await supabase.from('institutions').select('id').eq('admin_clerk_id', clerkId).maybeSingle();
  if (inst) return (inst as { id: string }).id;
  if (p.role === 'admin' && p.institution_id) return p.institution_id;
  return null;
}

async function requireAdmin(req: AuthenticatedRequest, res: Response): Promise<string | null> {
  const inst = await adminInstitution(req.auth!.clerkId);
  if (!inst) {
    res.status(403).json({ error: 'Institution admin access required' });
    return null;
  }
  return inst;
}

// GET /institution — account info + seat usage (§14.1, §14.4).
router.get('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const instId = await requireAdmin(req, res);
    if (!instId) return;
    const { data: inst } = await supabase.from('institutions').select('*').eq('id', instId).maybeSingle();
    const [{ count: teacherCount }, { data: classes }] = await Promise.all([
      supabase.from('profiles').select('id', { count: 'exact', head: true }).eq('institution_id', instId).eq('role', 'teacher'),
      supabase.from('classes').select('id').eq('institution_id', instId),
    ]);
    const classIds = ((classes ?? []) as { id: string }[]).map((c) => c.id);
    let studentCount = 0;
    if (classIds.length > 0) {
      const { data: enr } = await supabase.from('class_enrollments').select('student_clerk_id').in('class_id', classIds).eq('status', 'active');
      studentCount = new Set(((enr ?? []) as { student_clerk_id: string }[]).map((e) => e.student_clerk_id)).size;
    }
    const row = inst as Record<string, unknown>;
    return res.json({
      ...row,
      seat_usage: {
        teachers: teacherCount ?? 0,
        students: studentCount,
        seat_limit_teachers: row.seat_limit_teachers ?? null,
        seat_limit_students: row.seat_limit_students ?? null,
      },
    });
  } catch (err) {
    console.error('Institution info error:', err);
    return res.status(500).json({ error: 'Failed to load institution' });
  }
});

// PATCH /institution — account settings, academic year, retention, floor.
router.patch('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const instId = await requireAdmin(req, res);
    if (!instId) return;
    const b = req.body ?? {};
    const update: Record<string, unknown> = { updated_at: new Date().toISOString() };
    for (const key of ['name', 'address', 'country', 'contact_email', 'logo_url']) {
      if (typeof b[key] === 'string') update[key] = b[key];
    }
    if (b.academic_year_start) update.academic_year_start = b.academic_year_start;
    if (b.academic_year_end) update.academic_year_end = b.academic_year_end;
    if (b.auto_approve_floor !== undefined) update.auto_approve_floor = b.auto_approve_floor === null ? null : Number(b.auto_approve_floor);
    if (b.retention_policy && typeof b.retention_policy === 'object') update.retention_policy = b.retention_policy;
    const { data, error } = await supabase.from('institutions').update(update).eq('id', instId).select('*').single();
    if (error) throw error;
    return res.json(data);
  } catch (err) {
    console.error('Institution update error:', err);
    return res.status(500).json({ error: 'Failed to update institution' });
  }
});

// GET /institution/teachers — teacher activity (§14.3). No script access.
router.get('/teachers', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const instId = await requireAdmin(req, res);
    if (!instId) return;
    const { data: teachers } = await supabase.from('profiles').select('clerk_id, full_name, email, deactivated_at, updated_at').eq('institution_id', instId).eq('role', 'teacher');
    const rows = (teachers ?? []) as Record<string, unknown>[];
    const ids = rows.map((t) => t.clerk_id as string);
    const classCount = new Map<string, number>();
    const asgCount = new Map<string, number>();
    if (ids.length > 0) {
      const { data: classes } = await supabase.from('classes').select('id, owner_clerk_id').in('owner_clerk_id', ids);
      for (const c of (classes ?? []) as { owner_clerk_id: string }[]) classCount.set(c.owner_clerk_id, (classCount.get(c.owner_clerk_id) ?? 0) + 1);
      const { data: asgs } = await supabase.from('assignments').select('owner_clerk_id').in('owner_clerk_id', ids);
      for (const a of (asgs ?? []) as { owner_clerk_id: string }[]) asgCount.set(a.owner_clerk_id, (asgCount.get(a.owner_clerk_id) ?? 0) + 1);
    }
    return res.json(
      rows.map((t) => ({
        clerk_id: t.clerk_id,
        full_name: t.full_name,
        email: t.email,
        deactivated: Boolean(t.deactivated_at),
        classes: classCount.get(t.clerk_id as string) ?? 0,
        assignments: asgCount.get(t.clerk_id as string) ?? 0,
      }))
    );
  } catch (err) {
    console.error('Teachers error:', err);
    return res.status(500).json({ error: 'Failed to load teachers' });
  }
});

router.post('/teachers/:clerkId/deactivate', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const instId = await requireAdmin(req, res);
    if (!instId) return;
    const active = req.body?.active === true;
    await supabase.from('profiles').update({ deactivated_at: active ? null : new Date().toISOString() }).eq('clerk_id', req.params.clerkId).eq('institution_id', instId);
    return res.json({ ok: true });
  } catch (err) {
    console.error('Deactivate teacher error:', err);
    return res.status(500).json({ error: 'Failed to update teacher' });
  }
});

// GET /institution/performance — aggregate only, never scripts (§14.5).
router.get('/performance', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const instId = await requireAdmin(req, res);
    if (!instId) return;
    const { data: classes } = await supabase.from('classes').select('id, name, subject, level, syllabus_code').eq('institution_id', instId);
    const rows = (classes ?? []) as Record<string, unknown>[];
    const classIds = rows.map((c) => c.id as string);
    const perf = new Map<string, { students: number; scoreSum: number; scoreN: number }>();
    if (classIds.length > 0) {
      const { data: enr } = await supabase.from('class_enrollments').select('class_id, student_clerk_id').in('class_id', classIds).eq('status', 'active');
      for (const e of (enr ?? []) as { class_id: string; student_clerk_id: string }[]) {
        const p = perf.get(e.class_id) ?? { students: 0, scoreSum: 0, scoreN: 0 };
        p.students += 1;
        perf.set(e.class_id, p);
      }
      const { data: asgs } = await supabase.from('assignments').select('id, class_id').in('class_id', classIds);
      const asgToClass = new Map(((asgs ?? []) as { id: string; class_id: string }[]).map((a) => [a.id, a.class_id]));
      const asgIds = Array.from(asgToClass.keys());
      if (asgIds.length > 0) {
        const { data: subs } = await supabase.from('submissions').select('assignment_id, total_score, total_marks, status').in('assignment_id', asgIds).in('status', ['submitted', 'late']);
        for (const s of (subs ?? []) as Record<string, unknown>[]) {
          if (!(Number(s.total_marks) > 0)) continue;
          const cid = asgToClass.get(s.assignment_id as string);
          if (!cid) continue;
          const p = perf.get(cid) ?? { students: 0, scoreSum: 0, scoreN: 0 };
          p.scoreSum += Number(s.total_score) / Number(s.total_marks);
          p.scoreN += 1;
          perf.set(cid, p);
        }
      }
    }
    return res.json(
      rows.map((c) => {
        const p = perf.get(c.id as string);
        return { id: c.id, name: c.name, subject: c.subject, level: c.level, students: p?.students ?? 0, avg: p && p.scoreN > 0 ? Math.round((p.scoreSum / p.scoreN) * 100) : null };
      })
    );
  } catch (err) {
    console.error('Performance error:', err);
    return res.status(500).json({ error: 'Failed to load performance' });
  }
});

// ---- Scope grants (§14.6 / §4.2) ----
router.get('/scope-grants', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const instId = await requireAdmin(req, res);
    if (!instId) return;
    const { data } = await supabase.from('scope_grants').select('*').eq('institution_id', instId);
    return res.json(data ?? []);
  } catch (err) {
    console.error('Grants list error:', err);
    return res.status(500).json({ error: 'Failed to load grants' });
  }
});

router.post('/scope-grants', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const instId = await requireAdmin(req, res);
    if (!instId) return;
    const email = String(req.body?.email ?? '').trim().toLowerCase();
    if (!email) return res.status(400).json({ error: 'email is required' });
    const { data: target } = await supabase.from('profiles').select('clerk_id').ilike('email', email).maybeSingle();
    if (!target) return res.status(404).json({ error: 'No user with that email' });

    const { data, error } = await supabase
      .from('scope_grants')
      .insert({
        user_clerk_id: (target as { clerk_id: string }).clerk_id,
        institution_id: instId,
        filter_subjects: Array.isArray(req.body?.filter_subjects) ? req.body.filter_subjects : [],
        filter_levels: Array.isArray(req.body?.filter_levels) ? req.body.filter_levels : [],
        capabilities: Array.isArray(req.body?.capabilities) && req.body.capabilities.length ? req.body.capabilities : ['view'],
        label: req.body?.label?.trim() || null,
      })
      .select('*')
      .single();
    if (error) throw error;
    // §14.6: grant holders receive a notification when granted.
    await supabase.from('notifications').insert({
      recipient_clerk_id: (target as { clerk_id: string }).clerk_id,
      type: 'moderator_flag',
      body: `You were granted oversight: ${req.body?.label || 'scope grant'}`,
    });
    await supabase.from('activity_log').insert({ actor_clerk_id: req.auth!.clerkId, institution_id: instId, event_type: 'scope_grant_change', target_type: 'scope_grant', target_id: (data as { id: string }).id, detail: { action: 'create' } });
    return res.status(201).json(data);
  } catch (err) {
    console.error('Grant create error:', err);
    return res.status(500).json({ error: 'Failed to create grant' });
  }
});

router.delete('/scope-grants/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const instId = await requireAdmin(req, res);
    if (!instId) return;
    await supabase.from('scope_grants').delete().eq('id', req.params.id).eq('institution_id', instId);
    await supabase.from('activity_log').insert({ actor_clerk_id: req.auth!.clerkId, institution_id: instId, event_type: 'scope_grant_change', target_type: 'scope_grant', target_id: req.params.id, detail: { action: 'revoke' } });
    return res.json({ ok: true });
  } catch (err) {
    console.error('Grant delete error:', err);
    return res.status(500).json({ error: 'Failed to revoke grant' });
  }
});

// GET /institution/activity-log — immutable audit for the institution (§18.3).
router.get('/activity-log', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const instId = await requireAdmin(req, res);
    if (!instId) return;
    // Logs are written with institution_id where known; also include class-scoped
    // events by resolving the institution's class ids is heavier — return the
    // institution-tagged log for now.
    const { data } = await supabase.from('activity_log').select('*').eq('institution_id', instId).order('created_at', { ascending: false }).limit(200);
    return res.json(data ?? []);
  } catch (err) {
    console.error('Activity log error:', err);
    return res.status(500).json({ error: 'Failed to load activity log' });
  }
});

// ---- Account deletion (§18.2) — institution admin action ----
router.post('/deletion-requests', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const instId = await requireAdmin(req, res);
    if (!instId) return;
    const target = String(req.body?.target_clerk_id ?? '');
    if (!target) return res.status(400).json({ error: 'target_clerk_id is required' });
    const { data, error } = await supabase.from('deletion_requests').insert({ target_clerk_id: target, requested_by: req.auth!.clerkId, institution_id: instId, reason: req.body?.reason ?? null }).select('*').single();
    if (error) throw error;
    await supabase.from('activity_log').insert({ actor_clerk_id: req.auth!.clerkId, institution_id: instId, event_type: 'account_deletion', target_type: 'profile', target_id: null, detail: { target, reason: req.body?.reason ?? null } });
    return res.status(201).json(data);
  } catch (err) {
    console.error('Deletion request error:', err);
    return res.status(500).json({ error: 'Failed to record deletion request' });
  }
});

export default router;
