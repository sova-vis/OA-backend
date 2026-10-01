/**
 * Platform-owner console (spec §3). "The layer that protects margin" — creates
 * and governs schools, sets per-school limits/quotas/entitlements, and monitors
 * usage. Every limit here is enforced server-side (see lib/quota.ts).
 *
 * Guarded by requireOwner (role owner|admin). Mounted at /owner.
 */
import { Router, Response } from 'express';
import { clerkAuth } from './lib/clerkAuth';
import { ActorRequest, requireOwner } from './lib/roles';
import { supabase } from './lib/supabase';
import { logAudit } from './lib/audit';
import { getLimits, markingQuotaStatus, askAiUsage, countActiveTeachers, countStudents } from './lib/quota';
import { createStaffAccount, schoolShortCode, resetStaffPassword, updateStaffEmail } from './services/staffAccounts';
import { schoolTotals } from './lib/schoolStats';
import { deleteSchoolCascade } from './services/cascadeDelete';

const router = Router();
router.use(clerkAuth, requireOwner);

const LIMIT_FIELDS = [
  'max_teachers', 'max_students_per_teacher', 'max_classes_per_teacher',
  'max_students_total', 'marking_quota_units', 'quota_period', 'askai_allowance',
  'subject_entitlements', 'storage_cap_mb',
] as const;

const SCHOOL_PATCH_FIELDS = [
  'name', 'logo_url', 'status', 'licence_start', 'licence_expiry',
  'discount_pct', 'allow_admin_script_view', 'feature_flags',
] as const;

async function schoolUsage(schoolId: string) {
  const [limits, teachers, students, marking, askai] = await Promise.all([
    getLimits(schoolId),
    countActiveTeachers(schoolId),
    countStudents(schoolId),
    markingQuotaStatus(schoolId),
    askAiUsage(schoolId),
  ]);
  return {
    limits,
    usage: {
      teachers, students, marking, askai,
      seats: limits ? { teachers_used: teachers, teachers_max: limits.max_teachers, students_used: students, students_max: limits.max_students_total } : null,
    },
  };
}

// §6.2 upsell funnel for one school: how many classroom students were shown a
// Pro prompt vs how many converted (from funnel_events).
async function schoolFunnel(schoolId: string) {
  const { data } = await supabase.from('funnel_events').select('kind, student_clerk_id').eq('school_id', schoolId);
  const rows = (data ?? []) as { kind: string; student_clerk_id: string }[];
  const promptsShown = rows.filter((r) => r.kind === 'prompt_shown').length;
  const conversions = rows.filter((r) => r.kind === 'converted').length;
  const studentsPrompted = new Set(rows.filter((r) => r.kind === 'prompt_shown').map((r) => r.student_clerk_id)).size;
  const rate = studentsPrompted > 0 ? Math.round((conversions / studentsPrompted) * 100) : 0;
  return { prompts_shown: promptsShown, students_prompted: studentsPrompted, conversions, rate };
}

/** A school short code that's unique across schools (feature_flags.short_code). */
async function uniqueShortCode(name: string): Promise<string> {
  const base = schoolShortCode(name);
  const { data } = await supabase.from('schools').select('feature_flags');
  const taken = new Set(
    ((data ?? []) as { feature_flags?: Record<string, unknown> }[])
      .map((r) => r.feature_flags?.short_code).filter((x): x is string => typeof x === 'string'),
  );
  if (!taken.has(base)) return base;
  for (let i = 2; i < 100; i++) if (!taken.has(`${base}${i}`)) return `${base}${i}`;
  return `${base}${Date.now().toString().slice(-4)}`;
}

// ---------------------------------------------------------------------------
// §3.1 Create a school + its limits. The first school admin is added afterwards
// from the school page (by name → generated login), so no email is needed here.
// ---------------------------------------------------------------------------
router.post('/schools', async (req: ActorRequest, res: Response) => {
  try {
    const b = req.body || {};
    if (!b.name || typeof b.name !== 'string') return res.status(400).json({ error: 'School name is required' });

    const shortCode = await uniqueShortCode(b.name);
    const { data: sData, error: sErr } = await supabase.from('schools').insert({
      name: b.name.trim(),
      logo_url: b.logo_url ?? null,
      licence_start: b.licence_start ?? null,
      licence_expiry: b.licence_expiry ?? null,
      discount_pct: typeof b.discount_pct === 'number' ? b.discount_pct : 0,
      allow_admin_script_view: !!b.allow_admin_script_view,
      feature_flags: { ...(b.feature_flags ?? {}), short_code: shortCode },
      created_by: req.actor?.clerkId ?? null,
    }).select('*').single();
    if (sErr) throw sErr;
    const school = sData as { id: string; name: string };

    const limitRow: Record<string, unknown> = { school_id: school.id };
    if (b.limits && typeof b.limits === 'object') {
      for (const k of LIMIT_FIELDS) if (b.limits[k] !== undefined) limitRow[k] = b.limits[k];
    }
    const { data: limitsData } = await supabase.from('school_limits').insert(limitRow).select('*').single();

    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: 'school.create', targetType: 'school', targetId: school.id,
      after: { name: school.name, short_code: shortCode }, schoolId: school.id,
    });

    return res.status(201).json({ school, limits: limitsData, short_code: shortCode });
  } catch (err: unknown) {
    console.error('POST /owner/schools', err);
    return res.status(500).json({ error: (err as Error).message || 'Failed to create school' });
  }
});

// §3.4 List schools with seat + quota usage.
router.get('/schools', async (_req: ActorRequest, res: Response) => {
  try {
    const { data } = await supabase.from('schools').select('*').order('created_at', { ascending: false });
    const schools = (data as { id: string }[] | null) ?? [];
    const out = [];
    for (const s of schools) out.push({ ...s, ...(await schoolUsage(s.id)) });
    return res.json({ schools: out });
  } catch (err: unknown) {
    console.error('GET /owner/schools', err);
    return res.status(500).json({ error: 'Failed to list schools' });
  }
});

// §3.4 One school, full detail.
router.get('/schools/:id', async (req: ActorRequest, res: Response) => {
  try {
    const { data } = await supabase.from('schools').select('*').eq('id', req.params.id).maybeSingle();
    if (!data) return res.status(404).json({ error: 'School not found' });
    const [usage, funnel, totals] = await Promise.all([schoolUsage(req.params.id), schoolFunnel(req.params.id), schoolTotals(req.params.id)]);
    return res.json({ ...(data as object), ...usage, funnel, totals });
  } catch (err: unknown) {
    console.error('GET /owner/schools/:id', err);
    return res.status(500).json({ error: 'Failed to load school' });
  }
});

// §3.1 Suspend / reactivate, licence, discount, feature flags, branding.
router.patch('/schools/:id', async (req: ActorRequest, res: Response) => {
  try {
    const { data: before } = await supabase.from('schools').select('*').eq('id', req.params.id).maybeSingle();
    if (!before) return res.status(404).json({ error: 'School not found' });

    const patch: Record<string, unknown> = { updated_at: new Date().toISOString() };
    for (const k of SCHOOL_PATCH_FIELDS) if (req.body?.[k] !== undefined) patch[k] = req.body[k];
    if (patch.status && !['active', 'suspended', 'pending'].includes(String(patch.status))) {
      return res.status(400).json({ error: 'Invalid status' });
    }

    const { data: after, error } = await supabase.from('schools').update(patch).eq('id', req.params.id).select('*').single();
    if (error) throw error;

    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: 'school.update', targetType: 'school', targetId: req.params.id,
      before, after, schoolId: req.params.id,
    });
    return res.json({ school: after });
  } catch (err: unknown) {
    console.error('PATCH /owner/schools/:id', err);
    return res.status(500).json({ error: 'Failed to update school' });
  }
});

// §3.2 Set the per-school limits / quota / entitlements.
router.put('/schools/:id/limits', async (req: ActorRequest, res: Response) => {
  try {
    const { data: school } = await supabase.from('schools').select('id').eq('id', req.params.id).maybeSingle();
    if (!school) return res.status(404).json({ error: 'School not found' });

    const { data: before } = await supabase.from('school_limits').select('*').eq('school_id', req.params.id).maybeSingle();
    const row: Record<string, unknown> = { school_id: req.params.id, updated_at: new Date().toISOString() };
    for (const k of LIMIT_FIELDS) if (req.body?.[k] !== undefined) row[k] = req.body[k];
    if (row.quota_period && !['month', 'term'].includes(String(row.quota_period))) {
      return res.status(400).json({ error: 'quota_period must be month or term' });
    }

    const { data: after, error } = await supabase.from('school_limits').upsert(row, { onConflict: 'school_id' }).select('*').single();
    if (error) throw error;

    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: 'school.limits.update', targetType: 'school_limits', targetId: req.params.id,
      before, after, schoolId: req.params.id,
    });
    return res.json({ limits: after });
  } catch (err: unknown) {
    console.error('PUT /owner/schools/:id/limits', err);
    return res.status(500).json({ error: 'Failed to set limits' });
  }
});

// Add another school-admin to an existing school.
router.post('/schools/:id/admins', async (req: ActorRequest, res: Response) => {
  try {
    const { data: school } = await supabase.from('schools').select('id').eq('id', req.params.id).maybeSingle();
    if (!school) return res.status(404).json({ error: 'School not found' });
    const { name } = req.body || {};
    if (!name) return res.status(400).json({ error: 'name is required' });

    const result = await createStaffAccount({
      name, role: 'school_admin', schoolId: req.params.id, createdBy: req.actor?.clerkId,
    });
    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: 'school_admin.create', targetType: 'profile', targetId: result.clerkId, schoolId: req.params.id,
    });
    return res.status(201).json({
      admin: { email: result.email, tempPassword: result.tempPassword, existed: result.existed },
    });
  } catch (err: unknown) {
    const e = err as { statusCode?: number; message?: string };
    console.error('POST /owner/schools/:id/admins', err);
    return res.status(e.statusCode || 500).json({ error: e.message || 'Failed to add admin' });
  }
});

// List the school-admins of a school, so the owner can see who has access.
router.get('/schools/:id/admins', async (req: ActorRequest, res: Response) => {
  try {
    const { data, error } = await supabase
      .from('profiles')
      .select('clerk_id, full_name, email, deactivated_at, must_change_password, created_at')
      .eq('school_id', req.params.id).eq('role', 'school_admin')
      .order('created_at', { ascending: true });
    if (error) throw error;
    return res.json({ admins: data ?? [] });
  } catch (err: unknown) {
    const e = err as { message?: string };
    console.error('GET /owner/schools/:id/admins', err);
    return res.status(500).json({ error: e.message || 'Failed to load admins' });
  }
});

// Verify a school_admin belongs to this school (shared guard for the two routes below).
async function adminInSchool(schoolId: string, clerkId: string): Promise<{ email: string | null } | null> {
  const { data } = await supabase
    .from('profiles').select('clerk_id, school_id, role, email')
    .eq('clerk_id', clerkId).maybeSingle();
  const a = data as { school_id?: string; role?: string; email?: string } | null;
  if (!a || a.school_id !== schoolId || a.role !== 'school_admin') return null;
  return { email: a.email ?? null };
}

// Reset a school-admin's password → a fresh one-time password + forced reset.
router.post('/schools/:id/admins/:clerkId/reset-password', async (req: ActorRequest, res: Response) => {
  try {
    if (!(await adminInSchool(req.params.id, req.params.clerkId))) {
      return res.status(404).json({ error: 'Admin not found in this school' });
    }
    const tempPassword = await resetStaffPassword(req.params.clerkId);
    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: 'school_admin.reset_password', targetType: 'profile', targetId: req.params.clerkId, schoolId: req.params.id,
    });
    return res.json({ tempPassword });
  } catch (err: unknown) {
    const e = err as { statusCode?: number; message?: string };
    console.error('POST /owner/schools/:id/admins/:clerkId/reset-password', err);
    return res.status(e.statusCode || 500).json({ error: e.message || 'Failed to reset password' });
  }
});

// Change a school-admin's login email.
router.post('/schools/:id/admins/:clerkId/email', async (req: ActorRequest, res: Response) => {
  try {
    if (!(await adminInSchool(req.params.id, req.params.clerkId))) {
      return res.status(404).json({ error: 'Admin not found in this school' });
    }
    const email = await updateStaffEmail(req.params.clerkId, String(req.body?.email ?? ''));
    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: 'school_admin.update_email', targetType: 'profile', targetId: req.params.clerkId, schoolId: req.params.id,
    });
    return res.json({ email });
  } catch (err: unknown) {
    const e = err as { statusCode?: number; message?: string };
    console.error('POST /owner/schools/:id/admins/:clerkId/email', err);
    return res.status(e.statusCode || 500).json({ error: e.message || 'Failed to update email' });
  }
});

// DELETE /owner/schools/:id — permanently delete a school and cascade. Hard delete:
// removes the school + its school-admins + teachers (and their classes, assignments,
// submissions); enrolled students keep their accounts but are unenrolled and their
// school-tied answers are cleared. Requires the exact school name retyped in `confirm`.
router.delete('/schools/:id', async (req: ActorRequest, res: Response) => {
  try {
    const { data: school } = await supabase.from('schools').select('id, name').eq('id', req.params.id).maybeSingle();
    if (!school) return res.status(404).json({ error: 'School not found' });
    const name = ((school as { name?: string }).name || '').trim();
    const typed = String(req.body?.confirm ?? '').trim();
    if (!typed || typed.toLowerCase() !== name.toLowerCase()) {
      return res.status(400).json({ error: 'The name you typed does not match the school name.' });
    }
    // Log before the row is gone (audit_log.school_id becomes NULL after the delete).
    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: 'school.delete', targetType: 'school', targetId: req.params.id, schoolId: req.params.id,
    });
    const summary = await deleteSchoolCascade(req.params.id);
    return res.json({ ok: true, ...summary });
  } catch (err: unknown) {
    const e = err as { statusCode?: number; message?: string };
    console.error('DELETE /owner/schools/:id', err);
    return res.status(e.statusCode || 500).json({ error: e.message || 'Failed to delete school' });
  }
});

export default router;
