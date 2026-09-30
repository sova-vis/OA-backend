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
import { createStaffAccount } from './services/staffAccounts';

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

// ---------------------------------------------------------------------------
// §3.1 Create a school + its limits + the first school-admin (forced reset).
// ---------------------------------------------------------------------------
router.post('/schools', async (req: ActorRequest, res: Response) => {
  try {
    const b = req.body || {};
    if (!b.name || typeof b.name !== 'string') return res.status(400).json({ error: 'School name is required' });
    if (!b.admin || !b.admin.email || !b.admin.name) {
      return res.status(400).json({ error: 'First school admin (admin.email, admin.name) is required' });
    }

    const { data: sData, error: sErr } = await supabase.from('schools').insert({
      name: b.name.trim(),
      logo_url: b.logo_url ?? null,
      licence_start: b.licence_start ?? null,
      licence_expiry: b.licence_expiry ?? null,
      discount_pct: typeof b.discount_pct === 'number' ? b.discount_pct : 0,
      allow_admin_script_view: !!b.allow_admin_script_view,
      feature_flags: b.feature_flags ?? {},
      created_by: req.actor?.clerkId ?? null,
    }).select('*').single();
    if (sErr) throw sErr;
    const school = sData as { id: string; name: string };

    const limitRow: Record<string, unknown> = { school_id: school.id };
    if (b.limits && typeof b.limits === 'object') {
      for (const k of LIMIT_FIELDS) if (b.limits[k] !== undefined) limitRow[k] = b.limits[k];
    }
    const { data: limitsData } = await supabase.from('school_limits').insert(limitRow).select('*').single();

    let adminResult;
    try {
      adminResult = await createStaffAccount({
        email: b.admin.email, name: b.admin.name, role: 'school_admin',
        schoolId: school.id, password: b.admin.password, createdBy: req.actor?.clerkId,
      });
    } catch (e: unknown) {
      const err = e as { statusCode?: number; message?: string };
      return res.status(err.statusCode || 500).json({
        error: `School created but the admin account failed: ${err.message}. Add one via POST /owner/schools/${school.id}/admins.`,
        school,
      });
    }

    await logAudit({
      actorClerkId: req.actor?.clerkId, actorRole: req.actor?.role,
      action: 'school.create', targetType: 'school', targetId: school.id,
      after: { name: school.name }, schoolId: school.id,
    });

    return res.status(201).json({
      school, limits: limitsData,
      admin: {
        email: adminResult.email, tempPassword: adminResult.tempPassword,
        existed: adminResult.existed, mustChangePassword: !adminResult.existed,
      },
    });
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
    return res.json({ ...(data as object), ...(await schoolUsage(req.params.id)) });
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
    const { email, name, password } = req.body || {};
    if (!email || !name) return res.status(400).json({ error: 'email and name are required' });

    const result = await createStaffAccount({
      email, name, role: 'school_admin', schoolId: req.params.id, password, createdBy: req.actor?.clerkId,
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

export default router;
