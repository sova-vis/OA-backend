import { Router, Request, Response } from 'express';
import { createTeacherAccount } from './services/adminService';
import { AuthenticatedRequest, clerkAuth, requireRole } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import { activateManualPro, revokePro, computeAccess, MANUAL_PLAN_DAYS, BillingRow } from './lib/entitlements';
import { sendProWelcome } from './lib/proNotify';

const router = Router();

/** Build a default 'free' billing view for a user with no billing row yet. */
function accessForClerk(billingByClerk: Map<string, BillingRow>, clerkId: string) {
  const row = billingByClerk.get(clerkId);
  if (!row) {
    return { status: 'free', isPro: false, daysLeft: null, plan: null, currentPeriodEnd: null, trialEndsAt: null, autoRenew: false, trialAvailable: true };
  }
  return computeAccess(row);
}

// Deprecated legacy endpoint kept for backward compatibility
router.post('/login', (req: Request, res: Response) => {
  return res.status(410).json({
    error: 'Admin login is deprecated',
    message: 'Sign in on the landing page and ensure your profile role is admin.',
  });
});

// Add teacher using Supabase Admin
router.post('/add-teacher', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  const { email, password, name } = req.body;
  if (!email || !password || !name) return res.status(400).json({ error: 'Missing fields' });

  try {
    const user = await createTeacherAccount(email, password, name);
    return res.json({ message: 'Teacher created successfully', userId: user.id });
  } catch (error: any) {
    console.error("Create teacher error:", error);
    const statusCode = typeof error?.statusCode === 'number' ? error.statusCode : 500;
    return res.status(statusCode).json({ error: error.message || 'Failed to create teacher' });
  }
});

router.get('/teachers', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const { data: teachers, error: teacherError } = await supabase
      .from('profiles')
      .select('clerk_id, full_name, email, role')
      .eq('role', 'teacher')
      .order('full_name', { ascending: true });

    if (teacherError) {
      throw teacherError;
    }

    const teacherIds = (teachers ?? []).map((teacher) => teacher.clerk_id);

    let details: Array<Record<string, unknown>> = [];
    if (teacherIds.length > 0) {
      const { data, error } = await supabase
        .from('teacher_profiles')
        .select('clerk_id, headline, bio, subjects, availability, meeting_provider, is_active, updated_at')
        .in('clerk_id', teacherIds);

      if (error) {
        throw error;
      }

      details = data ?? [];
    }

    const detailsById = new Map<string, Record<string, unknown>>();
    for (const row of details) {
      const key = typeof row.clerk_id === 'string' ? row.clerk_id : '';
      if (key) {
        detailsById.set(key, row);
      }
    }

    return res.json({
      teachers: (teachers ?? []).map((teacher) => {
        const detail = detailsById.get(teacher.clerk_id);
        return {
          ...teacher,
          headline: detail?.headline ?? null,
          bio: detail?.bio ?? null,
          subjects: Array.isArray(detail?.subjects) ? detail.subjects : [],
          availability: Array.isArray(detail?.availability) ? detail.availability : [],
          meeting_provider: typeof detail?.meeting_provider === 'string' ? detail.meeting_provider : 'google_meet',
          is_active: typeof detail?.is_active === 'boolean' ? detail.is_active : true,
        };
      }),
    });
  } catch (error: any) {
    console.error('Failed to list teachers:', error);
    return res.status(500).json({ error: error.message || 'Failed to list teachers' });
  }
});

router.get('/users', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const { data, error } = await supabase
      .from('profiles')
      .select('*');

    if (error) {
      throw error;
    }

    return res.json({ users: data ?? [] });
  } catch (error: any) {
    console.error('Failed to list users:', error);
    return res.status(500).json({ error: error.message || 'Failed to list users' });
  }
});

router.get('/meetings', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const { data: meetings, error: meetingsError } = await supabase
      .from('mentoring_meetings')
      .select('*')
      .order('requested_at', { ascending: false });

    if (meetingsError) {
      throw meetingsError;
    }

    const participantIds = new Set<string>();
    for (const meeting of meetings ?? []) {
      if (meeting.student_clerk_id) participantIds.add(meeting.student_clerk_id);
      if (meeting.teacher_clerk_id) participantIds.add(meeting.teacher_clerk_id);
    }

    let participantsById = new Map<string, { clerk_id: string; full_name: string | null; email: string | null; role: string | null }>();
    if (participantIds.size > 0) {
      const { data: participants, error: participantsError } = await supabase
        .from('profiles')
        .select('clerk_id, full_name, email, role')
        .in('clerk_id', Array.from(participantIds));

      if (participantsError) {
        throw participantsError;
      }

      participantsById = new Map(
        (participants ?? []).map((row) => [
          row.clerk_id,
          {
            clerk_id: row.clerk_id,
            full_name: row.full_name ?? null,
            email: row.email ?? null,
            role: row.role ?? null,
          },
        ])
      );
    }

    return res.json({
      meetings: (meetings ?? []).map((meeting) => ({
        ...meeting,
        student_profile: participantsById.get(meeting.student_clerk_id) ?? null,
        teacher_profile: participantsById.get(meeting.teacher_clerk_id) ?? null,
      })),
    });
  } catch (error: any) {
    console.error('Failed to list meetings:', error);
    return res.status(500).json({ error: error.message || 'Failed to list meetings' });
  }
});

router.patch('/teacher-profile/:clerkId', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const teacherClerkId = req.params.clerkId;
    const body = (req.body ?? {}) as {
      headline?: string;
      bio?: string;
      subjects?: string[];
      availability?: Array<{ day: string; start: string; end: string }>;
      meeting_provider?: string;
      is_active?: boolean;
    };

    const payload: Record<string, unknown> = {
      clerk_id: teacherClerkId,
      updated_at: new Date().toISOString(),
    };

    if (typeof body.headline === 'string') payload.headline = body.headline.trim();
    if (typeof body.bio === 'string') payload.bio = body.bio.trim();
    if (Array.isArray(body.subjects)) {
      payload.subjects = body.subjects
        .filter((subject) => typeof subject === 'string')
        .map((subject) => subject.trim())
        .filter(Boolean);
    }
    if (Array.isArray(body.availability)) payload.availability = body.availability;
    if (typeof body.meeting_provider === 'string') payload.meeting_provider = body.meeting_provider.trim() || 'google_meet';
    if (typeof body.is_active === 'boolean') payload.is_active = body.is_active;

    const { data: existing, error: existingError } = await supabase
      .from('teacher_profiles')
      .select('*')
      .eq('clerk_id', teacherClerkId)
      .maybeSingle();

    if (existingError) {
      throw existingError;
    }

    if (existing) {
      const { data: updated, error: updateError } = await supabase
        .from('teacher_profiles')
        .update(payload)
        .eq('clerk_id', teacherClerkId)
        .select('*')
        .single();

      if (updateError) {
        throw updateError;
      }

      return res.json({ teacher_profile: updated });
    }

    const { data: created, error: createError } = await supabase
      .from('teacher_profiles')
      .insert(payload)
      .select('*')
      .single();

    if (createError) {
      throw createError;
    }

    return res.json({ teacher_profile: created });
  } catch (error: any) {
    console.error('Failed to update teacher profile:', error);
    return res.status(500).json({ error: error.message || 'Failed to update teacher profile' });
  }
});

// Update user profile (admin only)
// Update user profile (admin only) - Placeholder for future implementation using Supabase
router.put('/update-profile/:id', clerkAuth, requireRole('admin'), (req: Request, res: Response) => {
  return res.status(501).json({ error: 'Not implemented yet' });
});

/* ============================ Manual Pro flow (admin) ============================
 * Review manual payment requests, activate 30-day Pro, manage the QR + promo codes,
 * and monitor all users' trial/Pro status. Admin-only (requireRole('admin')).
 */

/** All Pro requests, enriched with the payer's current billing status + days left. */
router.get('/pro-requests', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const { data: requests, error } = await supabase
      .from('pro_requests').select('*').order('created_at', { ascending: false }).limit(500);
    if (error) throw error;

    const clerkIds = Array.from(new Set((requests ?? []).map((r) => r.clerk_id).filter(Boolean)));
    const billingByClerk = new Map<string, BillingRow>();
    const nameByClerk = new Map<string, string | null>();
    if (clerkIds.length > 0) {
      const [{ data: billing }, { data: profs }] = await Promise.all([
        supabase.from('student_billing').select('*').in('clerk_id', clerkIds),
        supabase.from('profiles').select('clerk_id, full_name').in('clerk_id', clerkIds),
      ]);
      for (const b of (billing ?? []) as BillingRow[]) billingByClerk.set(b.clerk_id, b);
      for (const p of profs ?? []) nameByClerk.set(p.clerk_id, p.full_name ?? null);
    }

    return res.json({
      requests: (requests ?? []).map((r) => {
        const access = accessForClerk(billingByClerk, r.clerk_id);
        return {
          ...r,
          full_name: nameByClerk.get(r.clerk_id) ?? r.name ?? null,
          billing_status: access.status,
          is_pro: access.isPro,
          days_left: access.daysLeft,
          current_period_end: access.currentPeriodEnd,
        };
      }),
    });
  } catch (error: any) {
    console.error('Failed to list pro requests:', error);
    return res.status(500).json({ error: error.message || 'Failed to list pro requests' });
  }
});

/** Activate 30-day Pro for a request's user, mark it approved, and email them. */
router.post('/pro-requests/:id/activate', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const id = req.params.id;
    const days = Number.parseInt(String(req.body?.days ?? ''), 10);
    const period = Number.isFinite(days) && days > 0 ? days : MANUAL_PLAN_DAYS;

    const { data: reqRow, error: reqError } = await supabase.from('pro_requests').select('*').eq('id', id).maybeSingle();
    if (reqError) throw reqError;
    if (!reqRow) return res.status(404).json({ error: 'not_found' });

    const { periodEnd, wasActive } = await activateManualPro(reqRow.clerk_id, { days: period, amountPkr: reqRow.amount_pkr ?? null });
    await supabase.from('pro_requests').update({
      status: 'approved', reviewed_at: new Date().toISOString(), reviewed_by: req.auth?.clerkId ?? null,
    }).eq('id', id);

    if (!wasActive) {
      void sendProWelcome(reqRow.clerk_id, { plan: 'manual', periodEndIso: periodEnd, amountPkr: reqRow.amount_pkr ?? null })
        .catch(() => { /* best-effort */ });
    }
    return res.json({ ok: true, periodEnd, days: period });
  } catch (error: any) {
    console.error('Failed to activate pro request:', error);
    return res.status(500).json({ error: error.message || 'Failed to activate' });
  }
});

/** Reject a Pro request. */
router.post('/pro-requests/:id/reject', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const id = req.params.id;
    const note = typeof req.body?.note === 'string' ? req.body.note.slice(0, 500) : null;
    const { error } = await supabase.from('pro_requests').update({
      status: 'rejected', note, reviewed_at: new Date().toISOString(), reviewed_by: req.auth?.clerkId ?? null,
    }).eq('id', id);
    if (error) throw error;
    return res.json({ ok: true });
  } catch (error: any) {
    console.error('Failed to reject pro request:', error);
    return res.status(500).json({ error: error.message || 'Failed to reject' });
  }
});

/** The general QR + payee details. */
router.get('/pay-config', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const { data, error } = await supabase.from('pay_config').select('*').eq('id', 1).maybeSingle();
    if (error) throw error;
    return res.json({ config: data ?? null });
  } catch (error: any) {
    console.error('Failed to read pay config:', error);
    return res.status(500).json({ error: error.message || 'Failed to read pay config' });
  }
});

/** Update the general QR + payee details. */
router.put('/pay-config', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const b = (req.body ?? {}) as Record<string, unknown>;
    const patch: Record<string, unknown> = { id: 1, updated_at: new Date().toISOString(), updated_by: req.auth?.clerkId ?? null };
    if (typeof b.qrImage === 'string') patch.qr_image = b.qrImage || null;
    if (typeof b.payeeName === 'string') patch.payee_name = b.payeeName.trim() || null;
    if (typeof b.accountNumber === 'string') patch.account_number = b.accountNumber.trim() || null;
    if (typeof b.bankName === 'string') patch.bank_name = b.bankName.trim() || null;
    if (typeof b.instructions === 'string') patch.instructions = b.instructions.trim() || null;
    if (b.amountPkr === null || b.amountPkr === '') patch.amount_pkr = null;
    else if (b.amountPkr !== undefined) { const n = Number.parseInt(String(b.amountPkr), 10); patch.amount_pkr = Number.isFinite(n) ? n : null; }

    const { data, error } = await supabase.from('pay_config').upsert(patch, { onConflict: 'id' }).select('*').single();
    if (error) throw error;
    return res.json({ ok: true, config: data });
  } catch (error: any) {
    console.error('Failed to save pay config:', error);
    return res.status(500).json({ error: error.message || 'Failed to save pay config' });
  }
});

/** List promo codes (with their QR + note). */
router.get('/promo-codes', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const { data, error } = await supabase.from('promo_codes').select('*').order('created_at', { ascending: false });
    if (error) throw error;
    return res.json({ promos: data ?? [] });
  } catch (error: any) {
    console.error('Failed to list promo codes:', error);
    return res.status(500).json({ error: error.message || 'Failed to list promo codes' });
  }
});

/** Create or update a promo code (attach a QR + optional discounted price/note). */
router.post('/promo-codes', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const b = (req.body ?? {}) as Record<string, unknown>;
    const code = String(b.code ?? '').trim().toUpperCase().slice(0, 40);
    if (!code && !b.id) return res.status(400).json({ error: 'code_required' });

    const row: Record<string, unknown> = { updated_at: new Date().toISOString() };
    if (code) row.code = code;
    if (typeof b.label === 'string') row.label = b.label.trim() || null;
    if (typeof b.note === 'string') row.note = b.note.trim() || null;
    if (typeof b.qrImage === 'string') row.qr_image = b.qrImage || null;
    if (typeof b.active === 'boolean') row.active = b.active;
    if (b.amountPkr === null || b.amountPkr === '') row.amount_pkr = null;
    else if (b.amountPkr !== undefined) { const n = Number.parseInt(String(b.amountPkr), 10); row.amount_pkr = Number.isFinite(n) ? n : null; }

    let saved;
    if (b.id) {
      const u = await supabase.from('promo_codes').update(row).eq('id', String(b.id)).select('*').single();
      if (u.error) throw u.error; saved = u.data;
    } else {
      const u = await supabase.from('promo_codes').upsert(row, { onConflict: 'code' }).select('*').single();
      if (u.error) throw u.error; saved = u.data;
    }
    return res.json({ ok: true, promo: saved });
  } catch (error: any) {
    console.error('Failed to save promo code:', error);
    return res.status(500).json({ error: error.message || 'Failed to save promo code' });
  }
});

/** Delete a promo code. */
router.delete('/promo-codes/:id', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { error } = await supabase.from('promo_codes').delete().eq('id', req.params.id);
    if (error) throw error;
    return res.json({ ok: true });
  } catch (error: any) {
    console.error('Failed to delete promo code:', error);
    return res.status(500).json({ error: error.message || 'Failed to delete promo code' });
  }
});

/** Immediately revoke a user's Pro (or trial) access. */
router.post('/users/:clerkId/revoke-pro', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.params.clerkId;
    if (!clerkId) return res.status(400).json({ error: 'clerk_id_required' });
    await revokePro(clerkId);
    return res.json({ ok: true });
  } catch (error: any) {
    console.error('Failed to revoke pro:', error);
    return res.status(500).json({ error: error.message || 'Failed to revoke' });
  }
});

/** All users with their role + trial/Pro status + days left (monitoring). */
router.get('/users-billing', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const [{ data: profs, error: pErr }, { data: billing, error: bErr }] = await Promise.all([
      supabase.from('profiles').select('clerk_id, full_name, email, role, onboarding_complete, created_at'),
      supabase.from('student_billing').select('*'),
    ]);
    if (pErr) throw pErr;
    if (bErr) throw bErr;
    const billingByClerk = new Map<string, BillingRow>();
    for (const b of (billing ?? []) as BillingRow[]) billingByClerk.set(b.clerk_id, b);

    const users = (profs ?? []).map((p) => {
      const access = accessForClerk(billingByClerk, p.clerk_id);
      return {
        clerk_id: p.clerk_id,
        full_name: p.full_name ?? null,
        email: p.email ?? null,
        role: p.role ?? 'student',
        onboarding_complete: !!p.onboarding_complete,
        created_at: p.created_at ?? null,
        billing_status: access.status,
        is_pro: access.isPro,
        plan: access.plan,
        days_left: access.daysLeft,
        current_period_end: access.currentPeriodEnd,
        trial_ends_at: access.trialEndsAt,
      };
    });
    return res.json({ users });
  } catch (error: any) {
    console.error('Failed to list users billing:', error);
    return res.status(500).json({ error: error.message || 'Failed to list users billing' });
  }
});

export default router;
