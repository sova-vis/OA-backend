import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';

/**
 * Teacher Portal — settings (spec feature group 17) and derived in-app
 * notifications (§16.1). Grading defaults apply to new assignments.
 */

const router = Router();
router.use(clerkAuth);

const SETTINGS_COLUMNS =
  'auto_approve_threshold, default_mark_scheme_visibility, default_release_content, notif_prefs, digest_frequency, timezone, institution_id';

router.get('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { data } = await supabase.from('profiles').select(SETTINGS_COLUMNS).eq('clerk_id', req.auth!.clerkId).maybeSingle();
    const profile = (data as Record<string, unknown>) ?? {};
    let floor: number | null = null;
    if (profile.institution_id) {
      const { data: inst } = await supabase.from('institutions').select('auto_approve_floor').eq('id', profile.institution_id as string).maybeSingle();
      floor = (inst as { auto_approve_floor?: number } | null)?.auto_approve_floor ?? null;
    }
    return res.json({ ...profile, auto_approve_floor: floor });
  } catch (err) {
    console.error('Get settings error:', err);
    return res.status(500).json({ error: 'Failed to load settings' });
  }
});

router.patch('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const body = req.body ?? {};
    const update: Record<string, unknown> = { updated_at: new Date().toISOString() };

    if (body.auto_approve_threshold !== undefined) {
      const v = Number(body.auto_approve_threshold);
      if (!Number.isFinite(v) || v < 0 || v > 1) return res.status(400).json({ error: 'threshold must be 0–1' });
      // Respect the institution floor (§9.6).
      const { data: prof } = await supabase.from('profiles').select('institution_id').eq('clerk_id', clerkId).maybeSingle();
      const instId = (prof as { institution_id?: string } | null)?.institution_id;
      if (instId) {
        const { data: inst } = await supabase.from('institutions').select('auto_approve_floor').eq('id', instId).maybeSingle();
        const floor = (inst as { auto_approve_floor?: number } | null)?.auto_approve_floor;
        if (floor != null && v < floor) return res.status(400).json({ error: `Institution requires a minimum threshold of ${floor}` });
      }
      update.auto_approve_threshold = v;
    }
    if (['never', 'after_submission', 'after_deadline', 'after_release'].includes(body.default_mark_scheme_visibility)) {
      update.default_mark_scheme_visibility = body.default_mark_scheme_visibility;
    }
    if (body.default_release_content && typeof body.default_release_content === 'object') update.default_release_content = body.default_release_content;
    if (body.notif_prefs && typeof body.notif_prefs === 'object') update.notif_prefs = body.notif_prefs;
    if (['daily', 'weekly', 'off'].includes(body.digest_frequency)) update.digest_frequency = body.digest_frequency;
    if (typeof body.timezone === 'string') update.timezone = body.timezone;

    const { data, error } = await supabase.from('profiles').update(update).eq('clerk_id', clerkId).select(SETTINGS_COLUMNS).single();
    if (error) throw error;
    return res.json(data);
  } catch (err) {
    console.error('Update settings error:', err);
    return res.status(500).json({ error: 'Failed to update settings' });
  }
});

// GET /settings/notifications — derived in-app notifications (§16.1), batched by
// assignment/class. Never one per submission.
router.get('/notifications', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const [{ data: owned }, { data: co }] = await Promise.all([
      supabase.from('classes').select('id, name').eq('owner_clerk_id', clerkId),
      supabase.from('class_co_teachers').select('class_id').eq('teacher_clerk_id', clerkId),
    ]);
    const classes = (owned ?? []) as { id: string; name: string }[];
    const classNameById = new Map(classes.map((c) => [c.id, c.name]));
    const classIds = [...classes.map((c) => c.id), ...((co ?? []) as { class_id: string }[]).map((c) => c.class_id)];
    if (classIds.length === 0) return res.json([]);

    const notifications: { type: string; body: string; class_id?: string; assignment_id?: string }[] = [];

    // Pending enrolment requests, batched by class.
    const { data: pending } = await supabase.from('class_enrollments').select('class_id').in('class_id', classIds).eq('status', 'pending');
    const pendingByClass = new Map<string, number>();
    for (const p of (pending ?? []) as { class_id: string }[]) pendingByClass.set(p.class_id, (pendingByClass.get(p.class_id) ?? 0) + 1);
    for (const [cid, n] of pendingByClass) {
      notifications.push({ type: 'enrolment_request', class_id: cid, body: `${n} student${n === 1 ? '' : 's'} waiting to join ${classNameById.get(cid) ?? 'a class'}` });
    }

    // Review backlog + new submissions, batched by assignment.
    const { data: assignments } = await supabase.from('assignments').select('id, title').in('class_id', classIds).eq('status', 'published');
    const asgRows = (assignments ?? []) as { id: string; title: string }[];
    const asgIds = asgRows.map((a) => a.id);
    if (asgIds.length > 0) {
      const { data: subs } = await supabase.from('submissions').select('id, assignment_id').in('assignment_id', asgIds);
      const subToAsg = new Map<string, string>();
      for (const s of (subs ?? []) as { id: string; assignment_id: string }[]) subToAsg.set(s.id, s.assignment_id);
      const subIds = Array.from(subToAsg.keys());
      if (subIds.length > 0) {
        const { data: marks } = await supabase.from('submission_marks').select('submission_id, status').in('submission_id', subIds).eq('status', 'needs_review');
        const backlogByAsg = new Map<string, number>();
        for (const m of (marks ?? []) as { submission_id: string }[]) {
          const aid = subToAsg.get(m.submission_id);
          if (aid) backlogByAsg.set(aid, (backlogByAsg.get(aid) ?? 0) + 1);
        }
        for (const [aid, n] of backlogByAsg) {
          const title = asgRows.find((a) => a.id === aid)?.title ?? 'an assignment';
          notifications.push({ type: 'review_backlog', assignment_id: aid, body: `${n} answer${n === 1 ? '' : 's'} awaiting review in ${title}` });
        }
      }
    }

    return res.json(notifications);
  } catch (err) {
    console.error('Notifications error:', err);
    return res.status(500).json({ error: 'Failed to load notifications' });
  }
});

export default router;
