import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import { resolveClassAccess } from './lib/portalAccess';
import { isSubmissionFullyReviewed, releaseSubmission } from './lib/release';

/**
 * Teacher Portal — result release (spec §11). Release gate, batch/single
 * release, per-assignment release-content control, and release status.
 */

const router = Router();
router.use(clerkAuth);

async function assignmentAccess(assignmentId: string, clerkId: string) {
  const { data: assignment } = await supabase.from('assignments').select('*').eq('id', assignmentId).maybeSingle();
  if (!assignment) return { assignment: null, access: null };
  const access = await resolveClassAccess((assignment as { class_id: string }).class_id, clerkId);
  return { assignment: assignment as Record<string, unknown>, access };
}

// PATCH /release/assignment/:id/config — release mode + content (§11.1, §11.3).
router.patch('/assignment/:id/config', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { assignment, access } = await assignmentAccess(req.params.id, req.auth!.clerkId);
    if (!assignment || !access) return res.status(404).json({ error: 'Assignment not found' });
    if (!access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const update: Record<string, unknown> = { updated_at: new Date().toISOString() };
    if (typeof req.body?.auto_release === 'boolean') update.auto_release = req.body.auto_release;
    if (req.body?.release_content && typeof req.body.release_content === 'object') {
      update.release_content = { ...(assignment.release_content as object), ...req.body.release_content };
    }
    const { data, error } = await supabase.from('assignments').update(update).eq('id', assignment.id).select('auto_release, release_content').single();
    if (error) throw error;
    return res.json(data);
  } catch (err) {
    console.error('Release config error:', err);
    return res.status(500).json({ error: 'Failed to update release settings' });
  }
});

// POST /release/assignment/:id — batch release (§11.2). mode: 'all' | 'reviewed'.
router.post('/assignment/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { assignment, access } = await assignmentAccess(req.params.id, clerkId);
    if (!assignment || !access) return res.status(404).json({ error: 'Assignment not found' });
    if (!access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const mode = req.body?.mode === 'all' ? 'all' : 'reviewed';
    const { data: subs } = await supabase
      .from('submissions')
      .select('id, released_at, status')
      .eq('assignment_id', assignment.id)
      .in('status', ['submitted', 'late']);

    let released = 0;
    for (const s of (subs ?? []) as { id: string; released_at: string | null }[]) {
      if (s.released_at) continue;
      if (mode === 'reviewed' && !(await isSubmissionFullyReviewed(s.id))) continue;
      await releaseSubmission(s.id, clerkId);
      released += 1;
    }

    return res.json({ released });
  } catch (err) {
    console.error('Batch release error:', err);
    return res.status(500).json({ error: 'Failed to release results' });
  }
});

// POST /release/submission/:id — release a single script (§11.2).
router.post('/submission/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { data: submission } = await supabase.from('submissions').select('id, assignment_id, released_at').eq('id', req.params.id).maybeSingle();
    if (!submission) return res.status(404).json({ error: 'Submission not found' });
    const { access } = await assignmentAccess((submission as { assignment_id: string }).assignment_id, clerkId);
    if (!access || !access.canGrade) return res.status(403).json({ error: 'No grading access' });

    await releaseSubmission(req.params.id, clerkId);
    return res.json({ ok: true });
  } catch (err) {
    console.error('Single release error:', err);
    return res.status(500).json({ error: 'Failed to release' });
  }
});

// GET /release/assignment/:id/status — who has received results (§11.4).
router.get('/assignment/:id/status', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { assignment, access } = await assignmentAccess(req.params.id, req.auth!.clerkId);
    if (!assignment || !access) return res.status(404).json({ error: 'Assignment not found' });

    const { data: subs } = await supabase
      .from('submissions')
      .select('id, student_clerk_id, status, released_at, reopened_at')
      .eq('assignment_id', assignment.id);
    const rows = (subs ?? []) as { id: string; student_clerk_id: string; status: string; released_at: string | null; reopened_at: string | null }[];

    const ids = Array.from(new Set(rows.map((r) => r.student_clerk_id)));
    const nameMap = new Map<string, string>();
    if (ids.length > 0) {
      const { data: profiles } = await supabase.from('profiles').select('clerk_id, full_name, email').in('clerk_id', ids);
      for (const p of (profiles ?? []) as { clerk_id: string; full_name: string | null; email: string | null }[]) nameMap.set(p.clerk_id, p.full_name || p.email || 'Student');
    }

    const released = rows.filter((r) => r.released_at).length;
    return res.json({
      auto_release: assignment.auto_release,
      release_content: assignment.release_content,
      released_count: released,
      total: rows.length,
      rows: rows.map((r) => ({
        submission_id: r.id,
        student_name: nameMap.get(r.student_clerk_id) || 'Student',
        released: Boolean(r.released_at),
        // A reopen after release withdraws the result (§7.4/§11.4).
        withdrawn: Boolean(r.reopened_at) && !r.released_at,
      })),
    });
  } catch (err) {
    console.error('Release status error:', err);
    return res.status(500).json({ error: 'Failed to load release status' });
  }
});

export default router;
