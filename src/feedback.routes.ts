import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import { resolveClassAccess } from './lib/portalAccess';
import { grokChatJson, grokEnabled } from './lib/grok';
import { rateLimit } from './lib/rateLimit';

/**
 * Teacher Portal — feedback (spec feature group 10). Per-criterion comments
 * (§10.1), AI-drafted overall feedback (§10.2), topic-filtered comment bank (§10.3).
 */

const router = Router();
router.use(clerkAuth);

// Throttle the paid-AI feedback-draft endpoint against cost-abuse loops.
const aiLimit = rateLimit({ windowMs: 60_000, max: 40, name: 'ai' });

interface Criterion {
  criterion_text: string;
  marks_available: number;
  marks_awarded: number;
  awarded: boolean;
}

async function markAccess(markId: string, clerkId: string) {
  const { data: mark } = await supabase.from('submission_marks').select('*').eq('id', markId).maybeSingle();
  if (!mark) return null;
  const { data: sub } = await supabase.from('submissions').select('id, assignment_id').eq('id', (mark as { submission_id: string }).submission_id).maybeSingle();
  if (!sub) return null;
  const { data: asg } = await supabase.from('assignments').select('id, class_id').eq('id', (sub as { assignment_id: string }).assignment_id).maybeSingle();
  if (!asg) return null;
  const access = await resolveClassAccess((asg as { class_id: string }).class_id, clerkId);
  return { mark: mark as Record<string, unknown>, access };
}

async function submissionAccess(submissionId: string, clerkId: string) {
  const { data: sub } = await supabase.from('submissions').select('*').eq('id', submissionId).maybeSingle();
  if (!sub) return null;
  const { data: asg } = await supabase.from('assignments').select('id, class_id, title').eq('id', (sub as { assignment_id: string }).assignment_id).maybeSingle();
  if (!asg) return null;
  const access = await resolveClassAccess((asg as { class_id: string }).class_id, clerkId);
  return { submission: sub as Record<string, unknown>, assignment: asg as Record<string, unknown>, access };
}

// POST /feedback/marks/:id/comment — comment on a specific criterion (§10.1).
router.post('/marks/:id/comment', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const ctx = await markAccess(req.params.id, req.auth!.clerkId);
    if (!ctx || !ctx.access) return res.status(404).json({ error: 'Mark not found' });
    if (!ctx.access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const index = Number(req.body?.criterion_index);
    if (!Number.isInteger(index) || index < 0) return res.status(400).json({ error: 'criterion_index is required' });
    const text = typeof req.body?.text === 'string' ? req.body.text.trim() : '';

    const comments = Array.isArray(ctx.mark.criterion_comments) ? [...(ctx.mark.criterion_comments as { index: number; text: string }[])] : [];
    const existing = comments.findIndex((c) => c.index === index);
    if (text) {
      if (existing >= 0) comments[existing] = { index, text };
      else comments.push({ index, text });
    } else if (existing >= 0) {
      comments.splice(existing, 1);
    }

    await supabase.from('submission_marks').update({ criterion_comments: comments, updated_at: new Date().toISOString() }).eq('id', req.params.id);
    return res.json({ ok: true, criterion_comments: comments });
  } catch (err) {
    console.error('Comment error:', err);
    return res.status(500).json({ error: 'Failed to save comment' });
  }
});

// POST /feedback/submissions/:id/overall — save the overall script feedback.
router.post('/submissions/:id/overall', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const ctx = await submissionAccess(req.params.id, req.auth!.clerkId);
    if (!ctx || !ctx.access) return res.status(404).json({ error: 'Submission not found' });
    if (!ctx.access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const text = typeof req.body?.text === 'string' ? req.body.text : '';
    const isDraft = Boolean(req.body?.is_draft);
    await supabase.from('submissions').update({ overall_feedback: text || null, feedback_is_draft: isDraft, updated_at: new Date().toISOString() }).eq('id', req.params.id);
    return res.json({ ok: true });
  } catch (err) {
    console.error('Overall feedback error:', err);
    return res.status(500).json({ error: 'Failed to save feedback' });
  }
});

// POST /feedback/submissions/:id/draft — AI-draft overall feedback from marks (§10.2).
router.post('/submissions/:id/draft', aiLimit, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const ctx = await submissionAccess(req.params.id, req.auth!.clerkId);
    if (!ctx || !ctx.access) return res.status(404).json({ error: 'Submission not found' });
    if (!ctx.access.canGrade) return res.status(403).json({ error: 'No grading access' });

    // Gather per-topic performance from the marks.
    const { data: marks } = await supabase
      .from('submission_marks')
      .select('assignment_question_id, ai_criteria, final_criteria, ai_score, final_score, ai_marks_available')
      .eq('submission_id', req.params.id);
    const { data: aqs } = await supabase
      .from('assignment_questions')
      .select('id, snapshot')
      .eq('assignment_id', ctx.submission.assignment_id as string);
    const topicByAq = new Map<string, string>();
    for (const q of (aqs ?? []) as { id: string; snapshot: Record<string, unknown> }[]) topicByAq.set(q.id, String(q.snapshot?.topic ?? 'General'));

    const byTopic = new Map<string, { earned: number; available: number }>();
    for (const m of (marks ?? []) as Record<string, unknown>[]) {
      const topic = topicByAq.get(m.assignment_question_id as string) || 'General';
      const criteria = (m.final_criteria ?? m.ai_criteria ?? []) as Criterion[];
      const earned = criteria.reduce((s, c) => s + (Number(c.marks_awarded) || 0), 0);
      const available = Number(m.ai_marks_available ?? criteria.reduce((s, c) => s + c.marks_available, 0));
      const cur = byTopic.get(topic) ?? { earned: 0, available: 0 };
      cur.earned += earned;
      cur.available += available;
      byTopic.set(topic, cur);
    }

    const topics = Array.from(byTopic, ([topic, v]) => ({ topic, pct: v.available > 0 ? v.earned / v.available : 0 }));
    topics.sort((a, b) => b.pct - a.pct);
    const strongest = topics.slice(0, 2).filter((t) => t.pct >= 0.6);
    const weakest = topics.slice(-2).filter((t) => t.pct < 0.6);
    const totalEarned = Array.from(byTopic.values()).reduce((s, v) => s + v.earned, 0);
    const totalAvailable = Array.from(byTopic.values()).reduce((s, v) => s + v.available, 0);

    let draft = '';
    if (grokEnabled()) {
      try {
        const summary = topics.map((t) => `${t.topic}: ${Math.round(t.pct * 100)}%`).join(', ');
        const result = await grokChatJson({
          system:
            'You are a supportive UK secondary-school teacher writing brief feedback to a student on a past-paper assignment. 2–4 sentences, encouraging but specific. Name the strongest and weakest topics and one concrete pattern to improve. Return JSON {"feedback": "..."}.',
          user: `Score ${totalEarned}/${totalAvailable}. Topic breakdown: ${summary || 'n/a'}.`,
          temperature: 0.4,
          maxTokens: 300,
        });
        if (typeof result.feedback === 'string') draft = result.feedback.trim();
      } catch {
        /* fall through to template */
      }
    }

    if (!draft) {
      // Deterministic template so drafting works without the AI service.
      const parts: string[] = [];
      parts.push(`You scored ${totalEarned} out of ${totalAvailable} on this assignment.`);
      if (strongest.length) parts.push(`Your strongest work was on ${strongest.map((t) => t.topic).join(' and ')}.`);
      if (weakest.length) parts.push(`Focus your next revision on ${weakest.map((t) => t.topic).join(' and ')}, where you lost the most marks.`);
      draft = parts.join(' ');
    }

    // Store as a draft (labelled until the teacher accepts).
    await supabase.from('submissions').update({ overall_feedback: draft, feedback_is_draft: true, updated_at: new Date().toISOString() }).eq('id', req.params.id);
    return res.json({ draft });
  } catch (err) {
    console.error('Draft feedback error:', err);
    return res.status(500).json({ error: 'Failed to draft feedback' });
  }
});

// POST /feedback/assignment/:assignmentId/question/:aqId/apply-missed
// Add guidance as a comment to every student who missed a criterion, in one
// step (§10.4). criterion_index selects the criterion.
router.post('/assignment/:assignmentId/question/:aqId/apply-missed', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { data: assignment } = await supabase.from('assignments').select('id, class_id').eq('id', req.params.assignmentId).maybeSingle();
    if (!assignment) return res.status(404).json({ error: 'Assignment not found' });
    const access = await resolveClassAccess((assignment as { class_id: string }).class_id, clerkId);
    if (!access || !access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const index = Number(req.body?.criterion_index);
    const text = typeof req.body?.text === 'string' ? req.body.text.trim() : '';
    if (!Number.isInteger(index) || index < 0 || !text) return res.status(400).json({ error: 'criterion_index and text are required' });

    const { data: subs } = await supabase.from('submissions').select('id').eq('assignment_id', req.params.assignmentId);
    const subIds = ((subs ?? []) as { id: string }[]).map((s) => s.id);
    if (subIds.length === 0) return res.json({ applied: 0 });

    const { data: marks } = await supabase
      .from('submission_marks')
      .select('id, ai_criteria, final_criteria, criterion_comments')
      .eq('assignment_question_id', req.params.aqId)
      .in('submission_id', subIds);

    let applied = 0;
    for (const m of (marks ?? []) as Record<string, unknown>[]) {
      const criteria = (m.final_criteria ?? m.ai_criteria ?? []) as Criterion[];
      if (criteria[index]?.awarded) continue; // only those who MISSED it
      const comments = Array.isArray(m.criterion_comments) ? [...(m.criterion_comments as { index: number; text: string }[])] : [];
      const existing = comments.findIndex((c) => c.index === index);
      if (existing >= 0) comments[existing] = { index, text };
      else comments.push({ index, text });
      await supabase.from('submission_marks').update({ criterion_comments: comments, updated_at: new Date().toISOString() }).eq('id', m.id);
      applied += 1;
    }

    return res.json({ applied });
  } catch (err) {
    console.error('Apply-missed error:', err);
    return res.status(500).json({ error: 'Failed to apply guidance' });
  }
});

// ---- Comment bank (§10.3) ----
async function institutionId(clerkId: string): Promise<string | null> {
  const { data } = await supabase.from('profiles').select('institution_id').eq('clerk_id', clerkId).maybeSingle();
  return (data as { institution_id: string | null } | null)?.institution_id ?? null;
}

router.get('/bank', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const topic = typeof req.query.topic === 'string' ? req.query.topic : null;
    let owned = supabase.from('comment_bank').select('*').eq('owner_clerk_id', clerkId);
    if (topic) owned = owned.eq('topic', topic);
    const { data: mine } = await owned;

    let shared: Record<string, unknown>[] = [];
    const inst = await institutionId(clerkId);
    if (inst) {
      let q = supabase.from('comment_bank').select('*').eq('institution_id', inst).eq('shared_to_institution', true).neq('owner_clerk_id', clerkId);
      if (topic) q = q.eq('topic', topic);
      const { data } = await q;
      shared = (data as Record<string, unknown>[]) ?? [];
    }
    return res.json([...((mine as Record<string, unknown>[]) ?? []), ...shared]);
  } catch (err) {
    console.error('Comment bank list error:', err);
    return res.status(500).json({ error: 'Failed to load comment bank' });
  }
});

router.post('/bank', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const text = typeof req.body?.text === 'string' ? req.body.text.trim() : '';
    if (!text) return res.status(400).json({ error: 'text is required' });
    const { data, error } = await supabase
      .from('comment_bank')
      .insert({
        owner_clerk_id: clerkId,
        institution_id: await institutionId(clerkId),
        topic: req.body?.topic?.trim() || null,
        text,
        shared_to_institution: Boolean(req.body?.shared_to_institution),
      })
      .select('*')
      .single();
    if (error) throw error;
    return res.status(201).json(data);
  } catch (err) {
    console.error('Comment bank add error:', err);
    return res.status(500).json({ error: 'Failed to save to bank' });
  }
});

router.delete('/bank/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { error } = await supabase.from('comment_bank').delete().eq('id', req.params.id).eq('owner_clerk_id', req.auth!.clerkId);
    if (error) throw error;
    return res.json({ ok: true });
  } catch (err) {
    console.error('Comment bank delete error:', err);
    return res.status(500).json({ error: 'Failed to delete' });
  }
});

export default router;
