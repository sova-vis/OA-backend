import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import { resolveClassAccess } from './lib/portalAccess';
import { maybeAutoRelease } from './lib/release';

/**
 * Teacher Portal — grading review (spec feature group 9). The module that
 * determines whether the product succeeds: confidence-triaged queue (§9.1),
 * per-criterion override (§9.4), bulk approve (§9.5), auto-approve threshold
 * (§9.6), sampling (§9.7), audit trail (§9.9). The teacher's mark always wins.
 */

const router = Router();
router.use(clerkAuth);

interface Criterion {
  criterion_text: string;
  marks_available: number;
  marks_awarded: number;
  awarded: boolean;
  reasoning: string;
  confidence: number | null;
}

// Pull an examiner-report note out of a question's `reference` jsonb if present
// (§10.4). Content ingestion hasn't structured this yet, so we probe the likely
// keys and return null otherwise — the UI degrades gracefully.
function examinerNote(reference: unknown): string | null {
  if (!reference || typeof reference !== 'object') return null;
  const ref = reference as Record<string, unknown>;
  for (const key of ['examiner_report', 'examiner', 'examiner_note', 'common_mistakes', 'commentary']) {
    const v = ref[key];
    if (typeof v === 'string' && v.trim()) return v.trim();
    if (Array.isArray(v) && v.length > 0) return v.map(String).join('; ');
  }
  return null;
}

async function logActivity(actor: string, eventType: string, targetType: string, targetId: string, detail: Record<string, unknown>) {
  await supabase.from('activity_log').insert({ actor_clerk_id: actor, event_type: eventType, target_type: targetType, target_id: targetId, detail });
}

async function getThreshold(clerkId: string): Promise<{ threshold: number; floor: number | null }> {
  const { data } = await supabase.from('profiles').select('auto_approve_threshold, institution_id').eq('clerk_id', clerkId).maybeSingle();
  const threshold = Number((data as { auto_approve_threshold?: number } | null)?.auto_approve_threshold ?? 0.85);
  let floor: number | null = null;
  const institutionId = (data as { institution_id?: string } | null)?.institution_id;
  if (institutionId) {
    const { data: inst } = await supabase.from('institutions').select('auto_approve_floor').eq('id', institutionId).maybeSingle();
    floor = (inst as { auto_approve_floor?: number } | null)?.auto_approve_floor ?? null;
  }
  return { threshold, floor };
}

// Load a mark plus its submission/assignment and the caller's class access.
async function loadMarkContext(markId: string, clerkId: string) {
  const { data: mark } = await supabase.from('submission_marks').select('*').eq('id', markId).maybeSingle();
  if (!mark) return null;
  const { data: submission } = await supabase.from('submissions').select('id, assignment_id, student_clerk_id').eq('id', (mark as { submission_id: string }).submission_id).maybeSingle();
  if (!submission) return null;
  const { data: assignment } = await supabase.from('assignments').select('id, class_id, title').eq('id', (submission as { assignment_id: string }).assignment_id).maybeSingle();
  if (!assignment) return null;
  const access = await resolveClassAccess((assignment as { class_id: string }).class_id, clerkId);
  return { mark: mark as Record<string, unknown>, submission: submission as Record<string, unknown>, assignment: assignment as Record<string, unknown>, access };
}

// ---------------------------------------------------------------------------
// GET /review/queue?assignment_id=&filter= — confidence-triaged queue (§9.1/9.2)
// ---------------------------------------------------------------------------
router.get('/queue', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const assignmentId = typeof req.query.assignment_id === 'string' ? req.query.assignment_id : '';
    const filter = typeof req.query.filter === 'string' ? req.query.filter : 'all';
    if (!assignmentId) return res.status(400).json({ error: 'assignment_id is required' });

    const { data: assignment } = await supabase.from('assignments').select('id, class_id, title').eq('id', assignmentId).maybeSingle();
    if (!assignment) return res.status(404).json({ error: 'Assignment not found' });
    const access = await resolveClassAccess((assignment as { class_id: string }).class_id, clerkId);
    if (!access) return res.status(404).json({ error: 'Assignment not found' });

    const { threshold } = await getThreshold(clerkId);

    // All marks for this assignment's submissions.
    const { data: submissions } = await supabase.from('submissions').select('id, student_clerk_id').eq('assignment_id', assignmentId);
    const subRows = (submissions ?? []) as { id: string; student_clerk_id: string }[];
    const subIds = subRows.map((s) => s.id);
    if (subIds.length === 0) {
      return res.json({ assignment: { id: assignment.id, title: (assignment as { title: string }).title, can_grade: access.canGrade }, threshold, counts: {}, items: [] });
    }

    const [{ data: marks }, { data: answers }, { data: aqs }] = await Promise.all([
      supabase.from('submission_marks').select('*').in('submission_id', subIds),
      supabase.from('submission_answers').select('submission_id, assignment_question_id, answer_text, selected_option, ocr_text, ocr_status, images').in('submission_id', subIds),
      supabase.from('assignment_questions').select('id, order_index, marks, snapshot, source, question_uid, custom_question_id').eq('assignment_id', assignmentId),
    ]);

    const markRows = (marks ?? []) as Record<string, unknown>[];
    const answerMap = new Map<string, Record<string, unknown>>();
    for (const a of (answers ?? []) as Record<string, unknown>[]) answerMap.set(`${a.submission_id}:${a.assignment_question_id}`, a);
    const aqMap = new Map<string, Record<string, unknown>>();
    for (const q of (aqs ?? []) as Record<string, unknown>[]) aqMap.set(q.id as string, q);
    const studentMap = new Map<string, string>(subRows.map((s) => [s.id, s.student_clerk_id]));

    // Examiner report notes (§10.4) — best-effort from the bank question's
    // `reference` jsonb; null when content ingestion hasn't provided it.
    const bankUids = ((aqs ?? []) as Record<string, unknown>[])
      .filter((q) => q.source === 'bank' && q.question_uid)
      .map((q) => q.question_uid as string);
    const examinerByAq = new Map<string, string>();
    if (bankUids.length > 0) {
      const { data: refs } = await supabase.from('questions').select('id, reference').in('id', bankUids);
      const refByUid = new Map<string, unknown>();
      for (const r of (refs ?? []) as { id: string; reference: unknown }[]) refByUid.set(r.id, r.reference);
      for (const q of (aqs ?? []) as Record<string, unknown>[]) {
        if (q.source === 'bank' && q.question_uid) {
          const note = examinerNote(refByUid.get(q.question_uid as string));
          if (note) examinerByAq.set(q.id as string, note);
        }
      }
    }

    // Names.
    const studentIds = Array.from(new Set(subRows.map((s) => s.student_clerk_id)));
    const nameMap = new Map<string, string>();
    if (studentIds.length > 0) {
      const { data: profiles } = await supabase.from('profiles').select('clerk_id, full_name, email').in('clerk_id', studentIds);
      for (const p of (profiles ?? []) as { clerk_id: string; full_name: string | null; email: string | null }[]) {
        nameMap.set(p.clerk_id, p.full_name || p.email || 'Student');
      }
    }

    // Counts across all marks (MCQ auto-approved bypass the queue, §8.1).
    const counts = {
      total: markRows.length,
      needs_review: markRows.filter((m) => m.status === 'needs_review').length,
      below_threshold: markRows.filter((m) => m.status === 'needs_review' && m.confidence !== null && Number(m.confidence) < threshold).length,
      ready_bulk: markRows.filter((m) => m.status === 'needs_review' && m.confidence !== null && Number(m.confidence) >= threshold).length,
      ocr_failed: markRows.filter((m) => m.status === 'ocr_failed').length,
      overridden: markRows.filter((m) => m.status === 'overridden').length,
      approved: markRows.filter((m) => m.status === 'approved' || m.status === 'auto_approved').length,
      flagged: markRows.filter((m) => m.flagged).length,
    };

    // Reviewable items exclude MCQ auto-approved.
    let items = markRows.filter((m) => m.question_type !== 'mcq' || m.status !== 'auto_approved');
    if (filter === 'below_threshold') items = items.filter((m) => m.status === 'needs_review' && m.confidence !== null && Number(m.confidence) < threshold);
    else if (filter === 'overridden') items = items.filter((m) => m.status === 'overridden');
    else if (filter === 'flagged') items = items.filter((m) => m.flagged);
    else if (filter === 'ocr_failed') items = items.filter((m) => m.status === 'ocr_failed');
    else if (filter === 'needs_review') items = items.filter((m) => m.status === 'needs_review' || m.status === 'ocr_failed');

    // Sort by confidence ascending — least certain first (null first).
    items.sort((a, b) => {
      const ca = a.confidence === null ? -1 : Number(a.confidence);
      const cb = b.confidence === null ? -1 : Number(b.confidence);
      return ca - cb;
    });

    const payload = items.map((m) => {
      const aq = aqMap.get(m.assignment_question_id as string) || {};
      const snap = (aq.snapshot as Record<string, unknown>) || {};
      const ans = answerMap.get(`${m.submission_id}:${m.assignment_question_id}`) || {};
      return {
        mark_id: m.id,
        submission_id: m.submission_id,
        assignment_question_id: m.assignment_question_id,
        student_name: nameMap.get(studentMap.get(m.submission_id as string) || '') || 'Student',
        question: {
          number: snap.question_number ?? aq.order_index,
          topic: snap.topic ?? null,
          marks: m.ai_marks_available ?? aq.marks ?? null,
          text: snap.text ?? '',
          type: m.question_type,
        },
        answer: {
          text: ans.answer_text ?? null,
          selected_option: ans.selected_option ?? null,
          ocr_text: ans.ocr_text ?? null,
          ocr_status: ans.ocr_status ?? 'none',
          images: ans.images ?? [],
        },
        ai_criteria: m.ai_criteria ?? [],
        final_criteria: m.final_criteria ?? null,
        criterion_comments: m.criterion_comments ?? [],
        ai_score: m.ai_score ?? null,
        ai_marks_available: m.ai_marks_available ?? null,
        final_score: m.final_score ?? null,
        confidence: m.confidence ?? null,
        status: m.status,
        flagged: m.flagged ?? false,
        examiner_note: examinerByAq.get(m.assignment_question_id as string) ?? null,
      };
    });

    return res.json({
      assignment: { id: assignment.id, title: (assignment as { title: string }).title, can_grade: access.canGrade },
      threshold,
      counts,
      items: payload,
    });
  } catch (err) {
    console.error('Review queue error:', err);
    return res.status(500).json({ error: 'Failed to load review queue' });
  }
});

// ---------------------------------------------------------------------------
// POST /review/marks/:id/approve — accept the AI mark as-is (§9.3).
// ---------------------------------------------------------------------------
router.post('/marks/:id/approve', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const ctx = await loadMarkContext(req.params.id, req.auth!.clerkId);
    if (!ctx || !ctx.access) return res.status(404).json({ error: 'Mark not found' });
    if (!ctx.access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const aiCriteria = (ctx.mark.ai_criteria as Criterion[]) ?? [];
    const finalScore = aiCriteria.reduce((s, c) => s + (Number(c.marks_awarded) || 0), 0);

    await supabase
      .from('submission_marks')
      .update({
        final_criteria: aiCriteria,
        final_score: ctx.mark.ai_score ?? finalScore,
        status: 'approved',
        reviewed_by: req.auth!.clerkId,
        reviewed_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      .eq('id', req.params.id);

    // Automatic release fires when the submission is now fully reviewed (§11.1).
    await maybeAutoRelease(ctx.assignment.id as string, req.auth!.clerkId);
    return res.json({ ok: true });
  } catch (err) {
    console.error('Approve mark error:', err);
    return res.status(500).json({ error: 'Failed to approve' });
  }
});

// ---------------------------------------------------------------------------
// POST /review/marks/:id/override — per-criterion override (§9.4). Teacher wins.
// ---------------------------------------------------------------------------
router.post('/marks/:id/override', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const ctx = await loadMarkContext(req.params.id, req.auth!.clerkId);
    if (!ctx || !ctx.access) return res.status(404).json({ error: 'Mark not found' });
    if (!ctx.access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const incoming = req.body?.criteria;
    if (!Array.isArray(incoming)) return res.status(400).json({ error: 'criteria array is required' });

    const aiCriteria = (ctx.mark.ai_criteria as Criterion[]) ?? [];
    const finalCriteria: Criterion[] = incoming.map((c: Record<string, unknown>, i: number) => {
      const base = aiCriteria[i] ?? ({} as Criterion);
      const marksAvailable = Number(base.marks_available ?? c.marks_available ?? 0);
      const awarded = Boolean(c.awarded);
      // Partial marks supported where provided; else binary on marks_available.
      const marksAwarded = Number.isFinite(Number(c.marks_awarded))
        ? Math.max(0, Math.min(marksAvailable, Number(c.marks_awarded)))
        : awarded
          ? marksAvailable
          : 0;
      return {
        criterion_text: base.criterion_text ?? String(c.criterion_text ?? ''),
        marks_available: marksAvailable,
        marks_awarded: marksAwarded,
        awarded: marksAwarded > 0,
        reasoning: base.reasoning ?? '',
        confidence: base.confidence ?? null,
      };
    });
    const finalScore = finalCriteria.reduce((s, c) => s + c.marks_awarded, 0);

    // §9.9 audit: record each changed criterion's before/after. AI score is
    // never overwritten (ai_criteria/ai_score untouched).
    const changes = finalCriteria
      .map((c, i) => ({ i, before: aiCriteria[i]?.marks_awarded ?? null, after: c.marks_awarded, criterion: c.criterion_text }))
      .filter((ch) => ch.before !== ch.after);

    await supabase
      .from('submission_marks')
      .update({
        final_criteria: finalCriteria,
        final_score: finalScore,
        status: 'overridden',
        reviewed_by: req.auth!.clerkId,
        reviewed_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      })
      .eq('id', req.params.id);

    if (changes.length > 0) {
      await logActivity(req.auth!.clerkId, 'override', 'mark', req.params.id, { changes, final_score: finalScore });
    }

    await maybeAutoRelease(ctx.assignment.id as string, req.auth!.clerkId);
    return res.json({ ok: true, final_score: finalScore });
  } catch (err) {
    console.error('Override mark error:', err);
    return res.status(500).json({ error: 'Failed to override' });
  }
});

// ---------------------------------------------------------------------------
// POST /review/marks/:id/flag — flag for re-marking (§9.1).
// ---------------------------------------------------------------------------
router.post('/marks/:id/flag', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const ctx = await loadMarkContext(req.params.id, req.auth!.clerkId);
    if (!ctx || !ctx.access) return res.status(404).json({ error: 'Mark not found' });
    // Flagging for re-marking is the `moderate` capability (§4.2) — teachers of
    // the class or oversight grant holders with moderate.
    if (!ctx.access.canGrade && !ctx.access.canModerate) return res.status(403).json({ error: 'No moderate access' });

    const flagged = req.body?.flagged !== false;
    await supabase
      .from('submission_marks')
      .update({
        flagged,
        flagged_by: flagged ? req.auth!.clerkId : null,
        flagged_at: flagged ? new Date().toISOString() : null,
        flag_reason: flagged ? (typeof req.body?.reason === 'string' ? req.body.reason : null) : null,
        updated_at: new Date().toISOString(),
      })
      .eq('id', req.params.id);

    await logActivity(req.auth!.clerkId, 'flag', 'mark', req.params.id, { flagged, reason: req.body?.reason ?? null });
    return res.json({ ok: true });
  } catch (err) {
    console.error('Flag mark error:', err);
    return res.status(500).json({ error: 'Failed to flag' });
  }
});

// ---------------------------------------------------------------------------
// POST /review/bulk-approve — approve all at/above threshold (§9.5).
// ---------------------------------------------------------------------------
router.post('/bulk-approve', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const assignmentId = String(req.body?.assignment_id ?? '');
    const { data: assignment } = await supabase.from('assignments').select('id, class_id').eq('id', assignmentId).maybeSingle();
    if (!assignment) return res.status(404).json({ error: 'Assignment not found' });
    const access = await resolveClassAccess((assignment as { class_id: string }).class_id, clerkId);
    if (!access || !access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const { threshold: defaultThreshold } = await getThreshold(clerkId);
    const threshold = Number.isFinite(Number(req.body?.threshold)) ? Number(req.body.threshold) : defaultThreshold;

    const { data: subs } = await supabase.from('submissions').select('id').eq('assignment_id', assignmentId);
    const subIds = ((subs ?? []) as { id: string }[]).map((s) => s.id);
    if (subIds.length === 0) return res.json({ approved: 0 });

    const { data: marks } = await supabase
      .from('submission_marks')
      .select('id, ai_criteria, ai_score, confidence, status')
      .in('submission_id', subIds)
      .eq('status', 'needs_review');

    const eligible = ((marks ?? []) as Record<string, unknown>[]).filter((m) => m.confidence !== null && Number(m.confidence) >= threshold);
    const now = new Date().toISOString();
    for (const m of eligible) {
      const ai = (m.ai_criteria as Criterion[]) ?? [];
      await supabase
        .from('submission_marks')
        .update({ final_criteria: ai, final_score: m.ai_score ?? ai.reduce((s, c) => s + (Number(c.marks_awarded) || 0), 0), status: 'approved', reviewed_by: clerkId, reviewed_at: now, updated_at: now })
        .eq('id', m.id);
    }

    // §9.5: recorded as a single event with an item count.
    await logActivity(clerkId, 'bulk_approve', 'assignment', assignmentId, { count: eligible.length, threshold });
    await maybeAutoRelease(assignmentId, clerkId);
    return res.json({ approved: eligible.length, threshold });
  } catch (err) {
    console.error('Bulk approve error:', err);
    return res.status(500).json({ error: 'Failed to bulk approve' });
  }
});

// ---------------------------------------------------------------------------
// Sampling mode (§9.7).
// ---------------------------------------------------------------------------
router.post('/sample', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const assignmentId = String(req.body?.assignment_id ?? '');
    const rate = Math.min(Math.max(Number(req.body?.rate) || 0.1, 0.01), 1);
    const { data: assignment } = await supabase.from('assignments').select('id, class_id').eq('id', assignmentId).maybeSingle();
    if (!assignment) return res.status(404).json({ error: 'Assignment not found' });
    const access = await resolveClassAccess((assignment as { class_id: string }).class_id, clerkId);
    if (!access || !access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const { data: subs } = await supabase.from('submissions').select('id').eq('assignment_id', assignmentId);
    const subIds = ((subs ?? []) as { id: string }[]).map((s) => s.id);
    const { data: marks } = await supabase.from('submission_marks').select('id').in('submission_id', subIds).eq('status', 'needs_review').eq('question_type', 'theory');
    const ids = ((marks ?? []) as { id: string }[]).map((m) => m.id);

    // Genuinely random sample (Fisher–Yates), not the first n (§9.7).
    for (let i = ids.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [ids[i], ids[j]] = [ids[j], ids[i]];
    }
    const sampleSize = Math.max(1, Math.ceil(ids.length * rate));
    const sampleIds = ids.slice(0, sampleSize);

    return res.json({ total: ids.length, sample_size: sampleIds.length, rate, sample_ids: sampleIds });
  } catch (err) {
    console.error('Sample error:', err);
    return res.status(500).json({ error: 'Failed to build sample' });
  }
});

// POST /review/approve-remainder — approve everything not individually reviewed,
// recording how the assignment was sampled (§9.7).
router.post('/approve-remainder', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const assignmentId = String(req.body?.assignment_id ?? '');
    const rate = Number(req.body?.rate) || null;
    const overrideRate = Number(req.body?.override_rate);
    const { data: assignment } = await supabase.from('assignments').select('id, class_id').eq('id', assignmentId).maybeSingle();
    if (!assignment) return res.status(404).json({ error: 'Assignment not found' });
    const access = await resolveClassAccess((assignment as { class_id: string }).class_id, clerkId);
    if (!access || !access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const { data: subs } = await supabase.from('submissions').select('id').eq('assignment_id', assignmentId);
    const subIds = ((subs ?? []) as { id: string }[]).map((s) => s.id);
    const { data: marks } = await supabase.from('submission_marks').select('id, ai_criteria, ai_score').in('submission_id', subIds).eq('status', 'needs_review');
    const now = new Date().toISOString();
    for (const m of (marks ?? []) as Record<string, unknown>[]) {
      const ai = (m.ai_criteria as Criterion[]) ?? [];
      await supabase
        .from('submission_marks')
        .update({ final_criteria: ai, final_score: m.ai_score ?? ai.reduce((s, c) => s + (Number(c.marks_awarded) || 0), 0), status: 'approved', reviewed_by: clerkId, reviewed_at: now, updated_at: now })
        .eq('id', m.id);
    }

    await supabase
      .from('assignments')
      .update({ marked_via: 'sample', sample_rate: rate, sample_override_rate: Number.isFinite(overrideRate) ? overrideRate : null })
      .eq('id', assignmentId);
    await logActivity(clerkId, 'sample_approve', 'assignment', assignmentId, { count: (marks ?? []).length, rate, override_rate: overrideRate });
    await maybeAutoRelease(assignmentId, clerkId);

    return res.json({ approved: (marks ?? []).length });
  } catch (err) {
    console.error('Approve remainder error:', err);
    return res.status(500).json({ error: 'Failed to approve remainder' });
  }
});

// ---------------------------------------------------------------------------
// Auto-approve threshold settings (§9.6).
// ---------------------------------------------------------------------------
router.get('/settings', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { threshold, floor } = await getThreshold(req.auth!.clerkId);
    return res.json({ auto_approve_threshold: threshold, floor });
  } catch (err) {
    console.error('Get review settings error:', err);
    return res.status(500).json({ error: 'Failed to load settings' });
  }
});

router.patch('/settings', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const requested = Number(req.body?.auto_approve_threshold);
    if (!Number.isFinite(requested) || requested < 0 || requested > 1) {
      return res.status(400).json({ error: 'auto_approve_threshold must be between 0 and 1' });
    }
    const { floor } = await getThreshold(clerkId);
    // Teachers can tighten above the institution floor, never loosen below it (§9.6).
    if (floor !== null && requested < floor) {
      return res.status(400).json({ error: `Your institution requires a minimum threshold of ${floor}` });
    }
    await supabase.from('profiles').update({ auto_approve_threshold: requested, updated_at: new Date().toISOString() }).eq('clerk_id', clerkId);
    return res.json({ auto_approve_threshold: requested, floor });
  } catch (err) {
    console.error('Update review settings error:', err);
    return res.status(500).json({ error: 'Failed to update settings' });
  }
});

export default router;
