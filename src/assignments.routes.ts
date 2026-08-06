import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import { resolveClassAccess } from './lib/portalAccess';

/**
 * Teacher Portal — assignment creation & tracking (spec feature group 6).
 * All routes authenticated; class-scoped access enforced in-code.
 */

const router = Router();
router.use(clerkAuth);

type SourceMode = 'full_paper' | 'selected' | 'topic';
type MarkSchemeVisibility = 'never' | 'after_submission' | 'after_deadline' | 'after_release';

interface IncomingQuestion {
  source?: 'bank' | 'custom';
  question_uid?: string;
  custom_question_id?: string;
  question_ref?: string;
  marks?: number;
  order_index?: number;
  excluded?: boolean;
  snapshot?: Record<string, unknown>;
}

function normalizeQuestions(raw: unknown): IncomingQuestion[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((q, i) => {
      const obj = q as IncomingQuestion;
      const source = obj.source === 'custom' ? 'custom' : 'bank';
      return {
        source,
        question_uid: typeof obj.question_uid === 'string' ? obj.question_uid : undefined,
        custom_question_id: typeof obj.custom_question_id === 'string' ? obj.custom_question_id : undefined,
        question_ref: typeof obj.question_ref === 'string' ? obj.question_ref : undefined,
        marks: Number.isFinite(obj.marks) ? Number(obj.marks) : undefined,
        order_index: Number.isFinite(obj.order_index) ? Number(obj.order_index) : i,
        excluded: Boolean(obj.excluded),
        snapshot: obj.snapshot && typeof obj.snapshot === 'object' ? obj.snapshot : {},
      } as IncomingQuestion;
    })
    // Keep bank questions (have a uid) and custom questions (have a custom id).
    .filter((q) => q.question_uid || q.custom_question_id);
}

function totalMarks(questions: IncomingQuestion[]): number {
  return questions
    .filter((q) => !q.excluded)
    .reduce((sum, q) => sum + (Number.isFinite(q.marks) ? Number(q.marks) : 0), 0);
}

const VALID_VISIBILITY: MarkSchemeVisibility[] = ['never', 'after_submission', 'after_deadline', 'after_release'];

interface RulesInput {
  deadline_at?: string | null;
  timed?: boolean;
  duration_minutes?: number | null;
  attempt_limit?: number | null;
  mark_scheme_visibility?: MarkSchemeVisibility;
}

function normalizeRules(raw: RulesInput | undefined) {
  const rules = raw ?? {};
  const visibility = VALID_VISIBILITY.includes(rules.mark_scheme_visibility as MarkSchemeVisibility)
    ? rules.mark_scheme_visibility
    : 'after_release';
  const attemptLimit =
    rules.attempt_limit === null || rules.attempt_limit === undefined
      ? null
      : Number.isFinite(rules.attempt_limit)
        ? Number(rules.attempt_limit)
        : null;
  return {
    deadline_at: rules.deadline_at || null,
    timed: Boolean(rules.timed),
    duration_minutes: rules.timed && Number.isFinite(rules.duration_minutes) ? Number(rules.duration_minutes) : null,
    attempt_limit: attemptLimit,
    mark_scheme_visibility: visibility,
  };
}

async function replaceQuestions(assignmentId: string, questions: IncomingQuestion[]) {
  await supabase.from('assignment_questions').delete().eq('assignment_id', assignmentId);
  if (questions.length === 0) return;
  const rows = questions.map((q, i) => ({
    assignment_id: assignmentId,
    source: q.source ?? 'bank',
    question_uid: q.question_uid ?? null,
    custom_question_id: q.custom_question_id ?? null,
    question_ref: q.question_ref ?? null,
    order_index: Number.isFinite(q.order_index) ? q.order_index : i,
    marks: Number.isFinite(q.marks) ? q.marks : null,
    excluded: Boolean(q.excluded),
    snapshot: q.snapshot ?? {},
  }));
  const { error } = await supabase.from('assignment_questions').insert(rows);
  if (error) throw error;
}

async function replaceRecipients(assignmentId: string, targetAll: boolean, ids: string[]) {
  await supabase.from('assignment_recipients').delete().eq('assignment_id', assignmentId);
  if (targetAll || ids.length === 0) return;
  const rows = ids.map((id) => ({ assignment_id: assignmentId, student_clerk_id: id }));
  const { error } = await supabase.from('assignment_recipients').insert(rows);
  if (error) throw error;
}

// ---------------------------------------------------------------------------
// POST /assignments — create a draft (or publish/schedule directly).
// ---------------------------------------------------------------------------
router.post('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const body = req.body ?? {};
    const classId = String(body.class_id ?? '');
    if (!classId) return res.status(400).json({ error: 'class_id is required' });

    const access = await resolveClassAccess(classId, clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    if (!access.canGrade) return res.status(403).json({ error: 'You do not have grading access to this class' });

    const title = String(body.title ?? '').trim();
    if (!title) return res.status(400).json({ error: 'title is required' });

    const sourceMode: SourceMode = ['full_paper', 'selected', 'topic'].includes(body.source_mode)
      ? body.source_mode
      : 'selected';
    const questions = normalizeQuestions(body.questions);
    if (questions.length === 0) return res.status(400).json({ error: 'At least one question is required' });

    const rules = normalizeRules(body.rules);
    const targetAll = body.target_all !== false;
    const recipientIds: string[] = Array.isArray(body.recipient_ids)
      ? body.recipient_ids.filter((x: unknown) => typeof x === 'string')
      : [];

    const requestedStatus = ['draft', 'published', 'scheduled'].includes(body.status) ? body.status : 'draft';
    const scheduledAt = body.scheduled_publish_at || null;
    let status = requestedStatus;
    let publishedAt: string | null = null;
    if (requestedStatus === 'published') publishedAt = new Date().toISOString();
    if (requestedStatus === 'scheduled' && !scheduledAt) status = 'draft';

    const { data: assignment, error } = await supabase
      .from('assignments')
      .insert({
        class_id: classId,
        owner_clerk_id: clerkId,
        title,
        source_mode: sourceMode,
        source_meta: body.source_meta && typeof body.source_meta === 'object' ? body.source_meta : {},
        status,
        ...rules,
        total_marks: totalMarks(questions),
        target_all: targetAll,
        scheduled_publish_at: status === 'scheduled' ? scheduledAt : null,
        published_at: publishedAt,
      })
      .select('*')
      .single();
    if (error) throw error;

    const assignmentId = (assignment as { id: string }).id;
    await replaceQuestions(assignmentId, questions);
    await replaceRecipients(assignmentId, targetAll, recipientIds);

    return res.status(201).json(assignment);
  } catch (err) {
    console.error('Create assignment error:', err);
    return res.status(500).json({ error: 'Failed to create assignment' });
  }
});

// ---------------------------------------------------------------------------
// GET /assignments?class_id= — list (all classes the caller can access if omitted)
// ---------------------------------------------------------------------------
router.get('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const classId = typeof req.query.class_id === 'string' ? req.query.class_id : null;

    // Determine accessible class ids.
    const [{ data: owned }, { data: coRows }] = await Promise.all([
      supabase.from('classes').select('id').eq('owner_clerk_id', clerkId),
      supabase.from('class_co_teachers').select('class_id').eq('teacher_clerk_id', clerkId),
    ]);
    const accessibleIds = new Set<string>([
      ...((owned ?? []) as { id: string }[]).map((c) => c.id),
      ...((coRows ?? []) as { class_id: string }[]).map((c) => c.class_id),
    ]);
    if (classId && !accessibleIds.has(classId)) {
      return res.status(404).json({ error: 'Class not found' });
    }
    const filterIds = classId ? [classId] : Array.from(accessibleIds);
    if (filterIds.length === 0) return res.json([]);

    const { data: assignments, error } = await supabase
      .from('assignments')
      .select('*')
      .in('class_id', filterIds)
      .order('created_at', { ascending: false });
    if (error) throw error;

    // Attach question counts.
    const ids = ((assignments ?? []) as { id: string }[]).map((a) => a.id);
    const qCounts = new Map<string, number>();
    if (ids.length > 0) {
      const { data: qs } = await supabase.from('assignment_questions').select('assignment_id').in('assignment_id', ids);
      for (const r of (qs ?? []) as { assignment_id: string }[]) {
        qCounts.set(r.assignment_id, (qCounts.get(r.assignment_id) ?? 0) + 1);
      }
    }

    // Attach pending-review counts (marks needing review) per assignment, so the
    // list can split Pending vs Done.
    const pendingCounts = new Map<string, number>();
    if (ids.length > 0) {
      const { data: subs } = await supabase.from('submissions').select('id, assignment_id').in('assignment_id', ids);
      const subToAsg = new Map<string, string>();
      for (const s of (subs ?? []) as { id: string; assignment_id: string }[]) subToAsg.set(s.id, s.assignment_id);
      const subIds = Array.from(subToAsg.keys());
      if (subIds.length > 0) {
        const { data: marks } = await supabase.from('submission_marks').select('submission_id, status').in('submission_id', subIds).in('status', ['needs_review', 'ocr_failed']);
        for (const m of (marks ?? []) as { submission_id: string }[]) {
          const aid = subToAsg.get(m.submission_id);
          if (aid) pendingCounts.set(aid, (pendingCounts.get(aid) ?? 0) + 1);
        }
      }
    }

    return res.json(
      ((assignments ?? []) as Record<string, unknown>[]).map((a) => ({
        ...a,
        question_count: qCounts.get(a.id as string) ?? 0,
        pending_reviews: pendingCounts.get(a.id as string) ?? 0,
      }))
    );
  } catch (err) {
    console.error('List assignments error:', err);
    return res.status(500).json({ error: 'Failed to list assignments' });
  }
});

async function loadAssignmentWithAccess(assignmentId: string, clerkId: string) {
  const { data: assignment, error } = await supabase
    .from('assignments')
    .select('*')
    .eq('id', assignmentId)
    .maybeSingle();
  if (error) throw error;
  if (!assignment) return { assignment: null, access: null };
  const access = await resolveClassAccess((assignment as { class_id: string }).class_id, clerkId);
  return { assignment: assignment as Record<string, unknown>, access };
}

// ---------------------------------------------------------------------------
// GET /assignments/:id — detail with questions + recipients.
// ---------------------------------------------------------------------------
router.get('/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { assignment, access } = await loadAssignmentWithAccess(req.params.id, req.auth!.clerkId);
    if (!assignment || !access) return res.status(404).json({ error: 'Assignment not found' });

    const [{ data: questions }, { data: recipients }] = await Promise.all([
      supabase.from('assignment_questions').select('*').eq('assignment_id', assignment.id).order('order_index'),
      supabase.from('assignment_recipients').select('student_clerk_id').eq('assignment_id', assignment.id),
    ]);

    return res.json({
      ...assignment,
      can_grade: access.canGrade,
      is_owner: access.isOwner,
      questions: questions ?? [],
      recipient_ids: ((recipients ?? []) as { student_clerk_id: string }[]).map((r) => r.student_clerk_id),
    });
  } catch (err) {
    console.error('Get assignment error:', err);
    return res.status(500).json({ error: 'Failed to load assignment' });
  }
});

// ---------------------------------------------------------------------------
// PATCH /assignments/:id — edit rules/title/questions/recipients.
// ---------------------------------------------------------------------------
router.patch('/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { assignment, access } = await loadAssignmentWithAccess(req.params.id, req.auth!.clerkId);
    if (!assignment || !access) return res.status(404).json({ error: 'Assignment not found' });
    if (!access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const body = req.body ?? {};
    const update: Record<string, unknown> = { updated_at: new Date().toISOString() };
    if (typeof body.title === 'string' && body.title.trim()) update.title = body.title.trim();
    if (body.source_meta && typeof body.source_meta === 'object') update.source_meta = body.source_meta;
    if (body.rules) Object.assign(update, normalizeRules(body.rules));
    if (typeof body.target_all === 'boolean') update.target_all = body.target_all;

    let newTotal: number | undefined;
    if (body.questions !== undefined) {
      const questions = normalizeQuestions(body.questions);
      await replaceQuestions(assignment.id as string, questions);
      newTotal = totalMarks(questions);
      update.total_marks = newTotal;
    }
    if (body.target_all !== undefined || body.recipient_ids !== undefined) {
      const targetAll = body.target_all !== false;
      const ids = Array.isArray(body.recipient_ids) ? body.recipient_ids.filter((x: unknown) => typeof x === 'string') : [];
      await replaceRecipients(assignment.id as string, targetAll, ids);
    }

    const { data, error } = await supabase.from('assignments').update(update).eq('id', assignment.id).select('*').single();
    if (error) throw error;
    return res.json(data);
  } catch (err) {
    console.error('Update assignment error:', err);
    return res.status(500).json({ error: 'Failed to update assignment' });
  }
});

// ---------------------------------------------------------------------------
// POST /assignments/:id/publish — publish now or schedule.
// ---------------------------------------------------------------------------
router.post('/:id/publish', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { assignment, access } = await loadAssignmentWithAccess(req.params.id, req.auth!.clerkId);
    if (!assignment || !access) return res.status(404).json({ error: 'Assignment not found' });
    if (!access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const scheduledAt = req.body?.scheduled_publish_at || null;
    const update: Record<string, unknown> = { updated_at: new Date().toISOString() };
    if (scheduledAt && new Date(scheduledAt) > new Date()) {
      update.status = 'scheduled';
      update.scheduled_publish_at = scheduledAt;
      update.published_at = null;
    } else {
      update.status = 'published';
      update.published_at = new Date().toISOString();
      update.scheduled_publish_at = null;
    }

    const { data, error } = await supabase.from('assignments').update(update).eq('id', assignment.id).select('*').single();
    if (error) throw error;
    return res.json(data);
  } catch (err) {
    console.error('Publish assignment error:', err);
    return res.status(500).json({ error: 'Failed to publish assignment' });
  }
});

// ---------------------------------------------------------------------------
// POST /assignments/:id/duplicate — copy questions+rules to another/same class.
// Deadline is never carried over (§6.7); submissions are not copied.
// ---------------------------------------------------------------------------
router.post('/:id/duplicate', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { assignment, access } = await loadAssignmentWithAccess(req.params.id, clerkId);
    if (!assignment || !access) return res.status(404).json({ error: 'Assignment not found' });

    const targetClassId = String(req.body?.target_class_id ?? assignment.class_id);
    const targetAccess = await resolveClassAccess(targetClassId, clerkId);
    if (!targetAccess || !targetAccess.canGrade) {
      return res.status(403).json({ error: 'No grading access to the target class' });
    }

    const { data: srcQuestions } = await supabase
      .from('assignment_questions')
      .select('*')
      .eq('assignment_id', assignment.id)
      .order('order_index');

    const { data: copy, error } = await supabase
      .from('assignments')
      .insert({
        class_id: targetClassId,
        owner_clerk_id: clerkId,
        title: `${assignment.title} (copy)`,
        source_mode: assignment.source_mode,
        source_meta: assignment.source_meta ?? {},
        status: 'draft',
        deadline_at: req.body?.deadline_at || null, // must be re-entered
        timed: assignment.timed,
        duration_minutes: assignment.duration_minutes,
        attempt_limit: assignment.attempt_limit,
        mark_scheme_visibility: assignment.mark_scheme_visibility,
        total_marks: assignment.total_marks,
        target_all: true,
      })
      .select('*')
      .single();
    if (error) throw error;

    const copyId = (copy as { id: string }).id;
    if (srcQuestions && srcQuestions.length > 0) {
      const rows = (srcQuestions as Record<string, unknown>[]).map((q) => ({
        assignment_id: copyId,
        source: q.source ?? 'bank',
        question_uid: q.question_uid,
        custom_question_id: q.custom_question_id,
        question_ref: q.question_ref,
        order_index: q.order_index,
        marks: q.marks,
        excluded: q.excluded,
        snapshot: q.snapshot,
      }));
      await supabase.from('assignment_questions').insert(rows);
    }

    return res.status(201).json(copy);
  } catch (err) {
    console.error('Duplicate assignment error:', err);
    return res.status(500).json({ error: 'Failed to duplicate assignment' });
  }
});

// ---------------------------------------------------------------------------
// DELETE /assignments/:id — draft only (published assignments are archived, not deleted).
// ---------------------------------------------------------------------------
router.delete('/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { assignment, access } = await loadAssignmentWithAccess(req.params.id, req.auth!.clerkId);
    if (!assignment || !access) return res.status(404).json({ error: 'Assignment not found' });
    if (!access.canGrade) return res.status(403).json({ error: 'No grading access' });
    // Deleting an assignment removes its questions and all submissions (cascade).
    const { error } = await supabase.from('assignments').delete().eq('id', assignment.id);
    if (error) throw error;
    return res.json({ ok: true });
  } catch (err) {
    console.error('Delete assignment error:', err);
    return res.status(500).json({ error: 'Failed to delete assignment' });
  }
});

export default router;
