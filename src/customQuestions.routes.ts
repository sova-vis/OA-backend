import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';

/**
 * Teacher Portal — custom questions with discrete criteria (spec §5.4).
 * Private to the uploading teacher by default, optionally shared to institution.
 */

const router = Router();
router.use(clerkAuth);

interface CriterionInput {
  criterion_text?: string;
  marks?: number;
  order_index?: number;
}

function normalizeCriteria(raw: unknown): { criterion_text: string; marks: number; order_index: number }[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((c, i) => {
      const obj = c as CriterionInput;
      return {
        criterion_text: typeof obj.criterion_text === 'string' ? obj.criterion_text.trim() : '',
        marks: Number.isFinite(obj.marks) ? Math.max(0, Number(obj.marks)) : 1,
        order_index: Number.isFinite(obj.order_index) ? Number(obj.order_index) : i,
      };
    })
    .filter((c) => c.criterion_text);
}

async function getInstitutionId(clerkId: string): Promise<string | null> {
  const { data } = await supabase.from('profiles').select('institution_id').eq('clerk_id', clerkId).maybeSingle();
  return (data as { institution_id: string | null } | null)?.institution_id ?? null;
}

async function loadCriteria(questionIds: string[]) {
  const map = new Map<string, { id: string; criterion_text: string; marks: number; order_index: number }[]>();
  if (questionIds.length === 0) return map;
  const { data } = await supabase
    .from('custom_question_criteria')
    .select('id, custom_question_id, criterion_text, marks, order_index')
    .in('custom_question_id', questionIds)
    .order('order_index');
  for (const c of (data ?? []) as { custom_question_id: string; id: string; criterion_text: string; marks: number; order_index: number }[]) {
    const list = map.get(c.custom_question_id) ?? [];
    list.push({ id: c.id, criterion_text: c.criterion_text, marks: c.marks, order_index: c.order_index });
    map.set(c.custom_question_id, list);
  }
  return map;
}

// POST /custom-questions — create with discrete criteria.
router.post('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const body = req.body ?? {};
    const subject = String(body.subject ?? '').trim();
    const questionText = String(body.question_text ?? '').trim();
    if (!subject || !questionText) return res.status(400).json({ error: 'subject and question_text are required' });

    const criteria = normalizeCriteria(body.criteria);
    if (criteria.length === 0) return res.status(400).json({ error: 'At least one mark-scheme criterion is required' });

    const questionType = ['mcq', 'structured', 'extended'].includes(body.question_type) ? body.question_type : 'structured';
    const marks = criteria.reduce((s, c) => s + c.marks, 0);

    const { data: created, error } = await supabase
      .from('custom_questions')
      .insert({
        owner_clerk_id: clerkId,
        institution_id: await getInstitutionId(clerkId),
        subject,
        syllabus_code: body.syllabus_code?.trim() || null,
        topic: body.topic?.trim() || null,
        sub_topic: body.sub_topic?.trim() || null,
        question_text: questionText,
        question_type: questionType,
        marks,
        shared_to_institution: Boolean(body.shared_to_institution),
      })
      .select('*')
      .single();
    if (error) throw error;

    const qId = (created as { id: string }).id;
    const rows = criteria.map((c) => ({ custom_question_id: qId, ...c }));
    const { error: critErr } = await supabase.from('custom_question_criteria').insert(rows);
    if (critErr) throw critErr;

    return res.status(201).json({ ...created, criteria });
  } catch (err) {
    console.error('Create custom question error:', err);
    return res.status(500).json({ error: 'Failed to create custom question' });
  }
});

// GET /custom-questions?subject= — list owned + institution-shared.
router.get('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const subject = typeof req.query.subject === 'string' ? req.query.subject : null;
    const institutionId = await getInstitutionId(clerkId);

    // Owned questions.
    let ownedQuery = supabase.from('custom_questions').select('*').eq('owner_clerk_id', clerkId);
    if (subject) ownedQuery = ownedQuery.ilike('subject', subject);
    const { data: owned, error: ownedErr } = await ownedQuery;
    if (ownedErr) throw ownedErr;

    // Institution-shared by others.
    let shared: Record<string, unknown>[] = [];
    if (institutionId) {
      let sharedQuery = supabase
        .from('custom_questions')
        .select('*')
        .eq('institution_id', institutionId)
        .eq('shared_to_institution', true)
        .neq('owner_clerk_id', clerkId);
      if (subject) sharedQuery = sharedQuery.ilike('subject', subject);
      const { data } = await sharedQuery;
      shared = (data as Record<string, unknown>[]) ?? [];
    }

    const all = [...((owned as Record<string, unknown>[]) ?? []), ...shared];
    all.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
    const criteriaMap = await loadCriteria(all.map((q) => q.id as string));

    return res.json(
      all.map((q) => ({ ...q, is_owner: q.owner_clerk_id === clerkId, criteria: criteriaMap.get(q.id as string) ?? [] }))
    );
  } catch (err) {
    console.error('List custom questions error:', err);
    return res.status(500).json({ error: 'Failed to list custom questions' });
  }
});

async function loadOwned(id: string, clerkId: string) {
  const { data, error } = await supabase.from('custom_questions').select('*').eq('id', id).maybeSingle();
  if (error) throw error;
  if (!data) return { question: null, isOwner: false };
  return { question: data as Record<string, unknown>, isOwner: (data as { owner_clerk_id: string }).owner_clerk_id === clerkId };
}

// GET /custom-questions/:id
router.get('/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { question, isOwner } = await loadOwned(req.params.id, clerkId);
    if (!question) return res.status(404).json({ error: 'Not found' });
    const criteriaMap = await loadCriteria([question.id as string]);
    return res.json({ ...question, is_owner: isOwner, criteria: criteriaMap.get(question.id as string) ?? [] });
  } catch (err) {
    console.error('Get custom question error:', err);
    return res.status(500).json({ error: 'Failed to load custom question' });
  }
});

// PATCH /custom-questions/:id — owner only; replaces criteria if provided.
router.patch('/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { question, isOwner } = await loadOwned(req.params.id, clerkId);
    if (!question) return res.status(404).json({ error: 'Not found' });
    if (!isOwner) return res.status(403).json({ error: 'Only the owner can edit this question' });

    const body = req.body ?? {};
    const update: Record<string, unknown> = { updated_at: new Date().toISOString() };
    if (typeof body.question_text === 'string' && body.question_text.trim()) update.question_text = body.question_text.trim();
    if (typeof body.subject === 'string' && body.subject.trim()) update.subject = body.subject.trim();
    if ('topic' in body) update.topic = body.topic?.trim() || null;
    if ('sub_topic' in body) update.sub_topic = body.sub_topic?.trim() || null;
    if ('syllabus_code' in body) update.syllabus_code = body.syllabus_code?.trim() || null;
    if (['mcq', 'structured', 'extended'].includes(body.question_type)) update.question_type = body.question_type;
    if (typeof body.shared_to_institution === 'boolean') update.shared_to_institution = body.shared_to_institution;

    if (body.criteria !== undefined) {
      const criteria = normalizeCriteria(body.criteria);
      if (criteria.length === 0) return res.status(400).json({ error: 'At least one criterion is required' });
      await supabase.from('custom_question_criteria').delete().eq('custom_question_id', question.id);
      await supabase.from('custom_question_criteria').insert(criteria.map((c) => ({ custom_question_id: question.id, ...c })));
      update.marks = criteria.reduce((s, c) => s + c.marks, 0);
    }

    const { data, error } = await supabase.from('custom_questions').update(update).eq('id', question.id).select('*').single();
    if (error) throw error;
    const criteriaMap = await loadCriteria([question.id as string]);
    return res.json({ ...data, criteria: criteriaMap.get(question.id as string) ?? [] });
  } catch (err) {
    console.error('Update custom question error:', err);
    return res.status(500).json({ error: 'Failed to update custom question' });
  }
});

// DELETE /custom-questions/:id — owner only.
router.delete('/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { question, isOwner } = await loadOwned(req.params.id, clerkId);
    if (!question) return res.status(404).json({ error: 'Not found' });
    if (!isOwner) return res.status(403).json({ error: 'Only the owner can delete this question' });
    const { error } = await supabase.from('custom_questions').delete().eq('id', question.id);
    if (error) throw error;
    return res.json({ ok: true });
  } catch (err) {
    console.error('Delete custom question error:', err);
    return res.status(500).json({ error: 'Failed to delete custom question' });
  }
});

export default router;
