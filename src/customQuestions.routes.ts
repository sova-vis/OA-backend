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

interface ImageInput { data_url?: string; url?: string; public_url?: string; alt?: string; caption?: string | null }
interface StoredImage { data_url: string; alt: string; caption: string | null }
// Figure images are stored inline as data URLs (teacher-authored, low volume).
function normalizeImages(raw: unknown): StoredImage[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((im) => {
      const o = (im ?? {}) as ImageInput;
      const src = [o.data_url, o.url, o.public_url].find((s) => typeof s === 'string' && s) || '';
      return {
        data_url: String(src),
        alt: typeof o.alt === 'string' && o.alt.trim() ? o.alt.trim() : 'Question figure',
        caption: typeof o.caption === 'string' && o.caption.trim() ? o.caption.trim() : null,
      };
    })
    .filter((im) => im.data_url && im.data_url.length < 8_000_000);
}

interface PartInput { label?: string; body?: string; text?: string; marks?: number | null; images?: unknown }
interface StoredPart { label: string; body: string; marks: number | null; images: StoredImage[] }
// A labelled sub-part: (a), (b)… — its own prompt, optional marks, own figures.
function normalizeParts(raw: unknown): StoredPart[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((p, i) => {
      const o = (p ?? {}) as PartInput;
      const body = (typeof o.body === 'string' ? o.body : typeof o.text === 'string' ? o.text : '').trim();
      const label = typeof o.label === 'string' && o.label.trim() ? o.label.trim() : `(${String.fromCharCode(97 + i)})`;
      const marks = Number.isFinite(o.marks as number) ? Math.max(0, Number(o.marks)) : null;
      return { label, body, marks, images: normalizeImages(o.images) };
    })
    .filter((p) => p.body || p.images.length);
}

/** True when an insert/update failed only because the parts/images columns aren't
 * migrated yet (032) — lets creation keep working before the migration is applied. */
function isMissingRichColumn(err: unknown): boolean {
  const e = err as { message?: string; details?: string; code?: string } | null;
  if (e?.code === 'PGRST204') return true;
  const blob = `${e?.message ?? ''} ${e?.details ?? ''}`.toLowerCase();
  return blob.includes('schema cache') || ((blob.includes('parts') || blob.includes('images')) && blob.includes('column'));
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

    const baseRow = {
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
    };
    const parts = normalizeParts(body.parts);
    const images = normalizeImages(body.images);

    let created: Record<string, unknown> | null = null;
    let error: unknown = null;
    ({ data: created, error } = await supabase
      .from('custom_questions').insert({ ...baseRow, parts, images }).select('*').single());
    if (error && isMissingRichColumn(error)) {
      // Migration 032 not applied yet — still create the question (without the
      // rich parts/images) so the teacher isn't blocked.
      ({ data: created, error } = await supabase
        .from('custom_questions').insert(baseRow).select('*').single());
    }
    if (error) throw error;

    const qId = (created as { id: string }).id;
    const rows = criteria.map((c) => ({ custom_question_id: qId, ...c }));
    const { error: critErr } = await supabase.from('custom_question_criteria').insert(rows);
    if (critErr) throw critErr;

    return res.status(201).json({ ...created, criteria, parts: created?.parts ?? [], images: created?.images ?? [] });
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
      all.map((q) => ({ ...q, is_owner: q.owner_clerk_id === clerkId, criteria: criteriaMap.get(q.id as string) ?? [], parts: q.parts ?? [], images: q.images ?? [] }))
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
    return res.json({ ...question, is_owner: isOwner, criteria: criteriaMap.get(question.id as string) ?? [], parts: question.parts ?? [], images: question.images ?? [] });
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
    if ('parts' in body) update.parts = normalizeParts(body.parts);
    if ('images' in body) update.images = normalizeImages(body.images);

    if (body.criteria !== undefined) {
      const criteria = normalizeCriteria(body.criteria);
      if (criteria.length === 0) return res.status(400).json({ error: 'At least one criterion is required' });
      await supabase.from('custom_question_criteria').delete().eq('custom_question_id', question.id);
      await supabase.from('custom_question_criteria').insert(criteria.map((c) => ({ custom_question_id: question.id, ...c })));
      update.marks = criteria.reduce((s, c) => s + c.marks, 0);
    }

    let data: Record<string, unknown> | null = null;
    let error: unknown = null;
    ({ data, error } = await supabase.from('custom_questions').update(update).eq('id', question.id).select('*').single());
    if (error && isMissingRichColumn(error)) {
      // Migration 032 not applied yet — save everything except the rich columns.
      const rest: Record<string, unknown> = { ...update };
      delete rest.parts;
      delete rest.images;
      ({ data, error } = await supabase.from('custom_questions').update(rest).eq('id', question.id).select('*').single());
    }
    if (error) throw error;
    const criteriaMap = await loadCriteria([question.id as string]);
    return res.json({ ...data, criteria: criteriaMap.get(question.id as string) ?? [], parts: data?.parts ?? [], images: data?.images ?? [] });
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
