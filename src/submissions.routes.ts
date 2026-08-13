import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import { resolveClassAccess } from './lib/portalAccess';
import { markSubmission } from './lib/marking';
import { grokChatJson, grokEnabled } from './lib/grok';

/**
 * Teacher Portal — submissions: student take/submit flow and teacher tracking
 * (spec feature groups 7 & 8). Mounted at /submissions.
 */

const router = Router();
router.use(clerkAuth);

interface AssignmentRow {
  id: string;
  class_id: string;
  title: string;
  status: string;
  deadline_at: string | null;
  timed: boolean;
  duration_minutes: number | null;
  attempt_limit: number | null;
  target_all: boolean;
  mark_scheme_visibility: string;
}

async function loadAssignment(id: string): Promise<AssignmentRow | null> {
  const { data } = await supabase
    .from('assignments')
    .select('id, class_id, title, status, deadline_at, timed, duration_minutes, attempt_limit, target_all, mark_scheme_visibility')
    .eq('id', id)
    .maybeSingle();
  return (data as AssignmentRow | null) ?? null;
}

/** Set of student clerk ids an assignment targets (§6.5). */
async function targetedStudents(assignment: AssignmentRow): Promise<Set<string>> {
  if (assignment.target_all) {
    const { data } = await supabase
      .from('class_enrollments')
      .select('student_clerk_id')
      .eq('class_id', assignment.class_id)
      .eq('status', 'active');
    return new Set(((data ?? []) as { student_clerk_id: string }[]).map((r) => r.student_clerk_id));
  }
  const { data } = await supabase.from('assignment_recipients').select('student_clerk_id').eq('assignment_id', assignment.id);
  return new Set(((data ?? []) as { student_clerk_id: string }[]).map((r) => r.student_clerk_id));
}

function effectiveDeadline(assignment: AssignmentRow, extensionUntil: string | null): Date | null {
  const base = assignment.deadline_at ? new Date(assignment.deadline_at) : null;
  const ext = extensionUntil ? new Date(extensionUntil) : null;
  if (base && ext) return ext > base ? ext : base;
  return ext ?? base;
}

// Question content sent to a student — never includes the mark scheme or the
// correct option (§6.4 visibility; releasing the scheme invalidates AI marking).
async function studentQuestionPayload(assignmentId: string) {
  const { data: aqs } = await supabase
    .from('assignment_questions')
    .select('id, order_index, marks, source, question_uid, custom_question_id, snapshot')
    .eq('assignment_id', assignmentId)
    .eq('excluded', false)
    .order('order_index');

  const rows = (aqs ?? []) as {
    id: string;
    order_index: number;
    marks: number | null;
    source: string;
    question_uid: string | null;
    custom_question_id: string | null;
    snapshot: Record<string, unknown>;
  }[];

  const bankUids = rows.filter((r) => r.source === 'bank' && r.question_uid).map((r) => r.question_uid as string);
  const customIds = rows.filter((r) => r.source === 'custom' && r.custom_question_id).map((r) => r.custom_question_id as string);

  const bankMap = new Map<string, { type: string; question_text: string; options: unknown; marks: number | null; images: unknown }>();
  if (bankUids.length > 0) {
    const { data } = await supabase.from('questions').select('id, type, question_text, options, marks, images').in('id', bankUids);
    for (const q of (data ?? []) as { id: string; type: string; question_text: string; options: unknown; marks: number | null; images: unknown }[]) {
      bankMap.set(q.id, { type: q.type, question_text: q.question_text, options: q.options, marks: q.marks, images: q.images });
    }
  }
  const customMap = new Map<string, { question_type: string; question_text: string; marks: number }>();
  if (customIds.length > 0) {
    const { data } = await supabase.from('custom_questions').select('id, question_type, question_text, marks').in('id', customIds);
    for (const q of (data ?? []) as { id: string; question_type: string; question_text: string; marks: number }[]) {
      customMap.set(q.id, { question_type: q.question_type, question_text: q.question_text, marks: q.marks });
    }
  }
  // Structured sub-parts (a, b, c…) so the full question reaches the student —
  // answers are NOT included (they're solving it).
  const partsByUid = new Map<string, { label: string; body: string; marks: number | null }[]>();
  if (bankUids.length > 0) {
    const { data } = await supabase.from('question_parts').select('question_uid, label, body, marks, order_index').in('question_uid', bankUids).order('order_index', { ascending: true });
    for (const p of (data ?? []) as { question_uid: string; label: string; body: string; marks: number | null }[]) {
      const list = partsByUid.get(p.question_uid) ?? [];
      list.push({ label: p.label, body: p.body, marks: p.marks });
      partsByUid.set(p.question_uid, list);
    }
  }

  return rows.map((r) => {
    if (r.source === 'custom' && r.custom_question_id) {
      const c = customMap.get(r.custom_question_id);
      return {
        assignment_question_id: r.id,
        order_index: r.order_index,
        type: c?.question_type === 'mcq' ? 'mcq' : 'theory',
        question_text: c?.question_text ?? '',
        options: [],
        marks: c?.marks ?? r.marks ?? 0,
      };
    }
    const b = r.question_uid ? bankMap.get(r.question_uid) : undefined;
    return {
      assignment_question_id: r.id,
      order_index: r.order_index,
      type: b?.type === 'mcq' ? 'mcq' : 'theory',
      question_text: b?.question_text ?? String((r.snapshot as { text?: string })?.text ?? ''),
      options: b?.type === 'mcq' ? normalizeOptions(b?.options) : [],
      marks: b?.marks ?? r.marks ?? 0,
      // Question figures so the student sees the full question (mark-scheme
      // "answer" images are excluded — never leak them while they're solving).
      images: normalizeQuestionImages(b?.images),
      parts: r.question_uid ? partsByUid.get(r.question_uid) ?? [] : [],
    };
  });
}

/** Question figures for the student, from a bank question's images jsonb. */
function normalizeQuestionImages(images: unknown): { src: string; alt: string; caption: string | null }[] {
  if (!Array.isArray(images)) return [];
  return images
    .filter((im) => im && typeof im === 'object' && (im as { role?: string }).role !== 'answer')
    .map((im) => {
      const o = im as { data_url?: string; public_url?: string; url?: string; alt?: string; caption?: string };
      return { src: o.data_url || o.public_url || o.url || '', alt: o.alt || 'Question figure', caption: o.caption || null };
    })
    .filter((im) => im.src);
}

function normalizeOptions(options: unknown): { label: string; text: string }[] {
  if (!options) return [];
  if (Array.isArray(options)) {
    return options.map((o, i) => {
      if (o && typeof o === 'object') {
        const obj = o as Record<string, unknown>;
        return { label: String(obj.label ?? String.fromCharCode(65 + i)).toUpperCase(), text: String(obj.text ?? obj.value ?? '') };
      }
      return { label: String.fromCharCode(65 + i), text: String(o) };
    });
  }
  return Object.entries(options as Record<string, unknown>).map(([label, text]) => ({ label: label.toUpperCase(), text: String(text) }));
}

// ===========================================================================
// STUDENT endpoints
// ===========================================================================

// GET /submissions/available — published assignments targeting this student.
router.get('/available', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { data: enrollments } = await supabase
      .from('class_enrollments')
      .select('class_id, approved_at, requested_at, created_at')
      .eq('student_clerk_id', clerkId)
      .eq('status', 'active');
    const enrollRows = (enrollments ?? []) as { class_id: string; approved_at: string | null; requested_at: string | null; created_at: string | null }[];
    const classIds = enrollRows.map((e) => e.class_id);
    if (classIds.length === 0) return res.json([]);
    // When the student joined each class — used to hide assignments that were
    // already set before they joined (new students don't inherit old work).
    const joinedAtByClass = new Map<string, string>();
    for (const e of enrollRows) { const t = e.approved_at || e.requested_at || e.created_at; if (t) joinedAtByClass.set(e.class_id, t); }

    // Class names for grouping in the classroom hub.
    const classNameById = new Map<string, string>();
    const { data: classRows } = await supabase.from('classes').select('id, name').in('id', classIds);
    for (const c of (classRows ?? []) as { id: string; name: string }[]) classNameById.set(c.id, c.name);

    const { data: assignments } = await supabase
      .from('assignments')
      .select('id, class_id, title, status, deadline_at, timed, duration_minutes, attempt_limit, target_all, mark_scheme_visibility, created_at')
      .in('class_id', classIds)
      .eq('status', 'published')
      .order('deadline_at', { ascending: true });

    const rows = (assignments ?? []) as AssignmentRow[];
    // Filter to those targeting the student and attach the student's submission state.
    const ids = rows.map((a) => a.id);
    const subByAssignment = new Map<string, { id: string; status: string; released: boolean }>();
    if (ids.length > 0) {
      const { data: subs } = await supabase
        .from('submissions')
        .select('id, assignment_id, status, released_at')
        .eq('student_clerk_id', clerkId)
        .in('assignment_id', ids);
      for (const s of (subs ?? []) as { id: string; assignment_id: string; status: string; released_at: string | null }[]) {
        subByAssignment.set(s.assignment_id, { id: s.id, status: s.status, released: Boolean(s.released_at) });
      }
    }

    const result = [];
    for (const a of rows) {
      const targeted = await targetedStudents(a);
      if (!targeted.has(clerkId)) continue;
      const sub = subByAssignment.get(a.id);
      // Skip assignments created before this student joined the class — unless
      // they already have a submission (were genuinely assigned earlier).
      const joinedAt = joinedAtByClass.get(a.class_id);
      const createdAt = (a as AssignmentRow & { created_at?: string | null }).created_at;
      if (!sub && joinedAt && createdAt && createdAt < joinedAt) continue;
      result.push({
        id: a.id,
        title: a.title,
        class_id: a.class_id,
        class_name: classNameById.get(a.class_id) ?? 'Class',
        deadline_at: a.deadline_at,
        timed: a.timed,
        duration_minutes: a.duration_minutes,
        submission_id: sub?.id ?? null,
        submission_status: sub?.status ?? 'not_started',
        released: sub?.released ?? false,
      });
    }
    return res.json(result);
  } catch (err) {
    console.error('Available assignments error:', err);
    return res.status(500).json({ error: 'Failed to load assignments' });
  }
});

// POST /submissions/start — get or create the student's submission for an assignment.
router.post('/start', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const assignmentId = String(req.body?.assignment_id ?? '');
    const assignment = await loadAssignment(assignmentId);
    if (!assignment || assignment.status !== 'published') return res.status(404).json({ error: 'Assignment not available' });

    const targeted = await targetedStudents(assignment);
    if (!targeted.has(clerkId)) return res.status(403).json({ error: 'This assignment is not assigned to you' });

    const { data: existing } = await supabase
      .from('submissions')
      .select('*')
      .eq('assignment_id', assignmentId)
      .eq('student_clerk_id', clerkId)
      .order('attempt_number', { ascending: false });
    const latest = ((existing ?? []) as Record<string, unknown>[])[0];

    let submission = latest;
    const attemptsUsed = (existing ?? []).length;
    const limit = assignment.attempt_limit;

    if (latest && (latest.status === 'in_progress' || latest.status === 'returned')) {
      submission = latest;
    } else if (latest && (latest.status === 'submitted' || latest.status === 'late')) {
      if (limit !== null && attemptsUsed >= limit) {
        // No attempts left — return the submitted one read-only.
        submission = latest;
      } else {
        const { data: created, error } = await supabase
          .from('submissions')
          .insert({ assignment_id: assignmentId, student_clerk_id: clerkId, attempt_number: attemptsUsed + 1, status: 'in_progress' })
          .select('*')
          .single();
        if (error) throw error;
        submission = created as Record<string, unknown>;
      }
    } else if (!latest) {
      const { data: created, error } = await supabase
        .from('submissions')
        .insert({ assignment_id: assignmentId, student_clerk_id: clerkId, attempt_number: 1, status: 'in_progress' })
        .select('*')
        .single();
      if (error) throw error;
      submission = created as Record<string, unknown>;
    }

    const [questions, { data: answers }] = await Promise.all([
      studentQuestionPayload(assignmentId),
      supabase
        .from('submission_answers')
        .select('assignment_question_id, answer_text, selected_option, answered')
        .eq('submission_id', submission!.id as string),
    ]);

    return res.json({
      submission,
      assignment: {
        id: assignment.id,
        title: assignment.title,
        deadline_at: assignment.deadline_at,
        timed: assignment.timed,
        duration_minutes: assignment.duration_minutes,
        effective_deadline: effectiveDeadline(assignment, (submission!.extension_until as string) ?? null)?.toISOString() ?? null,
      },
      questions,
      answers: answers ?? [],
      read_only: submission!.status === 'submitted' || submission!.status === 'late',
    });
  } catch (err) {
    console.error('Start submission error:', err);
    return res.status(500).json({ error: 'Failed to start assignment' });
  }
});

// POST /submissions/:id/answer — save an answer (autosave).
router.post('/:id/answer', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { data: submission } = await supabase.from('submissions').select('id, student_clerk_id, status').eq('id', req.params.id).maybeSingle();
    if (!submission) return res.status(404).json({ error: 'Submission not found' });
    if ((submission as { student_clerk_id: string }).student_clerk_id !== clerkId) return res.status(403).json({ error: 'Not your submission' });
    if (!['in_progress', 'returned'].includes((submission as { status: string }).status)) {
      return res.status(400).json({ error: 'This submission is not open for editing' });
    }

    const aqId = String(req.body?.assignment_question_id ?? '');
    if (!aqId) return res.status(400).json({ error: 'assignment_question_id is required' });
    const answerText = typeof req.body?.answer_text === 'string' ? req.body.answer_text : null;
    const selectedOption = typeof req.body?.selected_option === 'string' ? req.body.selected_option : null;
    const answered = Boolean((answerText && answerText.trim()) || selectedOption);

    const { error } = await supabase.from('submission_answers').upsert(
      {
        submission_id: req.params.id,
        assignment_question_id: aqId,
        answer_text: answerText,
        selected_option: selectedOption,
        answered,
        updated_at: new Date().toISOString(),
      },
      { onConflict: 'submission_id,assignment_question_id' }
    );
    if (error) throw error;
    return res.json({ ok: true });
  } catch (err) {
    console.error('Save answer error:', err);
    return res.status(500).json({ error: 'Failed to save answer' });
  }
});

// POST /submissions/:id/answer/ocr — upload a handwritten answer photo, OCR it
// with Grok vision, and store image + extracted text (§8.2). The original image
// is retained and shown to the teacher alongside the text at review time.
router.post('/:id/answer/ocr', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { data: submission } = await supabase.from('submissions').select('id, student_clerk_id, status').eq('id', req.params.id).maybeSingle();
    if (!submission) return res.status(404).json({ error: 'Submission not found' });
    if ((submission as { student_clerk_id: string }).student_clerk_id !== clerkId) return res.status(403).json({ error: 'Not your submission' });
    if (!['in_progress', 'returned'].includes((submission as { status: string }).status)) return res.status(400).json({ error: 'Not open for editing' });

    const aqId = String(req.body?.assignment_question_id ?? '');
    const rawImage = String(req.body?.image ?? '');
    if (!aqId || !rawImage) return res.status(400).json({ error: 'assignment_question_id and image are required' });

    // Accept a data URL or bare base64.
    const m = rawImage.match(/^data:(image\/[a-zA-Z+]+);base64,(.+)$/);
    const mime = m ? m[1] : 'image/jpeg';
    const base64 = m ? m[2] : rawImage;

    let ocrText = '';
    let confidence = 0;
    if (grokEnabled()) {
      try {
        const result = await grokChatJson({
          system: 'You transcribe a student\'s HANDWRITTEN exam answer from an image into plain text. Preserve the answer faithfully; do not solve or correct it. Return JSON {"text": "...", "confidence": 0..1} where confidence is how legible the handwriting was.',
          user: 'Transcribe this handwritten answer.',
          images: [{ base64, mimeType: mime }],
          maxTokens: 1500,
        });
        if (typeof result.text === 'string') ocrText = result.text.trim();
        const c = Number(result.confidence);
        confidence = Number.isFinite(c) ? Math.max(0, Math.min(1, c)) : 0.5;
      } catch {
        /* leave as failed */
      }
    }

    const OCR_FLOOR = 0.4;
    const ocrStatus = ocrText && confidence >= OCR_FLOOR ? 'ok' : 'failed';

    // Merge image into the answer's images array (retained for review).
    const { data: existing } = await supabase.from('submission_answers').select('images').eq('submission_id', req.params.id).eq('assignment_question_id', aqId).maybeSingle();
    const images = Array.isArray((existing as { images?: unknown[] } | null)?.images) ? (existing as { images: unknown[] }).images : [];
    images.push({ data_url: `data:${mime};base64,${base64}` });

    await supabase.from('submission_answers').upsert(
      {
        submission_id: req.params.id,
        assignment_question_id: aqId,
        images,
        ocr_text: ocrText || null,
        ocr_confidence: confidence,
        ocr_status: ocrStatus,
        // OCR text becomes the answer used for marking; teacher can correct it.
        answer_text: ocrText || null,
        answered: Boolean(ocrText),
        updated_at: new Date().toISOString(),
      },
      { onConflict: 'submission_id,assignment_question_id' }
    );

    return res.json({ ocr_text: ocrText, ocr_confidence: confidence, ocr_status: ocrStatus });
  } catch (err) {
    console.error('OCR upload error:', err);
    return res.status(500).json({ error: 'Failed to process handwriting' });
  }
});

// POST /submissions/:id/submit — finalize and run auto-marking.
router.post('/:id/submit', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { data: submission } = await supabase.from('submissions').select('*').eq('id', req.params.id).maybeSingle();
    if (!submission) return res.status(404).json({ error: 'Submission not found' });
    const sub = submission as Record<string, unknown>;
    if (sub.student_clerk_id !== clerkId) return res.status(403).json({ error: 'Not your submission' });
    if (!['in_progress', 'returned'].includes(sub.status as string)) return res.status(400).json({ error: 'Already submitted' });

    const assignment = await loadAssignment(sub.assignment_id as string);
    const deadline = assignment ? effectiveDeadline(assignment, (sub.extension_until as string) ?? null) : null;
    const now = new Date();
    const isLate = deadline ? now > deadline : false;

    await supabase
      .from('submissions')
      .update({
        status: isLate ? 'late' : 'submitted',
        submitted_at: now.toISOString(),
        time_spent_seconds: Number(req.body?.time_spent_seconds) || sub.time_spent_seconds || 0,
        updated_at: now.toISOString(),
      })
      .eq('id', req.params.id);

    // MCQ auto-marks instantly; theory is scaffolded for review (§8.1, §8.3).
    await markSubmission(req.params.id);

    return res.json({ ok: true, status: isLate ? 'late' : 'submitted' });
  } catch (err) {
    console.error('Submit error:', err);
    return res.status(500).json({ error: 'Failed to submit' });
  }
});

// GET /submissions/result/:assignmentId — a student's released results (§11).
// Content is filtered by the assignment's release_content config (§11.3).
router.get('/result/:assignmentId', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const { data: assignmentRow } = await supabase
      .from('assignments')
      .select('id, title, release_content')
      .eq('id', req.params.assignmentId)
      .maybeSingle();
    if (!assignmentRow) return res.status(404).json({ error: 'Assignment not found' });
    // Releasing means the student gets the full feedback — marks, per-criterion
    // breakdown, mark scheme, teacher comments and reasoning — UNLESS the teacher
    // explicitly turned a piece off. (Empty/absent config → everything shown.)
    const rc = (assignmentRow as { release_content: Record<string, boolean> }).release_content || {};
    const release = {
      marks: rc.marks !== false,
      breakdown: rc.breakdown !== false,
      comments: rc.comments !== false,
      scheme_missed: rc.scheme_missed !== false,
      ai_reasoning: rc.ai_reasoning !== false,
      examiner_notes: rc.examiner_notes !== false,
    };

    const { data: submission } = await supabase
      .from('submissions')
      .select('*')
      .eq('assignment_id', req.params.assignmentId)
      .eq('student_clerk_id', clerkId)
      .order('attempt_number', { ascending: false })
      .limit(1)
      .maybeSingle();
    if (!submission) return res.status(404).json({ error: 'No submission found' });
    const sub = submission as Record<string, unknown>;
    if (!sub.released_at) return res.json({ released: false });

    const [{ data: marks }, { data: aqs }, { data: answers }] = await Promise.all([
      supabase.from('submission_marks').select('*').eq('submission_id', sub.id as string),
      supabase.from('assignment_questions').select('id, order_index, snapshot').eq('assignment_id', req.params.assignmentId).order('order_index'),
      supabase.from('submission_answers').select('assignment_question_id, answer_text, selected_option, ocr_text, images').eq('submission_id', sub.id as string),
    ]);
    const snapByAq = new Map<string, Record<string, unknown>>();
    for (const q of (aqs ?? []) as { id: string; snapshot: Record<string, unknown> }[]) snapByAq.set(q.id, q.snapshot);
    const answerByAq = new Map<string, Record<string, unknown>>();
    for (const a of (answers ?? []) as Record<string, unknown>[]) answerByAq.set(a.assignment_question_id as string, a);

    const questions = ((marks ?? []) as Record<string, unknown>[]).map((m) => {
      const finalCriteria = (m.final_criteria ?? m.ai_criteria ?? []) as Record<string, unknown>[];
      const comments = (m.criterion_comments ?? []) as { index: number; text: string }[];
      const snap = snapByAq.get(m.assignment_question_id as string) || {};
      const ans = answerByAq.get(m.assignment_question_id as string) || {};
      const score = finalCriteria.reduce((s, c) => s + (Number(c.marks_awarded) || 0), 0);
      const available = Number(m.ai_marks_available ?? finalCriteria.reduce((s, c) => s + Number(c.marks_available || 0), 0));
      return {
        topic: snap.topic ?? null,
        number: snap.question_number ?? null,
        // The question, and what the student actually solved — so the marked
        // paper is meaningful when they read the feedback.
        question_text: snap.text ?? null,
        your_answer: (ans.answer_text as string) || (ans.ocr_text as string) || null,
        your_option: (ans.selected_option as string) || null,
        your_images: Array.isArray(ans.images) ? ans.images : [],
        score,
        available,
        // A voice note is deliberate per-question feedback the teacher recorded
        // for this student — always deliver it once results are released.
        voice_note: m.voice_note ?? null,
        // §11.3 breakdown: only if enabled.
        criteria: release.breakdown
          ? finalCriteria.map((c, i) => ({
              criterion_text: (release.scheme_missed || c.awarded) ? c.criterion_text : null, // scheme for missed criteria
              awarded: c.awarded,
              marks_available: c.marks_available,
              marks_awarded: c.marks_awarded,
              reasoning: release.ai_reasoning ? c.reasoning : null,
              comment: release.comments ? comments.find((cm) => cm.index === i)?.text ?? null : null,
            }))
          : null,
      };
    });

    const totalScore = questions.reduce((s, q) => s + q.score, 0);
    const totalAvailable = questions.reduce((s, q) => s + q.available, 0);

    return res.json({
      released: true,
      assignment_title: (assignmentRow as { title: string }).title,
      released_at: sub.released_at,
      total_score: release.marks ? totalScore : null,
      total_available: release.marks ? totalAvailable : null,
      overall_feedback: release.comments && !sub.feedback_is_draft ? sub.overall_feedback : null,
      questions,
    });
  } catch (err) {
    console.error('Student result error:', err);
    return res.status(500).json({ error: 'Failed to load results' });
  }
});

// GET /submissions/weak-spots — the student's mistakes grouped by topic, from
// their reviewed & released classroom work (for the Classroom dashboard).
router.get('/weak-spots', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const classId = typeof req.query.class_id === 'string' ? req.query.class_id : null;

    // Optionally scope to one class (classroom notebook).
    let assignmentFilter: string[] | null = null;
    if (classId) {
      const { data: asgs } = await supabase.from('assignments').select('id').eq('class_id', classId);
      assignmentFilter = ((asgs ?? []) as { id: string }[]).map((a) => a.id);
      if (assignmentFilter.length === 0) return res.json({ topics: [] });
    }

    let subQuery = supabase
      .from('submissions')
      .select('id, assignment_id, released_at')
      .eq('student_clerk_id', clerkId)
      .not('released_at', 'is', null);
    if (assignmentFilter) subQuery = subQuery.in('assignment_id', assignmentFilter);
    const { data: subs } = await subQuery;
    const subRows = (subs ?? []) as { id: string; assignment_id: string }[];
    if (subRows.length === 0) return res.json({ topics: [] });

    const subIds = subRows.map((s) => s.id);
    const asgIds = Array.from(new Set(subRows.map((s) => s.assignment_id)));

    const [{ data: marks }, { data: aqs }] = await Promise.all([
      supabase.from('submission_marks').select('assignment_question_id, final_criteria, ai_criteria').in('submission_id', subIds),
      supabase.from('assignment_questions').select('id, snapshot').in('assignment_id', asgIds),
    ]);
    const topicByAq = new Map<string, string>();
    for (const q of (aqs ?? []) as { id: string; snapshot: Record<string, unknown> }[]) topicByAq.set(q.id, String(q.snapshot?.topic ?? 'General'));

    // topic -> { missed, total, mistakes[] }
    const byTopic = new Map<string, { missed: number; total: number; mistakes: string[] }>();
    for (const m of (marks ?? []) as Record<string, unknown>[]) {
      const topic = topicByAq.get(m.assignment_question_id as string) || 'General';
      const criteria = (m.final_criteria ?? m.ai_criteria ?? []) as { criterion_text?: string; awarded?: boolean }[];
      const entry = byTopic.get(topic) ?? { missed: 0, total: 0, mistakes: [] };
      for (const c of criteria) {
        entry.total += 1;
        if (!c.awarded) {
          entry.missed += 1;
          if (c.criterion_text && entry.mistakes.length < 4 && !entry.mistakes.includes(c.criterion_text)) entry.mistakes.push(c.criterion_text);
        }
      }
      byTopic.set(topic, entry);
    }

    const topics = Array.from(byTopic, ([topic, v]) => ({
      topic,
      missed: v.missed,
      total: v.total,
      accuracy: v.total > 0 ? Math.round(((v.total - v.missed) / v.total) * 100) : 0,
      mistakes: v.mistakes,
    }))
      .filter((t) => t.missed > 0 && t.topic !== 'General')
      .sort((a, b) => b.missed - a.missed)
      .slice(0, 6);

    return res.json({ topics });
  } catch (err) {
    console.error('Weak spots error:', err);
    return res.status(500).json({ error: 'Failed to load weak spots' });
  }
});

// ===========================================================================
// TEACHER endpoints
// ===========================================================================

// GET /submissions/assignment/:assignmentId — status board (§7.1).
router.get('/assignment/:assignmentId', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const assignment = await loadAssignment(req.params.assignmentId);
    if (!assignment) return res.status(404).json({ error: 'Assignment not found' });
    const access = await resolveClassAccess(assignment.class_id, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Assignment not found' });

    const targeted = await targetedStudents(assignment);
    const targetedIds = Array.from(targeted);

    const [{ data: subs }, { data: profiles }] = await Promise.all([
      supabase.from('submissions').select('*').eq('assignment_id', assignment.id),
      targetedIds.length > 0
        ? supabase.from('profiles').select('clerk_id, full_name, email').in('clerk_id', targetedIds)
        : Promise.resolve({ data: [] as { clerk_id: string; full_name: string | null; email: string | null }[] }),
    ]);

    const nameMap = new Map<string, { full_name: string | null; email: string | null }>();
    for (const p of (profiles ?? []) as { clerk_id: string; full_name: string | null; email: string | null }[]) {
      nameMap.set(p.clerk_id, { full_name: p.full_name, email: p.email });
    }
    const subMap = new Map<string, Record<string, unknown>>();
    for (const s of (subs ?? []) as Record<string, unknown>[]) subMap.set(s.student_clerk_id as string, s);

    const now = new Date();
    const deadlinePassed = assignment.deadline_at ? now > new Date(assignment.deadline_at) : false;

    const rows = targetedIds.map((id) => {
      const s = subMap.get(id);
      const info = nameMap.get(id);
      let status: string;
      if (!s) status = deadlinePassed ? 'missed' : 'not_started';
      else if (s.status === 'submitted' || s.status === 'late') status = s.status as string;
      else if (s.status === 'returned') status = 'returned';
      else status = 'in_progress';
      return {
        student_clerk_id: id,
        full_name: info?.full_name ?? null,
        email: info?.email ?? null,
        submission_id: s?.id ?? null,
        status,
        submitted_at: s?.submitted_at ?? null,
        extension_until: s?.extension_until ?? null,
        total_score: s?.total_score ?? null,
        total_marks: s?.total_marks ?? null,
      };
    });

    const counts = rows.reduce((acc: Record<string, number>, r) => {
      acc[r.status] = (acc[r.status] ?? 0) + 1;
      return acc;
    }, {});

    return res.json({
      assignment: { id: assignment.id, title: assignment.title, deadline_at: assignment.deadline_at, total: targetedIds.length },
      counts,
      rows,
    });
  } catch (err) {
    console.error('Status board error:', err);
    return res.status(500).json({ error: 'Failed to load submissions' });
  }
});

async function logActivity(actor: string, eventType: string, targetType: string, targetId: string, detail: Record<string, unknown>) {
  await supabase.from('activity_log').insert({ actor_clerk_id: actor, event_type: eventType, target_type: targetType, target_id: targetId, detail });
}

// POST /submissions/assignment/:assignmentId/extend — extend deadline (§7.3).
router.post('/assignment/:assignmentId/extend', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const assignment = await loadAssignment(req.params.assignmentId);
    if (!assignment) return res.status(404).json({ error: 'Assignment not found' });
    const access = await resolveClassAccess(assignment.class_id, req.auth!.clerkId);
    if (!access || !access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const until = req.body?.extension_until;
    if (!until) return res.status(400).json({ error: 'extension_until is required' });
    const rawIds = req.body?.student_clerk_ids;
    const ids: string[] = Array.isArray(rawIds) ? rawIds.filter((x: unknown) => typeof x === 'string') : Array.from(await targetedStudents(assignment));

    // Upsert extension on each student's submission, reopening if already submitted.
    for (const id of ids) {
      const { data: existing } = await supabase
        .from('submissions')
        .select('id, status')
        .eq('assignment_id', assignment.id)
        .eq('student_clerk_id', id)
        .order('attempt_number', { ascending: false })
        .limit(1);
      const sub = ((existing ?? []) as { id: string; status: string }[])[0];
      if (sub) {
        const reopen = sub.status === 'submitted' || sub.status === 'late';
        await supabase
          .from('submissions')
          .update({ extension_until: until, ...(reopen ? { status: 'returned' } : {}), updated_at: new Date().toISOString() })
          .eq('id', sub.id);
      } else {
        await supabase
          .from('submissions')
          .insert({ assignment_id: assignment.id, student_clerk_id: id, attempt_number: 1, status: 'in_progress', extension_until: until });
      }
    }

    await logActivity(req.auth!.clerkId, 'extension', 'assignment', assignment.id, { extension_until: until, students: ids.length });
    return res.json({ ok: true, extended: ids.length });
  } catch (err) {
    console.error('Extend error:', err);
    return res.status(500).json({ error: 'Failed to extend deadline' });
  }
});

// POST /submissions/:id/reopen — return a submission to the student (§7.4).
router.post('/:id/reopen', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { data: submission } = await supabase.from('submissions').select('*').eq('id', req.params.id).maybeSingle();
    if (!submission) return res.status(404).json({ error: 'Submission not found' });
    const sub = submission as Record<string, unknown>;
    const assignment = await loadAssignment(sub.assignment_id as string);
    if (!assignment) return res.status(404).json({ error: 'Assignment not found' });
    const access = await resolveClassAccess(assignment.class_id, req.auth!.clerkId);
    if (!access || !access.canGrade) return res.status(403).json({ error: 'No grading access' });

    const wasReleased = Boolean(sub.released_at);
    await supabase
      .from('submissions')
      .update({
        status: 'returned',
        reopened_at: new Date().toISOString(),
        reopen_reason: typeof req.body?.reason === 'string' ? req.body.reason : null,
        released_at: null, // reopening a released result withdraws it (§7.4)
        submitted_at: null,
        updated_at: new Date().toISOString(),
      })
      .eq('id', req.params.id);

    await logActivity(req.auth!.clerkId, wasReleased ? 'withdraw' : 'reopen', 'submission', req.params.id, {
      reason: req.body?.reason ?? null,
      was_released: wasReleased,
    });
    // TODO(Phase 6): notify the student their result was withdrawn.
    return res.json({ ok: true, withdrawn: wasReleased });
  } catch (err) {
    console.error('Reopen error:', err);
    return res.status(500).json({ error: 'Failed to reopen submission' });
  }
});

export default router;
