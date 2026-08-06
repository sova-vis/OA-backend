import { supabase } from './supabase';
import { CRITERION_AWARD_THRESHOLD, evaluateCriterion, heuristicConfidence } from './gradingEngine';

/**
 * Marking orchestration (spec feature group 8).
 *
 * The portal marks MCQ itself (§8.1, deterministic vs the answer key) and builds
 * the per-criterion scaffold (§8.4) for theory answers. It does NOT perform
 * theory marking — that is the marking engine's job (§13). Until the engine
 * returns calibrated per-criterion confidence (Appendix A blocking dependency),
 * theory answers land as `needs_review` with an unmarked criteria scaffold that
 * the teacher fills in review (§9), or that the engine fills when wired.
 */

export interface Criterion {
  criterion_text: string;
  marks_available: number;
  marks_awarded: number;
  awarded: boolean;
  reasoning: string;
  confidence: number | null;
}

interface AssignmentQuestionRow {
  id: string;
  source: 'bank' | 'custom';
  question_uid: string | null;
  custom_question_id: string | null;
  marks: number | null;
  snapshot: Record<string, unknown>;
}

interface ResolvedQuestion {
  type: 'mcq' | 'theory';
  marks: number;
  questionText: string;
  subject: string | null;
  correctOption: string | null;
  criteria: { criterion_text: string; marks_available: number }[];
}

/** An unmarked criteria scaffold (awarded=false) for manual/deferred review. */
function scaffoldCriteria(criteria: { criterion_text: string; marks_available: number }[]): Criterion[] {
  return criteria.map((c) => ({
    criterion_text: c.criterion_text,
    marks_available: c.marks_available,
    marks_awarded: 0,
    awarded: false,
    reasoning: '',
    confidence: null,
  }));
}

/** Resolve marking metadata for one assignment question from its source. */
async function resolveQuestion(aq: AssignmentQuestionRow): Promise<ResolvedQuestion> {
  const fallbackMarks = Number.isFinite(aq.marks) ? Number(aq.marks) : 0;

  if (aq.source === 'custom' && aq.custom_question_id) {
    const [{ data: cq }, { data: criteria }] = await Promise.all([
      supabase.from('custom_questions').select('question_type, marks, question_text, subject').eq('id', aq.custom_question_id).maybeSingle(),
      supabase
        .from('custom_question_criteria')
        .select('criterion_text, marks, order_index')
        .eq('custom_question_id', aq.custom_question_id)
        .order('order_index'),
    ]);
    const row = cq as { question_type?: string; marks?: number; question_text?: string; subject?: string } | null;
    const type = row?.question_type === 'mcq' ? 'mcq' : 'theory';
    return {
      type,
      marks: Number(row?.marks ?? fallbackMarks),
      questionText: row?.question_text ?? '',
      subject: row?.subject ?? null,
      correctOption: null,
      criteria: ((criteria ?? []) as { criterion_text: string; marks: number }[]).map((c) => ({
        criterion_text: c.criterion_text,
        marks_available: c.marks,
      })),
    };
  }

  // Bank question.
  if (aq.question_uid) {
    const { data: q } = await supabase
      .from('questions')
      .select('type, marks, correct_option, marking_scheme, question_text, subject')
      .eq('id', aq.question_uid)
      .maybeSingle();
    const row = q as { type?: string; marks?: number; correct_option?: string; marking_scheme?: string; question_text?: string; subject?: string } | null;
    const marks = Number(row?.marks ?? fallbackMarks);
    if (row?.type === 'mcq') {
      return { type: 'mcq', marks, questionText: row?.question_text ?? '', subject: row?.subject ?? null, correctOption: (row.correct_option || '').toUpperCase() || null, criteria: [] };
    }
    // Structured/extended: the bank mark scheme is a single text blob, so it
    // becomes one criterion pending content-ingestion structuring (Appendix A).
    return {
      type: 'theory',
      marks,
      questionText: row?.question_text ?? '',
      subject: row?.subject ?? null,
      correctOption: null,
      criteria: [{ criterion_text: row?.marking_scheme?.trim() || 'Overall answer', marks_available: marks }],
    };
  }

  return { type: 'theory', marks: fallbackMarks, questionText: '', subject: null, correctOption: null, criteria: [{ criterion_text: 'Overall answer', marks_available: fallbackMarks }] };
}

/**
 * Mark a whole submission: auto-mark MCQ, scaffold theory for review.
 * Idempotent — upserts one submission_marks row per question.
 */
export async function markSubmission(submissionId: string): Promise<void> {
  const { data: submission } = await supabase
    .from('submissions')
    .select('id, assignment_id')
    .eq('id', submissionId)
    .maybeSingle();
  if (!submission) return;
  const assignmentId = (submission as { assignment_id: string }).assignment_id;

  const [{ data: aqs }, { data: answers }] = await Promise.all([
    supabase
      .from('assignment_questions')
      .select('id, source, question_uid, custom_question_id, marks, snapshot')
      .eq('assignment_id', assignmentId)
      .eq('excluded', false),
    supabase.from('submission_answers').select('assignment_question_id, selected_option, answer_text, ocr_text, ocr_status').eq('submission_id', submissionId),
  ]);

  const answerByQ = new Map<string, { selected_option: string | null; answer_text: string | null; ocr_text: string | null; ocr_status: string }>();
  for (const a of (answers ?? []) as { assignment_question_id: string; selected_option: string | null; answer_text: string | null; ocr_text: string | null; ocr_status: string }[]) {
    answerByQ.set(a.assignment_question_id, a);
  }

  let totalScore = 0;
  let totalMarks = 0;

  for (const aqRaw of (aqs ?? []) as AssignmentQuestionRow[]) {
    const resolved = await resolveQuestion(aqRaw);
    const answer = answerByQ.get(aqRaw.id);
    totalMarks += resolved.marks;

    if (resolved.type === 'mcq') {
      const chosen = (answer?.selected_option || '').toUpperCase();
      const correct = resolved.correctOption;
      const isCorrect = Boolean(correct) && chosen === correct;
      const awardedMarks = isCorrect ? resolved.marks : 0;
      totalScore += awardedMarks;

      const criteria: Criterion[] = [
        {
          criterion_text: 'Correct option selected',
          marks_available: resolved.marks,
          marks_awarded: awardedMarks,
          awarded: isCorrect,
          reasoning: correct
            ? isCorrect
              ? `Selected ${chosen}, which is correct.`
              : `Selected ${chosen || '—'}; the correct option is ${correct}.`
            : 'No answer key available.',
          confidence: 1, // deterministic
        },
      ];

      await upsertMark(submissionId, aqRaw.id, {
        question_type: 'mcq',
        ai_criteria: criteria,
        ai_score: awardedMarks,
        ai_marks_available: resolved.marks,
        confidence: 1,
        final_criteria: criteria,
        final_score: awardedMarks,
        status: 'auto_approved', // §8.1 MCQ bypasses review
      });
      continue;
    }

    // Theory: mark each criterion via the engine (§8.3); fall back to an
    // unmarked review scaffold when the engine is unavailable or there's no answer.
    const ocrFailed = answer?.ocr_status === 'failed';
    const studentText = (answer?.answer_text || answer?.ocr_text || '').trim();
    const marksAvailable = resolved.criteria.reduce((s, c) => s + c.marks_available, 0) || resolved.marks;

    if (ocrFailed) {
      await upsertMark(submissionId, aqRaw.id, {
        question_type: 'theory',
        ai_criteria: scaffoldCriteria(resolved.criteria),
        ai_score: null,
        ai_marks_available: marksAvailable,
        confidence: null,
        status: 'ocr_failed',
      });
      continue;
    }

    if (!studentText) {
      // Blank answer — no marks, but still teacher-reviewed.
      const blank: Criterion[] = resolved.criteria.map((c) => ({
        criterion_text: c.criterion_text,
        marks_available: c.marks_available,
        marks_awarded: 0,
        awarded: false,
        reasoning: 'No answer was provided.',
        confidence: 0.9,
      }));
      await upsertMark(submissionId, aqRaw.id, {
        question_type: 'theory',
        ai_criteria: blank,
        ai_score: 0,
        ai_marks_available: marksAvailable,
        confidence: 0.9,
        status: 'needs_review',
      });
      continue;
    }

    // Evaluate criteria concurrently.
    const evals = await Promise.all(
      resolved.criteria.map((c) =>
        evaluateCriterion({ question: resolved.questionText, studentAnswer: studentText, criterionText: c.criterion_text, subject: resolved.subject })
      )
    );

    if (evals.every((e) => e === null)) {
      // Engine unavailable → leave an unmarked scaffold for manual review.
      await upsertMark(submissionId, aqRaw.id, {
        question_type: 'theory',
        ai_criteria: scaffoldCriteria(resolved.criteria),
        ai_score: null,
        ai_marks_available: marksAvailable,
        confidence: null,
        status: 'needs_review',
      });
      continue;
    }

    const aiCriteria: Criterion[] = resolved.criteria.map((c, idx) => {
      const ev = evals[idx];
      if (!ev) {
        return { criterion_text: c.criterion_text, marks_available: c.marks_available, marks_awarded: 0, awarded: false, reasoning: '', confidence: null };
      }
      const awarded = ev.score >= CRITERION_AWARD_THRESHOLD;
      return {
        criterion_text: c.criterion_text,
        marks_available: c.marks_available,
        marks_awarded: awarded ? c.marks_available : 0,
        awarded,
        reasoning: ev.feedback || (awarded ? 'Answer satisfies this criterion.' : 'Answer does not clearly satisfy this criterion.'),
        confidence: heuristicConfidence(ev.score),
      };
    });

    const aiScore = aiCriteria.reduce((s, c) => s + c.marks_awarded, 0);
    // §8.6: question confidence is the MINIMUM of its criteria confidences.
    const confidences = aiCriteria.map((c) => c.confidence).filter((c): c is number => c !== null);
    const questionConfidence = confidences.length > 0 ? Math.min(...confidences) : null;

    await upsertMark(submissionId, aqRaw.id, {
      question_type: 'theory',
      ai_criteria: aiCriteria,
      ai_score: aiScore,
      ai_marks_available: marksAvailable,
      confidence: questionConfidence,
      status: 'needs_review',
    });
    totalScore += aiScore;
  }

  await supabase
    .from('submissions')
    .update({ total_score: totalScore, total_marks: totalMarks, updated_at: new Date().toISOString() })
    .eq('id', submissionId);
}

async function upsertMark(submissionId: string, assignmentQuestionId: string, fields: Record<string, unknown>) {
  const { error } = await supabase.from('submission_marks').upsert(
    {
      submission_id: submissionId,
      assignment_question_id: assignmentQuestionId,
      ...fields,
      updated_at: new Date().toISOString(),
    },
    { onConflict: 'submission_id,assignment_question_id' }
  );
  if (error) throw error;
}
