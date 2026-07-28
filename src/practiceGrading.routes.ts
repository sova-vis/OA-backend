import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import {
  grokEnabled, grokChatJson, grokVisionModel, grokTextModel, grokErrorMessage, GrokError, GrokImage,
} from './lib/grok';
import {
  PracticeProgressDoc, PracticeReport, GradedQuestion, SolveMode,
  ensureBucket, isValidPaperKey, readDoc, writeDoc, signUploads, downloadUploadImages,
} from './lib/practiceStore';

/**
 * AI marking for practice papers.
 *
 *   POST /practice-grading/grade   (Clerk auth)
 *
 * Marking-scheme-first: each question is graded against the scheme carried in
 * the request (it already ships with the question bank). MCQs are graded
 * deterministically (no AI needed). Written answers go to Grok's text model;
 * when a question has no scheme, Grok grades as an expert O/A-Level examiner.
 * Handwritten attempts are read + graded from the uploaded images via Grok
 * vision. The resulting report is saved onto the practice-progress document.
 */

interface GradePart {
  label: string;
  body: string;
  marks: number | null;
  answer: string | null;
}

interface GradeQuestion {
  id: string;
  questionNumber: string;
  type: 'mcq' | 'structured';
  questionText: string;
  maxMarks: number;
  correctOption?: string | null;
  markingScheme?: string | null;
  parts?: GradePart[];
  studentOption?: string | null;
  studentParts?: Record<string, string>;
  studentAnswer?: string | null;
}

const MAX_QUESTIONS = 60;
const WRITTEN_CONCURRENCY = 4;

const router = Router();

function clampMarks(value: unknown, fallback: number): number {
  const n = typeof value === 'number' ? value : Number.parseInt(String(value), 10);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.min(100, Math.round(n));
}

function verdictFromRatio(ratio: number): GradedQuestion['verdict'] {
  if (ratio >= 0.85) return 'correct';
  if (ratio >= 0.4) return 'partial';
  return 'weak';
}

function gradeBand(percent: number): string {
  if (percent >= 90) return 'A* (indicative)';
  if (percent >= 80) return 'A (indicative)';
  if (percent >= 70) return 'B (indicative)';
  if (percent >= 60) return 'C (indicative)';
  if (percent >= 50) return 'D (indicative)';
  if (percent >= 40) return 'E (indicative)';
  return 'U (indicative)';
}

function schemeText(question: GradeQuestion): string {
  const chunks: string[] = [];
  if (question.markingScheme && question.markingScheme.trim()) chunks.push(question.markingScheme.trim());
  for (const part of question.parts ?? []) {
    if (part.answer && part.answer.trim()) {
      chunks.push(`${part.label ? part.label + ' ' : ''}${part.answer.trim()}${part.marks != null ? ` [${part.marks}]` : ''}`);
    }
  }
  return chunks.join('\n');
}

function studentText(question: GradeQuestion): string {
  if (question.studentParts && Object.keys(question.studentParts).length > 0) {
    return Object.entries(question.studentParts)
      .map(([label, answer]) => `${label}: ${answer}`)
      .join('\n');
  }
  return (question.studentAnswer || '').trim();
}

function hasStudentWork(question: GradeQuestion): boolean {
  if (question.type === 'mcq') return Boolean((question.studentOption || '').trim());
  return studentText(question).length > 0;
}

function asStringArray(value: unknown, max = 6): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean).slice(0, max);
}

/** MCQ: deterministic against the correct option. */
function gradeMcq(question: GradeQuestion): GradedQuestion {
  const correct = (question.correctOption || '').trim().toUpperCase();
  const chosen = (question.studentOption || '').trim().toUpperCase();
  const max = clampMarks(question.maxMarks, 1);
  if (!chosen) {
    return {
      id: question.id, questionNumber: question.questionNumber, earned: 0, max,
      verdict: 'unanswered', feedback: 'Not answered.', expectedPoints: correct ? [`Correct option: ${correct}`] : [],
      missingPoints: [], gradingSource: 'deterministic',
    };
  }
  const isCorrect = Boolean(correct) && chosen === correct;
  return {
    id: question.id, questionNumber: question.questionNumber,
    earned: isCorrect ? max : 0, max,
    verdict: isCorrect ? 'correct' : 'weak',
    feedback: isCorrect
      ? 'Correct.'
      : correct ? `Incorrect — you chose ${chosen}, the correct option is ${correct}.` : `You chose ${chosen}.`,
    expectedPoints: correct ? [`Correct option: ${correct}`] : [],
    missingPoints: isCorrect ? [] : correct ? [`Review why ${correct} is correct.`] : [],
    gradingSource: 'deterministic',
    schemeUsed: Boolean(correct), // the correct option is the scheme
  };
}

/** Written: Grok grades against the scheme, or as an expert examiner if none. */
async function gradeWritten(subject: string, question: GradeQuestion): Promise<GradedQuestion> {
  const max = clampMarks(question.maxMarks, Math.max(1, (question.parts ?? []).reduce((s, p) => s + (p.marks ?? 0), 0) || 1));
  const scheme = schemeText(question);
  if (!hasStudentWork(question)) {
    return {
      id: question.id, questionNumber: question.questionNumber, earned: 0, max,
      verdict: 'unanswered', feedback: 'Not answered.', expectedPoints: [], missingPoints: [],
      gradingSource: 'grok', schemeUsed: scheme.length > 0,
    };
  }

  const system = [
    'You are a strict but fair Cambridge O/A Level examiner.',
    'Grade the student answer out of max_marks (award whole marks only).',
    'When a marking scheme is provided, mark strictly against it.',
    'When marking_scheme is empty, grade using your own expert subject knowledge and standard Cambridge mark-scheme conventions.',
    'Return JSON ONLY with keys: earned_marks (number), verdict ("correct"|"partial"|"weak"), feedback (string, one or two sentences, addressed to the student), expected_points (array of short strings), missing_points (array of short strings).',
  ].join(' ');
  const user = JSON.stringify({
    subject,
    question: question.questionText,
    max_marks: max,
    marking_scheme: scheme,
    student_answer: studentText(question),
  });

  const parsed = await grokChatJson({ system, user, temperature: 0, maxTokens: 900 });
  const earnedRaw = Number(parsed.earned_marks);
  const earned = Number.isFinite(earnedRaw) ? Math.max(0, Math.min(max, Math.round(earnedRaw))) : 0;
  const rawVerdict = String(parsed.verdict || '').toLowerCase();
  const verdict: GradedQuestion['verdict'] =
    rawVerdict === 'correct' || rawVerdict === 'partial' || rawVerdict === 'weak'
      ? rawVerdict
      : verdictFromRatio(max ? earned / max : 0);

  return {
    id: question.id, questionNumber: question.questionNumber, earned, max, verdict,
    feedback: typeof parsed.feedback === 'string' && parsed.feedback.trim() ? parsed.feedback.trim() : 'Graded.',
    expectedPoints: asStringArray(parsed.expected_points),
    missingPoints: asStringArray(parsed.missing_points),
    gradingSource: 'grok', schemeUsed: scheme.length > 0,
  };
}

async function mapPool<T, R>(items: T[], size: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let cursor = 0;
  const workers = Array.from({ length: Math.min(size, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      results[index] = await fn(items[index], index);
    }
  });
  await Promise.all(workers);
  return results;
}

/** Handwritten: one Grok vision pass reads + grades every question from images. */
async function gradeHandwritten(
  subject: string,
  questions: GradeQuestion[],
  images: GrokImage[],
): Promise<GradedQuestion[]> {
  const outline = questions.map((q) => ({
    question_number: q.questionNumber,
    max_marks: clampMarks(q.maxMarks, 1),
    question: q.questionText.slice(0, 1500),
    marking_scheme: schemeText(q).slice(0, 2000),
  }));
  const system = [
    'You are a strict but fair Cambridge O/A Level examiner grading a scanned handwritten attempt.',
    'The images contain the student\'s handwritten answers. Read the handwriting, match each answer to the question by its number, then grade it.',
    'Mark against each question\'s marking_scheme when present, otherwise use your own expert subject knowledge.',
    'Award whole marks only, never more than that question\'s max_marks. If a question is not attempted in the images, give 0 and verdict "unanswered".',
    'Return JSON ONLY: { "results": [ { "question_number": string, "earned_marks": number, "verdict": "correct"|"partial"|"weak"|"unanswered", "feedback": string, "expected_points": string[], "missing_points": string[] } ] }.',
  ].join(' ');
  const user = `Subject: ${subject}\nGrade these questions from the attached handwritten pages:\n${JSON.stringify(outline)}`;

  const parsed = await grokChatJson({ system, user, images, model: grokVisionModel(), temperature: 0, maxTokens: 3000 });
  const results = Array.isArray(parsed.results) ? (parsed.results as Array<Record<string, unknown>>) : [];
  const byNumber = new Map(results.map((r) => [String(r.question_number).trim(), r]));

  return questions.map((q) => {
    const max = clampMarks(q.maxMarks, 1);
    const usedScheme = schemeText(q).length > 0;
    const r = byNumber.get(String(q.questionNumber).trim());
    if (!r) {
      return {
        id: q.id, questionNumber: q.questionNumber, earned: 0, max, verdict: 'unanswered',
        feedback: 'No matching answer found in the uploaded pages.', expectedPoints: [], missingPoints: [],
        gradingSource: 'grok-vision', schemeUsed: usedScheme,
      };
    }
    const earnedRaw = Number(r.earned_marks);
    const earned = Number.isFinite(earnedRaw) ? Math.max(0, Math.min(max, Math.round(earnedRaw))) : 0;
    const rawVerdict = String(r.verdict || '').toLowerCase();
    const verdict: GradedQuestion['verdict'] =
      rawVerdict === 'correct' || rawVerdict === 'partial' || rawVerdict === 'weak' || rawVerdict === 'unanswered'
        ? (rawVerdict as GradedQuestion['verdict'])
        : verdictFromRatio(max ? earned / max : 0);
    return {
      id: q.id, questionNumber: q.questionNumber, earned, max, verdict,
      feedback: typeof r.feedback === 'string' && r.feedback.trim() ? r.feedback.trim() : 'Graded from your uploaded answer.',
      expectedPoints: asStringArray(r.expected_points),
      missingPoints: asStringArray(r.missing_points),
      gradingSource: 'grok-vision', schemeUsed: usedScheme,
    };
  });
}

function buildReport(graded: GradedQuestion[], solveMode: SolveMode, model: string): PracticeReport {
  const earned = graded.reduce((s, g) => s + g.earned, 0);
  const total = graded.reduce((s, g) => s + g.max, 0);
  const percent = total > 0 ? Math.round((earned / total) * 100) : 0;

  const strengths = graded
    .filter((g) => g.verdict === 'correct')
    .map((g) => `Q${g.questionNumber}`);
  const improvements = graded
    .filter((g) => g.verdict === 'weak' || g.verdict === 'unanswered')
    .flatMap((g) => g.missingPoints.length ? g.missingPoints.slice(0, 1).map((p) => `Q${g.questionNumber}: ${p}`) : [`Q${g.questionNumber}: revisit this question`]);

  const summary =
    percent >= 80 ? 'Excellent — a strong, exam-ready attempt across most of the paper.'
      : percent >= 60 ? 'Solid work. A few questions need tightening to push into the top band.'
      : percent >= 40 ? 'A fair attempt. Focus on the flagged questions to build accuracy.'
      : 'Keep going — review the marking points below and re-attempt the weaker questions.';

  return {
    earned, total, percent, grade: gradeBand(percent), summary,
    strengths: strengths.slice(0, 8),
    improvements: improvements.slice(0, 8),
    perQuestion: graded,
    solveMode, model,
    gradedAt: new Date().toISOString(),
  };
}

router.post('/grade', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const body = (req.body ?? {}) as {
      paperKey?: string; subject?: string; solveMode?: SolveMode; questions?: GradeQuestion[];
    };
    const paperKey = body.paperKey;
    if (!isValidPaperKey(paperKey)) return res.status(400).json({ error: 'Invalid paper key' });

    const subject = (body.subject || paperKey.split('|')[0] || '').trim();
    const solveMode: SolveMode = body.solveMode === 'handwritten' ? 'handwritten' : 'digital';
    const questions = Array.isArray(body.questions) ? body.questions.slice(0, MAX_QUESTIONS) : [];
    if (questions.length === 0) return res.status(400).json({ error: 'No questions to grade' });

    await ensureBucket();

    // Split into instant (MCQ) and AI (written) work.
    const needsAi = questions.some((q) => q.type !== 'mcq' || solveMode === 'handwritten');
    if (needsAi && !grokEnabled()) {
      return res.status(503).json({ error: grokErrorMessage(new GrokError('', 'no_key')), code: 'no_key' });
    }

    let graded: GradedQuestion[];
    let model = grokTextModel();

    if (solveMode === 'handwritten') {
      const doc = await readDoc(clerkId, paperKey);
      const images = await downloadUploadImages(doc?.uploads ?? []);
      if (images.length === 0) {
        return res.status(400).json({ error: 'Upload a photo or scan of your handwritten answers first.' });
      }
      model = grokVisionModel();
      graded = await gradeHandwritten(subject, questions, images);
    } else {
      graded = await mapPool(questions, WRITTEN_CONCURRENCY, async (q) =>
        q.type === 'mcq' ? gradeMcq(q) : gradeWritten(subject, q),
      );
    }

    const report = buildReport(graded, solveMode, model);

    // Persist the report onto the progress doc and mark the paper completed.
    const segments = paperKey.split('|');
    const existing = await readDoc(clerkId, paperKey);
    const now = new Date().toISOString();
    const doc: PracticeProgressDoc = existing
      ? { ...existing, report, status: 'completed', updatedAt: now }
      : {
          paperKey, subject, year: segments[1], session: segments[2], paper: segments[3], variant: segments[4],
          isMcq: questions.every((q) => q.type === 'mcq'), solveMode, status: 'completed',
          answers: { mcq: {}, parts: {} }, uploads: [], report,
          answeredCount: graded.filter((g) => g.verdict !== 'unanswered').length, totalCount: graded.length,
          timerDurationSeconds: 0, timerElapsedSeconds: 0, startedAt: now, updatedAt: now,
        };
    await writeDoc(clerkId, doc);

    return res.json({ report, item: { ...doc, uploads: await signUploads(doc.uploads ?? []) } });
  } catch (error) {
    console.error('Practice grading error:', error);
    const message = grokErrorMessage(error);
    const status = error instanceof GrokError && (error.code === 'no_key' || error.code === 'invalid_key') ? 503 : 500;
    return res.status(status).json({ error: message });
  }
});

/**
 * POST /practice-grading/grade-one  (Clerk auth)
 * Grade a single question against its own marking scheme (topic-drill mode).
 * Stateless — nothing is persisted.
 */
router.post('/grade-one', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const body = (req.body ?? {}) as { subject?: string; question?: GradeQuestion };
    const question = body.question;
    if (!question || !question.id) return res.status(400).json({ error: 'question is required' });
    const subject = (body.subject || '').trim();

    if (question.type === 'mcq') {
      return res.json({ result: gradeMcq(question) });
    }
    if (!grokEnabled()) {
      return res.status(503).json({ error: grokErrorMessage(new GrokError('', 'no_key')), code: 'no_key' });
    }
    const result = await gradeWritten(subject, question);
    return res.json({ result });
  } catch (error) {
    console.error('Single-question grading error:', error);
    const message = grokErrorMessage(error);
    const status = error instanceof GrokError && (error.code === 'no_key' || error.code === 'invalid_key') ? 503 : 500;
    return res.status(status).json({ error: message });
  }
});

router.get('/health', (_req, res: Response) => {
  res.json({ status: 'ok', grok_enabled: grokEnabled(), text_model: grokTextModel(), vision_model: grokVisionModel() });
});

export default router;
