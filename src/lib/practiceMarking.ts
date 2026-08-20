/**
 * The single marking engine for practice papers.
 *
 * Every graded mark in the practice feature comes from here, whether the student
 * typed their answers ("Solve here") or uploaded a photo/PDF of a handwritten
 * script. Uploads differ only in how the answer text is obtained: it is
 * transcribed first (lib/handwrittenExtraction) and then handed to the same
 * gradeMcq/gradeWritten functions below. There is deliberately no second
 * "score straight from the image" path, so a scanned answer and the same answer
 * typed in receive the same marks.
 */

import { grokChatJson, grokErrorMessage, isFatalGrokError } from './grok';
import {
  GradedQuestion, MarkCategory, PracticeReport, SolveMode, ExtractionSummary,
} from './practiceStore';
import {
  GradePart, GradeQuestion, ExtractedAnswer,
  extractHandwrittenAnswers, applyExtraction,
} from './handwrittenExtraction';
import { PageImage } from './pdfPages';

export const WRITTEN_CONCURRENCY = 2;
/** Reasoning models (grok-4 / grok-4.5) regularly need well over the 60s client default. */
export const GRADING_TIMEOUT_MS = 180_000;

export function clampMarks(value: unknown, fallback: number): number {
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

/** Total marks available for a question, from its own value or its parts. */
function maxMarksOf(question: GradeQuestion): number {
  return clampMarks(
    question.maxMarks,
    Math.max(1, (question.parts ?? []).reduce((sum, part) => sum + (part.marks ?? 0), 0) || 1),
  );
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

/** Examiner furniture like "three from:" is not a usable scheme — treat it as empty. */
function stripSchemeFurniture(text: string): string {
  return text
    .replace(/\b(any\s+)?(one|two|three|four|five|six|1|2|3|4|5|6)\s+from:?/gi, ' ')
    .replace(/\bworking space\b/gi, ' ')
    .replace(/\badditional guidance\b/gi, ' ')
    .replace(/\bmark scheme\b/gi, ' ')
    .replace(/\b(?:max|maximum)\s+\d+\s+from\b/gi, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function schemeIsUsable(scheme: string): boolean {
  return stripSchemeFurniture(scheme).replace(/[^a-z0-9]/gi, '').length >= 16;
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
export function gradeMcq(question: GradeQuestion): GradedQuestion {
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

const BREAKDOWN_INSTRUCTION =
  'Also split the marks by assessment objective in "breakdown": an array of { "category": "Knowledge"|"Explanation"|"Evaluation", "earned": number, "max": number }. ' +
  'Only include the objectives this question actually tests, and the earned/max across the breakdown should reconcile with the total marks.';

/** Normalize a model "breakdown" into clamped MarkCategory[] (or undefined). */
function parseBreakdown(value: unknown, totalMax: number): MarkCategory[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const cats: MarkCategory['category'][] = ['Knowledge', 'Explanation', 'Evaluation'];
  const out: MarkCategory[] = [];
  for (const raw of value) {
    if (!raw || typeof raw !== 'object') continue;
    const r = raw as Record<string, unknown>;
    const label = String(r.category ?? '').trim();
    const category = cats.find((c) => c.toLowerCase() === label.toLowerCase());
    if (!category) continue;
    const max = Math.max(0, Math.min(totalMax, Math.round(Number(r.max) || 0)));
    const earned = Math.max(0, Math.min(max, Math.round(Number(r.earned) || 0)));
    if (max <= 0) continue;
    out.push({ category, earned, max });
  }
  return out.length ? out : undefined;
}

const EXAMINER_INSTRUCTION =
  'Also add: "command_word" (the question\'s command word such as Describe/Explain/Evaluate/State/Calculate/Suggest, or "" if none); ' +
  '"command_word_note" (if the student answered in a style that does not match the command word — e.g. describing when asked to evaluate — one short sentence explaining the gap, otherwise ""); ' +
  '"examiner_note" (one sentence in the style of a Cambridge examiner report, e.g. "Candidates commonly lose marks here because…").';

function examinerFields(parsed: Record<string, unknown>): Pick<GradedQuestion, 'commandWord' | 'commandWordNote' | 'examinerNote'> {
  const s = (v: unknown, n: number) => { const t = String(v ?? '').trim(); return t ? t.slice(0, n) : undefined; };
  return {
    commandWord: s(parsed.command_word, 40),
    commandWordNote: s(parsed.command_word_note, 300),
    examinerNote: s(parsed.examiner_note, 400),
  };
}

const PART_SCORES_INSTRUCTION =
  'Also return "part_scores": an array of { "label": string, "earned": number, "max": number }, one entry per marked sub-part, using the SAME part labels as the marking scheme (e.g. "(a)", "(a)(ii)"). "earned" is the marks you awarded that sub-part; "max" is that sub-part\'s available marks; the sum of earned across parts must equal earned_marks. Return [] when the question has no separate sub-parts.';

const normPartLabel = (value: string) => value.trim().toLowerCase().replace(/\s+/g, '');

/** Normalize a model "part_scores" array into clamped, scheme-labelled scores. */
function parsePartScores(value: unknown, parts?: GradePart[]): GradedQuestion['partScores'] {
  if (!Array.isArray(value)) return undefined;
  const maxByLabel = new Map(
    (parts ?? []).map((p) => [normPartLabel(p.label || ''), p.marks ?? 0]),
  );
  const out: NonNullable<GradedQuestion['partScores']> = [];
  for (const raw of value) {
    if (!raw || typeof raw !== 'object') continue;
    const r = raw as Record<string, unknown>;
    const label = String(r.label ?? '').trim();
    if (!label) continue;
    const schemeMax = maxByLabel.get(normPartLabel(label));
    const max = Math.max(0, Math.round(Number(r.max) || schemeMax || 0));
    const cap = max > 0 ? max : 999;
    const earned = Math.max(0, Math.min(cap, Math.round(Number(r.earned) || 0)));
    out.push({ label, earned, max: max || earned });
  }
  return out.length ? out : undefined;
}

function isAuthConfigError(error: unknown): boolean {
  return isFatalGrokError(error);
}

function gradingUnavailable(question: GradeQuestion, error: unknown): GradedQuestion {
  return {
    id: question.id,
    questionNumber: question.questionNumber,
    earned: 0,
    max: maxMarksOf(question),
    verdict: 'unanswered',
    feedback: `${grokErrorMessage(error)} This question was left unmarked so the rest of the paper could still be scored. Try marking again.`,
    expectedPoints: [],
    missingPoints: [],
    gradingSource: 'grok',
    schemeUsed: schemeIsUsable(schemeText(question)),
    marksWithheld: true,
    gradingFailed: true,
  };
}

/** Written: Grok grades against the scheme, or as an expert examiner if none. */
export async function gradeWritten(subject: string, question: GradeQuestion): Promise<GradedQuestion> {
  const max = maxMarksOf(question);
  const scheme = schemeText(question);
  const usableScheme = schemeIsUsable(scheme);
  if (!hasStudentWork(question)) {
    return {
      id: question.id, questionNumber: question.questionNumber, earned: 0, max,
      verdict: 'unanswered', feedback: 'Not answered.', expectedPoints: [], missingPoints: [],
      gradingSource: 'grok', schemeUsed: usableScheme,
    };
  }

  const system = [
    'You are a strict but fair Cambridge O/A Level examiner.',
    'You MUST mark this question and return a grade. Never skip it, never return an error, never leave it unmarked.',
    'Grade every sub-part. Award whole marks only. earned_marks is the sum of marks awarded across all parts, capped at max_marks.',
    'When marking_scheme contains real marking points, mark strictly against those points. Credit equivalent wording and correct technical alternatives.',
    'When marking_scheme is empty or is only examiner furniture (e.g. "three from:", "Working Space", "1 mark"), ignore it and grade using your own expert Cambridge subject knowledge and standard mark-scheme conventions for this question.',
    'You MUST always fill expected_points with the credit-worthy model answer a Cambridge mark scheme would list (fill-in terms, definitions, or marking points). Never leave expected_points empty after marking a written answer — especially when you had to use examiner judgement because no scheme was on file.',
    'Do not award or deduct marks for anything the student did not write. Do not treat printed question-paper or mark-scheme text that leaked into student_answer as the student\'s own work — if student_answer looks like copied mark-scheme furniture rather than an attempt, award 0 and say so.',
    'Return JSON ONLY with keys: earned_marks (number), verdict ("correct"|"partial"|"weak"), feedback (string, one or two sentences, addressed to the student), expected_points (array of short strings), missing_points (array of short strings).',
    BREAKDOWN_INSTRUCTION,
    EXAMINER_INSTRUCTION,
    PART_SCORES_INSTRUCTION,
  ].join(' ');
  const user = JSON.stringify({
    subject,
    question: question.questionText,
    max_marks: max,
    marking_scheme: usableScheme ? scheme.slice(0, 4000) : '',
    scheme_available: usableScheme,
    parts: (question.parts ?? []).map((p) => ({
      label: p.label,
      question: p.body || undefined,
      marks: p.marks,
      mark_scheme: p.answer && stripSchemeFurniture(p.answer).replace(/[^a-z0-9]/gi, '').length >= 12
        ? p.answer
        : undefined,
    })),
    student_answer: studentText(question),
  });

  try {
    const parsed = await grokChatJson({ system, user, temperature: 0, maxTokens: 2000, timeoutMs: GRADING_TIMEOUT_MS });
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
      expectedPoints: asStringArray(parsed.expected_points, 12),
      missingPoints: asStringArray(parsed.missing_points, 8),
      gradingSource: 'grok', schemeUsed: usableScheme,
      breakdown: parseBreakdown(parsed.breakdown, max),
      partScores: parsePartScores(parsed.part_scores, question.parts),
      ...examinerFields(parsed),
    };
  } catch (error) {
    if (isAuthConfigError(error)) throw error;
    console.warn(`Grading Q${question.questionNumber} failed:`, error);
    return gradingUnavailable(question, error);
  }
}

export async function mapPool<T, R>(items: T[], size: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let cursor = 0;
  const workers = Array.from({ length: Math.max(1, Math.min(size, items.length)) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      results[index] = await fn(items[index], index);
    }
  });
  await Promise.all(workers);
  return results;
}

/** Grade one question the way "Solve here" does — the shared entry point. */
export async function gradeOne(subject: string, question: GradeQuestion): Promise<GradedQuestion> {
  return question.type === 'mcq' ? gradeMcq(question) : gradeWritten(subject, question);
}

function isApiGradingFailure(graded: GradedQuestion): boolean {
  return graded.gradingFailed === true || (
    graded.marksWithheld === true
    && graded.extractionFlag !== 'unreadable'
    && graded.extractionFlag !== 'not_found'
  );
}

function mergeRetry(previous: GradedQuestion, next: GradedQuestion): GradedQuestion {
  return {
    ...next,
    extractedAnswer: next.extractedAnswer ?? previous.extractedAnswer,
    extractionConfidence: next.extractionConfidence ?? previous.extractionConfidence,
    extractionFlag: next.extractionFlag ?? previous.extractionFlag,
    extractionPages: next.extractionPages ?? previous.extractionPages,
    extractionNote: next.extractionNote ?? previous.extractionNote,
  };
}

/** Sequential second pass for questions the first wave left unmarked due to API errors. */
async function retryFailedGrading(
  subject: string,
  questions: GradeQuestion[],
  graded: GradedQuestion[],
  retry: (question: GradeQuestion, index: number) => Promise<GradedQuestion>,
): Promise<GradedQuestion[]> {
  const out = [...graded];
  for (let i = 0; i < out.length; i++) {
    if (!isApiGradingFailure(out[i])) continue;
    await new Promise((resolve) => setTimeout(resolve, 800));
    try {
      const again = await retry(questions[i], i);
      if (!isApiGradingFailure(again)) out[i] = mergeRetry(out[i], again);
    } catch (error) {
      if (isAuthConfigError(error)) throw error;
      console.warn(`Retry grading Q${questions[i].questionNumber} failed:`, error);
    }
  }
  return out;
}

/**
 * OCR'd mark schemes sometimes glue the next question onto the previous part
 * (e.g. Q6(c) ends with "2 7 One mark for each correct term… public key • private key 4").
 * If the following question has no scheme of its own, reclaim that tail so it is
 * marked against the official points instead of empty examiner-judgement.
 */
export function reclaimLeakedSchemes(questions: GradeQuestion[]): GradeQuestion[] {
  const out = questions.map((question) => ({
    ...question,
    parts: (question.parts ?? []).map((part) => ({ ...part })),
  }));

  for (let i = 0; i < out.length; i++) {
    const next = out[i];
    if (schemeIsUsable(schemeText(next))) continue;
    const num = String(next.questionNumber || '').trim();
    if (!num || !/^\d+[a-z]*$/i.test(num)) continue;
    const escaped = num.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const leakedRe = new RegExp(
      `(?:^|[\\s\\n])${escaped}\\s+((?:One mark|Award|Any\\b|•).+)$`,
      'is',
    );

    for (let j = 0; j < i; j++) {
      for (const part of out[j].parts ?? []) {
        const raw = part.answer || '';
        const match = raw.match(leakedRe);
        if (!match?.[1] || match[1].replace(/[^a-z0-9]/gi, '').length < 16) continue;
        const cutAt = match.index ?? -1;
        if (cutAt < 0) continue;
        part.answer = raw.slice(0, cutAt).trim();
        const scheme = match[1].trim();
        next.markingScheme = scheme;
        const marksMatch = scheme.match(/(\d+)\s*(?:\d{4}\s*\/\s*\d{2})?\s*$/);
        const recoveredMarks = marksMatch ? Number(marksMatch[1]) : 0;
        if (recoveredMarks > 0 && recoveredMarks <= 25 && (!next.maxMarks || next.maxMarks <= 0)) {
          next.maxMarks = recoveredMarks;
        }
        if (!next.parts?.length) {
          next.parts = [{
            label: '',
            body: next.questionText,
            marks: next.maxMarks || recoveredMarks || null,
            answer: scheme,
          }];
        } else {
          const empty = next.parts.find((p) => !(p.answer || '').trim());
          if (empty) empty.answer = scheme;
        }
        break;
      }
      if (schemeIsUsable(schemeText(next))) break;
    }
  }
  return out;
}

/** Grade a typed attempt: every question, straight through the shared scorer. */
export async function gradeTyped(subject: string, questions: GradeQuestion[]): Promise<GradedQuestion[]> {
  const prepared = reclaimLeakedSchemes(questions);
  const graded = await mapPool(prepared, WRITTEN_CONCURRENCY, (question) => gradeOne(subject, question));
  return retryFailedGrading(subject, prepared, graded, (question) => gradeOne(subject, question));
}

/* ============================================================
   Handwritten: read the pages, then grade with the functions above.
   No image ever reaches a scoring prompt.
   ============================================================ */

const NO_ANSWER: ExtractedAnswer = {
  flag: 'not_found', confidence: 0, pages: [], note: '', text: '', studentParts: {}, studentOption: null,
};

/** A question whose answer could not be read is reported, never scored. */
function withheldResult(question: GradeQuestion, extracted: ExtractedAnswer): GradedQuestion {
  return {
    id: question.id, questionNumber: question.questionNumber, earned: 0, max: maxMarksOf(question),
    verdict: 'unanswered',
    feedback: extracted.note
      ? `We could not read your handwriting for this question (${extracted.note}), so it has not been marked. Re-upload a clearer photo of this page to have it marked.`
      : 'We could not read your handwriting for this question, so it has not been marked. Re-upload a clearer photo of this page to have it marked.',
    expectedPoints: [], missingPoints: [],
    gradingSource: 'grok-vision',
    schemeUsed: schemeIsUsable(schemeText(question)),
    marksWithheld: true,
  };
}

function notFoundResult(question: GradeQuestion): GradedQuestion {
  return {
    id: question.id, questionNumber: question.questionNumber, earned: 0, max: maxMarksOf(question),
    verdict: 'unanswered',
    feedback: 'This question was not found on the pages you uploaded. If you answered it, upload the missing page.',
    expectedPoints: [], missingPoints: [],
    gradingSource: 'grok-vision',
    schemeUsed: schemeIsUsable(schemeText(question)),
    marksWithheld: true,
  };
}

/** Attach how the answer was read, so the student can audit every mark. */
export function withExtractionMeta(graded: GradedQuestion, extracted: ExtractedAnswer): GradedQuestion {
  const extractedParts = Object.keys(extracted.studentParts).length ? extracted.studentParts : undefined;
  return {
    ...graded,
    extractedAnswer: extracted.text || undefined,
    extractionConfidence: Math.round(extracted.confidence * 100) / 100,
    extractionFlag: extracted.flag,
    extractionPages: extracted.pages.length ? extracted.pages : undefined,
    extractionNote: extracted.note || undefined,
    extractedParts,
    extractedOption: extracted.studentOption || undefined,
  };
}

const slotKey = (value: string) => value.trim().toLowerCase().replace(/[^a-z0-9]/g, '');

/**
 * Map a graded handwritten attempt onto the same answer-slot keys the typed
 * "Solve here" flow uses (`${questionId}::${partIndex}` / mcq[questionId]) so
 * the report UI can show transcribed answers in the original boxes.
 */
export function typedAnswersFromGraded(
  questions: GradeQuestion[],
  graded: GradedQuestion[],
): { mcq: Record<string, string>; parts: Record<string, string> } {
  const byId = new Map(graded.map((item) => [item.id, item]));
  const mcq: Record<string, string> = {};
  const parts: Record<string, string> = {};

  for (const question of questions) {
    const result = byId.get(question.id);
    if (!result) continue;

    if (question.type === 'mcq') {
      const option = (result.extractedOption || '').trim();
      if (option) mcq[question.id] = option;
      continue;
    }

    const extracted = result.extractedParts ?? {};
    const qParts = question.parts ?? [];
    const byNorm = new Map(
      Object.entries(extracted).map(([label, text]) => [slotKey(label), text] as const),
    );
    let wrote = false;
    qParts.forEach((part, index) => {
      const text =
        (part.label && extracted[part.label])
        || byNorm.get(slotKey(part.label || ''))
        || '';
      if (text.trim()) {
        parts[`${question.id}::${index}`] = text.trim();
        wrote = true;
      }
    });
    if (!wrote) {
      const blob = (result.extractedAnswer || '').trim()
        || Object.entries(extracted).map(([label, text]) => `${label}: ${text}`).join('\n').trim();
      if (blob) parts[`${question.id}::0`] = blob;
    }
  }

  return { mcq, parts };
}

/** Grade an already-read answer. Illegible and missing answers are never scored. */
export async function gradeExtracted(
  subject: string,
  question: GradeQuestion,
  extracted: ExtractedAnswer | undefined,
): Promise<GradedQuestion> {
  if (!extracted) return withExtractionMeta(notFoundResult(question), NO_ANSWER);
  if (extracted.flag === 'not_found') return withExtractionMeta(notFoundResult(question), extracted);
  // Illegible: flag it rather than guessing at what it might have said.
  if (extracted.flag === 'unreadable') return withExtractionMeta(withheldResult(question, extracted), extracted);

  // Everything else — including a genuinely blank answer — goes through the same
  // scorer the typed flow calls. gradeMcq/gradeWritten already return verdict
  // "unanswered" with 0 marks for an empty answer, so a blank question is marked
  // unanswered rather than wrong.
  return withExtractionMeta(await gradeOne(subject, applyExtraction(question, extracted)), extracted);
}

export async function gradeHandwritten(
  subject: string,
  questions: GradeQuestion[],
  pages: PageImage[],
  isMcqPaper: boolean,
): Promise<{ graded: GradedQuestion[]; extraction: ExtractionSummary }> {
  const prepared = reclaimLeakedSchemes(questions);
  const read = await extractHandwrittenAnswers(prepared, pages, { isMcqPaper, subject });
  const gradedFirst = await mapPool(prepared, WRITTEN_CONCURRENCY, (question) =>
    gradeExtracted(subject, question, read.byQuestionId.get(question.id)),
  );
  const graded = await retryFailedGrading(subject, prepared, gradedFirst, (question) =>
    gradeExtracted(subject, question, read.byQuestionId.get(question.id)),
  );

  const flagCount = (flag: ExtractedAnswer['flag']) => graded.filter((g) => g.extractionFlag === flag).length;
  const ocrWithheld = graded.filter((g) =>
    g.marksWithheld && (g.extractionFlag === 'unreadable' || g.extractionFlag === 'not_found'),
  );

  return {
    graded,
    extraction: {
      pageCount: read.pageCount,
      readCount: flagCount('ok'),
      lowConfidenceCount: flagCount('low_confidence'),
      unreadableCount: flagCount('unreadable'),
      blankCount: flagCount('blank'),
      notFoundCount: flagCount('not_found'),
      withheldMarks: ocrWithheld.reduce((sum, g) => sum + g.max, 0),
      warnings: read.warnings,
      paperMismatch: read.paperMismatch,
      visionModel: read.visionModel,
    },
  };
}

export function buildReport(
  graded: GradedQuestion[],
  solveMode: SolveMode,
  model: string,
  extraction?: ExtractionSummary,
): PracticeReport {
  // Questions whose answer could not be read are excluded from the score
  // entirely. Counting them as 0/max would report an unreadable page as if the
  // student had got it wrong.
  const assessed = graded.filter((g) => !g.marksWithheld);
  const earned = assessed.reduce((s, g) => s + g.earned, 0);
  const total = assessed.reduce((s, g) => s + g.max, 0);
  const percent = total > 0 ? Math.round((earned / total) * 100) : 0;

  const strengths = graded.filter((g) => g.verdict === 'correct').map((g) => `Q${g.questionNumber}`);
  const improvements = graded
    .filter((g) => !g.marksWithheld && (g.verdict === 'weak' || g.verdict === 'unanswered'))
    .flatMap((g) => g.missingPoints.length
      ? g.missingPoints.slice(0, 1).map((p) => `Q${g.questionNumber}: ${p}`)
      : [`Q${g.questionNumber}: revisit this question`]);

  const withheld = graded.filter((g) => g.marksWithheld);
  const unread = withheld.filter((g) => g.extractionFlag === 'unreadable' || g.extractionFlag === 'not_found');
  const failed = withheld.filter((g) => g.gradingFailed || (g.extractionFlag !== 'unreadable' && g.extractionFlag !== 'not_found'));
  if (unread.length > 0) {
    improvements.unshift(
      `${unread.length} question${unread.length === 1 ? '' : 's'} could not be read and ${unread.length === 1 ? 'was' : 'were'} left unmarked (${unread.map((g) => `Q${g.questionNumber}`).join(', ')}) — re-upload those pages more clearly.`,
    );
  }
  if (failed.length > 0) {
    improvements.unshift(
      `${failed.length} question${failed.length === 1 ? '' : 's'} could not be marked this time (${failed.map((g) => `Q${g.questionNumber}`).join(', ')}) — tap Mark again to retry just those.`,
    );
  }

  const base =
    total === 0 ? 'We could not mark any of this attempt — see the notes below and re-upload clearer pages.'
      : percent >= 80 ? 'Excellent — a strong, exam-ready attempt across most of the paper.'
      : percent >= 60 ? 'Solid work. A few questions need tightening to push into the top band.'
      : percent >= 40 ? 'A fair attempt. Focus on the flagged questions to build accuracy.'
      : 'Keep going — review the marking points below and re-attempt the weaker questions.';
  const caveat =
    withheld.length > 0 && total > 0
      ? ` This score covers the ${assessed.length} question${assessed.length === 1 ? '' : 's'} we could read; ${withheld.reduce((s, g) => s + g.max, 0)} marks were not assessed.`
      : '';

  return {
    earned, total, percent, grade: gradeBand(percent), summary: base + caveat,
    strengths: strengths.slice(0, 8),
    improvements: improvements.slice(0, 8),
    perQuestion: graded,
    solveMode, model,
    gradedAt: new Date().toISOString(),
    ...(extraction ? { extraction } : {}),
  };
}
