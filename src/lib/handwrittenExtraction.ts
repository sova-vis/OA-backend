/**
 * Reading handwritten answers off uploaded pages.
 *
 * This module ONLY transcribes. It deliberately awards no marks: the extracted
 * text is handed back as the same `studentParts` / `studentOption` payload that
 * the "Solve here" flow builds from typed input, so an uploaded attempt and a
 * typed attempt go through one identical grading path.
 *
 * Pages are transcribed one vision call at a time rather than all-at-once. That
 * keeps each call's output well inside the token budget (a single call covering
 * 40 questions truncates and takes the whole submission down with it), lets
 * pages run concurrently, and gives every fragment a page number — which is what
 * makes answers that run across pages or appear out of order reassemblable.
 */

import { GrokImage, grokChatJson, grokVisionModel } from './grok';
import { PageImage } from './pdfPages';
import { ExtractionFlag } from './practiceStore';

/* ---------------- question shapes (shared with the grading route) ---------------- */

export interface GradePart {
  label: string;
  body: string;
  marks: number | null;
  answer: string | null;
}

export interface GradeQuestion {
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

/* ---------------- tuning ---------------- */

function envFloat(name: string, fallback: number): number {
  const parsed = Number.parseFloat(process.env[name] || '');
  return Number.isFinite(parsed) && parsed >= 0 && parsed <= 1 ? parsed : fallback;
}

/** At or above this the read is trusted and graded without a caveat. */
const CONFIDENCE_OK = envFloat('PRACTICE_OCR_CONFIDENCE_OK', 0.55);
/** Below this the handwriting is treated as illegible: not graded, marks withheld. */
const CONFIDENCE_FLOOR = envFloat('PRACTICE_OCR_CONFIDENCE_FLOOR', 0.3);

const PAGE_CONCURRENCY = 2;
const PAGE_TIMEOUT_MS = 120_000;
const PAGE_MAX_TOKENS = 8192;

/* ---------------- per-page transcription ---------------- */

interface RawFragment {
  questionNumber: string;
  partLabel: string;
  text: string;
  selectedOption: string;
  legible: boolean;
  confidence: number;
  note: string;
  page: number;
}

interface PageResult {
  page: number;
  quality: 'good' | 'fair' | 'poor';
  headerText: string;
  fragments: RawFragment[];
  /** set when the page could not be transcribed at all */
  error?: string;
}

const EXTRACTION_SYSTEM = [
  'You transcribe handwritten exam answers from a scanned page. You are an OCR engine, NOT an examiner.',
  'Never grade, correct, complete, improve or comment on the content. Transcribe only what is physically written in handwriting.',
  'CRITICAL: transcribe HANDWRITTEN student work only. Ignore printed question-paper text, printed mark schemes, headers, footers, "Working Space", "Answers", "three from", tick boxes, and any typed/printed table that is part of the question.',
  'CRITICAL: never invent an answer for a question that does not appear on this page. If a question number is not written on this page, omit it entirely.',
  'If a question number is written but nothing is written after it, return it with text "" and legible true (the student left it blank).',
  'If something is written but you genuinely cannot make out the words, return your best partial reading, set legible to false, and set a low confidence. Do NOT guess at plausible exam content to fill the gap.',
  'confidence is your honest 0-1 certainty that your transcription matches the ink on the page. Use a value below 0.3 when the writing is mostly unreadable.',
  'Match each answer to the question number the student wrote next to it. Use part_label for sub-parts, e.g. "(a)", "(b)(ii)". Use "" when the answer has no sub-part label.',
  'For a multiple-choice answer sheet, put the chosen letter in selected_option and leave text "".',
  'Transcribe mathematical working, tables the student filled in, and chemical formulae as written, in plain text.',
  'Return JSON ONLY: { "page_quality": "good"|"fair"|"poor", "header_text": string, "fragments": [ { "question_number": string, "part_label": string, "text": string, "selected_option": string, "legible": boolean, "confidence": number, "note": string } ] }',
  'header_text is any printed paper title, subject name or paper code visible on the page (used to detect a wrong upload); "" if none.',
  'page_quality describes the scan itself (focus, lighting, skew), not the handwriting.',
].join(' ');

function asFragment(raw: Record<string, unknown>, page: number): RawFragment | null {
  const questionNumber = String(raw.question_number ?? '').trim();
  if (!questionNumber) return null;
  const rawConfidence = Number(raw.confidence);
  const confidence = Number.isFinite(rawConfidence) ? Math.max(0, Math.min(1, rawConfidence)) : 0.5;
  return {
    questionNumber: questionNumber.slice(0, 20),
    partLabel: String(raw.part_label ?? '').trim().slice(0, 20),
    text: String(raw.text ?? '').trim().slice(0, 8000),
    selectedOption: String(raw.selected_option ?? '').trim().slice(0, 4).toUpperCase(),
    legible: raw.legible !== false,
    confidence,
    note: String(raw.note ?? '').trim().slice(0, 200),
    page,
  };
}

/** Numbers/labels the student may have written, so the model snaps to them. */
function labelVocabulary(questions: GradeQuestion[]): string {
  return JSON.stringify(
    questions.map((q) => ({
      question_number: q.questionNumber,
      part_labels: (q.parts ?? []).map((p) => p.label).filter(Boolean),
    })),
  );
}

async function transcribePage(page: PageImage, questions: GradeQuestion[], isMcqPaper: boolean): Promise<PageResult> {
  const images: GrokImage[] = [{ base64: page.base64, mimeType: page.mimeType }];
  const user = [
    `This is page ${page.page} of a student's handwritten answer sheet.`,
    isMcqPaper
      ? 'It is a multiple-choice answer sheet: report the option letter the student marked for each question number.'
      : 'Transcribe every answer written on this page.',
    'The paper being attempted has these question numbers and sub-part labels. Use them to normalise the labels you report,',
    'but ONLY report questions actually written on this page:',
    labelVocabulary(questions),
  ].join('\n');

  try {
    const parsed = await grokChatJson({
      system: EXTRACTION_SYSTEM,
      user,
      images,
      model: grokVisionModel(),
      temperature: 0,
      maxTokens: PAGE_MAX_TOKENS,
      timeoutMs: PAGE_TIMEOUT_MS,
    });
    const rawFragments = Array.isArray(parsed.fragments) ? (parsed.fragments as Array<Record<string, unknown>>) : [];
    const quality = String(parsed.page_quality ?? '').toLowerCase();
    return {
      page: page.page,
      quality: quality === 'poor' || quality === 'fair' ? quality : 'good',
      headerText: String(parsed.header_text ?? '').trim().slice(0, 300),
      fragments: rawFragments
        .map((raw) => asFragment(raw, page.page))
        .filter((fragment): fragment is RawFragment => fragment !== null),
    };
  } catch (error) {
    // One unreadable page must not sink the whole submission — record it and
    // carry on, so the questions on the other pages are still marked.
    return {
      page: page.page,
      quality: 'poor',
      headerText: '',
      fragments: [],
      error: error instanceof Error ? error.message : 'page could not be read',
    };
  }
}

async function mapPool<T, R>(items: T[], size: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let cursor = 0;
  const workers = Array.from({ length: Math.max(1, Math.min(size, items.length)) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      results[index] = await fn(items[index]);
    }
  });
  await Promise.all(workers);
  return results;
}

/* ---------------- merging fragments into per-question answers ---------------- */

/** "1 (a)(ii)" / "Q1a ii" / "1.a.ii" all normalise to the same key. */
function normNumber(value: string): string {
  return value.toLowerCase().replace(/^q(uestion)?\s*/, '').replace(/[^a-z0-9]/g, '');
}

function normLabel(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '');
}

export interface ExtractedAnswer {
  flag: ExtractionFlag;
  confidence: number;
  pages: number[];
  note: string;
  /** verbatim reading, for the student to check against their script */
  text: string;
  /** keyed by the paper's own part labels where they could be matched */
  studentParts: Record<string, string>;
  studentOption: string | null;
}

export interface ExtractionOutcome {
  byQuestionId: Map<string, ExtractedAnswer>;
  pageCount: number;
  warnings: string[];
  paperMismatch: boolean;
  visionModel: string;
}

/**
 * A fragment belongs to a question when its written number matches the paper's
 * number, or begins with it followed by a part label (the student wrote "3a"
 * where the paper calls it Q3 part (a)).
 */
function fragmentsForQuestion(question: GradeQuestion, fragments: RawFragment[]): RawFragment[] {
  const target = normNumber(question.questionNumber);
  if (!target) return [];
  return fragments.filter((fragment) => {
    const written = normNumber(fragment.questionNumber);
    if (written === target) return true;
    // "3a" -> Q3 (a); guard against Q1 swallowing Q10 by requiring the
    // remainder to be a part label (letters/roman numerals), not more digits.
    if (written.startsWith(target)) {
      const rest = written.slice(target.length);
      return rest.length > 0 && /^[a-z]+$/.test(rest);
    }
    return false;
  });
}

/** Part label the student wrote, recovered from either field. */
function fragmentLabel(question: GradeQuestion, fragment: RawFragment): string {
  if (fragment.partLabel) return fragment.partLabel;
  const target = normNumber(question.questionNumber);
  const written = normNumber(fragment.questionNumber);
  if (written.startsWith(target) && written.length > target.length) {
    return `(${written.slice(target.length)})`;
  }
  return '';
}

/**
 * Fold a question's fragments into one answer, keyed to the marking scheme's own
 * part labels so `studentParts` is shape-identical to what the typed flow sends.
 */
function buildAnswer(question: GradeQuestion, fragments: RawFragment[], sawAnyQuestionNumber: boolean): ExtractedAnswer {
  if (fragments.length === 0) {
    return {
      // Nothing written for this question. If the upload clearly belongs to this
      // paper the student skipped it; otherwise the page simply isn't here.
      flag: sawAnyQuestionNumber ? 'blank' : 'not_found',
      confidence: 1, pages: [], note: '', text: '', studentParts: {}, studentOption: null,
    };
  }

  const ordered = [...fragments].sort((a, b) => a.page - b.page);
  const pages = Array.from(new Set(ordered.map((f) => f.page))).sort((a, b) => a - b);
  const notes = Array.from(new Set(ordered.map((f) => f.note).filter(Boolean)));

  // Conservative: a question is only as trustworthy as its worst-read fragment.
  const confidence = ordered.reduce((min, f) => Math.min(min, f.confidence), 1);
  const anyIllegible = ordered.some((f) => !f.legible);
  const withText = ordered.filter((f) => f.text.length > 0 || f.selectedOption.length > 0);

  const option = ordered.find((f) => f.selectedOption)?.selectedOption ?? null;

  // Match the student's labels onto the scheme's labels; unmatched labels are
  // kept verbatim so nothing the student wrote is thrown away.
  const schemeLabels = new Map(
    (question.parts ?? []).map((part) => [normLabel(part.label || ''), part.label] as const),
  );
  const studentParts: Record<string, string> = {};
  for (const fragment of ordered) {
    if (!fragment.text) continue;
    const written = fragmentLabel(question, fragment);
    const key = schemeLabels.get(normLabel(written)) ?? (written || 'Answer');
    // Answers running across pages arrive as several fragments for one label.
    studentParts[key] = studentParts[key] ? `${studentParts[key]}\n${fragment.text}` : fragment.text;
  }

  // Only fragments with real content contribute. Composing from every fragment
  // would turn a question the student numbered and then left blank into the
  // text "(a)", which the marker would then score as a wrong answer.
  const text = withText
    .map((fragment) => {
      const label = fragmentLabel(question, fragment);
      const body = fragment.selectedOption ? `Selected: ${fragment.selectedOption}` : fragment.text;
      return label ? `${label} ${body}`.trim() : body;
    })
    .filter(Boolean)
    .join('\n');

  let flag: ExtractionFlag;
  if (withText.length === 0) {
    flag = 'blank';
  } else if (anyIllegible || confidence < CONFIDENCE_FLOOR) {
    flag = 'unreadable';
  } else if (confidence < CONFIDENCE_OK) {
    flag = 'low_confidence';
  } else {
    flag = 'ok';
  }

  return {
    flag,
    confidence,
    pages,
    note: notes.join('; ').slice(0, 300),
    text,
    studentParts: flag === 'unreadable' ? {} : studentParts,
    studentOption: flag === 'unreadable' ? null : option,
  };
}

/* ---------------- wrong-paper detection ---------------- */

/** Distinctive subject names, longest first so "additional mathematics" wins over "mathematics". */
const SUBJECT_HINTS = [
  'additional mathematics', 'english language', 'pakistan studies', 'business studies',
  'computer science', 'environmental management', 'art and design',
  'islamiyat', 'islamiat', 'chemistry', 'biology', 'physics', 'mathematics',
  'economics', 'accounting', 'commerce', 'history', 'geography', 'sociology',
];

function normSubject(value: string): string {
  return value.toLowerCase().replace(/[^a-z]+/g, ' ').trim();
}

/** Subject named in a printed page header, or null if none is recognisable. */
export function subjectFromHeader(headers: string[]): string | null {
  const blob = headers.join(' ').toLowerCase();
  if (!blob.trim()) return null;
  return SUBJECT_HINTS.find((name) => blob.includes(name)) ?? null;
}

/**
 * True when the printed header names a different subject than the paper the
 * student selected. Catches the case two papers share Q1–Q8 numbering so the
 * question-number check alone would miss the swap.
 */
export function headerConflictsWithSubject(headers: string[], subject: string): boolean {
  const hinted = subjectFromHeader(headers);
  if (!hinted || !subject.trim()) return false;
  const selected = normSubject(subject);
  const hintedNorm = normSubject(hinted);
  if (selected === hintedNorm) return false;
  if (selected.includes(hintedNorm) || hintedNorm.includes(selected)) return false;
  return true;
}

function detectMismatch(
  questions: GradeQuestion[],
  seen: Set<string>,
  sawAnyText: boolean,
): { mismatch: boolean; matched: number } {
  const paperNumbers = new Set(questions.map((q) => normNumber(q.questionNumber)).filter(Boolean));
  let matched = 0;
  for (const number of paperNumbers) if (seen.has(number)) matched += 1;

  if (!sawAnyText) return { mismatch: false, matched };
  // Nothing on the pages lines up with this paper at all.
  if (matched === 0) return { mismatch: true, matched };

  // Mostly foreign question numbers with barely any overlap: a partial upload of
  // one page is legitimately low-overlap, but its numbers all belong to the
  // paper, so the foreign ratio is what separates the two cases.
  const foreign = Array.from(seen).filter((number) => !paperNumbers.has(number)).length;
  const foreignRatio = seen.size > 0 ? foreign / seen.size : 0;
  const matchRatio = paperNumbers.size > 0 ? matched / paperNumbers.size : 0;
  return { mismatch: foreignRatio >= 0.7 && matchRatio < 0.2, matched };
}

/* ---------------- entry point ---------------- */

/**
 * Read every question's answer off the uploaded pages.
 * Awards no marks — see the module header.
 */
export async function extractHandwrittenAnswers(
  questions: GradeQuestion[],
  pages: PageImage[],
  options?: {
    isMcqPaper?: boolean;
    /** Selected paper's subject — used to spot a header that names a different one. */
    subject?: string;
    /**
     * Topic drill: the upload is known to answer the one question asked, so
     * every fragment belongs to it even when the student wrote no question
     * number. Never use this for a full paper, where the number is what maps an
     * answer to its question.
     */
    singleQuestion?: boolean;
  },
): Promise<ExtractionOutcome> {
  const warnings: string[] = [];
  const byQuestionId = new Map<string, ExtractedAnswer>();

  if (pages.length === 0) {
    for (const question of questions) {
      byQuestionId.set(question.id, {
        flag: 'not_found', confidence: 0, pages: [], note: '', text: '', studentParts: {}, studentOption: null,
      });
    }
    return { byQuestionId, pageCount: 0, warnings, paperMismatch: false, visionModel: grokVisionModel() };
  }

  const isMcqPaper = options?.isMcqPaper ?? questions.every((q) => q.type === 'mcq');
  const results = await mapPool(pages, PAGE_CONCURRENCY, (page) => transcribePage(page, questions, isMcqPaper));

  for (const result of results) {
    if (result.error) warnings.push(`Page ${result.page} could not be read (${result.error}).`);
  }
  const poorPages = results.filter((r) => !r.error && r.quality === 'poor').map((r) => r.page);
  if (poorPages.length > 0) {
    warnings.push(
      `Image quality is poor on page${poorPages.length > 1 ? 's' : ''} ${poorPages.join(', ')} — a sharper, well-lit scan will read more reliably.`,
    );
  }

  const allFragments = results.flatMap((result) => result.fragments);
  const seen = new Set(allFragments.map((fragment) => normNumber(fragment.questionNumber)).filter(Boolean));
  const sawAnyText = allFragments.some((f) => f.text.length > 0 || f.selectedOption.length > 0);
  const headers = results.map((result) => result.headerText).filter(Boolean);

  const single = options?.singleQuestion === true && questions.length === 1;
  const numberMismatch = single ? false : detectMismatch(questions, seen, sawAnyText).mismatch;
  const headerMismatch = !single && headerConflictsWithSubject(headers, options?.subject || '');
  const mismatch = numberMismatch || headerMismatch;
  if (numberMismatch) {
    warnings.push(
      'The question numbers on these pages do not match the paper you selected. Check you picked the right paper before trusting these marks.',
    );
  } else if (headerMismatch) {
    const hinted = subjectFromHeader(headers);
    warnings.push(
      `These pages look like a ${hinted} paper, but you selected ${options?.subject}. Check you picked the right paper before trusting these marks.`,
    );
  }

  // `blank` vs `not_found` hinges on whether this upload is plausibly this paper.
  const uploadBelongsToPaper = sawAnyText && !mismatch;
  for (const question of questions) {
    const matched = fragmentsForQuestion(question, allFragments);
    // A one-question upload answers that question whether or not it was numbered.
    const mine = single && matched.length === 0 ? allFragments : matched;
    byQuestionId.set(question.id, buildAnswer(question, mine, single || uploadBelongsToPaper));
  }

  return { byQuestionId, pageCount: pages.length, warnings, paperMismatch: mismatch, visionModel: grokVisionModel() };
}

/**
 * Merge an extracted answer into a question, producing exactly the payload the
 * typed "Solve here" flow produces. The result is graded by the same functions.
 */
export function applyExtraction(question: GradeQuestion, extracted: ExtractedAnswer): GradeQuestion {
  if (question.type === 'mcq') {
    return { ...question, studentOption: extracted.studentOption, studentParts: undefined, studentAnswer: null };
  }
  const hasParts = Object.keys(extracted.studentParts).length > 0;
  return {
    ...question,
    studentOption: null,
    studentParts: hasParts ? extracted.studentParts : {},
    studentAnswer: hasParts ? null : extracted.text || null,
  };
}

export { CONFIDENCE_OK, CONFIDENCE_FLOOR };
