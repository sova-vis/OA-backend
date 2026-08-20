/**
 * Deterministic tests for the handwritten-upload grading pipeline.
 *
 * The vision and text models are replaced with scripted responders so every edge
 * case (multi-page answers, out-of-order pages, illegible writing, blank
 * questions, wrong paper, corrupt files) is exercised exactly and repeatably.
 * The live-model behaviour is covered separately by live-grading.ts.
 *
 *   npx ts-node scripts/test/handwritten-pipeline.test.ts
 */

import * as path from 'path';
import * as dotenv from 'dotenv';

dotenv.config({ path: path.resolve(__dirname, '../../.env') });
// Pin the thresholds the assertions below assume, whatever the .env says.
process.env.PRACTICE_OCR_CONFIDENCE_OK = '0.55';
process.env.PRACTICE_OCR_CONFIDENCE_FLOOR = '0.3';

import { PDFDocument, StandardFonts } from 'pdf-lib';
import { group, test, run, ok, equal, deepEqual, includes, throws } from './harness';

import * as grok from '../../src/lib/grok';
import * as supabaseModule from '../../src/lib/supabase';
import { rasterizePdfPages, PageImage, isPracticeUploadType } from '../../src/lib/pdfPages';
import { GradeQuestion, extractHandwrittenAnswers } from '../../src/lib/handwrittenExtraction';
import { gradeHandwritten, gradeTyped, buildReport, schemeIsUsable, typedAnswersFromGraded, reclaimLeakedSchemes } from '../../src/lib/practiceMarking';
import { loadUploadPages, PracticeUpload } from '../../src/lib/practiceStore';

/* ============================================================
   Scripted model responders
   ============================================================ */

/** One page's scripted transcription, keyed by the page marker. */
interface ScriptedPage {
  quality?: 'good' | 'fair' | 'poor';
  header?: string;
  /** throw instead of answering, to simulate a vision failure */
  fail?: string;
  fragments: Array<{
    q: string;
    part?: string;
    text?: string;
    option?: string;
    legible?: boolean;
    confidence?: number;
    note?: string;
  }>;
}

let pageScript = new Map<string, ScriptedPage>();
/** every student_answer the grading model was asked to mark */
let gradedAnswers: Array<{ question: string; answer: string }> = [];
let visionCalls = 0;

/** A fake page image whose base64 encodes its marker, so the mock can identify it. */
function fakePage(marker: string, pageNumber: number): PageImage {
  return { base64: Buffer.from(marker, 'utf8').toString('base64'), mimeType: 'image/png', page: pageNumber, source: `${marker}.png` };
}

/**
 * Deterministic stand-in for the grading model: awards one mark per marking-scheme
 * keyword that appears in the student's answer. Purely a function of the answer
 * text, which is what lets the typed-vs-uploaded equivalence test mean something.
 */
function fakeGrade(user: string): Record<string, unknown> {
  const payload = JSON.parse(user) as {
    question: string; max_marks: number; marking_scheme: string; student_answer: string;
  };
  gradedAnswers.push({ question: payload.question, answer: payload.student_answer });

  const keywords = (payload.marking_scheme.match(/\*([^*]+)\*/g) ?? []).map((k) => k.replace(/\*/g, '').toLowerCase());
  const answer = payload.student_answer.toLowerCase();
  const hits = keywords.filter((keyword) => answer.includes(keyword));
  const earned = Math.min(payload.max_marks, hits.length);
  return {
    earned_marks: earned,
    verdict: earned >= payload.max_marks ? 'correct' : earned > 0 ? 'partial' : 'weak',
    feedback: `Matched ${hits.length} of ${keywords.length} marking points.`,
    expected_points: keywords,
    missing_points: keywords.filter((keyword) => !answer.includes(keyword)),
  };
}

function installMocks(): void {
  (grok as { grokChatJson: typeof grok.grokChatJson }).grokChatJson = (async (options: {
    user: string; images?: grok.GrokImage[];
  }) => {
    if (options.images && options.images.length > 0) {
      visionCalls += 1;
      const marker = Buffer.from(options.images[0].base64, 'base64').toString('utf8');
      const scripted = pageScript.get(marker);
      if (!scripted) throw new Error(`no scripted page for marker "${marker}"`);
      if (scripted.fail) throw new grok.GrokError(scripted.fail, 'other');
      return {
        page_quality: scripted.quality ?? 'good',
        header_text: scripted.header ?? '',
        fragments: scripted.fragments.map((fragment) => ({
          question_number: fragment.q,
          part_label: fragment.part ?? '',
          text: fragment.text ?? '',
          selected_option: fragment.option ?? '',
          legible: fragment.legible !== false,
          confidence: fragment.confidence ?? 0.95,
          note: fragment.note ?? '',
        })),
      };
    }
    return fakeGrade(options.user);
  }) as typeof grok.grokChatJson;
}

function reset(script: Record<string, ScriptedPage>): void {
  pageScript = new Map(Object.entries(script));
  gradedAnswers = [];
  visionCalls = 0;
}

/* ============================================================
   Fixtures
   ============================================================ */

/** A structured question whose scheme keywords are marked with *asterisks*. */
function structured(
  questionNumber: string,
  parts: Array<{ label: string; answer: string; marks: number }>,
): GradeQuestion {
  return {
    id: `q${questionNumber}`,
    questionNumber,
    type: 'structured',
    questionText: `Question ${questionNumber} stem`,
    maxMarks: parts.reduce((sum, part) => sum + part.marks, 0),
    markingScheme: null,
    parts: parts.map((part) => ({ label: part.label, body: '', marks: part.marks, answer: part.answer })),
  };
}

function mcq(questionNumber: string, correctOption: string): GradeQuestion {
  return {
    id: `q${questionNumber}`,
    questionNumber,
    type: 'mcq',
    questionText: `MCQ ${questionNumber} stem`,
    maxMarks: 1,
    correctOption,
  };
}

/** Three-question structured paper used across the mapping tests. */
function paperA(): GradeQuestion[] {
  return [
    structured('1', [{ label: '(a)', answer: '*photosynthesis*', marks: 2 }]),
    structured('2', [
      { label: '(a)', answer: '*chlorophyll*', marks: 2 },
      { label: '(b)', answer: '*stomata*', marks: 3 },
    ]),
    structured('3', [{ label: '(a)', answer: '*osmosis*', marks: 4 }]),
  ];
}

function find<T extends { questionNumber: string }>(list: T[], questionNumber: string): T {
  const found = list.find((item) => item.questionNumber === questionNumber);
  if (!found) throw new Error(`no result for Q${questionNumber}`);
  return found;
}

installMocks();

/* ============================================================
   1. Mapping answers to the right question
   ============================================================ */

group('extraction / mapping');

test('maps each answer to its question across two pages, out of order', async () => {
  reset({
    // deliberately reversed: Q3 is on page 1, Q1 on page 2
    p1: { fragments: [{ q: '3', part: '(a)', text: 'osmosis moves water' }] },
    p2: { fragments: [{ q: '1', part: '(a)', text: 'photosynthesis in leaves' }] },
  });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1), fakePage('p2', 2)]);

  const q1 = read.byQuestionId.get('q1')!;
  const q3 = read.byQuestionId.get('q3')!;
  equal(q1.flag, 'ok');
  includes(q1.text, 'photosynthesis in leaves');
  deepEqual(q1.pages, [2], 'Q1 was written on page 2');
  includes(q3.text, 'osmosis moves water');
  deepEqual(q3.pages, [1], 'Q3 was written on page 1');
  equal(visionCalls, 2, 'one vision call per page');
});

test('joins an answer that runs across two pages, in page order', async () => {
  reset({
    p1: { fragments: [{ q: '3', part: '(a)', text: 'osmosis is the movement' }] },
    p2: { fragments: [{ q: '3', part: '(a)', text: 'of water down a gradient' }] },
  });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1), fakePage('p2', 2)]);
  const q3 = read.byQuestionId.get('q3')!;

  equal(q3.studentParts['(a)'], 'osmosis is the movement\nof water down a gradient');
  deepEqual(q3.pages, [1, 2]);
});

test('joins the continuation even when the later page is supplied first', async () => {
  reset({
    p1: { fragments: [{ q: '3', part: '(a)', text: 'first half' }] },
    p2: { fragments: [{ q: '3', part: '(a)', text: 'second half' }] },
  });
  // page objects handed over in the wrong order; page numbers are what count
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p2', 2), fakePage('p1', 1)]);
  equal(read.byQuestionId.get('q3')!.studentParts['(a)'], 'first half\nsecond half');
});

test('normalises loosely written labels onto the scheme labels', async () => {
  reset({
    p1: {
      fragments: [
        { q: '2a', text: 'chlorophyll absorbs light' },   // no part_label at all
        { q: '2', part: 'b', text: 'stomata open' },       // bare letter, no brackets
      ],
    },
  });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1)]);
  const q2 = read.byQuestionId.get('q2')!;

  deepEqual(Object.keys(q2.studentParts).sort(), ['(a)', '(b)'], 'both map onto the scheme labels');
  includes(q2.studentParts['(a)'], 'chlorophyll');
  includes(q2.studentParts['(b)'], 'stomata');
});

test('Q1 does not swallow Q10', async () => {
  const paper = [
    structured('1', [{ label: '(a)', answer: '*alpha*', marks: 2 }]),
    structured('10', [{ label: '(a)', answer: '*omega*', marks: 2 }]),
  ];
  reset({
    p1: { fragments: [{ q: '10', part: '(a)', text: 'omega answer' }] },
  });
  const read = await extractHandwrittenAnswers(paper, [fakePage('p1', 1)]);

  equal(read.byQuestionId.get('q10')!.flag, 'ok', 'Q10 got its answer');
  includes(read.byQuestionId.get('q10')!.text, 'omega answer');
  equal(read.byQuestionId.get('q1')!.flag, 'blank', 'Q1 must not claim Q10\'s answer');
  equal(read.byQuestionId.get('q1')!.text, '');
});

/* ============================================================
   2. Blank, illegible and missing answers
   ============================================================ */

group('extraction / quality flags');

test('a question number with nothing written is blank, not unreadable', async () => {
  reset({
    p1: {
      fragments: [
        { q: '1', part: '(a)', text: 'photosynthesis' },
        { q: '3', part: '(a)', text: '' }, // student wrote "3." then left it
      ],
    },
  });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1)]);
  equal(read.byQuestionId.get('q3')!.flag, 'blank');
});

test('a question absent from a matching upload is blank (skipped), not not_found', async () => {
  reset({ p1: { fragments: [{ q: '1', part: '(a)', text: 'photosynthesis' }] } });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1)]);

  equal(read.byQuestionId.get('q2')!.flag, 'blank');
  equal(read.byQuestionId.get('q3')!.flag, 'blank');
  equal(read.paperMismatch, false);
});

test('illegible writing is flagged unreadable and its text is not kept for marking', async () => {
  reset({
    p1: { fragments: [{ q: '1', part: '(a)', text: 'phot??yn??sis', legible: false, confidence: 0.2, note: 'very faint pencil' }] },
  });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1)]);
  const q1 = read.byQuestionId.get('q1')!;

  equal(q1.flag, 'unreadable');
  deepEqual(q1.studentParts, {}, 'unreadable text must not reach the marker');
  equal(q1.studentOption, null);
  includes(q1.note, 'faint pencil');
});

test('confidence below the floor is unreadable even when marked legible', async () => {
  reset({ p1: { fragments: [{ q: '1', part: '(a)', text: 'maybe photosynthesis', legible: true, confidence: 0.25 }] } });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1)]);
  equal(read.byQuestionId.get('q1')!.flag, 'unreadable');
});

test('middling confidence is low_confidence and is still marked', async () => {
  reset({ p1: { fragments: [{ q: '1', part: '(a)', text: 'photosynthesis happens', confidence: 0.45 }] } });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1)]);
  const q1 = read.byQuestionId.get('q1')!;

  equal(q1.flag, 'low_confidence');
  includes(q1.studentParts['(a)'], 'photosynthesis', 'the reading is kept so it can be marked');
});

test('a question is only as trustworthy as its worst fragment', async () => {
  reset({
    p1: { fragments: [{ q: '3', part: '(a)', text: 'clear first half', confidence: 0.99 }] },
    p2: { fragments: [{ q: '3', part: '(a)', text: 'messy second half', confidence: 0.4 }] },
  });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1), fakePage('p2', 2)]);
  const q3 = read.byQuestionId.get('q3')!;

  equal(q3.flag, 'low_confidence');
  equal(q3.confidence, 0.4);
});

test('a page the model cannot process is reported and the other pages still read', async () => {
  reset({
    p1: { fragments: [{ q: '1', part: '(a)', text: 'photosynthesis' }] },
    p2: { fail: 'vision timed out', fragments: [] },
  });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1), fakePage('p2', 2)]);

  equal(read.byQuestionId.get('q1')!.flag, 'ok', 'page 1 still read');
  equal(read.warnings.length >= 1, true);
  includes(read.warnings.join(' '), 'Page 2 could not be read');
});

test('poor scan quality raises a warning', async () => {
  reset({ p1: { quality: 'poor', fragments: [{ q: '1', part: '(a)', text: 'photosynthesis' }] } });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1)]);
  includes(read.warnings.join(' '), 'Image quality is poor on page 1');
});

/* ============================================================
   3. Wrong paper
   ============================================================ */

group('extraction / wrong paper');

test('an upload for a different paper is flagged as a mismatch', async () => {
  reset({
    p1: {
      header: 'Chemistry Paper 4',
      fragments: [
        { q: '41', text: 'titration answer' },
        { q: '42', text: 'moles answer' },
        { q: '43', text: 'electrolysis answer' },
      ],
    },
  });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1)]);

  equal(read.paperMismatch, true);
  includes(read.warnings.join(' '), 'do not match the paper you selected');
  equal(read.byQuestionId.get('q1')!.flag, 'not_found', 'nothing is claimed as blank on a mismatch');
});

test('a Chemistry header on a Biology paper is a mismatch even when Q numbers overlap', async () => {
  reset({
    p1: {
      header: 'CHEMISTRY 5070/21 Paper 2 May/June 2023',
      fragments: [
        { q: '1', part: '(a)', text: 'titration of HCl' },
        { q: '2', part: '(a)', text: 'moles of NaOH' },
        { q: '3', part: '(a)', text: 'electrolysis of brine' },
      ],
    },
  });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1)], { subject: 'Biology' });

  equal(read.paperMismatch, true, 'header names a different subject');
  includes(read.warnings.join(' '), 'chemistry');
});

test('the matching subject in the header is NOT a mismatch', async () => {
  reset({
    p1: {
      header: 'BIOLOGY 5090/21 Paper 2',
      fragments: [{ q: '1', part: '(a)', text: 'photosynthesis' }],
    },
  });
  const read = await extractHandwrittenAnswers(paperA(), [fakePage('p1', 1)], { subject: 'Biology' });
  equal(read.paperMismatch, false);
});

test('a partial upload of the right paper is NOT a mismatch', async () => {
  // 40-question paper, student uploads only the page holding Q1-Q2.
  const paper = Array.from({ length: 40 }, (_, index) =>
    structured(String(index + 1), [{ label: '(a)', answer: '*point*', marks: 2 }]),
  );
  reset({
    p1: { fragments: [{ q: '1', part: '(a)', text: 'point one' }, { q: '2', part: '(a)', text: 'point two' }] },
  });
  const read = await extractHandwrittenAnswers(paper, [fakePage('p1', 1)]);

  equal(read.paperMismatch, false, 'low overlap alone must not trip the mismatch check');
  equal(read.byQuestionId.get('q1')!.flag, 'ok');
  equal(read.byQuestionId.get('q40')!.flag, 'blank');
});

/* ============================================================
   4. Grading reuses the typed path
   ============================================================ */

group('marking / shared engine');

test('an uploaded attempt scores exactly the same as the same answers typed', async () => {
  const answers: Record<string, Record<string, string>> = {
    q1: { '(a)': 'this is about photosynthesis' },
    q2: { '(a)': 'chlorophyll pigment', '(b)': 'stomata regulate gas exchange' },
    q3: { '(a)': 'osmosis explained fully' },
  };

  // (a) typed, exactly as the Solve Here flow builds it
  reset({});
  const typed = await gradeTyped('Biology', paperA().map((question) => ({ ...question, studentParts: answers[question.id] })));
  const typedAnswers = [...gradedAnswers];

  // (b) the same text, arriving from an upload
  reset({
    p1: {
      fragments: [
        { q: '1', part: '(a)', text: answers.q1['(a)'] },
        { q: '2', part: '(a)', text: answers.q2['(a)'] },
        { q: '2', part: '(b)', text: answers.q2['(b)'] },
        { q: '3', part: '(a)', text: answers.q3['(a)'] },
      ],
    },
  });
  const { graded: uploaded } = await gradeHandwritten('Biology', paperA(), [fakePage('p1', 1)], false);

  for (const question of paperA()) {
    const t = find(typed, question.questionNumber);
    const u = find(uploaded, question.questionNumber);
    equal(u.earned, t.earned, `Q${question.questionNumber} earned marks must match the typed attempt`);
    equal(u.max, t.max, `Q${question.questionNumber} max`);
    equal(u.verdict, t.verdict, `Q${question.questionNumber} verdict`);
  }

  // and the marker literally received the same payloads
  const norm = (list: Array<{ question: string; answer: string }>) =>
    list.map((entry) => `${entry.question}::${entry.answer}`).sort();
  deepEqual(norm(gradedAnswers), norm(typedAnswers), 'the grading model saw identical student answers both ways');
});

test('unreadable answers are never sent to the marker', async () => {
  reset({
    p1: {
      fragments: [
        { q: '1', part: '(a)', text: 'photosynthesis' },
        { q: '3', part: '(a)', text: 'illegible scrawl', legible: false, confidence: 0.1 },
      ],
    },
  });
  const { graded } = await gradeHandwritten('Biology', paperA(), [fakePage('p1', 1)], false);

  ok(!gradedAnswers.some((entry) => entry.answer.includes('illegible scrawl')), 'no guessing at illegible text');
  const q3 = find(graded, '3');
  equal(q3.marksWithheld, true);
  equal(q3.earned, 0);
  includes(q3.feedback, 'could not read');
});

test('a blank question is marked unanswered, not wrong', async () => {
  reset({
    p1: { fragments: [{ q: '1', part: '(a)', text: 'photosynthesis' }, { q: '3', part: '(a)', text: '' }] },
  });
  const { graded } = await gradeHandwritten('Biology', paperA(), [fakePage('p1', 1)], false);
  const q3 = find(graded, '3');

  equal(q3.verdict, 'unanswered');
  equal(q3.earned, 0);
  equal(q3.marksWithheld, undefined, 'a blank answer still counts towards the total');
  equal(q3.extractionFlag, 'blank');
});

test('handwritten MCQs are graded deterministically, same as typed', async () => {
  const paper = [mcq('1', 'B'), mcq('2', 'C'), mcq('3', 'A')];
  reset({
    p1: {
      fragments: [
        { q: '1', option: 'B' },  // correct
        { q: '2', option: 'D' },  // wrong
        // Q3 not attempted
      ],
    },
  });
  const { graded } = await gradeHandwritten('Physics', paper, [fakePage('p1', 1)], true);

  equal(find(graded, '1').earned, 1);
  equal(find(graded, '1').verdict, 'correct');
  equal(find(graded, '2').earned, 0);
  equal(find(graded, '2').verdict, 'weak');
  equal(find(graded, '3').verdict, 'unanswered');
  equal(find(graded, '1').gradingSource, 'deterministic', 'no AI needed to mark an MCQ');
  equal(find(graded, '1').extractedOption, 'B');
  equal(find(graded, '2').extractedOption, 'D');
  const slotted = typedAnswersFromGraded(paper, graded);
  equal(slotted.mcq['q1'], 'B');
  equal(slotted.mcq['q2'], 'D');
  equal(gradedAnswers.length, 0, 'the text model is never called for an MCQ paper');
});

test('every graded question carries the reading it was marked from', async () => {
  reset({ p1: { fragments: [{ q: '1', part: '(a)', text: 'photosynthesis in the leaf', confidence: 0.9 }] } });
  const { graded } = await gradeHandwritten('Biology', paperA(), [fakePage('p1', 1)], false);
  const q1 = find(graded, '1');

  includes(q1.extractedAnswer || '', 'photosynthesis in the leaf');
  equal(q1.extractedParts?.['(a)'], 'photosynthesis in the leaf');
  equal(q1.extractionFlag, 'ok');
  equal(q1.extractionConfidence, 0.9);
  deepEqual(q1.extractionPages, [1]);
});

test('extracted answers land in the same Solve-here slots the typed flow uses', async () => {
  reset({
    p1: {
      fragments: [
        { q: '1', part: '(a)', text: 'this is about photosynthesis' },
        { q: '2', part: '(a)', text: 'chlorophyll pigment' },
        { q: '2', part: '(b)', text: 'stomata regulate gas exchange' },
      ],
    },
  });
  const paper = paperA();
  const { graded } = await gradeHandwritten('Biology', paper, [fakePage('p1', 1)], false);
  const answers = typedAnswersFromGraded(paper, graded);

  equal(answers.parts['q1::0'], 'this is about photosynthesis');
  equal(answers.parts['q2::0'], 'chlorophyll pigment');
  equal(answers.parts['q2::1'], 'stomata regulate gas exchange');
});

/* ============================================================
   5. Report arithmetic
   ============================================================ */

group('report');

test('withheld marks are excluded from the score rather than counted wrong', async () => {
  reset({
    p1: {
      fragments: [
        // the stub marker awards one mark per marking-scheme keyword matched
        { q: '1', part: '(a)', text: 'photosynthesis' },                          // 1 of 2
        { q: '2', part: '(a)', text: 'chlorophyll' },                             // 1 of 5, (b) not attempted
        { q: '3', part: '(a)', text: 'scrawl', legible: false, confidence: 0.1 }, // withheld, worth 4
      ],
    },
  });
  const { graded, extraction } = await gradeHandwritten('Biology', paperA(), [fakePage('p1', 1)], false);
  const report = buildReport(graded, 'handwritten', 'grok-vision-test', extraction);

  equal(extraction.unreadableCount, 1);
  equal(extraction.withheldMarks, 4, 'Q3 is worth 4 marks');
  equal(report.total, 7, 'total covers only Q1 (2) + Q2 (5); Q3 excluded');
  equal(report.earned, 2, 'one keyword matched on each of Q1 and Q2');
  // 2/7 = 29%. Had Q3 been counted as 0/4 the score would read 2/11 = 18%.
  equal(report.percent, 29);
  includes(report.summary, 'marks were not assessed');
  includes(report.improvements.join(' '), 'could not be read');
});

test('a report where nothing could be read scores 0 and says so', async () => {
  reset({
    p1: {
      fragments: [
        { q: '1', part: '(a)', text: 'x', legible: false, confidence: 0.1 },
        { q: '2', part: '(a)', text: 'x', legible: false, confidence: 0.1 },
        { q: '3', part: '(a)', text: 'x', legible: false, confidence: 0.1 },
      ],
    },
  });
  const { graded, extraction } = await gradeHandwritten('Biology', paperA(), [fakePage('p1', 1)], false);
  const report = buildReport(graded, 'handwritten', 'grok-vision-test', extraction);

  equal(report.total, 0);
  equal(report.percent, 0);
  includes(report.summary, 'could not mark any of this attempt');
  equal(report.extraction?.withheldMarks, 11, 'all 11 marks withheld');
});

test('a typed report carries no extraction block', async () => {
  reset({});
  const graded = await gradeTyped('Biology', paperA().map((q) => ({ ...q, studentParts: { '(a)': 'photosynthesis' } })));
  const report = buildReport(graded, 'digital', 'grok-text-test');

  equal(report.extraction, undefined);
  equal(report.total, 11, 'nothing is withheld on a typed attempt');
});

/* ============================================================
   6. File handling: PDFs, images, corrupt and unsupported files
   ============================================================ */

group('files');

/** Build a real multi-page PDF so rasterization is genuinely exercised. */
async function makePdf(pageCount: number): Promise<Buffer> {
  const pdf = await PDFDocument.create();
  const font = await pdf.embedFont(StandardFonts.Helvetica);
  for (let index = 0; index < pageCount; index++) {
    const page = pdf.addPage([595, 842]);
    page.drawText(`Answer page ${index + 1}`, { x: 60, y: 760, size: 24, font });
  }
  return Buffer.from(await pdf.save());
}

test('a real multi-page PDF rasterizes to one image per page, numbered in order', async () => {
  const pdf = await makePdf(3);
  const result = await rasterizePdfPages(pdf, 1, 10, 'attempt.pdf');

  equal(result.pages.length, 3);
  equal(result.totalPages, 3);
  equal(result.dropped, 0);
  deepEqual(result.pages.map((page) => page.page), [1, 2, 3]);
  ok(result.pages.every((page) => page.mimeType === 'image/png' && page.base64.length > 500), 'each page produced a real PNG');
});

test('the page budget truncates instead of silently dropping the rest', async () => {
  const pdf = await makePdf(5);
  const result = await rasterizePdfPages(pdf, 1, 2, 'attempt.pdf');

  equal(result.pages.length, 2);
  equal(result.totalPages, 5);
  equal(result.dropped, 3);
});

test('a corrupt PDF fails with a message naming the file', async () => {
  await throws(() => rasterizePdfPages(Buffer.from('%PDF-1.4 this is not a real pdf'), 1, 10, 'broken.pdf'), 'broken.pdf');
});

/** Fake Supabase storage serving a fixed set of files. */
function stubStorage(files: Record<string, Buffer | null>): void {
  (supabaseModule as { supabase: unknown }).supabase = {
    storage: {
      from: () => ({
        download: async (storagePath: string) => {
          const buffer = files[storagePath];
          if (buffer === undefined) return { data: null, error: new Error('not found') };
          if (buffer === null) return { data: null, error: new Error('download failed') };
          return { data: { arrayBuffer: async () => buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) }, error: null };
        },
        list: async (prefix: string) => ({
          data: Object.keys(files)
            .filter((key) => key.startsWith(`${prefix}/`))
            .map((key) => ({ name: key.slice(prefix.length + 1), id: '1', metadata: { size: files[key]?.byteLength ?? 0 } })),
          error: null,
        }),
      }),
    },
  };
}

const uploadRecord = (name: string, type: string, size = 1024): PracticeUpload => ({
  path: `files/u1/paper/${name}`, name, type, size, at: new Date().toISOString(),
});

test('a PDF upload is rasterized into pages instead of being skipped', async () => {
  const pdf = await makePdf(4);
  stubStorage({ 'files/u1/paper/attempt.pdf': pdf });
  const loaded = await loadUploadPages([uploadRecord('attempt.pdf', 'application/pdf')]);

  equal(loaded.pages.length, 4, 'this is the bug that used to score every PDF page as unanswered');
  deepEqual(loaded.skipped, []);
  equal(loaded.truncated, false);
  equal(loaded.usedFiles, 1);
});

test('images and PDFs mix into one correctly numbered page sequence', async () => {
  const pdf = await makePdf(2);
  stubStorage({
    'files/u1/paper/page1.jpg': Buffer.from('fake-jpeg-bytes'),
    'files/u1/paper/rest.pdf': pdf,
    'files/u1/paper/last.png': Buffer.from('fake-png-bytes'),
  });
  const loaded = await loadUploadPages([
    uploadRecord('page1.jpg', 'image/jpeg'),
    uploadRecord('rest.pdf', 'application/pdf'),
    uploadRecord('last.png', 'image/png'),
  ]);

  equal(loaded.pages.length, 4);
  deepEqual(loaded.pages.map((page) => page.page), [1, 2, 3, 4]);
  deepEqual(loaded.pages.map((page) => page.source), ['page1.jpg', 'rest.pdf', 'rest.pdf', 'last.png']);
});

test('JPG, PNG and PDF are accepted; other types and empty files are rejected by type helper', () => {
  equal(isPracticeUploadType('image/jpeg', 'a.jpg'), true);
  equal(isPracticeUploadType('image/png', 'a.png'), true);
  equal(isPracticeUploadType('application/pdf', 'a.pdf'), true);
  equal(isPracticeUploadType('application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'notes.docx'), false);
  equal(isPracticeUploadType('', 'scan.PDF'), true, 'empty MIME falls back to extension');
  equal(isPracticeUploadType('application/octet-stream', 'page.png'), true);
  equal(isPracticeUploadType('text/plain', 'answers.txt'), false);
});

test('an unsupported file type is reported, not silently dropped', async () => {
  stubStorage({ 'files/u1/paper/notes.docx': Buffer.from('PK zip bytes') });
  const loaded = await loadUploadPages([uploadRecord('notes.docx', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')]);

  equal(loaded.pages.length, 0);
  equal(loaded.skipped.length, 1);
  includes(loaded.skipped[0], 'unsupported file type');
  includes(loaded.skipped[0], 'notes.docx');
});

test('a corrupt PDF among good images is reported while the images still grade', async () => {
  stubStorage({
    'files/u1/paper/good.jpg': Buffer.from('fake-jpeg-bytes'),
    'files/u1/paper/broken.pdf': Buffer.from('not a pdf at all'),
  });
  const loaded = await loadUploadPages([uploadRecord('good.jpg', 'image/jpeg'), uploadRecord('broken.pdf', 'application/pdf')]);

  equal(loaded.pages.length, 1, 'the readable image is kept');
  equal(loaded.skipped.length, 1);
  includes(loaded.skipped[0], 'broken.pdf');
});

test('an empty file is reported', async () => {
  stubStorage({ 'files/u1/paper/empty.png': Buffer.alloc(0) });
  const loaded = await loadUploadPages([uploadRecord('empty.png', 'image/png', 0)]);

  equal(loaded.pages.length, 0);
  includes(loaded.skipped[0], 'empty');
});

test('a file missing from storage is reported', async () => {
  stubStorage({});
  const loaded = await loadUploadPages([uploadRecord('gone.png', 'image/png')]);

  equal(loaded.pages.length, 0);
  includes(loaded.skipped[0], 'could not be downloaded');
});

test('duplicate filenames are collapsed to the latest upload', async () => {
  stubStorage({
    'files/u1/paper/old.png': Buffer.from('fake-png-bytes'),
    'files/u1/paper/new.png': Buffer.from('fake-png-bytes'),
  });
  const loaded = await loadUploadPages([
    { ...uploadRecord('handtext-page-1.pdf', 'image/png'), path: 'files/u1/paper/old.png' },
    { ...uploadRecord('handtext-page-1.pdf', 'image/png'), path: 'files/u1/paper/new.png' },
  ]);
  equal(loaded.usedFiles, 1);
  equal(loaded.pages[0].source, 'handtext-page-1.pdf');
});

test('a signed URL path is normalized back to the storage key', async () => {
  stubStorage({ 'files/u1/paper/scan.png': Buffer.from('fake-png-bytes') });
  const loaded = await loadUploadPages([{
    ...uploadRecord('scan.png', 'image/png'),
    path: 'https://example.supabase.co/storage/v1/object/sign/practice-progress/files/u1/paper/scan.png?token=abc',
  }]);
  equal(loaded.pages.length, 1);
});

test('stale JSON paths recover files that are still in the paper folder', async () => {
  stubStorage({ 'files/u1/Chem_2021_May_June_Paper_2_Variant_1/1_scan.png': Buffer.from('fake-png-bytes') });
  const loaded = await loadUploadPages(
    [{ path: 'files/u1/Chem_2021_May_June_Paper_2_Variant_1/old_gone.png', name: 'scan.png', type: 'image/png', size: 10, at: '' }],
    undefined,
    { clerkId: 'u1', paperKey: 'Chem|2021|May_June|Paper_2|Variant_1' },
  );
  equal(loaded.pages.length, 1);
  equal(loaded.usedFiles, 1);
});

test('exceeding the page budget across files is reported as truncation', async () => {
  const pdf = await makePdf(6);
  stubStorage({ 'files/u1/paper/big.pdf': pdf, 'files/u1/paper/extra.png': Buffer.from('fake-png') });
  const loaded = await loadUploadPages([uploadRecord('big.pdf', 'application/pdf'), uploadRecord('extra.png', 'image/png')], 4);

  equal(loaded.pages.length, 4);
  equal(loaded.truncated, true);
  includes(loaded.skipped.join(' '), 'only the first 4 of 6 pages');
  includes(loaded.skipped.join(' '), 'extra.png');
});

test('no pages at all yields no pages and every reason listed', async () => {
  stubStorage({ 'files/u1/paper/a.pdf': Buffer.from('junk'), 'files/u1/paper/b.txt': Buffer.from('hello') });
  const loaded = await loadUploadPages([uploadRecord('a.pdf', 'application/pdf'), uploadRecord('b.txt', 'text/plain')]);

  equal(loaded.pages.length, 0);
  equal(loaded.skipped.length, 2, 'the student is told about both files');
});

/* ============================================================
   7. Whole-paper run with no pages
   ============================================================ */

group('degenerate input');

test('grading with zero pages marks everything not_found and withholds all marks', async () => {
  reset({});
  const { graded, extraction } = await gradeHandwritten('Biology', paperA(), [], false);
  const report = buildReport(graded, 'handwritten', 'grok-vision-test', extraction);

  equal(extraction.pageCount, 0);
  equal(extraction.notFoundCount, 3);
  equal(report.total, 0);
  equal(gradedAnswers.length, 0, 'nothing is sent to the marker');
});

test('a timeout on one question does not fail the rest of the paper', async () => {
  reset({
    p1: {
      fragments: [
        { q: '1', part: '(a)', text: 'photosynthesis in palisade cells' },
        { q: '2', part: '(a)', text: 'chlorophyll absorbs light' },
        { q: '3', part: '(a)', text: 'osmosis of water' },
      ],
    },
  });
  const original = grok.grokChatJson;
  (grok as { grokChatJson: typeof grok.grokChatJson }).grokChatJson = (async (options) => {
    if (options.images && options.images.length > 0) return original(options);
    const payload = JSON.parse(options.user) as { question: string };
    if (payload.question.includes('Question 2')) {
      throw new grok.GrokError('Grok request timed out', 'timeout');
    }
    return original(options);
  }) as typeof grok.grokChatJson;

  const { graded } = await gradeHandwritten('Biology', paperA(), [fakePage('p1', 1)], false);
  equal(graded.length, 3);
  equal(graded[1].marksWithheld, true, 'timed-out question is withheld, not a 500');
  equal(graded[1].gradingFailed, true, 'timeout is a marking failure, not unread handwriting');
  equal(graded[0].marksWithheld, undefined);
  equal(graded[2].marksWithheld, undefined);

  installMocks();
});

test('a transient grading failure is retried and the question is still marked', async () => {
  reset({
    p1: {
      fragments: [
        { q: '1', part: '(a)', text: 'photosynthesis in palisade cells' },
        { q: '2', part: '(a)', text: 'chlorophyll absorbs light' },
        { q: '3', part: '(a)', text: 'osmosis of water' },
      ],
    },
  });
  const original = grok.grokChatJson;
  let q2Attempts = 0;
  (grok as { grokChatJson: typeof grok.grokChatJson }).grokChatJson = (async (options) => {
    if (options.images && options.images.length > 0) return original(options);
    const payload = JSON.parse(options.user) as { question: string };
    if (payload.question.includes('Question 2')) {
      q2Attempts += 1;
      if (q2Attempts === 1) {
        throw new grok.GrokError('gemini API error 503: high demand UNAVAILABLE', 'rate_limit', 503);
      }
    }
    return original(options);
  }) as typeof grok.grokChatJson;

  const { graded } = await gradeHandwritten('Biology', paperA(), [fakePage('p1', 1)], false);
  equal(graded.length, 3);
  equal(find(graded, '2').marksWithheld, undefined, 'retried question is marked');
  equal(find(graded, '2').gradingFailed, undefined);
  ok(q2Attempts >= 2, 'the failed question was retried');
  equal(find(graded, '1').marksWithheld, undefined);
  equal(find(graded, '3').marksWithheld, undefined);

  installMocks();
});

group('scheme + json helpers');

test('furniture-only mark schemes are treated as empty so examiner judgement can run', () => {
  ok(!schemeIsUsable('three from:'));
  ok(!schemeIsUsable('Working Space'));
  ok(!schemeIsUsable(''));
  ok(schemeIsUsable('four from: / a match is found / door is closed'));
  ok(schemeIsUsable('*chlorophyll* absorbs light'));
});

test('a mark scheme glued onto the previous question is reclaimed for the empty next question', () => {
  const paper = [
    structured('6', [{
      label: '(c)',
      answer: 'Any two from: • Operating system • Utility software NOTE: Two examples of utility software can be awarded 2 7 One mark for each correct term in the correct place. • plain text • cipher text • public key • private key 4 2210/11',
      marks: 2,
    }]),
    {
      id: 'q7',
      questionNumber: '7',
      type: 'structured' as const,
      questionText: 'Complete the paragraph about asymmetric encryption.',
      maxMarks: 0,
      markingScheme: null,
      parts: [],
    },
  ];
  const recovered = reclaimLeakedSchemes(paper);
  includes(recovered[1].markingScheme || '', 'plain text');
  includes(recovered[1].markingScheme || '', 'private key');
  equal(recovered[1].maxMarks, 4, 'marks recovered from the leaked tail');
  ok(schemeIsUsable(recovered[1].markingScheme || ''), 'Q7 now has a usable scheme');
  ok(!(recovered[0].parts?.[0].answer || '').includes('plain text'), 'leak stripped off Q6');
  includes(recovered[0].parts?.[0].answer || '', 'Operating system');
});

test('HTTP 503 high-demand is a retryable rate limit, not a dead question', () => {
  equal(
    grok.classifyGrokHttpError(503, '[{ "error": { "code": 503, "status": "UNAVAILABLE", "message": "high demand" } }]'),
    'rate_limit',
  );
  equal(grok.classifyGrokHttpError(429, 'rate limited'), 'rate_limit');
  equal(
    grok.classifyGrokHttpError(429, 'You exceeded your current quota, please check your plan and billing details. Please retry in 17.3s.'),
    'rate_limit',
    'Gemini free-tier 429s mention quota but are a pause, not a dead key',
  );
  equal(
    grok.classifyGrokHttpError(400, '{"error":{"message":"Failed to validate JSON.","code":"json_validate_failed"}}'),
    'parse',
    'Groq JSON-mode 400 is a parse miss, not a missing model',
  );
  equal(grok.classifyGrokHttpError(403, 'Your team has used all available credits'), 'quota');
  equal(
    grok.classifyGrokHttpError(404, '{"error":{"message":"The model `meta-llama/llama-4-scout-17b-16e-instruct` does not exist or you do not have access to it.","code":"model_not_found"}}'),
    'model',
  );
  equal(
    grok.classifyGrokHttpError(413, 'Request too large for model qwen/qwen3.6-27b on tokens per minute (TPM): Limit 8000, Requested 11330, please reduce your message size'),
    'rate_limit',
  );
  equal(grok.retryAfterMsFrom('Please retry in 17s.'), 17000);
});

test('retired Groq vision/text IDs are skipped so production env cannot 404 every page', () => {
  const prevVision = process.env.GROQ_VISION_MODEL;
  const prevGrading = process.env.GROQ_GRADING_MODEL;
  const prevModel = process.env.GROQ_MODEL;
  process.env.GROQ_VISION_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct';
  process.env.GROQ_GRADING_MODEL = 'llama-3.3-70b-versatile';
  process.env.GROQ_MODEL = 'llama-3.1-8b-instant';
  try {
    equal(grok.groqVisionModel(), 'qwen/qwen3.6-27b');
    equal(grok.groqTextModel(), 'openai/gpt-oss-120b');
  } finally {
    if (prevVision === undefined) delete process.env.GROQ_VISION_MODEL;
    else process.env.GROQ_VISION_MODEL = prevVision;
    if (prevGrading === undefined) delete process.env.GROQ_GRADING_MODEL;
    else process.env.GROQ_GRADING_MODEL = prevGrading;
    if (prevModel === undefined) delete process.env.GROQ_MODEL;
    else process.env.GROQ_MODEL = prevModel;
  }
});

test('JSON wrapped in an array or markdown still parses', () => {
  const inner = { earned_marks: 2, verdict: 'partial', feedback: 'ok' };
  deepEqual(grok.parseJsonObject(JSON.stringify([inner])), inner);
  deepEqual(grok.parseJsonObject('```json\n' + JSON.stringify(inner) + '\n```'), inner);
});

void run();
