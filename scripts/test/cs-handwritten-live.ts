/**
 * Live evaluation of Upload handwritten against the 10 CS May/June Paper 1
 * Variant 1 fixtures (2016–2025).
 *
 * Uses the same path as the UI: rasterize the PDF → extractHandwrittenAnswers
 * → gradeHandwritten (shared Solve-here marker) → buildReport.
 *
 *   npx ts-node --transpile-only scripts/test/cs-handwritten-live.ts
 *   npx ts-node --transpile-only scripts/test/cs-handwritten-live.ts --year=2016
 *
 * Writes per-paper JSON under test-fixtures/cs-handwritten/_cache/live/
 */

import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';

dotenv.config({ path: path.resolve(__dirname, '../../.env') });

import { grokEnabled } from '../../src/lib/grok';
import { GradeQuestion } from '../../src/lib/handwrittenExtraction';
import { gradeHandwritten, buildReport, schemeIsUsable } from '../../src/lib/practiceMarking';
import { rasterizePdfPages } from '../../src/lib/pdfPages';

const FIXTURES = path.resolve(__dirname, '../../../test-fixtures/cs-handwritten');
const OUT_DIR = path.join(FIXTURES, '_cache', 'live');
const YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];

interface KeyPart {
  label: string;
  marks: number;
  studentAnswer: string;
  markScheme: string;
  expectedMarks: number;
}
interface KeyQuestion {
  questionNumber: string;
  maxMarks: number;
  stem: string;
  parts: KeyPart[];
}
interface KeyPaper {
  subject: string;
  year: number;
  expectedEarned: number;
  expectedTotal: number;
  expectedPercent: number;
  questions: KeyQuestion[];
}
interface BankPart {
  label: string;
  body: string;
  marks: number | null;
  answer: string;
}
interface BankQuestion {
  id: string;
  questionNumber: string;
  questionText: string;
  marks: number;
  markingScheme: string;
  parts: BankPart[];
}
interface BankPaper {
  year: number;
  questions: BankQuestion[];
}

function coverage(got: string, wanted: string): number {
  const words = wanted.toLowerCase().split(/\W+/).filter((w) => w.length >= 4);
  if (words.length === 0) return got.trim().length > 0 ? 1 : 0;
  const hay = got.toLowerCase();
  return words.filter((w) => hay.includes(w)).length / words.length;
}

function schemeText(question: GradeQuestion): string {
  const chunks: string[] = [];
  if (question.markingScheme?.trim()) chunks.push(question.markingScheme.trim());
  for (const part of question.parts ?? []) {
    if (part.answer?.trim()) {
      chunks.push(`${part.label ? part.label + ' ' : ''}${part.answer.trim()}`);
    }
  }
  return chunks.join('\n');
}

function toGradeQuestions(bank: BankPaper): GradeQuestion[] {
  return bank.questions.map((q) => ({
    id: q.id,
    questionNumber: String(q.questionNumber),
    type: 'structured' as const,
    questionText: q.questionText || `Question ${q.questionNumber}`,
    maxMarks: Number(q.marks) > 0
      ? Number(q.marks)
      : Math.max(1, (q.parts || []).reduce((s, p) => s + (p.marks ?? 0), 0) || 1),
    markingScheme: q.markingScheme || null,
    parts: (q.parts || []).map((p) => ({
      label: p.label || '',
      body: p.body || '',
      marks: p.marks,
      answer: p.answer || null,
    })),
  }));
}

function parseYearArg(): number[] {
  const force = process.argv.includes('--force');
  const raw = process.argv.find((a) => a.startsWith('--year='));
  const wanted = raw
    ? [Number.parseInt(raw.slice('--year='.length), 10)].filter((year) => YEARS.includes(year))
    : YEARS;
  if (raw && wanted.length === 0) throw new Error(`Unknown year ${raw}`);
  if (force) return wanted;
  return wanted.filter((year) => !fs.existsSync(path.join(OUT_DIR, `${year}.json`)));
}

if (!grokEnabled()) {
  console.log('SKIP cs-handwritten-live: no XAI/GROQ/GEMINI key.');
  process.exit(0);
}

void (async () => {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const years = parseYearArg();
  if (years.length === 0) {
    console.log('Nothing to run (existing results in _cache/live). Pass --force to re-grade.');
    return;
  }
  const summary: Array<Record<string, unknown>> = [];
  for (const year of YEARS) {
    const existing = path.join(OUT_DIR, `${year}.json`);
    if (fs.existsSync(existing) && !years.includes(year)) {
      summary.push(JSON.parse(fs.readFileSync(existing, 'utf8')) as Record<string, unknown>);
    }
  }

  for (const year of years) {
    const stem = `CS_${year}_May_June_Paper_1_Variant_1`;
    const pdfPath = path.join(FIXTURES, `${stem}.pdf`);
    const keyPath = path.join(FIXTURES, `${stem}.key.json`);
    const bankPath = path.join(FIXTURES, '_cache', `${year}.bank.json`);
    if (!fs.existsSync(pdfPath) || !fs.existsSync(keyPath) || !fs.existsSync(bankPath)) {
      throw new Error(`Missing fixture for ${year}`);
    }

    const key = JSON.parse(fs.readFileSync(keyPath, 'utf8')) as KeyPaper;
    const bank = JSON.parse(fs.readFileSync(bankPath, 'utf8')) as BankPaper;
    const questions = toGradeQuestions(bank);
    const keyByNumber = new Map(key.questions.map((q) => [String(q.questionNumber), q]));

    console.log(`\n======== ${year}  ${questions.length} questions  expected ${key.expectedEarned}/${key.expectedTotal} (${key.expectedPercent}%) ========`);

    const t0 = Date.now();
    const raster = await rasterizePdfPages(fs.readFileSync(pdfPath), 1, 40, `${stem}.pdf`);
    console.log(`  rasterized ${raster.pages.length}/${raster.totalPages} pages (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
    if (raster.pages.length === 0) throw new Error(`${stem}.pdf produced no pages`);

    const { graded, extraction } = await gradeHandwritten(
      'Computer Science',
      questions,
      raster.pages,
      false,
    );
    const report = buildReport(graded, 'handwritten', extraction.visionModel, extraction);

    const perQuestion = questions.map((question) => {
      const g = graded.find((item) => item.id === question.id);
      const intended = keyByNumber.get(question.questionNumber);
      const wanted = (intended?.parts ?? []).map((p) => p.studentAnswer).filter(Boolean).join('\n');
      const got = g?.extractedAnswer || Object.values(g?.extractedParts ?? {}).join('\n');
      const usable = schemeIsUsable(schemeText(question));
      return {
        questionNumber: question.questionNumber,
        max: g?.max ?? question.maxMarks,
        earned: g?.earned ?? 0,
        expectedMarks: (intended?.parts ?? []).reduce((s, p) => s + (p.expectedMarks || 0), 0),
        verdict: g?.verdict,
        extractionFlag: g?.extractionFlag,
        marksWithheld: Boolean(g?.marksWithheld),
        gradingFailed: Boolean(g?.gradingFailed),
        schemeUsed: g?.schemeUsed,
        schemeShouldBeUsed: usable,
        mapping: coverage(got, wanted),
        extractedPreview: (got || '').slice(0, 220),
        intendedPreview: wanted.slice(0, 160),
        feedback: (g?.feedback || '').slice(0, 240),
        pages: g?.extractionPages ?? [],
      };
    });

    const mapped = perQuestion.filter((q) => q.extractionFlag === 'ok' || q.extractionFlag === 'low_confidence').length;
    const schemeOk = perQuestion.filter((q) => q.schemeUsed === q.schemeShouldBeUsed).length;
    const deltaPct = report.percent - key.expectedPercent;
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

    const row = {
      year,
      pages: raster.pages.length,
      questions: questions.length,
      elapsedSec: Number(elapsed),
      expected: `${key.expectedEarned}/${key.expectedTotal} (${key.expectedPercent}%)`,
      got: `${report.earned}/${report.total} (${report.percent}%)`,
      grade: report.grade,
      deltaPct,
      paperMismatch: extraction.paperMismatch,
      readCount: extraction.readCount,
      lowConfidenceCount: extraction.lowConfidenceCount,
      unreadableCount: extraction.unreadableCount,
      blankCount: extraction.blankCount,
      notFoundCount: extraction.notFoundCount,
      withheldMarks: extraction.withheldMarks,
      mapped,
      schemeAlignment: `${schemeOk}/${perQuestion.length}`,
      warnings: extraction.warnings,
      summary: report.summary,
      perQuestion,
    };
    summary.push(row);
    fs.writeFileSync(path.join(OUT_DIR, `${year}.json`), JSON.stringify(row, null, 2));

    console.log(`  score     ${row.got}  vs expected ${row.expected}  Δ${deltaPct >= 0 ? '+' : ''}${deltaPct}pp`);
    console.log(`  extract   read=${extraction.readCount} low=${extraction.lowConfidenceCount} unread=${extraction.unreadableCount} blank=${extraction.blankCount} missing=${extraction.notFoundCount} mismatch=${extraction.paperMismatch}`);
    console.log(`  mapped    ${mapped}/${questions.length}   scheme-mode ${schemeOk}/${perQuestion.length}   ${elapsed}s`);
    if (extraction.warnings.length) console.log(`  warnings  ${extraction.warnings.join(' | ')}`);
    for (const q of perQuestion) {
      const flag = q.marksWithheld ? 'WITHHELD' : String(q.verdict);
      const scheme = q.schemeUsed ? 'scheme' : 'judgement';
      console.log(`    Q${q.questionNumber.padEnd(4)} ${String(q.earned).padStart(2)}/${q.max}  ${flag.padEnd(12)} ${scheme.padEnd(10)} map=${q.mapping.toFixed(2)}  ${q.extractionFlag}`);
    }
  }

  const all = YEARS.map((year) => {
    const file = path.join(OUT_DIR, `${year}.json`);
    return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) as Record<string, unknown> : null;
  }).filter((row): row is Record<string, unknown> => Boolean(row));
  fs.writeFileSync(path.join(OUT_DIR, 'summary.json'), JSON.stringify(all, null, 2));
  console.log('\n======== SUMMARY ========');
  for (const row of all) {
    console.log(`  ${row.year}  ${row.got}  expected ${row.expected}  mapped ${row.mapped}/${row.questions}  mismatch=${row.paperMismatch}`);
  }
  console.log(`\nWrote ${OUT_DIR}`);
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
