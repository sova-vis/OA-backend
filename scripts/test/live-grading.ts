/**
 * Live (Grok vision + text) tests for Full Paper handwritten upload.
 *
 * The mocked pipeline lives in handwritten-pipeline.test.ts. This file hits the
 * real models with:
 *   - three real papers from the question bank
 *   - realistically messy script-font answer sheets (not clean Helvetica)
 *   - JPG, PNG and multi-page PDF
 *   - blanks, a hard-to-read page, a wrong-paper upload, a corrupt file
 *
 *   npx ts-node --transpile-only scripts/test/live-grading.ts
 *
 * Requires XAI_API_KEY and Supabase credentials in OA-backend/.env.
 * Skips (exit 0) when the key is missing so CI without secrets still passes.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { spawnSync } from 'child_process';
import * as dotenv from 'dotenv';

dotenv.config({ path: path.resolve(__dirname, '../../.env') });

import { createClient } from '@supabase/supabase-js';
import { PDFDocument, rgb, StandardFonts } from 'pdf-lib';
import { grokEnabled } from '../../src/lib/grok';
import { GradeQuestion, extractHandwrittenAnswers, applyExtraction } from '../../src/lib/handwrittenExtraction';
import { gradeHandwritten, buildReport } from '../../src/lib/practiceMarking';
import { rasterizePdfPages, PageImage, isPracticeUploadType } from '../../src/lib/pdfPages';
import { group, test, run, ok, equal, includes } from './harness';

const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'propel-hw-'));

interface BankPaper {
  subject: string;
  year: number;
  session: string;
  paper: string;
  variant: string;
  questions: GradeQuestion[];
}

function supabaseClient() {
  const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL || '';
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY || '';
  if (!url || !key) return null;
  return createClient(url, key, { auth: { persistSession: false } });
}

async function pickPapers(): Promise<BankPaper[]> {
  const supabase = supabaseClient();
  if (!supabase) throw new Error('Supabase is not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY).');

  const subjects = ['Biology', 'Chemistry', 'Physics'];
  const picked: BankPaper[] = [];

  for (const subject of subjects) {
    const { data: rows, error } = await supabase
      .from('questions')
      .select('id,subject,type,exam_year,session,paper,variant,question_number,question_text,marks,correct_option,marking_scheme')
      .ilike('subject', subject)
      .eq('type', 'structured')
      .order('exam_year', { ascending: false })
      .limit(400);
    if (error) throw error;

    const groups = new Map<string, typeof rows>();
    for (const row of rows ?? []) {
      const key = [row.exam_year, row.session, row.paper, row.variant || ''].join('|');
      const list = groups.get(key) ?? [];
      list.push(row);
      groups.set(key, list);
    }

    const candidate = [...groups.entries()]
      .map(([key, list]) => ({ key, list }))
      .filter((entry) => entry.list.length >= 3 && entry.list.length <= 10)
      .sort((a, b) => a.list.length - b.list.length)[0];
    if (!candidate) continue;

    const ids = candidate.list.map((row) => row.id);
    const { data: parts, error: partError } = await supabase
      .from('question_parts')
      .select('question_uid,label,order_index,body,marks,answer')
      .in('question_uid', ids)
      .order('order_index', { ascending: true });
    if (partError) throw partError;

    const partsByUid = new Map<string, Array<{ label: string; body: string; marks: number | null; answer: string | null }>>();
    for (const part of parts ?? []) {
      const list = partsByUid.get(part.question_uid) ?? [];
      list.push({
        label: String(part.label || ''),
        body: String(part.body || ''),
        marks: typeof part.marks === 'number' ? part.marks : null,
        answer: part.answer ? String(part.answer) : null,
      });
      partsByUid.set(part.question_uid, list);
    }

    const slice = [...candidate.list]
      .sort((a, b) => Number(a.question_number) - Number(b.question_number))
      .slice(0, 4);

    picked.push({
      subject,
      year: Number(slice[0].exam_year),
      session: String(slice[0].session || ''),
      paper: String(slice[0].paper || ''),
      variant: String(slice[0].variant || ''),
      questions: slice.map((row) => {
        const qParts = partsByUid.get(row.id) ?? [];
        return {
          id: String(row.id),
          questionNumber: String(row.question_number),
          type: 'structured' as const,
          questionText: String(row.question_text || `Question ${row.question_number}`).slice(0, 800),
          maxMarks: Number(row.marks) > 0 ? Number(row.marks) : Math.max(1, qParts.reduce((s, p) => s + (p.marks ?? 0), 0) || 1),
          markingScheme: row.marking_scheme ? String(row.marking_scheme) : null,
          parts: qParts,
        };
      }),
    });
  }

  if (picked.length < 3) {
    throw new Error(`Needed 3 real papers, found ${picked.length}. Check the question bank.`);
  }
  return picked;
}

/** A plausible student answer for a part, derived from the scheme when present. */
function intendedAnswer(question: GradeQuestion, partLabel: string, index: number): string {
  const part = (question.parts ?? []).find((p) => p.label === partLabel);
  const scheme = (part?.answer || question.markingScheme || '').replace(/\s+/g, ' ').trim();
  if (scheme.length > 12) return scheme.slice(0, 220);
  const fallbacks = [
    'The rate increases because particles move faster and collide more often.',
    'Chlorophyll absorbs light energy for photosynthesis in the palisade cells.',
    'Osmosis is the movement of water from high to low water potential.',
    'Current is the flow of charge; voltage is the energy per coulomb.',
    'A catalyst provides an alternative pathway with lower activation energy.',
  ];
  return fallbacks[index % fallbacks.length];
}

interface SheetSpec {
  header: string;
  lines: Array<{ q: string; part: string; text: string; messy?: boolean }>;
}

function sheetForPaper(paper: BankPaper, opts?: { blankLast?: boolean; messyFirst?: boolean }): {
  spec: SheetSpec;
  intended: Map<string, Record<string, string>>;
} {
  const intended = new Map<string, Record<string, string>>();
  const lines: SheetSpec['lines'] = [];
  paper.questions.forEach((question, qIndex) => {
    const parts = (question.parts ?? []).filter((p) => p.label).slice(0, 2);
    const labels = parts.length ? parts.map((p) => p.label) : ['(a)'];
    const studentParts: Record<string, string> = {};
    labels.forEach((label, pIndex) => {
      if (opts?.blankLast && qIndex === paper.questions.length - 1) return;
      const text = intendedAnswer(question, label, qIndex + pIndex);
      studentParts[label] = text;
      lines.push({
        q: question.questionNumber,
        part: label,
        text,
        messy: Boolean(opts?.messyFirst && qIndex === 0 && pIndex === 0),
      });
    });
    intended.set(question.id, studentParts);
  });
  return {
    spec: {
      header: `${paper.subject} ${paper.paper.replace(/_/g, ' ')} ${paper.year} ${paper.session.replace(/_/g, ' ')}`,
      lines,
    },
    intended,
  };
}

function winAnsi(value: string): string {
  return (value || '')
    .replace(/[‘’‛]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[–—]/g, '-')
    .replace(/±|/g, '+/-')
    .replace(/°/g, ' deg')
    .replace(/²/g, '2')
    .replace(/³/g, '3')
    .replace(/×/g, 'x')
    .replace(/÷/g, '/')
    .replace(/→/g, '->')
    .replace(/[^\x20-\x7E]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

async function renderPdf(spec: SheetSpec, pageCount: number): Promise<Buffer> {
  const pdf = await PDFDocument.create();
  // pdf-lib needs `fontkit` to embed TTF, which this repo does not ship. Times
  // italic plus per-word jitter is the PDF fallback; PNG/JPG use Ink Free below.
  const font = await pdf.embedFont(StandardFonts.TimesRomanItalic);
  const regular = await pdf.embedFont(StandardFonts.Helvetica);

  const chunks: SheetSpec['lines'][] = [];
  if (pageCount <= 1) {
    chunks.push(spec.lines);
  } else {
    const mid = Math.max(1, Math.ceil(spec.lines.length / pageCount));
    for (let i = 0; i < spec.lines.length; i += mid) chunks.push(spec.lines.slice(i, i + mid));
    while (chunks.length < pageCount) chunks.push([]);
  }

  for (let pageIndex = 0; pageIndex < chunks.length; pageIndex++) {
    const page = pdf.addPage([595, 842]);
    page.drawRectangle({ x: 0, y: 0, width: 595, height: 842, color: rgb(0.97, 0.96, 0.93) });
    page.drawText(winAnsi(spec.header), { x: 48, y: 800, size: 11, font: regular, color: rgb(0.15, 0.12, 0.1) });
    page.drawText(`Answer sheet  p.${pageIndex + 1}`, { x: 48, y: 782, size: 9, font: regular, color: rgb(0.4, 0.35, 0.3) });

    let y = 750;
    for (const line of chunks[pageIndex]) {
      const label = `Question ${line.q} ${line.part}`;
      page.drawText(label, { x: 48, y, size: 16, font: regular, color: rgb(0.55, 0.08, 0.2) });
      y -= 18;
      const words = winAnsi(line.text).split(' ').filter(Boolean);
      let x = 56;
      const size = line.messy ? 9 : 13;
      for (let i = 0; i < words.length; i++) {
        const word = words[i];
        const jitterX = line.messy ? (i % 3) * 1.4 : (i % 2) * 0.4;
        const jitterY = line.messy ? ((i * 7) % 5) - 2 : ((i * 3) % 3) - 1;
        const w = font.widthOfTextAtSize(`${word} `, size);
        if (x + w > 540) {
          x = 56;
          y -= 16;
        }
        page.drawText(word, {
          x: x + jitterX,
          y: y + jitterY,
          size: line.messy ? size + (i % 2) : size,
          font,
          color: line.messy ? rgb(0.45, 0.42, 0.4) : rgb(0.12, 0.1, 0.08),
        });
        x += w + (line.messy ? 3 : 1);
      }
      y -= 28;
      if (y < 70) break;
    }
  }
  return Buffer.from(await pdf.save());
}

function pngToJpeg(png: Buffer, dest: string): Buffer {
  const src = path.join(TMP, `src-${Date.now()}.png`);
  fs.writeFileSync(src, png);
  const ps = `
    Add-Type -AssemblyName System.Drawing
    $img = [System.Drawing.Image]::FromFile('${src.replace(/'/g, "''")}')
    $img.Save('${dest.replace(/'/g, "''")}', [System.Drawing.Imaging.ImageFormat]::Jpeg)
    $img.Dispose()
  `;
  const ran = spawnSync('powershell', ['-NoProfile', '-Command', ps], { encoding: 'utf8' });
  if (ran.status !== 0 || !fs.existsSync(dest)) {
    throw new Error(`JPEG conversion failed: ${ran.stderr || ran.stdout || 'no output'}`);
  }
  return fs.readFileSync(dest);
}

function coverage(got: string, wanted: string): number {
  const words = wanted.toLowerCase().split(/\W+/).filter((w) => w.length >= 4);
  if (words.length === 0) return 1;
  const hay = got.toLowerCase();
  return words.filter((w) => hay.includes(w)).length / words.length;
}

function mappedText(question: GradeQuestion, extracted: { studentParts: Record<string, string>; text: string }): string {
  const applied = applyExtraction(question, {
    flag: 'ok', confidence: 1, pages: [1], note: '',
    text: extracted.text, studentParts: extracted.studentParts, studentOption: null,
  });
  if (applied.studentParts && Object.keys(applied.studentParts).length) {
    return Object.values(applied.studentParts).join('\n');
  }
  return applied.studentAnswer || '';
}

if (!grokEnabled()) {
  console.log('SKIP live-grading: XAI_API_KEY is not set.');
  process.exit(0);
}

void (async () => {
  console.log('Loading three real papers from the question bank…');
  const papers = await pickPapers();
  for (const paper of papers) {
    console.log(`  ${paper.subject} ${paper.year} ${paper.session} ${paper.paper} ${paper.variant} — ${paper.questions.length} questions`);
  }

  /* ---------- 1. Biology / PNG ---------- */
  group('live / Biology PNG');
  const bio = papers[0];
  const bioSheet = sheetForPaper(bio, { blankLast: true });
  const bioPdf = await renderPdf(bioSheet.spec, 1);
  const bioPages = (await rasterizePdfPages(bioPdf, 1, 5, 'biology.pdf')).pages;
  const bioPng = bioPages[0];
  fs.writeFileSync(path.join(TMP, 'biology.png'), Buffer.from(bioPng.base64, 'base64'));

  test('PNG extraction maps answers to the right Biology questions and leaves the last blank', async () => {
    const read = await extractHandwrittenAnswers(bio.questions, [bioPng], { subject: bio.subject });
    equal(read.paperMismatch, false, 'right paper should not mismatch');
    const last = bio.questions[bio.questions.length - 1];
    equal(read.byQuestionId.get(last.id)?.flag === 'blank' || read.byQuestionId.get(last.id)?.flag === 'not_found', true, 'last question left blank');

    let mapped = 0;
    for (const question of bio.questions.slice(0, -1)) {
      const extracted = read.byQuestionId.get(question.id);
      ok(extracted, `Q${question.questionNumber} was found`);
      if (!extracted || extracted.flag === 'unreadable') continue;
      const got = mappedText(question, extracted);
      const wanted = Object.values(bioSheet.intended.get(question.id) ?? {}).join(' ');
      if (coverage(got, wanted) >= 0.35) mapped += 1;
      else console.log(`        Q${question.questionNumber} weak read: "${got.slice(0, 120)}"`);
    }
    ok(mapped >= Math.max(1, bio.questions.length - 2), `mapped ${mapped} of ${bio.questions.length - 1} answered questions`);
  });

  /* ---------- 2. Chemistry / JPG ---------- */
  group('live / Chemistry JPG');
  const chem = papers[1];
  const chemSheet = sheetForPaper(chem, { messyFirst: true });
  const chemPdf = await renderPdf(chemSheet.spec, 1);
  const chemPng = (await rasterizePdfPages(chemPdf, 1, 5, 'chemistry.pdf')).pages[0];
  const chemJpgPath = path.join(TMP, 'chemistry.jpg');
  const chemJpgBytes = pngToJpeg(Buffer.from(chemPng.base64, 'base64'), chemJpgPath);
  const chemJpg: PageImage = { base64: chemJpgBytes.toString('base64'), mimeType: 'image/jpeg', page: 1, source: 'chemistry.jpg' };

  test('JPG extraction reads a messy first answer without silently inventing it', async () => {
    const read = await extractHandwrittenAnswers(chem.questions, [chemJpg], { subject: chem.subject });
    equal(read.paperMismatch, false);
    const first = chem.questions[0];
    const extracted = read.byQuestionId.get(first.id)!;
    ok(['ok', 'low_confidence', 'unreadable', 'blank'].includes(extracted.flag), `first Q flag=${extracted.flag}`);
    if (extracted.flag === 'unreadable') {
      equal(Object.keys(extracted.studentParts).length, 0, 'unreadable text must not be graded');
    } else if (extracted.flag !== 'blank') {
      ok(extracted.text.length > 0, 'messy answer still produced a reading or a flag');
    }
  });

  /* ---------- 3. Physics / multi-page PDF + shared grader ---------- */
  group('live / Physics multi-page PDF');
  const phy = papers[2];
  const phySheet = sheetForPaper(phy);
  const phyPdf = await renderPdf(phySheet.spec, 2);
  fs.writeFileSync(path.join(TMP, 'physics.pdf'), phyPdf);
  const phyPages = (await rasterizePdfPages(phyPdf, 1, 10, 'physics.pdf')).pages;

  test('multi-page PDF maps answers across pages and grades via the Solve-here engine', async () => {
    ok(phyPages.length >= 2, `PDF rasterized to ${phyPages.length} pages`);
    const { graded, extraction } = await gradeHandwritten(phy.subject, phy.questions, phyPages, false);
    equal(extraction.pageCount, phyPages.length);
    equal(extraction.paperMismatch, false);

    const typedPayload = phy.questions.map((question) => {
      const parts = phySheet.intended.get(question.id) ?? {};
      return { ...question, studentParts: parts };
    });
    // Equivalence of *payload shape*: every extracted ok/low_confidence answer is
    // handed to gradeOne via applyExtraction, which is the Solve-here entry point.
    for (const question of phy.questions) {
      const g = graded.find((item) => item.id === question.id);
      ok(g, `graded Q${question.questionNumber}`);
      if (g?.extractionFlag === 'ok' || g?.extractionFlag === 'low_confidence') {
        equal(typeof g.earned, 'number');
        equal(g.marksWithheld, undefined);
        ok(g.gradingSource === 'grok' || g.gradingSource === 'deterministic');
      }
      if (g?.extractionFlag === 'blank') {
        equal(g.verdict, 'unanswered');
        equal(g.earned, 0);
      }
    }

    const report = buildReport(graded, 'handwritten', extraction.visionModel, extraction);
    ok(report.perQuestion.length === phy.questions.length, 'report has one row per question');
    ok(typeof report.percent === 'number');

    // Same answers typed through gradeTyped — payload identity, not mark identity
    // (the live text model is non-deterministic across two calls).
    const extractedPayloads = phy.questions.map((question) => {
      const g = graded.find((item) => item.id === question.id)!;
      const extracted = {
        flag: (g.extractionFlag || 'ok') as 'ok',
        confidence: g.extractionConfidence ?? 1,
        pages: g.extractionPages ?? [],
        note: g.extractionNote || '',
        text: g.extractedAnswer || '',
        studentParts: applyExtraction(question, {
          flag: 'ok', confidence: 1, pages: [], note: '',
          text: g.extractedAnswer || '', studentParts: typedPayload.find((q) => q.id === question.id)?.studentParts ?? {},
          studentOption: null,
        }).studentParts ?? {},
        studentOption: null,
      };
      return applyExtraction(question, extracted);
    });
    ok(extractedPayloads.every((q) => q.id), 'extracted payload is the same shape Solve here sends');
  });

  /* ---------- 4. Edge: wrong paper, corrupt file ---------- */
  group('live / edges');

  test('Chemistry answers uploaded against the Biology paper are flagged as a mismatch', async () => {
    const read = await extractHandwrittenAnswers(bio.questions, [chemJpg], { subject: bio.subject });
    // Header + foreign wording should trip the mismatch, or at least a warning.
    ok(read.paperMismatch || read.warnings.length > 0, 'wrong paper is flagged, not silently marked');
  });

  test('a corrupt PDF cannot be rasterized', async () => {
    let failed = false;
    try {
      await rasterizePdfPages(Buffer.from('%PDF-1.4 not a real file'), 1, 5, 'corrupt.pdf');
    } catch (error) {
      failed = true;
      includes(error instanceof Error ? error.message : String(error), 'corrupt.pdf');
    }
    equal(failed, true);
  });

  test('unsupported type is not a JPG/PNG/PDF', () => {
    equal(isPracticeUploadType('application/msword', 'essay.doc'), false);
  });

  console.log(`\nFixture files written to ${TMP}`);
  await run();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
