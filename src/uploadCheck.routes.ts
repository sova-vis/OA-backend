import { Router, Response } from 'express';
import multer from 'multer';
import { PDFDocument, rgb, StandardFonts, PDFFont, PDFPage } from 'pdf-lib';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import { PRACTICE_BUCKET, ensureBucket, SIGNED_URL_TTL } from './lib/practiceStore';
import {
  grokEnabled, grokChatJson, grokVisionModel, grokErrorMessage, GrokError, GrokImage,
} from './lib/grok';

/**
 * Standalone "upload & mark" flow (separate from the question-bank practice).
 *
 *   POST /upload-check   (Clerk auth, multipart)
 *     fields: subject, paperType (topical|questions|mcqs|mixed)
 *     files:  one or more images, and/or a PDF of the solved work
 *
 * Pipeline: rasterize any PDF pages -> Grok vision reads + grades every page
 * against expert O/A-Level knowledge -> an annotated PDF is produced (each page
 * with examiner notes + a marking cover page) -> stored privately, report
 * returned with a signed URL. Nothing here is tied to a specific bank paper.
 */

const MAX_FILE_BYTES = 20 * 1024 * 1024;
const MAX_PAGES = 12;
const PAGE_W = 595; // A4 points

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: MAX_FILE_BYTES, files: 12 } });
const router = Router();

// Preserve a real ESM import in the CJS build (mupdf ships as ESM w/ top-level await).
const importEsm = new Function('m', 'return import(m)') as (m: string) => Promise<any>;
let mupdfPromise: Promise<any> | null = null;
function loadMupdf(): Promise<any> {
  if (!mupdfPromise) mupdfPromise = importEsm('mupdf');
  return mupdfPromise;
}

export type PaperType = 'topical' | 'questions' | 'mcqs' | 'mixed';
const PAPER_TYPES: PaperType[] = ['topical', 'questions', 'mcqs', 'mixed'];

interface PageImage {
  base64: string;
  mimeType: string; // image/png | image/jpeg
  page: number;
}

interface UploadGradedQuestion {
  questionNumber: string;
  page: number;
  question: string;
  studentAnswer: string;
  verdict: 'correct' | 'partial' | 'weak' | 'unanswered';
  earned: number;
  max: number;
  correctAnswer: string;
  corrections: string[];
  suggestions: string[];
  annotation: string;
}

export interface UploadCheckReport {
  subject: string;
  paperType: PaperType;
  earned: number;
  total: number;
  percent: number;
  grade: string;
  summary: string;
  shortcomings: string[];
  suggestions: string[];
  questions: UploadGradedQuestion[];
  pageCount: number;
  model: string;
  gradedAt: string;
}

/* ---------------- helpers ---------------- */

function gradeBand(percent: number): string {
  if (percent >= 90) return 'A* (indicative)';
  if (percent >= 80) return 'A (indicative)';
  if (percent >= 70) return 'B (indicative)';
  if (percent >= 60) return 'C (indicative)';
  if (percent >= 50) return 'D (indicative)';
  if (percent >= 40) return 'E (indicative)';
  return 'U (indicative)';
}

function asStringArray(value: unknown, max = 8): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item).trim()).filter(Boolean).slice(0, max);
}

function clampNum(value: unknown, min: number, max: number): number {
  const n = typeof value === 'number' ? value : Number.parseInt(String(value), 10);
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, Math.round(n)));
}

function verdictOf(value: unknown): UploadGradedQuestion['verdict'] {
  const v = String(value || '').toLowerCase();
  return v === 'correct' || v === 'partial' || v === 'weak' || v === 'unanswered' ? v : 'weak';
}

/** Rasterize a PDF (buffer) to page PNGs at ~2x via mupdf. */
export async function rasterizePdf(buffer: Buffer, startPage: number, budget: number): Promise<PageImage[]> {
  const mupdf = await loadMupdf();
  const doc = mupdf.Document.openDocument(new Uint8Array(buffer), 'application/pdf');
  const count = Math.min(doc.countPages(), budget);
  const scale = mupdf.Matrix.scale(2, 2);
  const out: PageImage[] = [];
  for (let i = 0; i < count; i++) {
    const page = doc.loadPage(i);
    const pixmap = page.toPixmap(scale, mupdf.ColorSpace.DeviceRGB, false, true);
    const png = pixmap.asPNG();
    out.push({ base64: Buffer.from(png).toString('base64'), mimeType: 'image/png', page: startPage + i });
  }
  return out;
}

/** Turn uploaded files (images + PDFs) into an ordered list of page images. */
async function filesToPages(files: Express.Multer.File[]): Promise<PageImage[]> {
  const pages: PageImage[] = [];
  for (const file of files) {
    if (pages.length >= MAX_PAGES) break;
    const budget = MAX_PAGES - pages.length;
    if (file.mimetype === 'application/pdf') {
      pages.push(...(await rasterizePdf(file.buffer, pages.length + 1, budget)));
    } else if (/^image\/(png|jpe?g)$/.test(file.mimetype)) {
      pages.push({ base64: file.buffer.toString('base64'), mimeType: file.mimetype, page: pages.length + 1 });
    } else if (/^image\//.test(file.mimetype)) {
      // webp/gif/etc: gradable by Grok but not embeddable by pdf-lib; keep for OCR
      pages.push({ base64: file.buffer.toString('base64'), mimeType: file.mimetype, page: pages.length + 1 });
    }
  }
  return pages;
}

/* ---------------- Grok grading ---------------- */

function systemPrompt(paperType: PaperType): string {
  const typeHint =
    paperType === 'mcqs'
      ? 'This is a multiple-choice paper. For each question, read the option the student selected, decide the correct option, and mark 1 for correct else 0.'
      : paperType === 'topical'
        ? 'This is a set of topical practice questions. Grade each written answer.'
        : paperType === 'mixed'
          ? 'This is a mixed paper (MCQs and written questions). Grade each accordingly.'
          : 'These are structured/written exam questions. Grade each written answer.';
  return [
    'You are a strict but fair Cambridge O/A Level examiner marking a student\'s scanned/handwritten attempt.',
    typeHint,
    'Read the handwriting on every page image (pages are given in order). For each question: extract the student answer, decide a verdict, award whole marks out of a sensible max, give the correct/expected answer, and concrete corrections + suggestions.',
    'Also produce one short "annotation" per question (max ~90 chars) — the note an examiner would write in the margin.',
    'Grade using your own expert subject knowledge and standard Cambridge mark-scheme conventions.',
    'Return JSON ONLY with this shape:',
    '{ "overall": { "summary": string, "shortcomings": string[], "suggestions": string[] },',
    '  "questions": [ { "question_number": string, "page": number, "question": string, "student_answer": string,',
    '    "verdict": "correct"|"partial"|"weak"|"unanswered", "earned_marks": number, "max_marks": number,',
    '    "correct_answer": string, "corrections": string[], "suggestions": string[], "annotation": string } ] }',
  ].join(' ');
}

async function gradeUpload(subject: string, paperType: PaperType, pages: PageImage[]): Promise<UploadCheckReport> {
  const images: GrokImage[] = pages.map((p) => ({ base64: p.base64, mimeType: p.mimeType }));
  const user = `Subject: ${subject}\nPaper type: ${paperType}\nThere are ${pages.length} page image(s), in order (page 1 to ${pages.length}). Mark the whole attempt.`;
  const parsed = await grokChatJson({
    system: systemPrompt(paperType), user, images,
    model: grokVisionModel(), temperature: 0, maxTokens: 4000, timeoutMs: 120_000,
  });

  const rawQuestions = Array.isArray(parsed.questions) ? (parsed.questions as Array<Record<string, unknown>>) : [];
  const questions: UploadGradedQuestion[] = rawQuestions.slice(0, 60).map((q, index) => {
    const max = clampNum(q.max_marks, 1, 100);
    const earned = clampNum(q.earned_marks, 0, max);
    return {
      questionNumber: String(q.question_number ?? index + 1).slice(0, 12),
      page: clampNum(q.page, 1, pages.length || 1),
      question: String(q.question ?? '').slice(0, 400),
      studentAnswer: String(q.student_answer ?? '').slice(0, 600),
      verdict: verdictOf(q.verdict),
      earned, max,
      correctAnswer: String(q.correct_answer ?? '').slice(0, 600),
      corrections: asStringArray(q.corrections),
      suggestions: asStringArray(q.suggestions),
      annotation: String(q.annotation ?? '').slice(0, 140),
    };
  });

  const earned = questions.reduce((s, q) => s + q.earned, 0);
  const total = questions.reduce((s, q) => s + q.max, 0);
  const percent = total > 0 ? Math.round((earned / total) * 100) : 0;
  const overall = (parsed.overall ?? {}) as Record<string, unknown>;

  return {
    subject, paperType, earned, total, percent, grade: gradeBand(percent),
    summary: String(overall.summary ?? 'Marked.').slice(0, 800),
    shortcomings: asStringArray(overall.shortcomings),
    suggestions: asStringArray(overall.suggestions),
    questions, pageCount: pages.length, model: grokVisionModel(), gradedAt: new Date().toISOString(),
  };
}

/* ---------------- annotated PDF ---------------- */

// pdf-lib's standard Helvetica is Latin-1 only; transliterate common symbols in
// the examiner notes and drop anything else so drawing never throws.
function sanitize(text: string): string {
  return (text || '')
    .replace(/[‘’‛]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[–—]/g, '-')
    .replace(/→/g, '->').replace(/←/g, '<-').replace(/⇒/g, '=>')
    .replace(/≥/g, '>=').replace(/≤/g, '<=').replace(/≠/g, '!=').replace(/≈/g, '~')
    .replace(/×/g, 'x').replace(/÷/g, '/')
    .replace(/°/g, ' deg').replace(/²/g, '2').replace(/³/g, '3')
    .replace(/[…]/g, '...').replace(/[•]/g, '-')
    .replace(/[^\x09\x0A\x0D\x20-\x7E\xA0-\xFF]/g, '');
}

function wrapText(font: PDFFont, raw: string, size: number, maxWidth: number): string[] {
  const text = sanitize(raw);
  const words = text.replace(/\s+/g, ' ').trim().split(' ');
  const lines: string[] = [];
  let line = '';
  for (const word of words) {
    const attempt = line ? `${line} ${word}` : word;
    if (font.widthOfTextAtSize(attempt, size) > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = attempt;
    }
  }
  if (line) lines.push(line);
  return lines;
}

const VERDICT_RGB: Record<UploadGradedQuestion['verdict'], [number, number, number]> = {
  correct: [0.06, 0.48, 0.36],
  partial: [0.6, 0.42, 0],
  weak: [0.69, 0.16, 0.1],
  unanswered: [0.42, 0.37, 0.33],
};

export async function buildAnnotatedPdf(report: UploadCheckReport, pages: PageImage[]): Promise<Uint8Array> {
  const pdf = await PDFDocument.create();
  const font = await pdf.embedFont(StandardFonts.Helvetica);
  const bold = await pdf.embedFont(StandardFonts.HelveticaBold);

  /* --- cover / marking summary page --- */
  const cover = pdf.addPage([PAGE_W, 842]);
  cover.drawRectangle({ x: 0, y: 742, width: PAGE_W, height: 100, color: rgb(0.66, 0.07, 0.24) });
  cover.drawText('Marking report', { x: 40, y: 800, size: 22, font: bold, color: rgb(1, 1, 1) });
  cover.drawText(sanitize(`${report.subject} · ${report.paperType} · ${new Date(report.gradedAt).toLocaleString()}`),
    { x: 40, y: 774, size: 11, font, color: rgb(1, 1, 1) });

  let y = 700;
  cover.drawText(`${report.earned} / ${report.total}`, { x: 40, y, size: 30, font: bold, color: rgb(0.66, 0.07, 0.24) });
  cover.drawText(sanitize(`${report.percent}%  ·  ${report.grade}`), { x: 210, y: y + 8, size: 15, font: bold, color: rgb(0.1, 0.1, 0.1) });
  y -= 34;
  for (const line of wrapText(font, report.summary, 11, PAGE_W - 80)) {
    cover.drawText(line, { x: 40, y, size: 11, font, color: rgb(0.15, 0.13, 0.11) });
    y -= 16;
  }

  const section = (title: string, items: string[]) => {
    if (items.length === 0) return;
    y -= 14;
    cover.drawText(title, { x: 40, y, size: 12.5, font: bold, color: rgb(0.66, 0.07, 0.24) });
    y -= 18;
    items.forEach((item) => {
      const lines = wrapText(font, `•  ${item}`, 10.5, PAGE_W - 90);
      lines.forEach((line, li) => {
        cover.drawText(line, { x: li === 0 ? 46 : 56, y, size: 10.5, font, color: rgb(0.2, 0.18, 0.16) });
        y -= 14.5;
      });
    });
  };
  section('Where you lost marks', report.shortcomings);
  section('How to improve', report.suggestions);

  /* --- one annotated page per uploaded page --- */
  const byPage = new Map<number, UploadGradedQuestion[]>();
  for (const q of report.questions) {
    const list = byPage.get(q.page) ?? [];
    list.push(q);
    byPage.set(q.page, list);
  }

  for (const pageImg of pages) {
    const notes = byPage.get(pageImg.page) ?? [];
    let embedded: Awaited<ReturnType<typeof pdf.embedPng>> | null = null;
    try {
      embedded = pageImg.mimeType === 'image/jpeg'
        ? await pdf.embedJpg(Buffer.from(pageImg.base64, 'base64'))
        : await pdf.embedPng(Buffer.from(pageImg.base64, 'base64'));
    } catch {
      embedded = null; // unsupported image (e.g. webp) — render a notes-only page
    }

    const imgW = embedded?.width ?? PAGE_W * 2;
    const imgH = embedded?.height ?? PAGE_W * 2;
    const scale = PAGE_W / imgW;
    const drawH = imgH * scale;
    // examiner panel height grows with the number of note lines
    const panelLines = notes.flatMap((n) =>
      wrapText(font, `Q${n.questionNumber} (${n.earned}/${n.max}) ${n.annotation || n.verdict}`, 9, PAGE_W - 28));
    const panelH = Math.min(230, 30 + panelLines.length * 12);

    const page = pdf.addPage([PAGE_W, drawH + panelH]);
    if (embedded) page.drawImage(embedded, { x: 0, y: panelH, width: PAGE_W, height: drawH });

    // panel background
    page.drawRectangle({ x: 0, y: 0, width: PAGE_W, height: panelH, color: rgb(0.99, 0.96, 0.93) });
    page.drawRectangle({ x: 0, y: panelH - 2, width: PAGE_W, height: 2, color: rgb(0.66, 0.07, 0.24) });
    page.drawText(`Page ${pageImg.page} - examiner notes`, { x: 14, y: panelH - 18, size: 9.5, font: bold, color: rgb(0.66, 0.07, 0.24) });

    let py = panelH - 34;
    for (const n of notes) {
      const [r, g, b] = VERDICT_RGB[n.verdict];
      const head = `Q${n.questionNumber} (${n.earned}/${n.max}) - ${n.annotation || n.verdict}`;
      for (const line of wrapText(font, head, 9, PAGE_W - 28)) {
        if (py < 8) break;
        page.drawText(line, { x: 14, y: py, size: 9, font, color: rgb(r, g, b) });
        py -= 12;
      }
    }
  }

  return pdf.save();
}

/* ---------------- storage ---------------- */

function checksPrefix(clerkId: string, checkId: string): string {
  return `checks/${clerkId}/${checkId}`;
}

async function storeCheck(clerkId: string, checkId: string, pdfBytes: Uint8Array, report: UploadCheckReport): Promise<string> {
  const prefix = checksPrefix(clerkId, checkId);
  const pdfPath = `${prefix}/annotated.pdf`;
  const up1 = await supabase.storage.from(PRACTICE_BUCKET).upload(pdfPath, Buffer.from(pdfBytes), {
    contentType: 'application/pdf', upsert: true,
  });
  if (up1.error) throw up1.error;
  await supabase.storage.from(PRACTICE_BUCKET).upload(`${prefix}/report.json`, Buffer.from(JSON.stringify({ ...report, checkId }), 'utf8'), {
    contentType: 'application/json', upsert: true,
  });
  const { data } = await supabase.storage.from(PRACTICE_BUCKET).createSignedUrl(pdfPath, SIGNED_URL_TTL);
  return data?.signedUrl ?? '';
}

/* ---------------- routes ---------------- */

router.post('/', clerkAuth, upload.array('files', 12), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const subject = String(req.body?.subject || '').trim().slice(0, 80) || 'General';
    const paperTypeRaw = String(req.body?.paperType || 'questions').toLowerCase();
    const paperType: PaperType = (PAPER_TYPES as string[]).includes(paperTypeRaw) ? (paperTypeRaw as PaperType) : 'questions';

    const files = (req.files as Express.Multer.File[] | undefined) ?? [];
    if (files.length === 0) return res.status(400).json({ error: 'Upload at least one image or PDF of your work.' });
    if (!grokEnabled()) {
      return res.status(503).json({ error: grokErrorMessage(new GrokError('', 'no_key')), code: 'no_key' });
    }

    const pages = await filesToPages(files);
    if (pages.length === 0) return res.status(400).json({ error: 'Could not read any pages from your upload.' });

    const report = await gradeUpload(subject, paperType, pages);
    const pdfBytes = await buildAnnotatedPdf(report, pages);

    await ensureBucket();
    const checkId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const annotatedUrl = await storeCheck(clerkId, checkId, pdfBytes, report);

    return res.json({ checkId, report, annotatedUrl });
  } catch (error) {
    console.error('Upload check error:', error);
    const message = grokErrorMessage(error);
    const status = error instanceof GrokError && (error.code === 'no_key' || error.code === 'invalid_key') ? 503 : 500;
    return res.status(status).json({ error: message });
  }
});

/** GET /upload-check/history — past checks for the student, newest first. */
router.get('/history', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });
    await ensureBucket();

    const { data: folders } = await supabase.storage.from(PRACTICE_BUCKET).list(`checks/${clerkId}`, { limit: 100, sortBy: { column: 'name', order: 'desc' } });
    const items = (
      await Promise.all(
        (folders ?? []).map(async (folder) => {
          const { data } = await supabase.storage.from(PRACTICE_BUCKET).download(`checks/${clerkId}/${folder.name}/report.json`);
          if (!data) return null;
          try {
            const report = JSON.parse(await data.text()) as UploadCheckReport & { checkId: string };
            const { data: signed } = await supabase.storage.from(PRACTICE_BUCKET).createSignedUrl(`checks/${clerkId}/${folder.name}/annotated.pdf`, SIGNED_URL_TTL);
            return { checkId: folder.name, report, annotatedUrl: signed?.signedUrl ?? '' };
          } catch {
            return null;
          }
        }),
      )
    ).filter(Boolean);

    items.sort((a, b) => (b!.report.gradedAt || '').localeCompare(a!.report.gradedAt || ''));
    return res.json({ items });
  } catch (error) {
    console.error('Upload check history error:', error);
    return res.status(500).json({ error: 'Failed to load history' });
  }
});

router.get('/health', (_req, res: Response) => {
  res.json({ status: 'ok', grok_enabled: grokEnabled(), vision_model: grokVisionModel() });
});

export default router;
