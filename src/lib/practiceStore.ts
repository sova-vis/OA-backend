import { supabase } from './supabase';
import { PageImage, isSupportedImageType, rasterizePdfPages } from './pdfPages';
import {
  studentDataSource,
  pgDualWriteEnabled,
  pgUpsertAttempts,
  pgReadAttempts,
  pgUpsertSession,
  pgReadSession,
  warnPgOnce,
} from './studentDataPg';

/**
 * Shared persistence for practice-paper sessions. Progress documents and
 * handwritten uploads live in a private Supabase Storage bucket:
 *
 *   s/<clerkId>/<paperKeySafe>.json          -> PracticeProgressDoc
 *   files/<clerkId>/<paperKeySafe>/<name>    -> handwritten upload binaries
 *
 * paper_key = "Subject|Year|Session|Paper|Variant" in question-bank naming,
 * e.g. "Chemistry|2024|May_June|Paper_2|Variant_1".
 */

export const PRACTICE_BUCKET = 'practice-progress';
export const MAX_DOC_BYTES = 1_500_000;
export const SIGNED_URL_TTL = 60 * 60; // 1h

export type SolveMode = 'digital' | 'handwritten';
export type PracticeStatus = 'in_progress' | 'completed';

export interface PracticeUpload {
  path: string;
  name: string;
  size: number;
  type: string;
  at: string;
  /** short-lived signed URL for viewing (present on API responses, not stored) */
  url?: string;
}

/** Marks split by assessment objective (Phase 1 · Marks Breakdown). */
export interface MarkCategory {
  category: 'Knowledge' | 'Explanation' | 'Evaluation';
  earned: number;
  max: number;
}

/**
 * Outcome of reading one question's answer off an uploaded page.
 *   ok             — read cleanly, graded normally
 *   low_confidence — read, but the model was unsure; graded AND flagged
 *   unreadable     — handwriting present but illegible; NOT graded, marks withheld
 *   blank          — question genuinely left unanswered
 *   not_found      — question number never appeared on any uploaded page
 */
export type ExtractionFlag = 'ok' | 'low_confidence' | 'unreadable' | 'blank' | 'not_found';

export interface GradedQuestion {
  id: string;
  questionNumber: string;
  earned: number;
  max: number;
  verdict: 'correct' | 'partial' | 'weak' | 'unanswered';
  feedback: string;
  expectedPoints: string[];
  missingPoints: string[];
  gradingSource: 'deterministic' | 'grok' | 'grok-vision';
  /** ---- handwritten-upload provenance (absent for typed attempts) ---- */
  /** what the vision pass read for this question, so the student can verify it */
  extractedAnswer?: string;
  /** 0..1 reading confidence reported by the vision pass */
  extractionConfidence?: number;
  extractionFlag?: ExtractionFlag;
  /** 1-based uploaded page numbers this answer was found on */
  extractionPages?: number[];
  /** why the read was imperfect, e.g. "second line unclear" */
  extractionNote?: string;
  /** true when marks were withheld because the answer could not be read */
  marksWithheld?: boolean;
  /** true when scoring failed (API/model error), distinct from unreadable handwriting */
  gradingFailed?: boolean;
  /** true when a marking scheme was matched for this question; false = examiner-judgement fallback */
  schemeUsed?: boolean;
  /** marks earned vs available per assessment objective */
  breakdown?: MarkCategory[];
  /** Phase 3 — examiner intelligence */
  commandWord?: string;      // the question's command word, e.g. "Describe", "Evaluate"
  commandWordNote?: string;  // coaching when the answer's style doesn't match the command word
  examinerNote?: string;     // "Candidates commonly lose marks here because…"
  /** per sub-part awarded marks (earned vs available), labelled to the scheme parts */
  partScores?: { label: string; earned: number; max: number }[];
  /** transcribed sub-parts, same keys the typed "Solve here" flow uses */
  extractedParts?: Record<string, string>;
  /** transcribed MCQ option letter */
  extractedOption?: string | null;
}

/** Summary of the read step, surfaced to the student on handwritten attempts. */
export interface ExtractionSummary {
  pageCount: number;
  /** questions read cleanly */
  readCount: number;
  lowConfidenceCount: number;
  unreadableCount: number;
  blankCount: number;
  notFoundCount: number;
  /** marks not awarded because the answer could not be read */
  withheldMarks: number;
  /** page-quality / wrong-paper / dropped-file warnings, safe to show a student */
  warnings: string[];
  /** true when the upload looks like it belongs to a different paper */
  paperMismatch: boolean;
  visionModel: string;
}

export interface PracticeReport {
  earned: number;
  total: number;
  percent: number;
  grade: string;
  summary: string;
  strengths: string[];
  improvements: string[];
  perQuestion: GradedQuestion[];
  solveMode: SolveMode;
  model: string;
  gradedAt: string;
  /** present only for handwritten attempts */
  extraction?: ExtractionSummary;
}

export interface PracticeProgressDoc {
  paperKey: string;
  subject: string;
  year: string;
  session: string;
  paper: string;
  variant: string;
  isMcq: boolean;
  solveMode: SolveMode;
  status: PracticeStatus;
  answers: { mcq: Record<string, string>; parts: Record<string, string> };
  uploads: PracticeUpload[];
  answeredCount: number;
  totalCount: number;
  timerDurationSeconds: number;
  timerElapsedSeconds: number;
  report: PracticeReport | null;
  startedAt: string;
  updatedAt: string;
}

let bucketReady = false;
export async function ensureBucket(): Promise<void> {
  if (bucketReady) return;
  const { data } = await supabase.storage.getBucket(PRACTICE_BUCKET);
  if (!data) {
    const { error } = await supabase.storage.createBucket(PRACTICE_BUCKET, { public: false });
    if (error && !/already exists/i.test(error.message)) throw error;
  }
  bucketReady = true;
}

export function safeKey(paperKey: string): string {
  return paperKey.replace(/[^A-Za-z0-9._-]/g, '_');
}

export function isValidPaperKey(paperKey: unknown): paperKey is string {
  if (typeof paperKey !== 'string' || paperKey.length === 0 || paperKey.length > 200) return false;
  const segments = paperKey.split('|');
  return segments.length === 5 && segments.every((segment) => segment.trim().length > 0);
}

export function docPath(clerkId: string, paperKey: string): string {
  return `s/${clerkId}/${safeKey(paperKey)}.json`;
}

export function filesPrefix(clerkId: string, paperKey: string): string {
  return `files/${clerkId}/${safeKey(paperKey)}`;
}

export async function readDoc(clerkId: string, paperKey: string): Promise<PracticeProgressDoc | null> {
  // Phase 2: read from Postgres once cut over; fall back to Storage if the row
  // isn't there yet (not backfilled) or Postgres errors.
  if (studentDataSource() === 'postgres') {
    try {
      const fromPg = await pgReadSession(clerkId, paperKey);
      if (fromPg) return fromPg;
    } catch (error) {
      warnPgOnce('practice_sessions read failed, using storage', error);
    }
  }
  const { data, error } = await supabase.storage.from(PRACTICE_BUCKET).download(docPath(clerkId, paperKey));
  if (error || !data) return null;
  try {
    return JSON.parse(await data.text()) as PracticeProgressDoc;
  } catch {
    return null;
  }
}

export async function writeDoc(clerkId: string, doc: PracticeProgressDoc): Promise<void> {
  const payload = Buffer.from(JSON.stringify(doc), 'utf8');
  if (payload.byteLength > MAX_DOC_BYTES) throw new Error('Progress document too large');
  const { error } = await supabase.storage
    .from(PRACTICE_BUCKET)
    .upload(docPath(clerkId, doc.paperKey), payload, { contentType: 'application/json', upsert: true });
  if (error) throw error;
  // Phase 2: mirror the session (and its graded report) into Postgres.
  // Best-effort — a Postgres failure (e.g. migration 005 not yet run) must never
  // fail the student's save, which already succeeded in Storage above.
  if (pgDualWriteEnabled()) {
    try {
      await pgUpsertSession(clerkId, doc);
    } catch (pgError) {
      warnPgOnce('practice_sessions upsert failed', pgError);
    }
  }
}

/* ============================================================
   Phase 1 — Attempts log (mistake-level data backbone)
   Every graded question (MCQ or written) is appended here, per
   student, so the Mistake Notebook, Weakness Map and everything
   downstream can attribute performance to a concept/topic.
     attempts/<clerkId>.json  ->  { items: AttemptRecord[] }
   ============================================================ */

export type AttemptVerdict = 'correct' | 'partial' | 'weak' | 'unanswered' | 'incorrect';

export interface AttemptRecord {
  id: string;            // questionId + timestamp, stable per logged attempt
  questionId: string;
  subject: string;
  topic: string;         // the concept this attempt is attributed to
  theme?: string;
  type: 'mcq' | 'structured';
  verdict: AttemptVerdict;
  earned: number;
  max: number;
  reason: string;        // short "why it was wrong" note
  year?: string;
  session?: string;
  paper?: string;
  variant?: string;
  at: string;            // ISO timestamp
}

export const MAX_ATTEMPTS = 3000;

function attemptsPath(clerkId: string): string {
  return `attempts/${clerkId}.json`;
}

function normalizeAttempt(raw: unknown): AttemptRecord | null {
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, unknown>;
  const questionId = String(r.questionId ?? '').slice(0, 200);
  if (!questionId) return null;
  const num = (v: unknown, min: number, max: number) => {
    const n = typeof v === 'number' ? v : Number.parseFloat(String(v));
    return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : min;
  };
  const verdicts: AttemptVerdict[] = ['correct', 'partial', 'weak', 'unanswered', 'incorrect'];
  const verdict = verdicts.includes(r.verdict as AttemptVerdict) ? (r.verdict as AttemptVerdict) : 'weak';
  const at = typeof r.at === 'string' && !Number.isNaN(new Date(r.at).getTime()) ? r.at : new Date().toISOString();
  return {
    id: String(r.id ?? `${questionId}_${at}`).slice(0, 260),
    questionId,
    subject: String(r.subject ?? '').slice(0, 80),
    topic: String(r.topic ?? '').slice(0, 120) || 'Uncategorised',
    theme: r.theme ? String(r.theme).slice(0, 120) : undefined,
    type: r.type === 'mcq' ? 'mcq' : 'structured',
    verdict,
    earned: num(r.earned, 0, 100),
    max: num(r.max, 0, 100),
    reason: String(r.reason ?? '').slice(0, 400),
    year: r.year ? String(r.year).slice(0, 8) : undefined,
    session: r.session ? String(r.session).slice(0, 40) : undefined,
    paper: r.paper ? String(r.paper).slice(0, 40) : undefined,
    variant: r.variant ? String(r.variant).slice(0, 40) : undefined,
    at,
  };
}

export async function readAttempts(clerkId: string): Promise<AttemptRecord[]> {
  // Phase 2: once STUDENT_DATA_SOURCE=postgres (set only AFTER backfilling), the
  // attempts log is served by a single indexed query instead of parsing a blob.
  // Only flip that flag after the backfill has run, or history will read empty.
  if (studentDataSource() === 'postgres') {
    try {
      return await pgReadAttempts(clerkId);
    } catch (error) {
      warnPgOnce('attempts read failed, using storage', error);
    }
  }
  const { data, error } = await supabase.storage.from(PRACTICE_BUCKET).download(attemptsPath(clerkId));
  if (error || !data) return [];
  try {
    const parsed = JSON.parse(await data.text()) as { items?: unknown };
    return Array.isArray(parsed.items) ? (parsed.items.map(normalizeAttempt).filter(Boolean) as AttemptRecord[]) : [];
  } catch {
    return [];
  }
}

/** Append new attempts (newest kept), deduped by id, capped at MAX_ATTEMPTS. */
export async function appendAttempts(clerkId: string, incoming: unknown[]): Promise<AttemptRecord[]> {
  const clean = (Array.isArray(incoming) ? incoming : []).map(normalizeAttempt).filter(Boolean) as AttemptRecord[];
  if (clean.length === 0) return readAttempts(clerkId);
  const existing = await readAttempts(clerkId);
  const byId = new Map<string, AttemptRecord>();
  for (const a of existing) byId.set(a.id, a);
  for (const a of clean) byId.set(a.id, a); // new overwrites same id
  const merged = Array.from(byId.values())
    .sort((a, b) => (b.at || '').localeCompare(a.at || ''))
    .slice(0, MAX_ATTEMPTS);
  const payload = Buffer.from(JSON.stringify({ items: merged }), 'utf8');
  const { error } = await supabase.storage
    .from(PRACTICE_BUCKET)
    .upload(attemptsPath(clerkId), payload, { contentType: 'application/json', upsert: true });
  if (error) throw error;
  // Phase 2: mirror only the NEW attempts into Postgres, each as an independent
  // upsert keyed on its stable id. This is what removes the concurrency data
  // loss — two devices finishing at once no longer overwrite each other here.
  // Best-effort: a failure never blocks the student (Storage above succeeded).
  if (pgDualWriteEnabled()) {
    try {
      await pgUpsertAttempts(clerkId, clean);
    } catch (pgError) {
      warnPgOnce('attempts upsert failed', pgError);
    }
  }
  return merged;
}

export async function signUploads(items: PracticeUpload[]): Promise<Array<PracticeUpload & { url?: string }>> {
  if (items.length === 0) return [];
  const paths = items.map((item) => normalizeStoragePath(item.path)).filter(Boolean);
  if (paths.length === 0) return items;
  const { data } = await supabase.storage
    .from(PRACTICE_BUCKET)
    .createSignedUrls(paths, SIGNED_URL_TTL);
  const byPath = new Map((data ?? []).map((entry) => [entry.path, entry.signedUrl]));
  return items.map((item) => {
    const path = normalizeStoragePath(item.path);
    return { ...item, path: path || item.path, url: byPath.get(path) ?? undefined };
  });
}

/** Pull a bucket-relative storage key out of a raw path or a signed URL. */
export function normalizeStoragePath(raw: string | undefined | null): string {
  const value = (raw || '').trim();
  if (!value) return '';
  if (/^https?:\/\//i.test(value)) {
    try {
      const url = new URL(value);
      const parts = url.pathname.split('/').filter(Boolean);
      const objectAt = parts.indexOf('object');
      if (objectAt >= 0 && parts.length > objectAt + 3) {
        return decodeURIComponent(parts.slice(objectAt + 3).join('/'));
      }
    } catch {
      return '';
    }
  }
  return value.replace(/^\/+/, '').replace(new RegExp(`^${PRACTICE_BUCKET}/`), '');
}

function mimeFromName(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase() || '';
  if (ext === 'pdf') return 'application/pdf';
  if (ext === 'png') return 'image/png';
  if (ext === 'jpg' || ext === 'jpeg') return 'image/jpeg';
  return '';
}

function isMissingStorageError(error: unknown): boolean {
  const message = String((error as { message?: string })?.message || error || '').toLowerCase();
  return /not found|not_found|does not exist|no such file|object not found/.test(message);
}

async function downloadStorageBytes(path: string): Promise<{ buffer: Buffer | null; error: unknown }> {
  const { data, error } = await supabase.storage.from(PRACTICE_BUCKET).download(path);
  if (error || !data) return { buffer: null, error: error || new Error('empty download') };
  try {
    return { buffer: Buffer.from(await data.arrayBuffer()), error: null };
  } catch (readError) {
    return { buffer: null, error: readError };
  }
}

/** List the binaries actually stored for this paper, used when JSON paths go stale. */
export async function listPaperUploadFiles(clerkId: string, paperKey: string): Promise<PracticeUpload[]> {
  const prefix = filesPrefix(clerkId, paperKey);
  try {
    const { data, error } = await supabase.storage.from(PRACTICE_BUCKET).list(prefix, {
      limit: 100,
      sortBy: { column: 'name', order: 'asc' },
    });
    if (error || !data) {
      if (error) console.warn('[practice] list uploads failed', prefix, error.message);
      return [];
    }
    return data
      .filter((entry) => entry.name && !entry.name.endsWith('/'))
      .map((entry) => ({
        path: `${prefix}/${entry.name}`,
        name: entry.name.replace(/^\d+_/, ''),
        size: Number((entry.metadata as { size?: number } | undefined)?.size) || 0,
        type: mimeFromName(entry.name) || 'application/octet-stream',
        at: entry.updated_at || new Date().toISOString(),
      }));
  } catch (error) {
    console.warn('[practice] list uploads threw', prefix, error);
    return [];
  }
}

/**
 * Turn a student's uploads into an ordered list of page images for vision
 * grading. Images pass through as-is; PDFs are rasterized page by page, so a
 * multi-page PDF submission becomes N pages instead of being dropped.
 *
 * Anything that could not be used is reported back in `skipped` rather than
 * being silently discarded — a page we never read would otherwise be graded as
 * an unanswered question and score zero.
 */
export interface LoadedPages {
  pages: PageImage[];
  /** human-readable reasons a file (or some of its pages) was not used */
  skipped: string[];
  /** true when the page budget cut the submission short */
  truncated: boolean;
  /** files that produced at least one page */
  usedFiles: number;
}

export const MAX_GRADING_PAGES = Number.parseInt(process.env.PRACTICE_MAX_PAGES || '', 10) > 0
  ? Number.parseInt(process.env.PRACTICE_MAX_PAGES || '', 10)
  : 40;

export async function loadUploadPages(
  uploads: PracticeUpload[],
  maxPages = MAX_GRADING_PAGES,
  opts?: { clerkId?: string; paperKey?: string },
): Promise<LoadedPages> {
  const pages: PageImage[] = [];
  const skipped: string[] = [];
  let truncated = false;
  let usedFiles = 0;
  const seen = new Set<string>();

  let records = uploads.filter((item) => item && (item.path || item.name));
  // Same original filename uploaded twice (re-mark without removing) — keep the latest.
  const byName = new Map<string, PracticeUpload>();
  for (const item of records) byName.set(item.name, item);
  if (byName.size < records.length) records = [...byName.values()];

  const consume = async (item: PracticeUpload): Promise<void> => {
    if (pages.length >= maxPages) {
      truncated = true;
      skipped.push(`${item.name}: not read (page limit of ${maxPages} reached)`);
      return;
    }

    const path = normalizeStoragePath(item.path);
    if (path && seen.has(path)) return;
    if (path) seen.add(path);

    const downloaded = path ? await downloadStorageBytes(path) : { buffer: null, error: new Error('missing path') };
    if (downloaded.buffer && downloaded.buffer.byteLength === 0) {
      skipped.push(`${item.name}: file is empty`);
      return;
    }
    let buffer = downloaded.buffer;
    if (!buffer && item.url && /^https?:\/\//i.test(item.url)) {
      try {
        const response = await fetch(item.url);
        if (response.ok) buffer = Buffer.from(await response.arrayBuffer());
      } catch (fetchError) {
        console.warn('[practice] signed-url fallback failed', item.name, fetchError);
      }
    }
    if (!buffer || buffer.byteLength === 0) {
      const detail = downloaded.error && !isMissingStorageError(downloaded.error)
        ? ` (${String((downloaded.error as { message?: string }).message || downloaded.error)})`
        : '';
      console.warn('[practice] storage download failed', { path: item.path, name: item.name, error: downloaded.error });
      skipped.push(`${item.name}: could not be downloaded from storage${detail}`);
      return;
    }

    const type = item.type || mimeFromName(item.name);
    if (type === 'application/pdf') {
      try {
        const result = await rasterizePdfPages(buffer, pages.length + 1, maxPages - pages.length, item.name);
        if (result.pages.length === 0) {
          skipped.push(`${item.name}: no readable pages found`);
          return;
        }
        pages.push(...result.pages);
        usedFiles += 1;
        if (result.dropped > 0) {
          truncated = true;
          skipped.push(`${item.name}: only the first ${result.pages.length} of ${result.totalPages} pages were read (page limit of ${maxPages})`);
        }
      } catch (pdfError) {
        skipped.push(pdfError instanceof Error ? pdfError.message : `${item.name}: could not be read`);
      }
      return;
    }

    if (isSupportedImageType(type)) {
      pages.push({
        base64: buffer.toString('base64'),
        mimeType: type,
        page: pages.length + 1,
        source: item.name,
      });
      usedFiles += 1;
      return;
    }

    skipped.push(`${item.name}: unsupported file type (${type || 'unknown'}) — upload a JPG, PNG or PDF`);
  };

  for (const item of records) await consume(item);

  // JSON can list files that were moved/replaced. If nothing readable came out,
  // grade whatever is actually sitting in this paper's storage folder.
  if (pages.length === 0 && opts?.clerkId && opts?.paperKey) {
    const stored = await listPaperUploadFiles(opts.clerkId, opts.paperKey);
    const fresh = stored.filter((item) => {
      const path = normalizeStoragePath(item.path);
      return path && !seen.has(path);
    });
    if (fresh.length > 0) {
      skipped.length = 0;
      for (const item of fresh) await consume(item);
    }
  }

  return { pages, skipped, truncated, usedFiles };
}
