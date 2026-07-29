import { supabase } from './supabase';

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
}

/** Marks split by assessment objective (Phase 1 · Marks Breakdown). */
export interface MarkCategory {
  category: 'Knowledge' | 'Explanation' | 'Evaluation';
  earned: number;
  max: number;
}

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
  /** true when a marking scheme was matched for this question; false = examiner-judgement fallback */
  schemeUsed?: boolean;
  /** marks earned vs available per assessment objective */
  breakdown?: MarkCategory[];
  /** Phase 3 — examiner intelligence */
  commandWord?: string;      // the question's command word, e.g. "Describe", "Evaluate"
  commandWordNote?: string;  // coaching when the answer's style doesn't match the command word
  examinerNote?: string;     // "Candidates commonly lose marks here because…"
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
  return merged;
}

export async function signUploads(items: PracticeUpload[]): Promise<Array<PracticeUpload & { url?: string }>> {
  if (items.length === 0) return [];
  const { data } = await supabase.storage
    .from(PRACTICE_BUCKET)
    .createSignedUrls(items.map((item) => item.path), SIGNED_URL_TTL);
  const byPath = new Map((data ?? []).map((entry) => [entry.path, entry.signedUrl]));
  return items.map((item) => ({ ...item, url: byPath.get(item.path) ?? undefined }));
}

/** Download upload binaries as base64 (for vision grading). Skips non-images. */
export async function downloadUploadImages(
  uploads: PracticeUpload[],
  maxImages = 8,
): Promise<Array<{ base64: string; mimeType: string; name: string }>> {
  const images = uploads.filter((item) => /^image\//.test(item.type)).slice(0, maxImages);
  const out: Array<{ base64: string; mimeType: string; name: string }> = [];
  for (const item of images) {
    const { data, error } = await supabase.storage.from(PRACTICE_BUCKET).download(item.path);
    if (error || !data) continue;
    const buffer = Buffer.from(await data.arrayBuffer());
    out.push({ base64: buffer.toString('base64'), mimeType: item.type, name: item.name });
  }
  return out;
}
