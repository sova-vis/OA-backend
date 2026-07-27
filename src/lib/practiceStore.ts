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
