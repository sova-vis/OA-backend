import { Router, Response } from 'express';
import multer from 'multer';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';

/**
 * Practice-paper progress, persisted as JSON objects in a private Supabase
 * Storage bucket (chosen over a SQL table because Storage rides the working
 * REST credentials). One object per (student, paper):
 *
 *   s/<clerkId>/<paperKeySafe>.json          -> PracticeProgress document
 *   files/<clerkId>/<paperKeySafe>/<name>    -> handwritten upload binaries
 *
 * paper_key = "Subject|Year|Session|Paper|Variant" in question-bank naming,
 * e.g. "Chemistry|2024|May_June|Paper_2|Variant_1".
 */

const BUCKET = 'practice-progress';
const MAX_DOC_BYTES = 1_000_000; // 1MB of typed answers is plenty
const MAX_UPLOAD_BYTES = 15 * 1024 * 1024;
const MAX_UPLOADS_PER_PAPER = 24;
const SIGNED_URL_TTL = 60 * 60; // 1h

type SolveMode = 'digital' | 'handwritten';
type PracticeStatus = 'in_progress' | 'completed';

interface PracticeUpload {
  path: string;
  name: string;
  size: number;
  type: string;
  at: string;
}

interface PracticeProgressDoc {
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
  startedAt: string;
  updatedAt: string;
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_UPLOAD_BYTES },
});

const router = Router();

let bucketReady = false;
async function ensureBucket() {
  if (bucketReady) return;
  const { data } = await supabase.storage.getBucket(BUCKET);
  if (!data) {
    const { error } = await supabase.storage.createBucket(BUCKET, { public: false });
    if (error && !/already exists/i.test(error.message)) throw error;
  }
  bucketReady = true;
}

function safeKey(paperKey: string): string {
  return paperKey.replace(/[^A-Za-z0-9._-]/g, '_');
}

function isValidPaperKey(paperKey: unknown): paperKey is string {
  if (typeof paperKey !== 'string' || paperKey.length === 0 || paperKey.length > 200) return false;
  const segments = paperKey.split('|');
  return segments.length === 5 && segments.every((segment) => segment.trim().length > 0);
}

function docPath(clerkId: string, paperKey: string) {
  return `s/${clerkId}/${safeKey(paperKey)}.json`;
}

function filesPrefix(clerkId: string, paperKey: string) {
  return `files/${clerkId}/${safeKey(paperKey)}`;
}

function cleanRecord(input: unknown, maxEntries: number): Record<string, string> {
  if (!input || typeof input !== 'object' || Array.isArray(input)) return {};
  const out: Record<string, string> = {};
  let count = 0;
  for (const [key, value] of Object.entries(input as Record<string, unknown>)) {
    if (count >= maxEntries) break;
    if (typeof key !== 'string' || key.length > 300) continue;
    if (typeof value !== 'string' || value.length === 0) continue;
    out[key] = value.slice(0, 20_000);
    count += 1;
  }
  return out;
}

function toIso(value: unknown, fallback: string): string {
  if (typeof value === 'string' && !Number.isNaN(new Date(value).getTime())) return value;
  return fallback;
}

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === 'number' ? value : Number.parseInt(String(value), 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.round(n)));
}

function normalizeDoc(paperKey: string, body: Record<string, unknown>, existing: PracticeProgressDoc | null): PracticeProgressDoc {
  const now = new Date().toISOString();
  const segments = paperKey.split('|');
  return {
    paperKey,
    subject: String(body.subject ?? segments[0]).slice(0, 80),
    year: String(body.year ?? segments[1]).slice(0, 8),
    session: String(body.session ?? segments[2]).slice(0, 40),
    paper: String(body.paper ?? segments[3]).slice(0, 40),
    variant: String(body.variant ?? segments[4]).slice(0, 40),
    isMcq: Boolean(body.isMcq),
    solveMode: body.solveMode === 'handwritten' ? 'handwritten' : 'digital',
    status: body.status === 'completed' ? 'completed' : 'in_progress',
    answers: {
      mcq: cleanRecord((body.answers as Record<string, unknown> | undefined)?.mcq, 200),
      parts: cleanRecord((body.answers as Record<string, unknown> | undefined)?.parts, 500),
    },
    // uploads are managed exclusively by the upload endpoints; a stale client
    // PUT must never clobber them
    uploads: existing?.uploads ?? [],
    answeredCount: clampInt(body.answeredCount, 0, 10_000, 0),
    totalCount: clampInt(body.totalCount, 0, 10_000, 0),
    timerDurationSeconds: clampInt(body.timerDurationSeconds, 0, 24 * 3600, 0),
    timerElapsedSeconds: clampInt(body.timerElapsedSeconds, 0, 7 * 24 * 3600, 0),
    startedAt: existing?.startedAt ?? toIso(body.startedAt, now),
    updatedAt: now,
  };
}

async function readDoc(clerkId: string, paperKey: string): Promise<PracticeProgressDoc | null> {
  const { data, error } = await supabase.storage.from(BUCKET).download(docPath(clerkId, paperKey));
  if (error || !data) return null;
  try {
    return JSON.parse(await data.text()) as PracticeProgressDoc;
  } catch {
    return null;
  }
}

async function writeDoc(clerkId: string, doc: PracticeProgressDoc): Promise<void> {
  const payload = Buffer.from(JSON.stringify(doc), 'utf8');
  if (payload.byteLength > MAX_DOC_BYTES) throw new Error('Progress document too large');
  const { error } = await supabase.storage
    .from(BUCKET)
    .upload(docPath(clerkId, doc.paperKey), payload, { contentType: 'application/json', upsert: true });
  if (error) throw error;
}

async function signUploads(items: PracticeUpload[]): Promise<Array<PracticeUpload & { url?: string }>> {
  if (items.length === 0) return [];
  const { data } = await supabase.storage
    .from(BUCKET)
    .createSignedUrls(items.map((item) => item.path), SIGNED_URL_TTL);
  const byPath = new Map((data ?? []).map((entry) => [entry.path, entry.signedUrl]));
  return items.map((item) => ({ ...item, url: byPath.get(item.path) ?? undefined }));
}

/** GET /practice/progress — every practice session for the student, newest first. */
router.get('/progress', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });
    await ensureBucket();

    const { data: entries, error } = await supabase.storage
      .from(BUCKET)
      .list(`s/${clerkId}`, { limit: 200 });
    if (error) throw error;

    const jsonNames = (entries ?? []).filter((entry) => entry.name.endsWith('.json'));
    const docs = (
      await Promise.all(
        jsonNames.map(async (entry) => {
          const { data } = await supabase.storage.from(BUCKET).download(`s/${clerkId}/${entry.name}`);
          if (!data) return null;
          try {
            return JSON.parse(await data.text()) as PracticeProgressDoc;
          } catch {
            return null;
          }
        }),
      )
    ).filter((doc): doc is PracticeProgressDoc => Boolean(doc && doc.paperKey));

    docs.sort((a, b) => (b.updatedAt || '').localeCompare(a.updatedAt || ''));
    const items = await Promise.all(
      docs.map(async (doc) => ({ ...doc, uploads: await signUploads(doc.uploads) })),
    );
    return res.json({ items });
  } catch (error) {
    console.error('Failed to load practice progress:', error);
    return res.status(500).json({ error: 'Failed to load practice progress' });
  }
});

/** PUT /practice/progress/:paperKey — upsert one session (autosave). */
router.put('/progress/:paperKey', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const paperKey = req.params.paperKey;
    if (!isValidPaperKey(paperKey)) return res.status(400).json({ error: 'Invalid paper key' });
    await ensureBucket();

    const existing = await readDoc(clerkId, paperKey);
    const doc = normalizeDoc(paperKey, (req.body ?? {}) as Record<string, unknown>, existing);
    await writeDoc(clerkId, doc);
    return res.json({ ok: true, item: { ...doc, uploads: await signUploads(doc.uploads) } });
  } catch (error) {
    console.error('Failed to save practice progress:', error);
    return res.status(500).json({ error: 'Failed to save practice progress' });
  }
});

/** DELETE /practice/progress/:paperKey — discard a session and its uploads. */
router.delete('/progress/:paperKey', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const paperKey = req.params.paperKey;
    if (!isValidPaperKey(paperKey)) return res.status(400).json({ error: 'Invalid paper key' });
    await ensureBucket();

    const prefix = filesPrefix(clerkId, paperKey);
    const { data: files } = await supabase.storage.from(BUCKET).list(prefix, { limit: 100 });
    const paths = (files ?? []).map((file) => `${prefix}/${file.name}`);
    paths.push(docPath(clerkId, paperKey));
    await supabase.storage.from(BUCKET).remove(paths);
    return res.json({ ok: true });
  } catch (error) {
    console.error('Failed to delete practice progress:', error);
    return res.status(500).json({ error: 'Failed to delete practice progress' });
  }
});

/** POST /practice/progress/:paperKey/uploads — attach one handwritten file. */
router.post(
  '/progress/:paperKey/uploads',
  clerkAuth,
  upload.single('file'),
  async (req: AuthenticatedRequest, res: Response) => {
    try {
      const clerkId = req.auth?.clerkId;
      if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

      const paperKey = req.params.paperKey;
      if (!isValidPaperKey(paperKey)) return res.status(400).json({ error: 'Invalid paper key' });

      const file = req.file;
      if (!file) return res.status(400).json({ error: 'No file provided' });
      const isAllowed = /^image\//.test(file.mimetype) || file.mimetype === 'application/pdf';
      if (!isAllowed) return res.status(400).json({ error: 'Only images and PDFs are allowed' });
      await ensureBucket();

      const existing = await readDoc(clerkId, paperKey);
      if ((existing?.uploads?.length ?? 0) >= MAX_UPLOADS_PER_PAPER) {
        return res.status(400).json({ error: `Upload limit reached (${MAX_UPLOADS_PER_PAPER} files)` });
      }

      const cleanName = (file.originalname || 'upload').replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 120);
      const path = `${filesPrefix(clerkId, paperKey)}/${Date.now()}_${cleanName}`;
      const { error: uploadError } = await supabase.storage
        .from(BUCKET)
        .upload(path, file.buffer, { contentType: file.mimetype, upsert: false });
      if (uploadError) throw uploadError;

      const record: PracticeUpload = {
        path,
        name: file.originalname || cleanName,
        size: file.size,
        type: file.mimetype,
        at: new Date().toISOString(),
      };

      // fold into the progress doc (create a skeleton if this is the first touch)
      const base = existing ?? normalizeDoc(paperKey, { solveMode: 'handwritten' }, null);
      base.uploads = [...(base.uploads ?? []), record];
      base.solveMode = 'handwritten';
      base.updatedAt = record.at;
      await writeDoc(clerkId, base);

      const [signed] = await signUploads([record]);
      return res.json({ ok: true, upload: signed, item: { ...base, uploads: await signUploads(base.uploads) } });
    } catch (error) {
      console.error('Failed to upload practice file:', error);
      return res.status(500).json({ error: 'Failed to upload file' });
    }
  },
);

/** DELETE /practice/progress/:paperKey/uploads — remove one handwritten file. */
router.delete('/progress/:paperKey/uploads', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const paperKey = req.params.paperKey;
    if (!isValidPaperKey(paperKey)) return res.status(400).json({ error: 'Invalid paper key' });

    const path = typeof req.body?.path === 'string' ? req.body.path : '';
    // a student may only ever delete files under their own prefix
    if (!path.startsWith(filesPrefix(clerkId, paperKey) + '/')) {
      return res.status(400).json({ error: 'Invalid upload path' });
    }
    await ensureBucket();

    await supabase.storage.from(BUCKET).remove([path]);
    const existing = await readDoc(clerkId, paperKey);
    if (existing) {
      existing.uploads = (existing.uploads ?? []).filter((item) => item.path !== path);
      existing.updatedAt = new Date().toISOString();
      await writeDoc(clerkId, existing);
      return res.json({ ok: true, item: { ...existing, uploads: await signUploads(existing.uploads) } });
    }
    return res.json({ ok: true });
  } catch (error) {
    console.error('Failed to remove practice upload:', error);
    return res.status(500).json({ error: 'Failed to remove upload' });
  }
});

export default router;
