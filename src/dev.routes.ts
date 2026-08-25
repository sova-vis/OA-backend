import { Router, Response, NextFunction } from 'express';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';

/**
 * Dev mode — a single SHARED password (bcrypt-hashed, one `dev_access` row)
 * unlocks in-place editing of the live question bank from the practice UI.
 * Unlock issues a short-lived signed token; the editor endpoints require it.
 * Edits overwrite public.questions / public.question_parts directly (no drafts,
 * no history) so all users see the fix immediately.
 */

const router = Router();

const DEV_SECRET = process.env.DEV_TOKEN_SECRET || process.env.CLERK_SECRET_KEY || 'dev-mode-fallback-secret';
const TOKEN_TTL = '8h';
const MIN_PW_LEN = 6;

// --- brute-force guard: throttle unlock attempts per clerk user -------------
const attempts = new Map<string, { n: number; first: number }>();
const WINDOW_MS = 10 * 60 * 1000;
const MAX_ATTEMPTS = 8;
function rateLimited(key: string): boolean {
  const now = Date.now();
  const rec = attempts.get(key);
  if (!rec || now - rec.first > WINDOW_MS) {
    attempts.set(key, { n: 1, first: now });
    return false;
  }
  rec.n += 1;
  return rec.n > MAX_ATTEMPTS;
}

async function getDevRow(): Promise<{ password_hash: string } | null> {
  const { data } = await supabase.from('dev_access').select('password_hash').eq('id', 1).maybeSingle();
  return (data as { password_hash: string }) || null;
}

function signDevToken(clerkId: string): string {
  return jwt.sign({ scope: 'dev', sub: clerkId }, DEV_SECRET, { expiresIn: TOKEN_TTL });
}

// gate: valid Clerk session (applied at mount) + a valid dev token header
function requireDev(req: AuthenticatedRequest, res: Response, next: NextFunction) {
  const token = String(req.headers['x-dev-token'] || '');
  if (!token) return res.status(401).json({ error: 'dev token required' });
  try {
    const payload = jwt.verify(token, DEV_SECRET) as { scope?: string };
    if (payload.scope !== 'dev') throw new Error('bad scope');
    return next();
  } catch {
    return res.status(401).json({ error: 'invalid or expired dev token' });
  }
}

// --- GET /dev/status — is a password set? ----------------------------------
router.get('/status', async (_req: AuthenticatedRequest, res: Response) => {
  const row = await getDevRow();
  res.json({ passwordSet: Boolean(row) });
});

// --- POST /dev/password — set (first time) or change (needs current) -------
router.post('/password', async (req: AuthenticatedRequest, res: Response) => {
  const password = String(req.body?.password || '');
  const current = String(req.body?.currentPassword || '');
  if (password.length < MIN_PW_LEN) {
    return res.status(400).json({ error: `password must be at least ${MIN_PW_LEN} characters` });
  }
  const row = await getDevRow();
  if (row) {
    // changing an existing password requires the current one
    if (!current || !(await bcrypt.compare(current, row.password_hash))) {
      return res.status(403).json({ error: 'current password is incorrect' });
    }
  }
  const hash = await bcrypt.hash(password, 10);
  const { error } = await supabase
    .from('dev_access')
    .upsert({ id: 1, password_hash: hash, set_by_clerk_id: req.auth?.clerkId || null, updated_at: new Date().toISOString() }, { onConflict: 'id' });
  if (error) return res.status(500).json({ error: 'could not save password' });
  res.json({ ok: true, token: signDevToken(req.auth?.clerkId || 'dev') });
});

// --- POST /dev/unlock — verify password -> issue dev token ------------------
router.post('/unlock', async (req: AuthenticatedRequest, res: Response) => {
  const key = req.auth?.clerkId || req.ip || 'anon';
  if (rateLimited(key)) return res.status(429).json({ error: 'too many attempts — try again later' });
  const password = String(req.body?.password || '');
  const row = await getDevRow();
  if (!row) return res.status(400).json({ error: 'no dev password set yet' });
  if (!(await bcrypt.compare(password, row.password_hash))) {
    return res.status(403).json({ error: 'incorrect password' });
  }
  attempts.delete(key);
  res.json({ ok: true, token: signDevToken(req.auth?.clerkId || 'dev') });
});

// --- editors (require the dev token) ---------------------------------------
const OPTION_LETTER = /^[A-D]$/;

// PATCH /dev/questions/:id — text / options / correct_option / marking / images
router.patch('/questions/:id', requireDev, async (req: AuthenticatedRequest, res: Response) => {
  const id = req.params.id;
  const b = req.body || {};
  const patch: Record<string, unknown> = {};

  if (typeof b.question_text === 'string') patch.question_text = b.question_text.slice(0, 20_000);
  if (typeof b.marking_scheme === 'string') patch.marking_scheme = b.marking_scheme.slice(0, 20_000);
  if (b.correct_option === null || (typeof b.correct_option === 'string' && OPTION_LETTER.test(b.correct_option.toUpperCase()))) {
    patch.correct_option = b.correct_option ? b.correct_option.toUpperCase() : null;
  }
  if (b.options && typeof b.options === 'object') patch.options = b.options;
  if (Array.isArray(b.images)) {
    // keep only well-formed image objects, cap count and per-image size
    patch.images = b.images
      .filter((im: unknown) => im && typeof im === 'object')
      .slice(0, 20)
      .map((im: Record<string, unknown>) => ({
        role: typeof im.role === 'string' ? im.role : 'figure',
        caption: typeof im.caption === 'string' ? im.caption : null,
        width: typeof im.width === 'number' ? im.width : null,
        height: typeof im.height === 'number' ? im.height : null,
        data_url: typeof im.data_url === 'string' ? im.data_url : null,
      }))
      .filter((im: { data_url: string | null }) => im.data_url && im.data_url.length < 8_000_000);
  }

  if (Object.keys(patch).length === 0) return res.status(400).json({ error: 'nothing to update' });

  const { error } = await supabase.from('questions').update(patch).eq('id', id);
  if (error) return res.status(500).json({ error: error.message });
  res.json({ ok: true, updated: Object.keys(patch) });
});

// PATCH /dev/parts/:id — a structured question's part (body / marks / answer)
router.patch('/parts/:id', requireDev, async (req: AuthenticatedRequest, res: Response) => {
  const id = req.params.id;
  const b = req.body || {};
  const patch: Record<string, unknown> = {};
  if (typeof b.body === 'string') patch.body = b.body.slice(0, 20_000);
  if (typeof b.answer === 'string') patch.answer = b.answer.slice(0, 20_000);
  if (b.marks === null || typeof b.marks === 'number') patch.marks = b.marks;
  if (Object.keys(patch).length === 0) return res.status(400).json({ error: 'nothing to update' });

  const { error } = await supabase.from('question_parts').update(patch).eq('id', id);
  if (error) return res.status(500).json({ error: error.message });
  res.json({ ok: true, updated: Object.keys(patch) });
});

export default router;
export { clerkAuth };
