import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { ensureBucket, readAttempts, appendAttempts } from './lib/practiceStore';

/**
 * Phase 1 — performance insights backbone.
 * Stores the per-student attempts log (mistake-level data) that powers the
 * Mistake Notebook, Weakness Map and everything downstream.
 *
 *   GET  /insights/attempts        -> { items: AttemptRecord[] }  (newest first)
 *   POST /insights/attempts        { items: [...] } -> { items }  (append + return all)
 */

const MAX_INCOMING = 200;
const router = Router();

router.get('/attempts', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });
    await ensureBucket();
    const items = await readAttempts(clerkId);
    return res.json({ items });
  } catch (error) {
    console.error('Failed to load attempts:', error);
    return res.status(500).json({ error: 'Failed to load attempts' });
  }
});

router.post('/attempts', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const body = (req.body ?? {}) as { items?: unknown };
    if (!Array.isArray(body.items)) {
      return res.status(400).json({ error: 'items[] is required' });
    }
    await ensureBucket();
    const items = await appendAttempts(clerkId, body.items.slice(0, MAX_INCOMING));
    return res.json({ ok: true, items });
  } catch (error) {
    console.error('Failed to append attempts:', error);
    return res.status(500).json({ error: 'Failed to save attempts' });
  }
});

export default router;
