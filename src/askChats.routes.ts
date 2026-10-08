import { Router, Response } from 'express';
import { AuthenticatedRequest } from './lib/clerkAuth';
import { supabase } from './lib/supabase';

/**
 * Account-synced Ask-AI chat history (migration 035). A student's Ask / Find
 * conversations are mirrored here so they show up on every device; the client
 * keeps localStorage as an offline cache and merges the two on load. Mounted
 * with clerkAuth only — no Pro gate or AI rate limit, this is just storage.
 *
 *   GET  /ask-chats?mode=ask|find   -> { sessions }
 *   POST /ask-chats { mode, sessions } -> replace the caller's list for that mode
 */
const router = Router();

router.get('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const mode = req.query.mode === 'find' ? 'find' : 'ask';
    const { data } = await supabase
      .from('ask_chats')
      .select('sessions')
      .eq('clerk_id', clerkId)
      .eq('mode', mode)
      .maybeSingle();
    const raw = (data as { sessions?: unknown } | null)?.sessions;
    return res.json({ sessions: Array.isArray(raw) ? raw : [] });
  } catch (err) {
    console.error('Ask chats load error:', err);
    return res.status(500).json({ error: 'Failed to load chats' });
  }
});

router.post('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const body = (req.body ?? {}) as { mode?: unknown; sessions?: unknown };
    const mode = body.mode === 'find' ? 'find' : 'ask';
    const sessions = Array.isArray(body.sessions) ? body.sessions.slice(0, 20) : [];
    const { error } = await supabase.from('ask_chats').upsert(
      { clerk_id: clerkId, mode, sessions, updated_at: new Date().toISOString() },
      { onConflict: 'clerk_id,mode' },
    );
    if (error) throw error;
    return res.json({ ok: true });
  } catch (err) {
    console.error('Ask chats save error:', err);
    return res.status(500).json({ error: 'Failed to save chats' });
  }
});

export default router;
