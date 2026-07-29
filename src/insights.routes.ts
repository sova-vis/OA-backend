import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { ensureBucket, readAttempts, appendAttempts, PRACTICE_BUCKET, AttemptRecord } from './lib/practiceStore';
import { supabase } from './lib/supabase';
import { grokEnabled, grokChatJson, grokTextModel } from './lib/grok';

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

/* ---------------- Pattern detection (Grok, cached) ---------------- */

interface PatternCache {
  patterns: { title: string; detail: string }[];
  generatedAt: string;
  basedOnCount: number;
}

const patternsPath = (clerkId: string) => `patterns/${clerkId}.json`;
const STALE_MS = 3 * 24 * 60 * 60 * 1000; // 3 days
const MIN_MISTAKES = 4;
const REGEN_DELTA = 8; // regenerate once this many new mistakes accrue

async function readPatternCache(clerkId: string): Promise<PatternCache | null> {
  const { data } = await supabase.storage.from(PRACTICE_BUCKET).download(patternsPath(clerkId));
  if (!data) return null;
  try { return JSON.parse(await data.text()) as PatternCache; } catch { return null; }
}

async function detectPatterns(subjectMistakes: AttemptRecord[]): Promise<{ title: string; detail: string }[]> {
  const sample = subjectMistakes.slice(0, 45).map((m) => ({ subject: m.subject, topic: m.topic, reason: m.reason }));
  const system = [
    'You are an experienced Cambridge O/A Level tutor reviewing a student\'s logged mistakes.',
    'Identify 2 to 4 RECURRING BEHAVIOURAL patterns across the mistakes — habits, not individual topics.',
    'Examples of behavioural patterns: "rushes calculations", "skips command words", "gives vague definitions", "forgets units", "misreads the question".',
    'Each pattern: a short title (max 6 words) and one concrete sentence of advice addressed to the student.',
    'Return JSON ONLY: { "patterns": [ { "title": string, "detail": string } ] }.',
  ].join(' ');
  const user = `Here are the student's recent mistakes (subject, topic, reason). Find their recurring habits:\n${JSON.stringify(sample)}`;
  const parsed = await grokChatJson({ system, user, model: grokTextModel(), temperature: 0.3, maxTokens: 700 });
  const raw = Array.isArray(parsed.patterns) ? (parsed.patterns as Array<Record<string, unknown>>) : [];
  return raw
    .map((p) => ({ title: String(p.title ?? '').slice(0, 60).trim(), detail: String(p.detail ?? '').slice(0, 300).trim() }))
    .filter((p) => p.title && p.detail)
    .slice(0, 4);
}

async function getOrBuildPatterns(clerkId: string, force: boolean): Promise<PatternCache> {
  const attempts = await readAttempts(clerkId);
  const mistakes = attempts.filter((a) => a.verdict !== 'correct');
  const cache = await readPatternCache(clerkId);
  const fresh = cache && Date.now() - new Date(cache.generatedAt).getTime() < STALE_MS && mistakes.length - cache.basedOnCount < REGEN_DELTA;

  if (mistakes.length < MIN_MISTAKES) {
    return { patterns: [], generatedAt: new Date().toISOString(), basedOnCount: mistakes.length };
  }
  if (!force && fresh && cache) return cache;
  if (!grokEnabled()) return cache ?? { patterns: [], generatedAt: new Date().toISOString(), basedOnCount: mistakes.length };

  const patterns = await detectPatterns(mistakes);
  const next: PatternCache = { patterns, generatedAt: new Date().toISOString(), basedOnCount: mistakes.length };
  await supabase.storage.from(PRACTICE_BUCKET).upload(patternsPath(clerkId), Buffer.from(JSON.stringify(next), 'utf8'), {
    contentType: 'application/json', upsert: true,
  });
  return next;
}

/** GET returns cached patterns (rebuilds if stale); POST ?refresh forces a rebuild. */
router.get('/patterns', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });
    await ensureBucket();
    return res.json(await getOrBuildPatterns(clerkId, false));
  } catch (error) {
    console.error('Pattern detection error:', error);
    return res.status(500).json({ error: 'Failed to load patterns' });
  }
});

router.post('/patterns', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });
    await ensureBucket();
    return res.json(await getOrBuildPatterns(clerkId, true));
  } catch (error) {
    console.error('Pattern detection error:', error);
    return res.status(500).json({ error: 'Failed to analyze patterns' });
  }
});

export default router;
