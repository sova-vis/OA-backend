import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';

type PaperStatus = 'in_progress' | 'completed' | 'important' | 'bookmarked';

interface TrackingItem {
  id: string;
  name: string;
  type: string;
  viewUrl?: string;
  downloadUrl?: string;
  embedUrl?: string;
  savedAt: string;
  statuses: PaperStatus[];
}

const VALID_STATUSES = new Set<PaperStatus>([
  'in_progress',
  'completed',
  'important',
  'bookmarked',
]);

const router = Router();

function buildTrackingTableHint(error: unknown): string | null {
  const maybeError = error as { code?: string; message?: string };
  if (maybeError?.code === '42P01') {
    return 'Table user_paper_tracking is missing. Apply migration OA-backend/migrations/001_create_user_paper_tracking.sql.';
  }
  if (typeof maybeError?.message === 'string' && maybeError.message.toLowerCase().includes('user_paper_tracking')) {
    return 'Tracking table is not available. Apply migration OA-backend/migrations/001_create_user_paper_tracking.sql.';
  }
  return null;
}

function normalizeStatuses(input: unknown): PaperStatus[] {
  if (!Array.isArray(input)) return [];
  const deduped = new Set<PaperStatus>();

  for (const value of input) {
    if (typeof value === 'string' && VALID_STATUSES.has(value as PaperStatus)) {
      deduped.add(value as PaperStatus);
    }
  }

  return Array.from(deduped);
}

function normalizeItem(raw: unknown): TrackingItem | null {
  if (!raw || typeof raw !== 'object') return null;

  const item = raw as Record<string, unknown>;
  if (!item.id || !item.name || !item.type) return null;

  const statuses = normalizeStatuses(item.statuses);
  if (statuses.length === 0) return null;

  const savedAt = typeof item.savedAt === 'string' ? item.savedAt : new Date().toISOString();
  if (Number.isNaN(new Date(savedAt).getTime())) return null;

  const normalizeUrl = (value: unknown): string | undefined => {
    if (typeof value !== 'string') return undefined;
    const trimmed = value.trim();
    return trimmed.length > 0 ? trimmed : undefined;
  };

  return {
    id: String(item.id),
    name: String(item.name),
    type: String(item.type),
    viewUrl: normalizeUrl(item.viewUrl),
    downloadUrl: normalizeUrl(item.downloadUrl),
    embedUrl: normalizeUrl(item.embedUrl),
    savedAt,
    statuses,
  };
}

router.get('/papers', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const { data, error } = await supabase
      .from('user_paper_tracking')
      .select('paper_id, name, type, view_url, download_url, embed_url, saved_at, statuses')
      .eq('clerk_id', clerkId)
      .order('updated_at', { ascending: false });

    if (error) {
      throw error;
    }

    const items: TrackingItem[] = (data ?? []).map((row) => ({
      id: String(row.paper_id),
      name: String(row.name),
      type: String(row.type),
      viewUrl: row.view_url ? String(row.view_url) : undefined,
      downloadUrl: row.download_url ? String(row.download_url) : undefined,
      embedUrl: row.embed_url ? String(row.embed_url) : undefined,
      savedAt: new Date(row.saved_at).toISOString(),
      statuses: normalizeStatuses(row.statuses),
    }));

    return res.json({ items });
  } catch (error) {
    console.error('Failed to fetch tracked papers:', error);
    const hint = buildTrackingTableHint(error);
    return res.status(500).json({
      error: 'Failed to load tracked papers',
      ...(hint ? { hint } : {}),
    });
  }
});

router.put('/papers', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const body = (req.body ?? {}) as { items?: unknown };
    if (!Array.isArray(body.items)) {
      return res.status(400).json({ error: 'Invalid payload: items[] is required' });
    }

    const sanitized = body.items
      .map(normalizeItem)
      .filter((item): item is TrackingItem => Boolean(item));

    if (sanitized.length > 1000) {
      return res.status(400).json({ error: 'Too many tracking items (max 1000)' });
    }

    const { error: deleteError } = await supabase
      .from('user_paper_tracking')
      .delete()
      .eq('clerk_id', clerkId);

    if (deleteError) {
      throw deleteError;
    }

    if (sanitized.length > 0) {
      const rows = sanitized.map((item) => ({
        clerk_id: clerkId,
        paper_id: item.id,
        name: item.name,
        type: item.type,
        view_url: item.viewUrl ?? null,
        download_url: item.downloadUrl ?? null,
        embed_url: item.embedUrl ?? null,
        saved_at: item.savedAt,
        statuses: item.statuses,
        updated_at: new Date().toISOString(),
      }));

      const { error: insertError } = await supabase
        .from('user_paper_tracking')
        .insert(rows);

      if (insertError) {
        throw insertError;
      }
    }

    return res.json({ ok: true, count: sanitized.length });
  } catch (error) {
    console.error('Failed to persist tracked papers:', error);
    const hint = buildTrackingTableHint(error);
    return res.status(500).json({
      error: 'Failed to save tracked papers',
      ...(hint ? { hint } : {}),
    });
  }
});

export default router;