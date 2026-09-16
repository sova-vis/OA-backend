/**
 * Match ranking for Ask-AI. The retrieval step returns candidates that are
 * near in embedding space or contain a keyword; this asks the LLM to judge each
 * one against what the student is actually looking for and sort them into:
 *
 *   best        — essentially the same question (same concept AND same demand)
 *   conceptual  — tests the same concept, different wording / values / context
 *   related     — different concept but technically relevant (same technique,
 *                 neighbouring syllabus point, needed as a step)
 *   (dropped)   — only shares a word
 *
 * If the LLM is unavailable a distance-based fallback keeps results flowing.
 */
import { chatJson, type LlmTier } from './llm';
import { refLabel, type Hit, type Level } from './retrieve';
import type { QueryPlan } from './planner';

export type Tier = 'best' | 'conceptual' | 'related';
export interface Ranked { hit: Hit; tier: Tier; why: string }
export interface RankResult {
  best: Ranked[];
  conceptual: Ranked[];
  related: Ranked[];
  source: 'llm' | 'heuristic';
}

const CAPS: Record<Tier, number> = { best: 3, conceptual: 8, related: 8 };

const RANK_SYSTEM = `You rank past-paper questions retrieved for a Cambridge student's search. Judge each candidate against what the student is looking for and classify it:
- "best": essentially the same question — same concept AND the same thing being asked or calculated. Rare: 0 to 3 candidates.
- "conceptual": tests the same concept/topic, just with different wording, values or context.
- "related": a different concept, but technically relevant — uses the same technique or skill, is a neighbouring point on the same syllabus topic, or needs this concept as a step.
- "drop": only shares a word with the search, or is about something else; not useful to this student.
Be strict: a question that merely mentions the keyword in passing is "drop" or at most "related". Prefer questions whose main demand is the searched concept.
Return JSON: {"matches":[{"id":"<id>","tier":"best|conceptual|related|drop","why":"<specific reason, max 14 words>"}]} — include every candidate id exactly once.`;

const snippet = (t: string, n: number) => t.replace(/\s+/g, ' ').trim().slice(0, n);

interface RawMatch { id: string; tier: Tier; why: string }

const asTier = (v: unknown): Tier | null =>
  v === 'best' || v === 'conceptual' || v === 'related' ? v : null;

/**
 * Accept the shapes smaller models actually produce: the requested
 * {"matches":[{id,tier,why}]}, a tier-keyed object {"best":[...ids or objects]},
 * or a bare array. Entries may name the id as id / question_id.
 */
function normalizeMatches(raw: unknown): RawMatch[] {
  const out: RawMatch[] = [];
  const pushEntry = (entry: unknown, tierHint: Tier | null) => {
    if (typeof entry === 'string') { if (tierHint) out.push({ id: entry, tier: tierHint, why: '' }); return; }
    if (!entry || typeof entry !== 'object') return;
    const e = entry as Record<string, unknown>;
    const id = String(e.id ?? e.question_id ?? e.questionId ?? '').trim();
    const tier = asTier(e.tier) ?? tierHint;
    if (!id || !tier) return;
    out.push({ id, tier, why: typeof e.why === 'string' ? e.why.trim() : typeof e.reason === 'string' ? e.reason.trim() : '' });
  };
  const root = raw as Record<string, unknown> | unknown[] | null;
  if (Array.isArray(root)) { root.forEach((e) => pushEntry(e, null)); return out; }
  if (!root || typeof root !== 'object') return out;
  const list = root.matches ?? root.results ?? root.candidates;
  if (Array.isArray(list)) { list.forEach((e) => pushEntry(e, null)); return out; }
  const grouped = (list && typeof list === 'object' ? list : root) as Record<string, unknown>;
  for (const t of ['best', 'conceptual', 'related'] as Tier[]) {
    if (Array.isArray(grouped[t])) (grouped[t] as unknown[]).forEach((e) => pushEntry(e, t));
  }
  return out;
}

// No-LLM fallback: deliberately conservative — only rows that literally contain
// a searched term, or are very close in embedding space, are shown at all.
export function heuristicRank(hits: Hit[]): RankResult {
  const out: RankResult = { best: [], conceptual: [], related: [], source: 'heuristic' };
  for (const hit of hits) {
    const d = hit.distance;
    let tier: Tier | null = null;
    if (hit.verified && d < 0.3) tier = 'best';
    else if (hit.verified) tier = 'conceptual';
    else if (d < 0.36) tier = 'related';
    if (tier && out[tier].length < CAPS[tier]) {
      out[tier].push({ hit, tier, why: hit.verified ? 'Contains the searched term' : 'Very close in meaning to the search' });
    }
  }
  return out;
}

export async function rankCandidates(
  query: string, plan: QueryPlan, hits: Hit[], level: Level, tier: LlmTier = 'smart',
): Promise<RankResult> {
  if (!hits.length) return { best: [], conceptual: [], related: [], source: 'llm' };
  const fallback = heuristicRank(hits);
  const lines = hits.map((h) =>
    `${h.id} | ${refLabel(h.metadata)} | ${h.metadata.type || '?'} | topic: ${h.metadata.topic || '-'} | ${snippet(h.text, 240)}`);
  const user =
    `Student is looking for: "${query.slice(0, 600)}"\n` +
    `Interpreted topic: ${plan.topic || '-'} | Subject: ${plan.subject || 'any'} | Level: ${level === 'olevel' ? 'O Level' : 'A Level'} | Intent: ${plan.intent}\n\n` +
    `Candidates (id | paper reference | type | topic | text):\n${lines.join('\n')}`;
  try {
    const raw = await chatJson<Record<string, unknown>>(
      RANK_SYSTEM, user, { tier, maxTokens: 1400, temperature: 0, timeoutMs: 45_000 },
    );
    const byId = new Map(hits.map((h) => [h.id, h] as const));
    const out: RankResult = { best: [], conceptual: [], related: [], source: 'llm' };
    const placed = new Set<string>();
    for (const m of normalizeMatches(raw)) {
      const hit = byId.get(m.id);
      if (!hit || placed.has(m.id)) continue;
      placed.add(m.id);
      out[m.tier].push({ hit, tier: m.tier, why: m.why.slice(0, 140) });
    }
    if (!placed.size) {
      console.warn('[askai] ranker returned no usable ids:', JSON.stringify(raw).slice(0, 300));
      return fallback;
    }
    // Within a tier keep the retrieval order (closest first), then cap.
    for (const t of ['best', 'conceptual', 'related'] as Tier[]) {
      out[t].sort((a, b) => a.hit.distance - b.hit.distance);
      out[t] = out[t].slice(0, CAPS[t]);
    }
    return out;
  } catch (e) {
    console.warn('[askai] ranker unavailable, using distance ranking:', e instanceof Error ? e.message : e);
    return fallback;
  }
}

export const rankedFlat = (r: RankResult): Ranked[] => [...r.best, ...r.conceptual, ...r.related];
