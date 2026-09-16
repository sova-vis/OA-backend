/**
 * Retrieval for Ask-AI, in-process over the Supabase REST client (pgvector via
 * the match_ask_ai_chunks RPC). Ported from the Python retrieve.py.
 *
 * Every search is scoped to ONE level (olevel|alevel) so O- and A-level results
 * never mix, and optionally to a subject.
 */
import { supabase } from '../supabase';
import { embedQuery } from './embed';

export interface HitMeta {
  subject: string | null; year: number | null; session: string | null;
  paper: string | null; variant: string | null; question_number: string | null;
  topic: string | null; level: string | null; type: string | null;
  source_file: string | null; page: number | null;
}
export interface Hit { text: string; metadata: HitMeta; distance: number; verified: boolean; }
export interface Occurrence extends HitMeta { question_text: string; }
export interface RouteResult {
  intent: 'paper_lookup' | 'general_qa';
  hits: Hit[];
  occurrences?: Occurrence[];
  year_limit?: number;
}

const PAPER_LOOKUP_RE = new RegExp([
  '\\bwhich years?\\b', '\\bwhat years?\\b', '\\bhow many times\\b', '\\bhow often\\b',
  '\\bhas .* (been )?asked\\b', '\\bwas .* asked\\b', '\\bpast papers?\\b',
  '\\bprevious (years?|papers?|exams?)\\b', '\\bshow me questions?\\b',
  '\\bfind questions?\\b', '\\blist questions?\\b',
].join('|'), 'i');

const YEAR_LIMIT_RE = /\b(?:last|past|previous|recent)\s+(\d{1,2})\s+years?\b/i;
const DEFAULT_YEAR_LIMIT = 5;

export function requestedYearLimit(query: string): number {
  const m = query.match(YEAR_LIMIT_RE);
  return m ? parseInt(m[1], 10) : DEFAULT_YEAR_LIMIT;
}

// Union of O/A-level subject names, longest first so "Business Studies" /
// "Computer Science" match before a shorter substring would.
const SUBJECTS = [
  'Accounting', 'Additional Maths', 'Art and Design', 'Biology', 'Business Studies',
  'Business', 'Chemistry', 'Commerce', 'Computer Science', 'Economics',
  'English General Paper', 'English Language', 'English', 'Environmental Management',
  'Further Mathematics', 'Geography', 'Global Perspectives', 'History',
  'Information Technology', 'Islamiyat', 'Law', 'Literature in English',
  'Mathematics', 'Pakistan Studies', 'Physics', 'Psychology', 'Religious Studies',
  'Sociology', 'Statistics',
];
const SUBJECT_RES = SUBJECTS
  .map((s) => [s, new RegExp('\\b' + s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i')] as const)
  .sort((a, b) => b[0].length - a[0].length);

const ACRONYM_RE = /\b[A-Z]{2,6}\b/g;
const GENERIC_WORDS = new Set([
  'in', 'the', 'a', 'an', 'was', 'were', 'is', 'are', 'asked', 'for', 'of', 'on',
  'about', 'come', 'up', 'to', 'this', 'that', 'topic', 'questions', 'question',
  'how', 'many', 'times', 'did', 'has', 'have', 'had', 'been', 'being', 'which',
  'what', 'years', 'year', 'show', 'find', 'list', 'past', 'papers', 'paper',
  'and', 'related', 'does', 'do',
]);

export function detectSubject(query: string): string | null {
  for (const [name, re] of SUBJECT_RES) if (re.test(query)) return name;
  return null;
}

export function extractCoreTopic(query: string, subject: string | null): string {
  let text = query.replace(YEAR_LIMIT_RE, ' ');
  if (subject) text = text.replace(new RegExp(subject.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'ig'), ' ');
  text = text.replace(/[?.!,]/g, ' ');
  return text.split(/\s+/).filter((w) => w && !GENERIC_WORDS.has(w.toLowerCase())).join(' ').trim();
}

export function normalizeLevel(level?: string | null): string | null {
  if (!level) return null;
  const s = String(level).toLowerCase().replace(/[^a-z]/g, '');
  if (s.startsWith('o')) return 'olevel';
  if (s.startsWith('a')) return 'alevel';
  return null;
}

function classifyIntent(query: string): 'paper_lookup' | 'general_qa' {
  return PAPER_LOOKUP_RE.test(query) ? 'paper_lookup' : 'general_qa';
}

const SELECT_COLS =
  'question_id,level,subject,type,exam_year,session,paper,variant,question_number,topic,content';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function toHit(r: any, verified: boolean): Hit {
  return {
    text: r.content || '',
    metadata: {
      subject: r.subject ?? null, year: r.exam_year ?? null, session: r.session ?? null,
      paper: r.paper ?? null, variant: r.variant ?? null, question_number: r.question_number ?? null,
      topic: r.topic ?? null, level: r.level ?? null, type: r.type ?? null,
      source_file: null, page: null,
    },
    distance: Number(r.distance ?? 0),
    verified,
  };
}

export class AskAiIndexUnavailable extends Error {}

async function search(
  query: string, subject: string | null, level: string | null, topK: number, coreTopic: string | null,
): Promise<Hit[]> {
  const qvec = await embedQuery(query);
  const lvl = normalizeLevel(level);

  const rpc = await supabase.rpc('match_ask_ai_chunks', {
    query_embedding: qvec, match_level: lvl, match_subject: subject, match_count: topK,
  });
  if (rpc.error) {
    throw new AskAiIndexUnavailable(
      `Ask-AI search failed (is migration 024 applied and the index built?): ${rpc.error.message}`,
    );
  }
  const semantic = (rpc.data as unknown[]) || [];

  // Exact-term boost: acronyms + the query's core topic, so a genuine match just
  // outside top-k (or an acronym embeddings rank poorly) isn't dropped.
  const terms = new Set<string>(query.match(ACRONYM_RE) || []);
  if (coreTopic && coreTopic.length >= 4) terms.add(coreTopic);
  const verifiedIds = new Set<string>();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const exact: any[] = [];
  for (const term of terms) {
    let q = supabase.from('ask_ai_chunks').select(SELECT_COLS).ilike('content', `*${term}*`).limit(200);
    if (lvl) q = q.eq('level', lvl);
    if (subject) q = q.eq('subject', subject);
    const { data } = await q;
    for (const r of (data as unknown[]) || []) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      verifiedIds.add((r as any).question_id);
      exact.push(r);
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const byId = new Map<string, Hit>();
  for (const r of semantic) byId.set((r as any).question_id, toHit(r, false));
  for (const r of exact) if (!byId.has(r.question_id)) byId.set(r.question_id, toHit(r, true));
  for (const id of verifiedIds) { const h = byId.get(id); if (h) h.verified = true; }

  const hits = [...byId.values()].sort((a, b) => a.distance - b.distance);
  const verifiedHits = hits.filter((h) => h.verified);
  const otherHits = hits.filter((h) => !h.verified);
  return [...verifiedHits, ...otherHits.slice(0, Math.max(0, topK - verifiedHits.length))];
}

export function paperLookupSummary(hits: Hit[], yearLimit?: number): Occurrence[] {
  const verified = hits.filter((h) => h.verified);
  const candidates = verified.length ? verified : hits;
  const seen = new Set<string>();
  const occ: Occurrence[] = [];
  for (const h of candidates) {
    const m = h.metadata;
    if (m.question_number == null) continue;
    const key = [m.subject, m.year, m.session, m.paper, m.variant, m.question_number].join('|');
    if (seen.has(key)) continue;
    seen.add(key);
    occ.push({ ...m, question_text: h.text });
  }
  let out = occ;
  if (yearLimit != null && occ.length) {
    const cutoff = new Date().getFullYear() - yearLimit + 1;
    out = occ.filter((m) => (m.year || 0) >= cutoff);
  }
  out.sort((a, b) => (a.year || 0) - (b.year || 0) || (a.session || '').localeCompare(b.session || ''));
  return out;
}

export async function route(
  query: string, opts: { subject?: string | null; level?: string | null; topK?: number } = {},
): Promise<RouteResult> {
  const topK = opts.topK ?? 10;
  const intent = classifyIntent(query);
  const subject = opts.subject ?? detectSubject(query);
  const searchQuery = query.replace(YEAR_LIMIT_RE, '').trim();
  const coreTopic = extractCoreTopic(query, subject);
  const hits = await search(searchQuery, subject, opts.level ?? null, topK, coreTopic);
  const result: RouteResult = { intent, hits };
  if (intent === 'paper_lookup') {
    const yearLimit = requestedYearLimit(query);
    result.year_limit = yearLimit;
    result.occurrences = paperLookupSummary(hits, yearLimit);
  }
  return result;
}
