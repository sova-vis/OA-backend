/**
 * Retrieval for Ask-AI, in-process over the Supabase REST client (pgvector via
 * the match_ask_ai_chunks RPC + a keyword pass over ask_ai_chunks.content).
 *
 * Every search is scoped to ONE level (olevel|alevel) so O- and A-level results
 * never mix, and optionally to a subject. The planner (planner.ts) decides the
 * search phrasings and keywords; rank.ts sorts the candidates into tiers.
 */
import { supabase } from '../supabase';
import { embedQuery } from './embed';

export type Level = 'olevel' | 'alevel';

export interface HitMeta {
  subject: string | null; year: number | null; session: string | null;
  paper: string | null; variant: string | null; question_number: string | null;
  topic: string | null; level: string | null; type: string | null;
}
export interface Hit {
  id: string;
  text: string;
  metadata: HitMeta;
  /** cosine distance from the query (lower = closer); keyword-only rows get a neutral 0.6 */
  distance: number;
  /** true when the row literally contains one of the planner's keywords */
  verified: boolean;
}

// Subjects present in the index, per level (probed from ask_ai_chunks). The
// planner may only pick from these, so a hallucinated subject can't zero a search.
export const SUBJECTS_BY_LEVEL: Record<Level, string[]> = {
  olevel: [
    'Accounting', 'Art and Design', 'Biology', 'Business Studies', 'Chemistry', 'Commerce',
    'Computer Science', 'Economics', 'English Language', 'Environmental Management', 'Geography',
    'History', 'Islamiyat', 'Mathematics', 'Pakistan Studies', 'Physics', 'Religious Studies',
    'Sociology', 'Statistics',
  ],
  alevel: [
    'Accounting', 'Biology', 'Business', 'Chemistry', 'Computer Science', 'Economics',
    'English General Paper', 'English Language', 'Further Mathematics', 'Geography',
    'Global Perspectives', 'History', 'Information Technology', 'Law', 'Literature in English',
    'Mathematics', 'Physics', 'Psychology', 'Sociology',
  ],
};
export const ALL_SUBJECTS = [...new Set([...SUBJECTS_BY_LEVEL.olevel, ...SUBJECTS_BY_LEVEL.alevel])].sort();

// Common student shorthand → canonical index subject (level-specific ones are
// resolved in resolveSubject).
const SUBJECT_ALIASES: Record<string, string> = {
  maths: 'Mathematics', math: 'Mathematics', mathematics: 'Mathematics',
  'further maths': 'Further Mathematics', 'further mathematics': 'Further Mathematics',
  bio: 'Biology', biology: 'Biology', chem: 'Chemistry', chemistry: 'Chemistry',
  phy: 'Physics', phys: 'Physics', physics: 'Physics',
  eco: 'Economics', econ: 'Economics', economics: 'Economics',
  accounts: 'Accounting', accounting: 'Accounting',
  'pak studies': 'Pakistan Studies', 'pakistan studies': 'Pakistan Studies', pst: 'Pakistan Studies',
  islamiat: 'Islamiyat', islamiyat: 'Islamiyat',
  cs: 'Computer Science', computer: 'Computer Science', 'computer science': 'Computer Science', computing: 'Computer Science',
  business: 'Business', 'business studies': 'Business Studies',
  english: 'English Language', 'english language': 'English Language',
  gp: 'English General Paper', 'general paper': 'English General Paper', 'english general paper': 'English General Paper',
  lit: 'Literature in English', literature: 'Literature in English', 'english literature': 'Literature in English', 'literature in english': 'Literature in English',
  it: 'Information Technology', ict: 'Information Technology', 'information technology': 'Information Technology',
  env: 'Environmental Management', evm: 'Environmental Management', 'environmental management': 'Environmental Management',
  stats: 'Statistics', statistics: 'Statistics', socio: 'Sociology', sociology: 'Sociology',
  psych: 'Psychology', psychology: 'Psychology', rs: 'Religious Studies', 'religious studies': 'Religious Studies',
  geo: 'Geography', geography: 'Geography', law: 'Law', history: 'History', commerce: 'Commerce',
  art: 'Art and Design', 'art and design': 'Art and Design', 'global perspectives': 'Global Perspectives',
};

export function normalizeLevel(level?: string | null): Level | null {
  if (!level) return null;
  const s = String(level).toLowerCase().replace(/[^a-z]/g, '');
  if (s.startsWith('o')) return 'olevel';
  if (s.startsWith('a')) return 'alevel';
  return null;
}

/** Map any subject spelling ("Chem", "Business (A Level)") to the index name at this level, or null. */
export function resolveSubject(raw: string | null | undefined, level: Level): string | null {
  if (!raw) return null;
  const s = String(raw).replace(/\s*\((?:A|O)[ -]?Levels?\)\s*$/i, '').trim();
  if (!s) return null;
  const list = SUBJECTS_BY_LEVEL[level];
  const lower = s.toLowerCase();
  let canonical = list.find((n) => n.toLowerCase() === lower) || SUBJECT_ALIASES[lower] || null;
  if (!canonical) canonical = list.find((n) => n.toLowerCase().startsWith(lower)) || null;
  if (!canonical) return null;
  // Business is "Business Studies" at O Level and "Business" at A Level.
  if (/^business/i.test(canonical)) return level === 'olevel' ? 'Business Studies' : 'Business';
  return list.includes(canonical) ? canonical : null;
}

// Subject mentions in free text, longest first so "Business Studies" wins over "Business".
const SUBJECT_MENTIONS = Object.keys(SUBJECT_ALIASES)
  .concat(ALL_SUBJECTS.map((s) => s.toLowerCase()))
  .filter((v, i, arr) => arr.indexOf(v) === i && v.length > 2)
  .sort((a, b) => b.length - a.length)
  .map((name) => [name, new RegExp('\\b' + name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i')] as const);

export function detectSubject(query: string, level: Level): string | null {
  for (const [name, re] of SUBJECT_MENTIONS) {
    if (re.test(query)) {
      const resolved = resolveSubject(name, level);
      if (resolved) return resolved;
    }
  }
  return null;
}

// ---- heuristics shared with the planner's no-LLM fallback ----------------------

const PAPER_LOOKUP_RE = new RegExp([
  '\\bwhich years?\\b', '\\bwhat years?\\b', '\\bhow many times\\b', '\\bhow often\\b',
  '\\bhas .* (been )?asked\\b', '\\bwas .* asked\\b', '\\bpast papers?\\b',
  '\\bprevious (years?|papers?|exams?)\\b', '\\bshow me questions?\\b',
  '\\bfind questions?\\b', '\\blist questions?\\b', '\\bwhere (did|does|has) .* (come|appear)\\b',
].join('|'), 'i');
export const looksLikePaperLookup = (q: string) => PAPER_LOOKUP_RE.test(q);

const YEAR_LIMIT_RE = /\b(?:last|past|previous|recent)\s+(\d{1,2})\s+years?\b/i;
export function requestedYearFrom(query: string): number | null {
  const m = query.match(YEAR_LIMIT_RE);
  return m ? new Date().getFullYear() - parseInt(m[1], 10) + 1 : null;
}
export const stripYearPhrase = (q: string) => q.replace(YEAR_LIMIT_RE, ' ').replace(/\s+/g, ' ').trim();

const ACRONYM_RE = /\b[A-Z]{2,6}\b/g;
const GENERIC_WORDS = new Set([
  'in', 'the', 'a', 'an', 'was', 'were', 'is', 'are', 'asked', 'for', 'of', 'on', 'about', 'come',
  'up', 'to', 'this', 'that', 'topic', 'questions', 'question', 'how', 'many', 'times', 'did',
  'has', 'have', 'had', 'been', 'being', 'which', 'what', 'years', 'year', 'show', 'find', 'list',
  'past', 'papers', 'paper', 'and', 'related', 'does', 'do', 'me', 'explain', 'give', 'please',
  'with', 'from', 'level', 'exam', 'exams', 'mcq', 'mcqs',
]);

/** Query minus subject/years/filler — used as a keyword when the planner is unavailable. */
export function extractCoreTopic(query: string, subject: string | null): string {
  let text = stripYearPhrase(query);
  if (subject) text = text.replace(new RegExp(subject.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'ig'), ' ');
  text = text.replace(/[?.!,;:"']/g, ' ');
  return text.split(/\s+/).filter((w) => w && !GENERIC_WORDS.has(w.toLowerCase())).join(' ').trim();
}

export function heuristicKeywords(query: string, subject: string | null): string[] {
  const terms = new Set<string>(query.match(ACRONYM_RE) || []);
  const core = extractCoreTopic(query, subject);
  if (core && core.length >= 4 && core.split(' ').length <= 3) terms.add(core);
  return [...terms].slice(0, 4);
}

// ---- search --------------------------------------------------------------------

const SELECT_COLS =
  'question_id,level,subject,type,exam_year,session,paper,variant,question_number,topic,content';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function toHit(r: any, verified: boolean, distance?: number): Hit {
  return {
    id: String(r.question_id),
    text: r.content || '',
    metadata: {
      subject: r.subject ?? null, year: r.exam_year ?? null, session: r.session ?? null,
      paper: r.paper ?? null, variant: r.variant ?? null, question_number: r.question_number ?? null,
      topic: r.topic ?? null, level: r.level ?? null, type: r.type ?? null,
    },
    distance: distance ?? Number(r.distance ?? 0.6),
    verified,
  };
}

export class AskAiIndexUnavailable extends Error {}

export interface SearchOptions {
  level: Level;
  subject?: string | null;
  topK?: number;
  /** exact terms a matching question would contain — boosts semantic hits and pulls in literal mentions */
  keywords?: string[];
  questionType?: 'mcq' | 'structured' | null;
}

const KEYWORD_CLEAN_RE = /[^\p{L}\p{N} +\-./]/gu;

/** One semantic search + keyword pass, level-scoped, returning up to topK hits (closest first). */
export async function search(query: string, opts: SearchOptions): Promise<Hit[]> {
  const topK = opts.topK ?? 12;
  const qvec = await embedQuery(query);

  const rpc = await supabase.rpc('match_ask_ai_chunks', {
    query_embedding: qvec, match_level: opts.level, match_subject: opts.subject ?? null,
    match_count: Math.max(topK * 3, 30),
  });
  if (rpc.error) {
    throw new AskAiIndexUnavailable(
      `Ask-AI search failed (is migration 024 applied and the index built?): ${rpc.error.message}`,
    );
  }
  const byId = new Map<string, Hit>();
  for (const r of (rpc.data as unknown[]) || []) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const hit = toHit(r as any, false);
    byId.set(hit.id, hit);
  }

  // Keyword pass: semantic hits that literally contain a term are "verified" and
  // nudged up; literal mentions the embedding missed are appended (most recent
  // first) so a genuine occurrence is never dropped.
  const terms = [...new Set((opts.keywords || [])
    .map((t) => t.replace(KEYWORD_CLEAN_RE, ' ').replace(/\s+/g, ' ').trim())
    .filter((t) => t.length >= 3))].slice(0, 4);
  const keywordIds = new Set<string>();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const literalOnly = new Map<string, any>();
  await Promise.all(terms.map(async (term) => {
    let q = supabase.from('ask_ai_chunks').select(SELECT_COLS)
      .eq('level', opts.level).ilike('content', `*${term}*`)
      .order('exam_year', { ascending: false }).limit(120);
    if (opts.subject) q = q.eq('subject', opts.subject);
    if (opts.questionType) q = q.eq('type', opts.questionType);
    const { data, error } = await q;
    if (error) { console.warn('[askai] keyword pass failed:', error.message); return; }
    for (const r of (data as unknown[]) || []) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const row = r as any;
      keywordIds.add(String(row.question_id));
      if (!byId.has(String(row.question_id))) literalOnly.set(String(row.question_id), row);
    }
  }));
  for (const id of keywordIds) {
    const h = byId.get(id);
    if (h) { h.verified = true; h.distance = Math.max(0, h.distance - 0.04); }
  }

  let ranked = [...byId.values()].sort((a, b) => a.distance - b.distance);
  if (opts.questionType) {
    // Prefer the requested type, but don't empty the result set over it.
    const typed = ranked.filter((h) => h.metadata.type === opts.questionType);
    if (typed.length >= Math.min(4, topK)) ranked = typed;
  }
  const tail = [...literalOnly.values()]
    .sort((a, b) => (b.exam_year || 0) - (a.exam_year || 0))
    .slice(0, Math.max(4, Math.floor(topK / 3)))
    .map((r) => toHit(r, true, 0.6));
  return [...ranked, ...tail].slice(0, topK);
}

/** Run several phrasings of the same need and merge (closest distance wins). */
export async function searchMany(queries: string[], opts: SearchOptions): Promise<Hit[]> {
  const topK = opts.topK ?? 12;
  const phrasings = [...new Set(queries.map((q) => q.trim()).filter(Boolean))].slice(0, 3);
  const results = await Promise.all(phrasings.map((q) => search(q, { ...opts, topK })));
  const merged = new Map<string, Hit>();
  for (const hits of results) {
    for (const h of hits) {
      const prev = merged.get(h.id);
      if (!prev) merged.set(h.id, { ...h });
      else { prev.distance = Math.min(prev.distance, h.distance); prev.verified = prev.verified || h.verified; }
    }
  }
  return [...merged.values()].sort((a, b) => a.distance - b.distance).slice(0, topK);
}

export function filterYears(hits: Hit[], from: number | null, to: number | null): Hit[] {
  if (from == null && to == null) return hits;
  return hits.filter((h) => {
    const y = h.metadata.year || 0;
    return (from == null || y >= from) && (to == null || y <= to);
  });
}

// ---- presentation helpers ------------------------------------------------------

export const prettySession = (s: string | null) => (s || '').replace(/_/g, '/');
export const prettyPaper = (p: string | null) => (p || '').replace(/_/g, ' ');
export function prettyVariant(v: string | null): string {
  if (!v) return '';
  const s = String(v).trim();
  return /^\d+$/.test(s) ? `Variant ${s}` : s.replace(/_/g, ' ');
}

/** "Chemistry 2023 May/June Paper 2 Variant 1 Q8" */
export function refLabel(m: HitMeta): string {
  return [m.subject, m.year, prettySession(m.session), prettyPaper(m.paper), prettyVariant(m.variant),
    m.question_number ? `Q${m.question_number}` : '']
    .filter(Boolean).join(' ');
}

export function dedupeByPaperQuestion(hits: Hit[]): Hit[] {
  const seen = new Set<string>();
  return hits.filter((h) => {
    const m = h.metadata;
    const key = [m.subject, m.year, m.session, m.paper, m.variant, m.question_number].join('|');
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
