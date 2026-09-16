/**
 * Query planner for Ask-AI — the "LLM first" step. Before touching the index it
 * decides what the student actually wants (find matching questions / explain /
 * solve / make practice questions / just chatting), whether a past-paper search
 * would help at all, and if so how to search (exam-wording phrasings, literal
 * keywords, subject, year window, question type).
 *
 * If the LLM is unavailable the heuristic plan keeps Ask AI working.
 */
import { chatJson, type Turn } from './llm';
import {
  SUBJECTS_BY_LEVEL, detectSubject, heuristicKeywords, looksLikePaperLookup,
  requestedYearFrom, resolveSubject, stripYearPhrase, type Level,
} from './retrieve';

export type Intent = 'find_questions' | 'explain' | 'solve' | 'make_questions' | 'chat';
export type Mode = 'ask' | 'find';
/** what the student is searching with: one specific question, or a topic/area */
export type SearchKind = 'question' | 'topic';

export interface QueryPlan {
  intent: Intent;
  needsSearch: boolean;
  subject: string | null;
  topic: string | null;
  searchQueries: string[];
  keywords: string[];
  yearFrom: number | null;
  yearTo: number | null;
  questionType: 'mcq' | 'structured' | null;
  /** how many questions the student asked for ("only 5", "3 MCQs") */
  count: number | null;
  /** a specific earlier-shown question the student is referring to (follow-ups) */
  reference: string | null;
  kind: SearchKind;
  /** direct reply when no search is needed (chat / meta) */
  reply: string | null;
  source: 'llm' | 'heuristic';
}

const INTENTS: Intent[] = ['find_questions', 'explain', 'solve', 'make_questions', 'chat'];

function plannerSystem(level: Level, mode: Mode): string {
  const year = new Date().getFullYear();
  const levelName = level === 'olevel' ? 'O Level' : 'A Level';
  return `You are the query planner for "Ask AI", a Cambridge ${levelName} past-paper study assistant. The student is on the "${mode === 'find' ? 'Find' : 'Ask'}" tab. Today is ${new Date().toISOString().slice(0, 10)} (current year ${year}).
Find = look up which real past-paper questions match a topic or question. Ask = explain, solve, or practise, grounded in real past-paper questions.
The searchable index holds ${levelName} past-paper questions (2010-${year}) for exactly these subjects: ${SUBJECTS_BY_LEVEL[level].join(', ')}. Each question has a subject, year, session (May/June, Oct/Nov, Feb/March), paper, variant, question number, topic and type (mcq or structured).

Decide how to handle the student's latest message (use the conversation so far to resolve follow-ups such as "another one" or "same topic but MCQs"). Return JSON:
{
  "intent": "find_questions" | "explain" | "solve" | "make_questions" | "chat",
  "needs_search": true | false,
  "subject": <one of the listed subjects, or null if unclear>,
  "topic": <short canonical topic phrase in syllabus wording, or null>,
  "search_queries": [<1-3 phrasings, in the wording exam questions actually use, that would retrieve matching past-paper questions>],
  "keywords": [<0-4 exact words, acronyms or formulae a matching question would literally contain, e.g. "electrolysis", "brine", "NaCl">],
  "year_from": <int or null>, "year_to": <int or null>,
  "question_type": "mcq" | "structured" | null,
  "count": <number of questions the student asked for ("only 5", "3 MCQs"), or null>,
  "search_kind": "question" | "topic",
  "reference": <paper reference of ONE specific earlier-shown question the student is referring to, copied exactly from the conversation, or null>,
  "reply": <string or null>
}
search_kind is "question" when the student gave or described ONE specific exam question (pasted text, or a reference to one shown earlier), and "topic" when they named a topic, concept or syllabus area.
Rules:
- The tab is a hint, not a constraint — decide from what the student actually wrote.
- Find tab: "find_questions" for a topic, concept, pasted question, or any request for N questions to practise/prepare (set "count"). But an explicit "explain …" / "solve …" / "why is … wrong" is "explain" or "solve" even here.
- Ask tab: "solve" when the message contains an actual exam-style question to answer; "make_questions" when they want practice questions/MCQs; "explain" for concept or topic explanations, revision, or "why do I lose marks on X"; "find_questions" when they ask which papers/years something appeared in.
- "chat" only for greetings, thanks, meta questions about the tool, or off-topic messages; then needs_search=false and "reply" is a brief friendly reply (max 3 sentences; if off-topic, say what you can help with). Every other intent has needs_search=true — real past-paper grounding is the point.
- Follow-ups: when the student refers to questions shown earlier ("one of them", "the first one", "the 2023 one", "Q5"), pick ONE concrete question from the conversation — copy its paper reference into "reference" and use its question text as the first search query so it can be retrieved. "Explain one of them" → intent "explain".
- year_from/year_to only when a period is stated: "last 3 years" → year_from=${year - 2}; "2020 to 2022" → 2020/2022; "in 2023" → 2023/2023.
- Never output a subject that is not in the list. If the student names a subject not offered at this level, set subject=null and note it in "topic".`;
}

function historyBlock(history: Turn[]): string {
  if (!history.length) return '';
  // Assistant turns keep more text: their paper references are what follow-ups point at.
  const lines = history.slice(-6).map((t) =>
    `${t.role === 'user' ? 'Student' : 'Assistant'}: ${t.content.replace(/\s+/g, ' ').slice(0, t.role === 'user' ? 300 : 800)}`);
  return `Conversation so far:\n${lines.join('\n')}\n\n`;
}

// Paper references exactly as the UI prints them (retrieve.ts refLabel).
const REF_RE = /\b(?:[A-Z][A-Za-z]+(?: (?:and|in|of|[A-Z][A-Za-z]+))* )(20[0-3]\d) (?:May\/June|Oct\/Nov|Feb\/March) Paper \d{1,2}(?: Variant \d)? Q(\d{1,2})\b/g;
const FOLLOW_UP_RE = /\b(them|those|one of|any one|first|second|third|fourth|fifth|last one|that one|this one|the above|it|above)\b/i;
const ORDINALS: Record<string, number> = { first: 0, second: 1, third: 2, fourth: 3, fifth: 4, '1st': 0, '2nd': 1, '3rd': 2, '4th': 3, '5th': 4 };

/**
 * Deterministic fallback for follow-ups ("explain one of them", "solve the
 * third one", "the 2023 one", "Q15"): pick a paper reference from the most
 * recent assistant answer. Small models often leave `reference` empty here.
 */
export function inferReferenceFromHistory(query: string, history: Turn[]): string | null {
  if (query.length > 160 || !FOLLOW_UP_RE.test(query)) return null;
  const lastAssistant = [...history].reverse().find((t) => t.role === 'assistant');
  if (!lastAssistant) return null;
  const refs = [...lastAssistant.content.matchAll(REF_RE)].map((m) => ({ ref: m[0], year: m[1], q: m[2] }));
  if (!refs.length) return null;
  const q = query.toLowerCase();
  const qn = q.match(/\bq(?:uestion)?\s*(\d{1,2})\b/)?.[1];
  if (qn) { const hit = refs.find((r) => r.q === qn); if (hit) return hit.ref; }
  const year = q.match(/\b(20[0-3]\d)\b/)?.[1];
  if (year) { const hit = refs.find((r) => r.year === year); if (hit) return hit.ref; }
  for (const [word, idx] of Object.entries(ORDINALS)) {
    if (new RegExp(`\\b${word}\\b`).test(q) && refs[idx]) return refs[idx].ref;
  }
  if (/\blast\b/.test(q)) return refs[refs.length - 1].ref;
  return refs[0].ref;
}

/**
 * Year window straight from the text — small models' idea of "this year" lags
 * their training data, so "last 4 years" is computed here, not by the LLM.
 * Explicit years ("2021 to 2023", "in 2024") are read literally.
 */
export function deriveYearWindow(query: string): { yearFrom: number | null; yearTo: number | null } {
  const year = new Date().getFullYear();
  const fromPhrase = requestedYearFrom(query);
  if (fromPhrase != null) return { yearFrom: fromPhrase, yearTo: null };
  const explicit = [...query.matchAll(/\b(20[0-3]\d)\b/g)].map((m) => parseInt(m[1], 10)).filter((y) => y <= year + 1);
  if (explicit.length) return { yearFrom: Math.min(...explicit), yearTo: Math.max(...explicit) };
  if (/\b(this|current) year\b/i.test(query)) return { yearFrom: year, yearTo: year };
  if (/\blast year\b/i.test(query)) return { yearFrom: year - 1, yearTo: year - 1 };
  return { yearFrom: null, yearTo: null };
}

function asInt(v: unknown, min: number, max: number): number | null {
  const n = typeof v === 'number' ? v : typeof v === 'string' ? parseInt(v, 10) : NaN;
  return Number.isFinite(n) && n >= min && n <= max ? Math.round(n) : null;
}
const asStrings = (v: unknown, maxItems: number, maxLen: number): string[] =>
  Array.isArray(v) ? v.filter((s): s is string => typeof s === 'string' && s.trim().length > 0)
    .map((s) => s.trim().slice(0, maxLen)).slice(0, maxItems) : [];

export function heuristicPlan(query: string, mode: Mode, level: Level, scopeSubject: string | null, history: Turn[] = []): QueryPlan {
  const subject = scopeSubject ?? detectSubject(query, level);
  const reference = inferReferenceFromHistory(query, history);
  const intent: Intent = /^\s*(explain|why|how|describe)\b/i.test(query) || (reference && /\b(explain|solve|answer|work)/i.test(query)) ? 'explain'
    : mode === 'find' || looksLikePaperLookup(query) ? 'find_questions'
    : /\b(mcqs?|questions?|quiz|practi[cs]e)\b/i.test(query) && /\b(give|make|generate|create|want|need|some|\d+)\b/i.test(query) ? 'make_questions'
    : 'explain';
  const countMatch = query.match(/\b(\d{1,2})\s*(?:mcqs?|questions?)\b/i);
  const years = deriveYearWindow(query);
  return {
    intent, needsSearch: true, subject, topic: null,
    searchQueries: [stripYearPhrase(query)],
    keywords: heuristicKeywords(query, subject),
    yearFrom: years.yearFrom, yearTo: years.yearTo,
    questionType: /\bmcqs?\b|multiple[- ]choice/i.test(query) ? 'mcq' : null,
    count: countMatch ? Math.min(8, parseInt(countMatch[1], 10)) : null,
    reference,
    // A long message with a question mark or numbers reads like a pasted question.
    kind: reference || (query.length > 80 && /[?]|\d/.test(query)) ? 'question' : 'topic',
    reply: null, source: 'heuristic',
  };
}

export async function planQuery(
  query: string,
  opts: { mode: Mode; level: Level; subject?: string | null; history?: Turn[] },
): Promise<QueryPlan> {
  const scopeSubject = resolveSubject(opts.subject, opts.level);
  const history = opts.history || [];
  const fallback = heuristicPlan(query, opts.mode, opts.level, scopeSubject, history);
  const year = new Date().getFullYear();
  try {
    const raw = await chatJson<Record<string, unknown>>(
      plannerSystem(opts.level, opts.mode),
      `${historyBlock(opts.history || [])}Latest message: ${query.slice(0, 1500)}` +
        (scopeSubject ? `\n(The student has scoped this chat to ${scopeSubject}.)` : ''),
      { tier: 'fast', maxTokens: 500, temperature: 0, timeoutMs: 25_000 },
    );
    const intent = INTENTS.includes(raw.intent as Intent) ? (raw.intent as Intent) : fallback.intent;
    // Only chat/meta skips the index. Every subject-content intent is grounded in
    // real past-paper questions — even a pasted question gets "this appeared in…".
    const needsSearch = intent !== 'chat';
    const searchQueries = asStrings(raw.search_queries, 3, 200);
    // Trust the LLM's years only when the text has no usable signal of its own.
    const derived = deriveYearWindow(query);
    const llmFrom = asInt(raw.year_from, 2000, year + 1);
    const llmTo = asInt(raw.year_to, 2000, year + 1);
    const hasDerived = derived.yearFrom != null || derived.yearTo != null;
    const yearFrom = hasDerived ? derived.yearFrom : llmFrom;
    const yearTo = hasDerived ? derived.yearTo : llmTo;
    const qt = raw.question_type === 'mcq' || raw.question_type === 'structured' ? raw.question_type : null;
    const reply = typeof raw.reply === 'string' && raw.reply.trim() ? raw.reply.trim().slice(0, 700) : null;
    // The model's reference if it gave one; otherwise resolve "one of them" ourselves.
    const reference = (typeof raw.reference === 'string' && raw.reference.trim() ? raw.reference.trim().slice(0, 120) : null)
      ?? inferReferenceFromHistory(query, history);
    return {
      intent,
      needsSearch,
      // The chat's subject scope always wins over the planner's guess.
      subject: scopeSubject ?? resolveSubject(typeof raw.subject === 'string' ? raw.subject : null, opts.level),
      topic: typeof raw.topic === 'string' && raw.topic.trim() ? raw.topic.trim().slice(0, 120) : null,
      searchQueries: searchQueries.length ? searchQueries : [stripYearPhrase(query)],
      keywords: asStrings(raw.keywords, 4, 40),
      yearFrom: yearFrom != null && yearTo != null && yearFrom > yearTo ? yearTo : yearFrom,
      yearTo,
      questionType: qt,
      count: asInt(raw.count, 1, 8),
      reference,
      kind: raw.search_kind === 'question' || reference ? 'question' : 'topic',
      reply,
      source: 'llm',
    };
  } catch (e) {
    console.warn('[askai] planner unavailable, using heuristic plan:', e instanceof Error ? e.message : e);
    return fallback;
  }
}
