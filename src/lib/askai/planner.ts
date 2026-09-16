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
  /** how many questions the student asked for (make_questions) */
  count: number | null;
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
  "count": <number of questions the student asked for, or null>,
  "reply": <string or null>
}
Rules:
- Find tab: intent "find_questions" with needs_search=true for anything that could be a topic, concept or exam question. Only greetings, thanks, or questions about how the tool works are "chat".
- Ask tab: "solve" when the message contains an actual exam-style question to answer; "make_questions" when they want practice questions/MCQs; "explain" for concept or topic explanations, revision, or "why do I lose marks on X"; "chat" only for greetings, meta questions or off-topic messages.
- needs_search is true for every subject-content intent — real past-paper grounding is the point. For "chat" set needs_search=false and write "reply": a brief friendly reply (max 3 sentences; if off-topic, say what you can help with).
- year_from/year_to only when a period is stated: "last 3 years" → year_from=${year - 2}; "2020 to 2022" → 2020/2022; "in 2023" → 2023/2023.
- Never output a subject that is not in the list. If the student names a subject not offered at this level, set subject=null and note it in "topic".`;
}

function historyBlock(history: Turn[]): string {
  if (!history.length) return '';
  const lines = history.slice(-6).map((t) =>
    `${t.role === 'user' ? 'Student' : 'Assistant'}: ${t.content.replace(/\s+/g, ' ').slice(0, 300)}`);
  return `Conversation so far:\n${lines.join('\n')}\n\n`;
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

export function heuristicPlan(query: string, mode: Mode, level: Level, scopeSubject: string | null): QueryPlan {
  const subject = scopeSubject ?? detectSubject(query, level);
  const intent: Intent = mode === 'find' || looksLikePaperLookup(query) ? 'find_questions'
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
    reply: null, source: 'heuristic',
  };
}

export async function planQuery(
  query: string,
  opts: { mode: Mode; level: Level; subject?: string | null; history?: Turn[] },
): Promise<QueryPlan> {
  const scopeSubject = resolveSubject(opts.subject, opts.level);
  const fallback = heuristicPlan(query, opts.mode, opts.level, scopeSubject);
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
      reply,
      source: 'llm',
    };
  } catch (e) {
    console.warn('[askai] planner unavailable, using heuristic plan:', e instanceof Error ? e.message : e);
    return fallback;
  }
}
