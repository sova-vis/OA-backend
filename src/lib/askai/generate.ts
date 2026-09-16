/**
 * Ask-AI orchestrator — LLM first. Every request goes: plan → (search → rank) →
 * answer.
 *
 *  - plan   (planner.ts): what does the student want, and is the past-paper index
 *           worth searching at all? Greetings/meta get a direct reply, no search.
 *  - search (retrieve.ts): level-scoped pgvector + keyword retrieval on the
 *           planner's exam-wording phrasings.
 *  - rank   (rank.ts): the LLM sorts candidates into best / conceptual / related
 *           matches, each with the paper it appeared in and a one-line reason.
 *  - answer: Find returns the ranked matches; Ask explains / solves / presents
 *           real questions, citing papers and never inventing references.
 */
import { chatText, type Turn } from './llm';
import { planQuery, type Intent, type Mode, type QueryPlan, type SearchKind } from './planner';
import { TIER_LABELS, TIER_TITLES, rankCandidates, rankedFlat, type RankResult, type Ranked, type Tier } from './rank';
import {
  dedupeByPaperQuestion, fetchByReference, filterYears, normalizeLevel, refLabel, searchMany,
  type Hit, type Level,
} from './retrieve';

export { chatText as callLLM };

export interface MatchOut {
  id: string;
  tier: Tier;
  why: string;
  subject: string | null; year: number | null; session: string | null; paper: string | null;
  variant: string | null; questionNumber: string | null; topic: string | null;
  /** kept for the existing citations table */
  topicGeneral: string | null;
  type: string | null; level: string | null;
  reference: string;
  preview: string;
  text: string;
}

export interface AskAiInput {
  query: string;
  mode: Mode;
  level?: string | null;
  subject?: string | null;
  history?: unknown;
}

export interface AskAiOutput {
  type: 'exam_question' | 'smalltalk';
  mode: Mode;
  intent: Intent;
  /** full Markdown answer (also what chat history keeps) */
  answer: string;
  /** one-line lead for Find results (the UI renders the structured matches under it) */
  summary: string | null;
  level: Level;
  subject: string | null;
  topic: string | null;
  searched: boolean;
  candidates: number;
  matches: { best: MatchOut[]; conceptual: MatchOut[]; related: MatchOut[] };
  /** UI section titles / badge labels for the tiers (differ for topic vs question searches) */
  tierTitles: Record<Tier, string>;
  tierLabels: Record<Tier, string>;
  citations: MatchOut[];
  planner: 'llm' | 'heuristic';
  ranker: 'llm' | 'heuristic' | null;
  /** what the planner decided (no secrets) — handy in the browser's network tab */
  plan: {
    intent: Intent; kind: SearchKind; subject: string | null; topic: string | null;
    searchQueries: string[]; keywords: string[]; count: number | null; reference: string | null;
    yearFrom: number | null; yearTo: number | null; questionType: string | null;
  };
}

const planOut = (p: QueryPlan): AskAiOutput['plan'] => ({
  intent: p.intent, kind: p.kind, subject: p.subject, topic: p.topic, searchQueries: p.searchQueries,
  keywords: p.keywords, count: p.count, reference: p.reference, yearFrom: p.yearFrom, yearTo: p.yearTo,
  questionType: p.questionType,
});

const levelName = (l: Level) => (l === 'olevel' ? 'O Level' : 'A Level');
const LEADING_QNUM_RE = /^\s*\d{1,2}\s*/;
const stripRedundantOrPrefix = (t: string) => t.replace(/^(\s*-\s+)or\b:?\s*/gim, '$1');

// Models sometimes print the prompt's internal section labels ("**Explanation
// (PART 1)**", "PART 2 — Worked examples:") despite being told not to.
// Only whole label lines are removed — a label with content on the same line
// is left alone rather than risk deleting the content.
const PART_LABEL_LINE_RE = /^[ \t]*(?:\*\*|__|#{1,6}[ \t]*)?[ \t]*(?:PART[ \t]*[12]\b[^\n:]{0,40}:?|Explanation(?:[ \t]*\(PART[ \t]*1\))?[ \t]*:?)[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*\r?\n/gim;
const stripPartLabels = (t: string) => t.replace(PART_LABEL_LINE_RE, '').replace(/^\s+/, '');

const GREEK: Record<string, string> = {
  alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', Delta: 'Δ', theta: 'θ', lambda: 'λ', mu: 'μ',
  pi: 'π', rho: 'ρ', sigma: 'σ', Sigma: 'Σ', omega: 'ω', Omega: 'Ω', phi: 'φ', epsilon: 'ε', tau: 'τ', nu: 'ν',
};

/**
 * The chat UI renders plain text/Markdown only, yet models still slip LaTeX
 * into worked solutions. Convert the common constructs to readable plain text
 * rather than showing raw backslash commands to a student.
 */
export function plainMath(text: string): string {
  let t = text
    .replace(/\\[()]/g, '')               // \( \)
    .replace(/\\\[|\\\]/g, '')            // \[ \]
    .replace(/\$\$?([^$]+?)\$\$?/g, '$1') // $...$ / $$...$$
    .replace(/\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}/g, '($1)/($2)')
    .replace(/\\sqrt\{([^{}]*)\}/g, '√($1)')
    .replace(/\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}/g, '$1')
    .replace(/\^\{\\circ\}|\^\\circ|\\degree/g, '°')
    .replace(/\^\{([^{}]*)\}/g, '^$1')
    .replace(/_\{([^{}]*)\}/g, '$1')
    .replace(/\\times/g, '×').replace(/\\cdot/g, '·').replace(/\\div/g, '÷').replace(/\\pm/g, '±')
    .replace(/\\approx/g, '≈').replace(/\\(?:le|leq)\b/g, '≤').replace(/\\(?:ge|geq)\b/g, '≥').replace(/\\(?:ne|neq)\b/g, '≠')
    .replace(/\\(?:rightarrow|to|longrightarrow)\b/g, '→').replace(/\\leftrightarrow\b/g, '⇌').replace(/\\rightleftharpoons\b/g, '⇌')
    .replace(/\\infty\b/g, '∞').replace(/\\propto\b/g, '∝').replace(/\\%/g, '%')
    .replace(/\\(?:,|;|:|!|quad|qquad)/g, ' ')
    .replace(/\\(alpha|beta|gamma|delta|Delta|theta|lambda|mu|pi|rho|sigma|Sigma|omega|Omega|phi|epsilon|tau|nu)\b/g, (_, g: string) => GREEK[g] || g)
    .replace(/\\left|\\right/g, '')
    .replace(/\\(?:displaystyle|,)/g, '');
  // Any leftover \command{...} → its content; bare \command → the word itself.
  t = t.replace(/\\[a-zA-Z]+\{([^{}]*)\}/g, '$1').replace(/\\([a-zA-Z]+)\b/g, '$1');
  return t;
}

export function questionPreview(text: string, numWords = 9): string {
  const words = text.replace(LEADING_QNUM_RE, '').replace(/\s+/g, ' ').trim().split(' ');
  let t = words.slice(0, numWords).join(' ');
  if (words.length > numWords) t += '…';
  return t.replace(/\|/g, '\\|');
}

function toMatch(r: Ranked): MatchOut {
  const m = r.hit.metadata;
  return {
    id: r.hit.id, tier: r.tier, why: r.why,
    subject: m.subject, year: m.year, session: m.session, paper: m.paper, variant: m.variant,
    questionNumber: m.question_number, topic: m.topic, topicGeneral: m.topic, type: m.type, level: m.level,
    reference: refLabel(m),
    preview: questionPreview(r.hit.text),
    text: r.hit.text.replace(LEADING_QNUM_RE, '').trim().slice(0, 2500),
  };
}

function tiersOut(rank: RankResult) {
  return { best: rank.best.map(toMatch), conceptual: rank.conceptual.map(toMatch), related: rank.related.map(toMatch) };
}

function sanitizeHistory(raw: unknown): Turn[] {
  if (!Array.isArray(raw)) return [];
  const turns: Turn[] = [];
  for (const t of raw) {
    const role = (t as { role?: unknown })?.role;
    const content = (t as { content?: unknown })?.content;
    if ((role !== 'user' && role !== 'assistant') || typeof content !== 'string' || !content.trim()) continue;
    turns.push({ role, content: content.trim().slice(0, 1200) });
  }
  return turns.slice(-6);
}

// ---- prompts -------------------------------------------------------------------

const FORMAT_RULES = String.raw`FORMATTING RULES (the chat UI renders only plain text and Markdown bold/lists/headings; anything else looks broken):
- Never use LaTeX or math markup of any kind: no \(...\), \[...\], \frac, ^{}, _{} or backslash commands. Write formulas in plain text, e.g. '6CO2 + 6H2O -> C6H12O6 + 6O2', 'v = u + at', 'x^2' as 'x squared' or 'x^2'.
- Never truncate quoted question text with '...' — quote the relevant part in full or paraphrase cleanly.
- Short paragraphs; Markdown '- ' bullets for lists of distinct facts or marking points. Every marking point, including alternatives, is its own bullet — never start a bullet with 'OR'.
- Cite papers exactly as they appear in the context (e.g. 'Chemistry 2023 May/June Paper 2 Variant 1 Q8'). Never invent, alter or guess a paper reference; if the context has no relevant question, say so plainly.`;

const EXPLAIN_SYSTEM = `You are Ask AI, a study assistant for Cambridge O/A Level students, grounded in real past-paper questions. Always respond in English.

${FORMAT_RULES}

Answer in two parts, in this order. The labels 'PART 1'/'PART 2' are instructions for you — never print them, and never put the explanation under its own heading; start straight in with the explanation:

PART 1 — Explanation: a clear, accurate, exam-focused explanation of what the student asked, at the right depth for the level. Use the past-paper context to anchor it in how examiners actually ask about this, citing the paper(s) that informed it. You may draw on standard Cambridge syllabus knowledge for the explanation itself. If the student is asking why they lose marks on something, focus on the common errors and what the mark scheme rewards.

PART 2 — Worked past-paper examples: exactly one heading '### Worked Past-Paper Examples', then answer EACH question listed under 'Questions to answer' in the order given — use the exact question text provided, never substitute different questions, and omit this whole part if none are listed. For each: a level-4 heading ('#### ') with ONLY the paper reference (no 'Question 1' prefix, not bold); restate the question cleanly, bolding each sub-part label together with its instruction sentence (e.g. '**(a) Explain how ...**'), extra data below it in normal weight; then a full answer in normal weight written like a top-scoring candidate: correct terminology, EVERY marking point as its own '- ' bullet, working step by step on its own line for calculations, MCQs as the correct option letter plus a one-line justification. Match depth to the marks available. If a needed numeric value is not present in the question, say so and state the method — never invent a number.`;

const SOLVE_SYSTEM = `You are Ask AI, a Cambridge O/A Level tutor. The student has pasted an exam-style question and wants it solved. Always respond in English.

${FORMAT_RULES}

Answer it the way a top-scoring candidate would, at the level's depth:
- If the context contains the same or a near-identical past-paper question, say on the first line '**Found in past papers:** <reference>' and use it to match the expected marking points. If not, do not claim it appears anywhere.
- Then give the full answer: each sub-part with its label bold (e.g. '**(a)**'), EVERY marking point as its own '- ' bullet, working shown step by step on its own line for calculations, MCQs as the correct option letter plus a one-line justification. Match depth to the marks available.
- Finish with a short '### Examiner tip' section: 2-3 bullets on what candidates typically lose marks on here (draw on the related past-paper questions in the context if useful).
- If a needed numeric value is missing from the question, say so and state the method — never invent a number.`;

const MAKE_SYSTEM = `You are Ask AI, a Cambridge O/A Level study assistant. The student wants practice questions. Always respond in English.

${FORMAT_RULES}

Use ONLY real past-paper questions from the context — never invent, merge or alter a question. Present the requested number (or all suitable ones if fewer exist, and say so):
- For each: a level-4 heading '#### <paper reference exactly as given>', then the question text cleanly — MCQ options each on their own line as '- A. ...', structured sub-parts each on their own line with their marks if shown.
- Then one heading '### Answers' and, per question (same order), a bold label with the paper reference and: for MCQs the correct option letter plus a one-line justification; for structured questions the marking points as '- ' bullets, matched to the marks available.
- Do not add explanations before the questions; a one-line intro at most.`;

const CHAT_SYSTEM = `You are Ask AI, a friendly study assistant for Cambridge O/A Level students inside the Propel app. Reply briefly (max 3 sentences), warmly, in plain text. You can: explain topics grounded in real past papers, find which past papers a topic or question appeared in (the Find tab), solve pasted exam questions, and give real past-paper practice questions.`;

// ---- context building ----------------------------------------------------------

const snip = (t: string, n: number) => t.replace(LEADING_QNUM_RE, '').replace(/\s+/g, ' ').trim().slice(0, n);

function contextBlock(items: Ranked[], maxChars: number, tierNote = true): string {
  return items.map((r, i) => {
    const m = r.hit.metadata;
    const note = tierNote && r.why ? ` — ${r.tier === 'best' ? 'same question' : r.tier}: ${r.why}` : '';
    return `[${i + 1}] ${refLabel(m)} (${m.type || 'question'}${m.topic ? `, topic: ${m.topic}` : ''})${note}\n${snip(r.hit.text, maxChars)}`;
  }).join('\n\n');
}

function pickWorkedExamples(rank: RankResult, limit = 2): Ranked[] {
  const recent = new Date().getFullYear() - 5;
  const pool = [...rank.best, ...rank.conceptual, ...rank.related].filter((r) => r.hit.metadata.type !== 'mcq');
  const out = pool.filter((r) => (r.hit.metadata.year || 0) >= recent).slice(0, limit);
  if (out.length < limit) for (const r of pool) { if (out.length >= limit) break; if (!out.includes(r)) out.push(r); }
  return out;
}

function describeYears(plan: QueryPlan): string {
  if (plan.yearFrom != null && plan.yearTo != null) return plan.yearFrom === plan.yearTo ? `${plan.yearFrom}` : `${plan.yearFrom}–${plan.yearTo}`;
  if (plan.yearFrom != null) return `${plan.yearFrom} onwards`;
  return `up to ${plan.yearTo}`;
}

function scopeLine(plan: QueryPlan, level: Level): string {
  return `${levelName(level)}${plan.subject ? ` ${plan.subject}` : ''}`;
}

// ---- Find -----------------------------------------------------------------------

/** "Only 5" → keep the first N in tier order (best, then same concept, then related). */
function capRank(rank: RankResult, count: number | null): RankResult {
  if (!count || rankedFlat(rank).length <= count) return rank;
  const out: RankResult = { best: [], conceptual: [], related: [], source: rank.source };
  let left = count;
  for (const t of ['best', 'conceptual', 'related'] as Tier[]) {
    out[t] = rank[t].slice(0, Math.max(0, left));
    left -= out[t].length;
  }
  return out;
}

function formatFindAnswer(query: string, plan: QueryPlan, rank: RankResult, candidates: number, level: Level, yearNote: string, totalBeforeCap: number) {
  const total = rankedFlat(rank).length;
  const scope = scopeLine(plan, level);
  const topic = plan.topic ? ` on **${plan.topic}**` : '';
  if (!total) {
    const summary = `Searched ${candidates} ${scope} past-paper questions${topic} — none is a genuine match.`;
    const answer = `${summary}\n\nNothing in the ${scope} index closely matches “${query}”. Try the wording the syllabus uses, pick the subject under Scope, or check you're on the right level (${levelName(level)}).`;
    return { summary, answer };
  }
  const isTopic = plan.kind === 'topic';
  const counts: string[] = [];
  if (rank.best.length) counts.push(isTopic ? `${rank.best.length} squarely on it` : `${rank.best.length} best match${rank.best.length > 1 ? 'es' : ''}`);
  if (rank.conceptual.length) counts.push(isTopic ? `${rank.conceptual.length} testing part of it` : `${rank.conceptual.length} on the same concept`);
  if (rank.related.length) counts.push(`${rank.related.length} related`);
  const window = plan.yearFrom != null || plan.yearTo != null ? ` (${describeYears(plan)})` : '';
  const capNote = totalBeforeCap > total ? ` Showing the ${total} you asked for (out of ${totalBeforeCap} matches).` : '';
  const rankNote = rank.source === 'heuristic' ? ' AI ranking was busy, so these are ordered by similarity only.' : '';
  const summary = `Searched ${candidates} ${scope} past-paper questions${topic}${window}: ${counts.join(', ')}.${capNote}${yearNote ? ` ${yearNote}` : ''}${rankNote}`;
  const sections: string[] = [summary];
  const titles = TIER_TITLES[plan.kind];
  for (const tier of ['best', 'conceptual', 'related'] as Tier[]) {
    if (!rank[tier].length) continue;
    sections.push(`### ${titles[tier]} (${rank[tier].length})\n` +
      rank[tier].map((r) => `- **${refLabel(r.hit.metadata)}**${r.why ? ` — ${r.why}` : ''}\n  “${questionPreview(r.hit.text, 14)}”`).join('\n'));
  }
  return { summary, answer: sections.join('\n\n') };
}

// ---- Ask ------------------------------------------------------------------------

function askUserPrompt(query: string, plan: QueryPlan, rank: RankResult, allHits: Hit[], level: Level): { system: string; user: string } {
  const ranked = rankedFlat(rank);
  const header = `Student (${scopeLine(plan, level)}) asks: ${query}${plan.topic ? `\nInterpreted topic: ${plan.topic}` : ''}` +
    (plan.reference ? `\nThe student is referring to this question from earlier in the conversation: ${plan.reference} — answer about THAT question (it is in the context below if it was found).` : '');

  if (plan.intent === 'solve') {
    const ctx = [...rank.best.slice(0, 2), ...rank.conceptual.slice(0, 3), ...rank.related.slice(0, 2)];
    return {
      system: SOLVE_SYSTEM,
      user: `${header}\n\nPast-paper context (closest first; "same question" = this is the pasted question):\n${ctx.length ? contextBlock(ctx, 1500) : '(no matching past-paper question found — solve from syllabus knowledge and do not claim a paper source)'}`,
    };
  }

  if (plan.intent === 'make_questions') {
    const count = plan.count ?? 5;
    const pool = (ranked.length ? ranked : allHits.map((h) => ({ hit: h, tier: 'related' as Tier, why: '' })))
      .filter((r) => !plan.questionType || r.hit.metadata.type === plan.questionType || ranked.length < count);
    const ctx = pool.slice(0, Math.max(count + 2, 6));
    return {
      system: MAKE_SYSTEM,
      user: `${header}\nRequested: ${count} ${plan.questionType === 'mcq' ? 'MCQs' : plan.questionType === 'structured' ? 'structured questions' : 'questions'}.\n\nReal past-paper questions available (most relevant first):\n${ctx.length ? contextBlock(ctx, 1400, false) : '(none found)'}`,
    };
  }

  const ctx = ranked.slice(0, 5);
  const worked = pickWorkedExamples(rank, 2);
  const examples = worked.length
    ? `\n\nQuestions to answer (worked examples, in this order):\n${worked.map((r) => `Question [${refLabel(r.hit.metadata)}]:\n${snip(r.hit.text, 1500)}`).join('\n\n')}`
    : '';
  return {
    system: EXPLAIN_SYSTEM,
    user: `${header}\n\nPast-paper context (most relevant first):\n${ctx.length ? contextBlock(ctx, 500) : '(no closely related past-paper question was found — explain from syllabus knowledge and say the papers had nothing directly on this)'}${examples}`,
  };
}

// ---- entry point ----------------------------------------------------------------

export async function askAi(input: AskAiInput): Promise<AskAiOutput> {
  const level = normalizeLevel(input.level) ?? 'olevel';
  const query = input.query.trim();
  const history = sanitizeHistory(input.history);
  const mode: Mode = input.mode === 'find' ? 'find' : 'ask';

  const plan = await planQuery(query, { mode, level, subject: input.subject, history });
  const empty = { best: [], conceptual: [], related: [] };

  if (!plan.needsSearch) {
    const answer = plainMath(plan.reply
      || await chatText(CHAT_SYSTEM, query, { tier: 'fast', maxTokens: 300, temperature: 0.5, history, timeoutMs: 30_000 }));
    return {
      type: 'smalltalk', mode, intent: plan.intent, answer, summary: null, level,
      subject: plan.subject, topic: plan.topic, searched: false, candidates: 0,
      matches: empty, tierTitles: TIER_TITLES[plan.kind], tierLabels: TIER_LABELS[plan.kind],
      citations: [], planner: plan.source, ranker: null, plan: planOut(plan),
    };
  }

  // The tab is only a hint to the planner. What the student actually asked for
  // decides the flow: "explain one of them" on the Find tab is an explanation;
  // "which years was X asked" on the Ask tab is a lookup. On the Find tab a
  // request for N questions is a lookup capped at N (Ask presents them with answers).
  const isFind = plan.intent === 'find_questions' || (mode === 'find' && plan.intent === 'make_questions');
  // Candidate counts are sized for Groq's per-minute token budget as much as quality.
  const topK = isFind ? 24 : plan.intent === 'make_questions' ? 14 : 12;
  const hasYears = plan.yearFrom != null || plan.yearTo != null;

  // A follow-up about a question shown earlier ("explain the first one"): fetch
  // exactly that question, search with ITS text (not "explain one of them"), and
  // keep it first so the answer is about that one, not a sibling.
  const exact = plan.reference ? await fetchByReference(plan.reference, level) : null;
  const queries = exact ? [exact.text.replace(LEADING_QNUM_RE, '').slice(0, 400), ...plan.searchQueries.slice(0, 1)] : plan.searchQueries;
  if (exact && !plan.subject) plan.subject = exact.metadata.subject;

  let hits = dedupeByPaperQuestion(await searchMany(queries, {
    level, subject: plan.subject, topK: hasYears ? topK * 2 : topK,
    keywords: plan.keywords, questionType: plan.questionType,
  }));
  let yearNote = '';
  if (hasYears) {
    const inWindow = filterYears(hits, plan.yearFrom, plan.yearTo);
    if (inWindow.length) hits = inWindow.slice(0, topK);
    else { yearNote = `Nothing matched in ${describeYears(plan)}, so these are the closest from other years.`; hits = hits.slice(0, topK); }
  }
  if (exact) hits = [exact, ...hits.filter((h) => h.id !== exact.id)];

  const fullRank = await rankCandidates(query, plan, hits, level, isFind ? 'smart' : 'fast');
  const rank = isFind ? capRank(fullRank, plan.count) : fullRank;
  const matches = tiersOut(rank);
  const citations = [...matches.best, ...matches.conceptual, ...matches.related];
  const tierTitles = TIER_TITLES[plan.kind];
  const tierLabels = TIER_LABELS[plan.kind];

  if (isFind) {
    const { summary, answer } = formatFindAnswer(query, plan, rank, hits.length, level, yearNote, rankedFlat(fullRank).length);
    return {
      type: 'exam_question', mode: 'find', intent: plan.intent, answer, summary, level,
      subject: plan.subject, topic: plan.topic, searched: true, candidates: hits.length,
      matches, tierTitles, tierLabels, citations, planner: plan.source, ranker: rank.source, plan: planOut(plan),
    };
  }

  // The UI lists best/conceptual matches ("Where this appears in past papers")
  // from `matches` under the answer, so nothing is appended to the Markdown here.
  const { system, user } = askUserPrompt(query, plan, rank, hits, level);
  const answer = plainMath(stripPartLabels(stripRedundantOrPrefix(await chatText(system, user, {
    tier: 'smart', maxTokens: 2600, temperature: 0.2, history, timeoutMs: 90_000,
  }))));

  return {
    type: 'exam_question', mode: 'ask', intent: plan.intent, answer, summary: null, level,
    subject: plan.subject, topic: plan.topic, searched: true, candidates: hits.length,
    matches, tierTitles, tierLabels, citations, planner: plan.source, ranker: rank.source, plan: planOut(plan),
  };
}
