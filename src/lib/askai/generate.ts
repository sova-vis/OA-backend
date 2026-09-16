/**
 * Answer generation for Ask-AI, in-process. Ported from the Python
 * generate_answer.py: retrieves grounding chunks, then asks a free/paid LLM for
 * an explanation + worked past-paper examples, citing the source papers.
 *
 * Find mode / paper_lookup returns the occurrences table directly (no LLM).
 * (Web search from the Python version is omitted here; the past-paper grounding
 * is the core — it can be added later behind a search API.)
 */
import {
  route, paperLookupSummary, requestedYearLimit,
  type Hit, type Occurrence, type RouteResult,
} from './retrieve';

interface Provider { name: string; url: string; apiKey: string | undefined; model: string; }

function providers(): Provider[] {
  const groq = (process.env.GROQ_API_KEY || '').trim();
  const groqModel = (process.env.GROQ_MODEL || 'llama-3.3-70b-versatile').trim();
  const xai = (process.env.XAI_API_KEY || process.env.GROK_API_KEY || '').trim();
  const samba = (process.env.SAMBANOVA_API_KEY || '').trim();
  const openrouter = (process.env.OPENROUTER_API_KEY || '').trim();
  return [
    // Groq first — it's the key that's configured in Railway, and it's fast.
    { name: 'groq', url: 'https://api.groq.com/openai/v1/chat/completions', apiKey: groq, model: groqModel },
    { name: 'xai', url: 'https://api.x.ai/v1/chat/completions', apiKey: xai, model: 'grok-4.5' },
    { name: 'sambanova', url: 'https://api.sambanova.ai/v1/chat/completions', apiKey: samba, model: 'Meta-Llama-3.3-70B-Instruct' },
    { name: 'openrouter', url: 'https://openrouter.ai/api/v1/chat/completions', apiKey: openrouter, model: 'openai/gpt-oss-20b:free' },
    { name: 'openrouter', url: 'https://openrouter.ai/api/v1/chat/completions', apiKey: openrouter, model: 'google/gemma-4-31b-it:free' },
    { name: 'openrouter', url: 'https://openrouter.ai/api/v1/chat/completions', apiKey: openrouter, model: 'nvidia/nemotron-3-super-120b-a12b:free' },
  ];
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export async function callLLMWithFallback(system: string, user: string): Promise<string> {
  let lastError = 'no provider';
  for (const p of providers()) {
    if (!p.apiKey) continue;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const res = await fetch(p.url, {
          method: 'POST',
          headers: { Authorization: `Bearer ${p.apiKey}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: p.model, messages: [
            { role: 'system', content: system }, { role: 'user', content: user },
          ] }),
          signal: AbortSignal.timeout(90_000),
        });
        if (res.status === 429) {
          lastError = `${p.name}/${p.model} rate-limited`;
          if (attempt < 1) await sleep(3000 * (attempt + 1));
          continue;
        }
        if (!res.ok) { lastError = `${p.name}/${p.model} ${res.status}`; break; }
        const data = await res.json();
        const content = data?.choices?.[0]?.message?.content;
        if (typeof content === 'string' && content.trim()) return content;
        lastError = `${p.name}/${p.model} empty`; break;
      } catch (e) { lastError = `${p.name}/${p.model} ${e instanceof Error ? e.message : e}`; break; }
    }
  }
  throw new Error(`All AI providers are currently unavailable. Last error: ${lastError}`);
}

const LEADING_QNUM_RE = /^\s*\d{1,2}\s*/;
const BULLET_OR_PREFIX_RE = /^(\s*-\s+)or\b:?\s*/gim;
const stripRedundantOrPrefix = (t: string) => t.replace(BULLET_OR_PREFIX_RE, '$1');

export function questionPreview(text: string, numWords = 7): string {
  let t = text.replace(LEADING_QNUM_RE, '').replace(/\s+/g, ' ').trim();
  const words = t.split(' ');
  t = words.slice(0, numWords).join(' ');
  if (words.length > numWords) t += '...';
  return t.replace(/\|/g, '\\|').replace(/\[/g, '(').replace(/\]/g, ')');
}

function buildContext(result: RouteResult): string {
  const lines: string[] = [];
  if (result.intent === 'paper_lookup') {
    lines.push('Matching past-paper questions found (full detail for each occurrence):');
    for (const m of result.occurrences || []) {
      lines.push(
        `- Subject: ${m.subject} | Year: ${m.year} | Session (month): ${m.session} | ` +
        `Paper: ${m.paper} | Variant: ${m.variant} | Question number: ${m.question_number}\n` +
        `  Full question text: ${(m.question_text || '').trim()}`,
      );
    }
  } else {
    lines.push('Retrieved past-paper excerpts:');
    for (const h of result.hits.slice(0, 6)) {
      const m = h.metadata;
      lines.push(`[${m.subject} ${m.year} ${m.session} Paper ${m.paper} Q${m.question_number}]\n${h.text.slice(0, 500)}`);
    }
  }
  return lines.join('\n\n');
}

function selectWorkedExamples(hits: Hit[], yearLimit: number, limit = 3): Hit[] {
  const cutoff = new Date().getFullYear() - yearLimit + 1;
  const seen = new Set<string>();
  const out: Hit[] = [];
  for (const h of hits) {
    const m = h.metadata;
    if (m.question_number == null) continue;
    if ((m.year || 0) < cutoff) continue;
    if (m.type === 'mcq') continue; // worked examples model full written answers, not MCQs
    const key = [m.subject, m.year, m.session, m.paper, m.variant, m.question_number].join('|');
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(h);
    if (out.length >= limit) break;
  }
  return out;
}

function formatWorkedExamplesBlock(examples: Hit[]): string {
  return examples.map((h) => {
    const m = h.metadata;
    const session = (m.session || '').replace(/_/g, '/');
    const variant = m.variant != null ? ` Variant ${m.variant}` : '';
    const ref = `${m.subject} ${m.year} ${session} Paper ${m.paper}${variant} Q${m.question_number}`;
    const text = h.text.trim().replace(LEADING_QNUM_RE, '');
    return `Question [${ref}]:\n${text}`;
  }).join('\n\n');
}

export function formatPaperLookupAnswer(occurrences: Occurrence[], yearLimit?: number): string {
  if (!occurrences.length) {
    return `No matching questions were found${yearLimit ? ` in the last ${yearLimit} years` : ''} for this query.`;
  }
  const note = yearLimit ? ` (last ${yearLimit} years)` : '';
  const lines = [
    `Found **${occurrences.length}** matching question(s)${note}:\n`,
    '| # | Year | Session (Month) | Paper | Variant | Question # | Question |',
    '|---|------|------------------|-------|---------|------------|----------|',
  ];
  occurrences.forEach((m, i) => {
    const session = (m.session || '').replace(/_/g, '/');
    const variant = m.variant != null ? m.variant : '-';
    lines.push(`| ${i + 1} | ${m.year} | ${session} | ${m.paper} | ${variant} | ${m.question_number} | ${questionPreview(m.question_text || '')} |`);
  });
  return lines.join('\n');
}

const SYSTEM_PROMPT = String.raw`You are a study assistant for Cambridge O/A Level past exam papers. Answer the user's question using ONLY the provided context (past-paper excerpts). Always respond in English.

FORMATTING RULES (the chat UI only renders plain text and Markdown bold/lists/headings - nothing else renders, so violating these makes the answer look broken):
- Never use LaTeX or math markup of any kind: no \(...\), \[...\], \mathrm{}, \ldots, ^{}, _{}, or any other backslash command. Write chemical and math formulas in plain text instead, e.g. '6CO2 + 6H2O -> C6H12O6 + 6O2' (plain digits, no subscript/superscript markup, '->' or '→' for arrows).
- Never truncate quoted question text with '...' or '\ldots' - either quote the relevant part in full or paraphrase it cleanly in your own words; don't leave a dangling ellipsis.
- Use short paragraphs and Markdown bullet points ('- ') for lists of distinct facts or marking points, not one dense run-on paragraph.
- Every distinct marking point in an answer, including alternative ('OR ...') marking points, must be its own Markdown bullet line starting with '- ' - never write alternatives as plain lines of text, even in a short answer with only 2-3 points.

Structure your response in two parts, in this order. These 'PART 1' / 'PART 2' labels below are instructions for YOU only - never print the words 'PART 1' or 'PART 2' in your actual answer, and never print the explanation under its own heading either; start straight in with the explanation text itself:

PART 1 - Explanation: a clear, accurate explanation of the topic grounded in the provided context, citing which paper(s) informed it (e.g. 'Accounting 2023 May_June Paper 1 Q16'). If the context doesn't contain enough information, say so honestly rather than making things up.

PART 2 - Worked past-paper examples: this part gets exactly one heading, '### Worked Past-Paper Examples', immediately followed by the answers - answer EACH question listed in 'Questions to answer' below, one at a time, in the order given - use the exact question text provided, do not invent or substitute different questions, and skip this part entirely if no questions are listed there. For each one: give the question a level-4 Markdown heading ('#### ') with ONLY its paper reference, no 'Question 1' / 'Question 2' numbering prefix and NOT bold text (e.g. '#### Biology 2025 Oct/Nov Paper 2 Q4', NOT '**Question 1 - Biology 2025 Oct/Nov Paper 2 Q4**'), restate the question text cleanly below it - bold each sub-part label TOGETHER WITH the instruction/question sentence that directly follows it on the same line (e.g. '**(a) Explain how the structure of a leaf is adapted for photosynthesis.**'); any extra data, passages, or context given below that sub-part's instruction (not the instruction itself) stays normal weight - then give a full answer in normal (non-bold) weight, written the way a top-scoring Cambridge O/A Level candidate would: correct subject terminology, EVERY marking point as its own '- ' bullet - never write the answer as flowing prose/sentences, even a single-sentence answer must be a bullet. Never write the word 'OR' at the start of a bullet - each bullet being its own line already shows it's a separate valid alternative. All working shown step-by-step on its own line for calculations, and for multiple-choice questions the correct option letter followed by a one-line justification. Match the depth of the answer to the marks available. If a calculation needs a numeric value that isn't present in the given question text or context, say plainly that the value isn't given and state the method/formula that would be used - never invent, guess, or assume a placeholder number.`;

export interface GenerateResult { answer: string; result: RouteResult; }

/** mode overrides the auto-classified intent: "find" -> paper lookup, "ask" -> explanation. */
export async function generate(
  query: string, opts: { subject?: string | null; level?: string | null; mode?: string | null } = {},
): Promise<GenerateResult> {
  const result = await route(query, { subject: opts.subject, level: opts.level, topK: 10 });
  const naturalIntent = result.intent;
  if (opts.mode === 'find') result.intent = 'paper_lookup';
  else if (opts.mode === 'ask') result.intent = 'general_qa';

  const yearLimit = requestedYearLimit(query);

  if (result.intent === 'paper_lookup') {
    if (!result.occurrences) {
      result.year_limit = yearLimit;
      result.occurrences = paperLookupSummary(result.hits, yearLimit);
    }
    return { answer: formatPaperLookupAnswer(result.occurrences, result.year_limit), result };
  }

  const context = buildContext(result);
  const worked = selectWorkedExamples(result.hits, yearLimit);
  const examplesBlock = worked.length
    ? `\n\nQuestions to answer (most relevant, last ${yearLimit} years):\n${formatWorkedExamplesBlock(worked)}`
    : '';
  const userPrompt = `Question: ${query}\n\nContext:\n${context}${examplesBlock}`;

  let answer = stripRedundantOrPrefix(await callLLMWithFallback(SYSTEM_PROMPT, userPrompt));

  // Surface which past-paper questions this topic shows up in (unless Ask mode is
  // answering a query naturally phrased as a paper lookup).
  if (!(opts.mode === 'ask' && naturalIntent === 'paper_lookup')) {
    const occurrences = paperLookupSummary(result.hits, yearLimit);
    result.occurrences = occurrences;
    result.year_limit = yearLimit;
    if (occurrences.length) {
      answer += `\n\n---\n\n### Related past-paper questions\n\n${formatPaperLookupAnswer(occurrences, yearLimit)}`;
    }
  }
  return { answer, result };
}
