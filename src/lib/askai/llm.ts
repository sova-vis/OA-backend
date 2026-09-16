/**
 * LLM access for Ask-AI — Groq first (the key configured in Railway), on model
 * ids Groq actually serves in 2026 (the llama-3.x ids were retired mid-2026 and
 * now 404), then any other configured OpenAI-compatible provider.
 *
 * chatText() returns Markdown; chatJson() returns a parsed object (JSON mode).
 * Two tiers: 'fast' for planning/ranking calls, 'smart' for the answer itself.
 * A provider that answers "bad key / out of credits" is skipped for 30 minutes,
 * so a dead key never adds a failed round-trip to every student question.
 */
import { classifyGrokHttpError, grokTextModel, parseJsonObject, retryAfterMsFrom } from '../grok';

export type LlmTier = 'fast' | 'smart';
export interface Turn { role: 'user' | 'assistant'; content: string }
export interface LlmOptions {
  tier?: LlmTier;
  maxTokens?: number;
  temperature?: number;
  timeoutMs?: number;
  /** Prior conversation turns, sent as real messages before the final user one. */
  history?: Turn[];
}

interface Attempt { provider: string; url: string; apiKey: string; model: string }

const env = (k: string) => (process.env[k] || '').trim();

// Groq ids retired during 2026 — still present in older env files; never call them.
const RETIRED_GROQ = new Set([
  'llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'llama3-70b-8192', 'llama3-8b-8192',
  'meta-llama/llama-4-scout-17b-16e-instruct', 'meta-llama/llama-4-maverick-17b-128e-instruct',
  'qwen/qwen3-32b', 'mixtral-8x7b-32768', 'gemma-7b-it', 'gemma2-9b-it', 'deepseek-r1-distill-llama-70b',
]);
// qwen last: on Groq's on-demand tier it allows only 1,000 output tokens/min.
const GROQ_SMART = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.8-27b'];
const GROQ_FAST = ['openai/gpt-oss-20b', 'openai/gpt-oss-120b', 'qwen/qwen3.8-27b'];

function groqModels(tier: LlmTier): string[] {
  const preferred = tier === 'smart' ? [env('GROQ_MODEL')] : [env('GROQ_FAST_MODEL'), env('GROQ_MODEL')];
  const out: string[] = [];
  for (const m of [...preferred, ...(tier === 'smart' ? GROQ_SMART : GROQ_FAST)]) {
    if (m && !RETIRED_GROQ.has(m) && !out.includes(m)) out.push(m);
  }
  return out;
}

function attempts(tier: LlmTier): Attempt[] {
  const list: Attempt[] = [];
  const groq = env('GROQ_API_KEY');
  for (const model of groq ? groqModels(tier) : []) {
    list.push({ provider: 'groq', url: 'https://api.groq.com/openai/v1/chat/completions', apiKey: groq, model });
  }
  const xai = env('XAI_API_KEY') || env('GROK_API_KEY');
  if (xai) list.push({ provider: 'xai', url: 'https://api.x.ai/v1/chat/completions', apiKey: xai, model: grokTextModel() });
  const samba = env('SAMBANOVA_API_KEY');
  if (samba) {
    list.push({ provider: 'sambanova', url: 'https://api.sambanova.ai/v1/chat/completions', apiKey: samba, model: env('SAMBANOVA_MODEL') || 'Meta-Llama-3.3-70B-Instruct' });
  }
  const openrouter = env('OPENROUTER_API_KEY');
  for (const model of openrouter ? ['openai/gpt-oss-120b', 'google/gemma-4-31b-it:free'] : []) {
    list.push({ provider: 'openrouter', url: 'https://openrouter.ai/api/v1/chat/completions', apiKey: openrouter, model });
  }
  return list;
}

/** True when at least one provider key is present (used by the /health report). */
export function llmConfigured(): boolean {
  return attempts('smart').length > 0;
}

export class LlmError extends Error {
  constructor(
    public provider: string, public model: string, public status: number,
    public code: string, detail: string, public retryAfterMs?: number,
  ) {
    super(`${provider}/${model} ${status || ''} ${code}: ${detail}`.replace(/\s+/g, ' ').trim());
  }
}

const cooldownUntil = new Map<string, number>();
const COOLDOWN_MS = 30 * 60_000;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function callOnce(a: Attempt, system: string, user: string, opts: LlmOptions, json: boolean): Promise<string> {
  const messages = [
    { role: 'system', content: system },
    ...(opts.history || []).map((t) => ({ role: t.role, content: t.content })),
    { role: 'user', content: user },
  ];
  let maxTokens = opts.maxTokens ?? 2048;
  const body: Record<string, unknown> = { model: a.model, messages, temperature: opts.temperature ?? 0.2 };
  if (a.provider === 'groq') {
    // Groq rejects (413/429) any request whose reserved output exceeds the model's
    // per-minute output cap; qwen's on-demand cap is 1,000.
    if (a.model.startsWith('qwen/')) maxTokens = Math.min(maxTokens, 1000);
    body.max_completion_tokens = maxTokens;
    // gpt-oss are reasoning models: keep the hidden thinking short for planning calls.
    if (a.model.startsWith('openai/gpt-oss')) body.reasoning_effort = opts.tier === 'fast' ? 'low' : 'medium';
  } else {
    body.max_tokens = maxTokens;
  }
  if (json) body.response_format = { type: 'json_object' };

  let res: Awaited<ReturnType<typeof fetch>>;
  try {
    res = await fetch(a.url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${a.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(opts.timeoutMs ?? 60_000),
    });
  } catch (e) {
    const timeout = e instanceof Error && e.name === 'TimeoutError';
    throw new LlmError(a.provider, a.model, 0, timeout ? 'timeout' : 'other', e instanceof Error ? e.message : String(e));
  }
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new LlmError(
      a.provider, a.model, res.status, classifyGrokHttpError(res.status, text),
      text.slice(0, 300), retryAfterMsFrom(text, res.headers.get('retry-after')),
    );
  }
  const data = (await res.json()) as { choices?: Array<{ message?: { content?: unknown } }> };
  let content = data.choices?.[0]?.message?.content;
  if (Array.isArray(content)) {
    content = content.map((p) => (typeof p === 'string' ? p : String((p as { text?: unknown })?.text || ''))).join('');
  }
  if (typeof content !== 'string' || !content.trim()) {
    throw new LlmError(a.provider, a.model, res.status, 'parse', 'empty completion');
  }
  if (json && !parseJsonObject(content)) {
    throw new LlmError(a.provider, a.model, res.status, 'parse', `not JSON: ${content.slice(0, 120)}`);
  }
  return content;
}

async function run(system: string, user: string, opts: LlmOptions, json: boolean): Promise<string> {
  const list = attempts(opts.tier ?? 'smart');
  if (!list.length) throw new Error('No AI provider is configured — set GROQ_API_KEY.');
  let last: unknown = null;
  let deadProvider: string | null = null;
  for (const a of list) {
    if (a.provider === deadProvider) continue;
    if ((cooldownUntil.get(a.provider) || 0) > Date.now()) continue;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        return await callOnce(a, system, user, opts, json);
      } catch (e) {
        last = e;
        if (!(e instanceof LlmError)) break;
        if (e.code === 'invalid_key' || e.code === 'quota') {
          // Dead key / no credits: don't try this provider's other models, and
          // skip it for a while so students don't pay the failed round-trip.
          cooldownUntil.set(a.provider, Date.now() + COOLDOWN_MS);
          console.warn(`[askai/llm] ${a.provider} disabled for 30 min: ${e.message}`);
          deadProvider = a.provider;
          break;
        }
        if (e.code === 'model' || e.status === 413) {
          console.warn(`[askai/llm] ${e.message} — trying next model`);
          break;
        }
        if (e.code === 'rate_limit' && attempt === 0 && (e.retryAfterMs ?? 0) <= 8_000) {
          await sleep(e.retryAfterMs || 2_500);
          continue;
        }
        if ((e.code === 'parse' || e.code === 'other') && attempt === 0) {
          await sleep(400);
          continue;
        }
        console.warn(`[askai/llm] ${e.message} — trying next`);
        break;
      }
    }
  }
  const detail = last instanceof Error ? last.message : String(last);
  throw new Error(`All AI providers are currently unavailable. Last error: ${detail}`);
}

/** Markdown/plain-text completion. */
export function chatText(system: string, user: string, opts: LlmOptions = {}): Promise<string> {
  return run(system, user, opts, false);
}

/** JSON-mode completion, parsed. Throws if no provider returns a JSON object. */
export async function chatJson<T = Record<string, unknown>>(system: string, user: string, opts: LlmOptions = {}): Promise<T> {
  const raw = await run(`${system}\n\nRespond with a single JSON object and nothing else.`, user, opts, true);
  const parsed = parseJsonObject(raw);
  if (!parsed) throw new Error('AI returned no parseable JSON');
  return parsed as T;
}
