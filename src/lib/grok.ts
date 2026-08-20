/**
 * OpenAI-compatible chat client for practice marking.
 *
 * Prefers xAI Grok (XAI_API_KEY). When Grok is out of credits or unavailable,
 * falls back to Groq (GROQ_API_KEY), then Gemini (GEMINI_API_KEY), so
 * handwritten OCR and written marking keep working.
 */

const GROK_URL = 'https://api.x.ai/v1/chat/completions';
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
const GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions';

type Provider = 'xai' | 'groq' | 'gemini';

export function grokApiKey(): string {
  return (process.env.XAI_API_KEY || process.env.GROK_API_KEY || '').trim();
}

export function groqApiKey(): string {
  return (process.env.GROQ_GRADING_API_KEY || process.env.GROQ_API_KEY || '').trim();
}

export function geminiApiKey(): string {
  return (process.env.GEMINI_API_KEY || '').trim();
}

export function grokEnabled(): boolean {
  return grokApiKey().length > 0 || groqApiKey().length > 0 || geminiApiKey().length > 0;
}

export function grokTextModel(): string {
  return (process.env.XAI_GRADING_MODEL || 'grok-4').trim();
}

export function grokVisionModel(): string {
  return (process.env.XAI_VISION_MODEL || 'grok-2-vision-1212').trim();
}

/**
 * Groq IDs that 404 on free/developer tiers after the July 2026 shutdowns.
 * Production may still have these in GROQ_VISION_MODEL / GROQ_GRADING_MODEL
 * from an older .env.example — skip them instead of failing every page.
 * @see https://console.groq.com/docs/deprecations
 */
const RETIRED_GROQ_MODELS = new Set([
  'meta-llama/llama-4-scout-17b-16e-instruct',
  'llama-3.3-70b-versatile',
  'llama-3.1-8b-instant',
  'qwen/qwen3-32b',
  'mixtral-8x7b-32768',
  'gemma-7b-it',
]);

const GROQ_TEXT_FALLBACKS = ['openai/gpt-oss-120b', 'qwen/qwen3.6-27b'];
const GROQ_VISION_FALLBACKS = ['qwen/qwen3.6-27b'];

function uniqueLiveGroqModels(names: string[]): string[] {
  const out: string[] = [];
  for (const name of names) {
    const trimmed = name.trim();
    if (!trimmed || RETIRED_GROQ_MODELS.has(trimmed) || out.includes(trimmed)) continue;
    out.push(trimmed);
  }
  return out;
}

export function groqTextModel(): string {
  return uniqueLiveGroqModels([
    process.env.GROQ_GRADING_MODEL || '',
    process.env.GROQ_MODEL || '',
    ...GROQ_TEXT_FALLBACKS,
  ])[0] || GROQ_TEXT_FALLBACKS[0];
}

export function groqVisionModel(): string {
  return uniqueLiveGroqModels([
    process.env.GROQ_VISION_MODEL || '',
    ...GROQ_VISION_FALLBACKS,
  ])[0] || GROQ_VISION_FALLBACKS[0];
}

export function geminiModel(): string {
  return (process.env.GEMINI_MODEL || process.env.GEMINI_VISION_MODEL || 'gemini-3.6-flash').trim();
}

export type GrokErrorCode = 'no_key' | 'invalid_key' | 'quota' | 'rate_limit' | 'model' | 'timeout' | 'parse' | 'other';

export class GrokError extends Error {
  code: GrokErrorCode;
  status?: number;
  retryAfterMs?: number;
  constructor(message: string, code: GrokErrorCode, status?: number, retryAfterMs?: number) {
    super(message);
    this.name = 'GrokError';
    this.code = code;
    this.status = status;
    this.retryAfterMs = retryAfterMs;
  }
}

/** Errors that mean "stop calling this provider / this request cannot be marked". */
export function isFatalGrokError(error: unknown): boolean {
  return error instanceof GrokError && (error.code === 'no_key' || error.code === 'invalid_key' || error.code === 'quota');
}

/** Transient failures that should be retried on the same provider, then the next. */
export function isTransientGrokError(error: unknown): boolean {
  if (!(error instanceof GrokError)) return true;
  return error.code === 'timeout' || error.code === 'rate_limit' || error.code === 'parse' || error.code === 'other';
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function grokHttpStatus(error: unknown): number {
  if (!(error instanceof GrokError)) return 500;
  if (error.code === 'no_key' || error.code === 'invalid_key' || error.code === 'quota' || error.code === 'model') {
    return 503;
  }
  return 500;
}

/** Human-facing message for a grading failure, safe to show a student. */
export function grokErrorMessage(error: unknown): string {
  if (error instanceof GrokError) {
    switch (error.code) {
      case 'no_key':
        return 'AI grading is not configured yet. Add XAI_API_KEY, GROQ_API_KEY or GEMINI_API_KEY in OA-backend/.env.';
      case 'invalid_key':
        return 'The AI API key is invalid or expired. Update XAI_API_KEY, GROQ_API_KEY or GEMINI_API_KEY in OA-backend/.env.';
      case 'quota':
        return 'AI grading credits are exhausted on the current provider. Add xAI credits or another working key (Groq / Gemini).';
      case 'rate_limit':
        return 'Grading is rate-limited right now. Please try again in a moment.';
      case 'model':
        return 'The configured AI model is unavailable. Check XAI_GRADING_MODEL / XAI_VISION_MODEL or GROQ_MODEL / GROQ_VISION_MODEL.';
      case 'timeout':
        return 'Grading timed out. Please try again.';
      case 'parse':
        return 'The marker returned an unreadable result. Please try marking again.';
      default:
        return error.message || 'Grading failed. Please try again.';
    }
  }
  return error instanceof Error ? error.message : 'Grading failed. Please try again.';
}

function asJsonObject(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (Array.isArray(value) && value.length === 1 && value[0] && typeof value[0] === 'object' && !Array.isArray(value[0])) {
    return value[0] as Record<string, unknown>;
  }
  return null;
}

/** Robust JSON-object extraction from a model completion. */
export function parseJsonObject(raw: string): Record<string, unknown> | null {
  const text = String(raw || '').trim();
  if (!text) return null;
  const stripped = text
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/m, '')
    .trim();
  const attempts = [stripped, text];
  const start = stripped.indexOf('{');
  const end = stripped.lastIndexOf('}');
  if (start >= 0 && end > start) attempts.push(stripped.slice(start, end + 1));
  const arrayStart = stripped.indexOf('[');
  const arrayEnd = stripped.lastIndexOf(']');
  if (arrayStart >= 0 && arrayEnd > arrayStart) attempts.push(stripped.slice(arrayStart, arrayEnd + 1));
  for (const candidate of attempts) {
    try {
      const parsed = asJsonObject(JSON.parse(candidate));
      if (parsed) return parsed;
    } catch {
      // try next
    }
  }
  return null;
}

export interface GrokImage {
  /** base64 (no data: prefix) */
  base64: string;
  mimeType: string;
}

export interface GrokChatOptions {
  system: string;
  user: string;
  images?: GrokImage[];
  model?: string;
  temperature?: number;
  maxTokens?: number;
  timeoutMs?: number;
}

/**
 * Gemini free-tier 429s say "exceeded your current quota" and "billing" even
 * when the body also has "Please retry in 17s" — that is a per-model RPM pause,
 * not a dead key. True credit exhaustion (xAI) is 403 / "out of funds".
 */
export function classifyGrokHttpError(status: number, bodyText: string): GrokErrorCode {
  const lowered = bodyText.toLowerCase();
  if (/api key|api_key|incorrect key|authenticat|invalid.token|credential/.test(lowered)) {
    return 'invalid_key';
  }
  if (status === 401) return 'invalid_key';
  // Groq JSON mode rejecting the completion is a parse miss, not a missing model.
  if (/json_validate_failed|failed to validate json|failed_generation/.test(lowered)) {
    return 'parse';
  }
  if (status === 429 || status === 502 || status === 503 || status === 529) return 'rate_limit';
  if (status === 413 || /request too large|tokens per minute|please reduce your message size/.test(lowered)) {
    return 'rate_limit';
  }
  if (/unavailable|high demand|overloaded|try again later|temporarily|resource_exhausted/.test(lowered)) {
    return 'rate_limit';
  }
  if (/credit|spending limit|out of funds|permission-denied/.test(lowered)) {
    return 'quota';
  }
  if (status === 404) return 'model';
  if (/no longer available|is not found|model/.test(lowered) && (status === 400 || status === 404)) {
    return 'model';
  }
  // xAI uses 403 for both a bad key and an empty credit balance. Key language
  // is handled above; remaining 403s are treated as quota, not invalid_key.
  if (status === 403) return 'quota';
  return 'other';
}

/** Parse Retry-After / "Please retry in 17.3s" so we wait the provider's pause. */
export function retryAfterMsFrom(bodyText: string, header?: string | null): number | undefined {
  if (header) {
    const seconds = Number(header);
    if (Number.isFinite(seconds) && seconds > 0) return Math.min(45_000, seconds * 1000);
    const when = Date.parse(header);
    if (Number.isFinite(when)) return Math.min(45_000, Math.max(0, when - Date.now()));
  }
  const match = String(bodyText || '').match(/retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s/i);
  if (match) return Math.min(45_000, Math.max(400, Number(match[1]) * 1000));
  return undefined;
}

function completionText(data: {
  choices?: Array<{ message?: { content?: unknown; reasoning?: unknown } }>;
}): string {
  const message = data.choices?.[0]?.message;
  if (!message) return '';
  const content = message.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') return part;
        if (part && typeof part === 'object' && 'text' in part) return String((part as { text?: unknown }).text || '');
        return '';
      })
      .join('');
  }
  if (typeof message.reasoning === 'string') return message.reasoning;
  return '';
}

function canFallback(error: unknown): boolean {
  if (!(error instanceof GrokError)) return true;
  // A missing key on this provider is fallbackable; a missing key overall is not.
  return error.code !== 'no_key';
}

function providerKey(provider: Provider): string {
  if (provider === 'xai') return grokApiKey();
  if (provider === 'groq') return groqApiKey();
  return geminiApiKey();
}

function providerUrl(provider: Provider): string {
  if (provider === 'xai') return GROK_URL;
  if (provider === 'groq') return GROQ_URL;
  return GEMINI_URL;
}

function providerModels(provider: Provider, hasImages: boolean, requested?: string): string[] {
  if (provider === 'xai') return [requested || (hasImages ? grokVisionModel() : grokTextModel())];
  if (provider === 'groq') {
    return hasImages
      ? uniqueLiveGroqModels([groqVisionModel(), ...GROQ_VISION_FALLBACKS])
      : uniqueLiveGroqModels([groqTextModel(), ...GROQ_TEXT_FALLBACKS]);
  }
  // gemini-2.5-flash / 2.0-flash 404 for new keys. Each flash SKU has its own
  // free-tier RPM, so a 429 on 3.6-flash can still succeed on lite/latest.
  const primary = geminiModel();
  return [...new Set([primary, 'gemini-3.5-flash-lite', 'gemini-flash-latest'])];
}

function providerLabel(provider: Provider): string {
  if (provider === 'xai') return 'XAI_API_KEY';
  if (provider === 'groq') return 'GROQ_API_KEY';
  return 'GEMINI_API_KEY';
}

async function chatCompletions(
  provider: Provider,
  options: GrokChatOptions & { model: string; jsonMode?: boolean },
): Promise<Record<string, unknown>> {
  const apiKey = providerKey(provider);
  if (!apiKey) throw new GrokError(`${providerLabel(provider)} is not set`, 'no_key');

  const jsonMode = options.jsonMode !== false;

  const userContent: unknown = Array.isArray(options.images) && options.images.length > 0
    ? [
        { type: 'text', text: options.user },
        ...options.images!.map((image) => ({
          type: 'image_url',
          image_url: { url: `data:${image.mimeType};base64,${image.base64}` },
        })),
      ]
    : options.user;

  // Reasoning models (Groq gpt-oss / Qwen, xAI grok-4.x) spend tokens on hidden
  // thinking; keep a floor so JSON mode is not starved into an empty completion.
  // Groq on_demand TPM is 8000 and counts reserved max_completion_tokens, so an
  // 8192 completion budget plus a page image 413s (Requested 11330). Cap Groq.
  const xaiReasoning = provider === 'xai' && /grok-4/i.test(String(options.model || ''));
  const hasImages = Array.isArray(options.images) && options.images.length > 0;
  let tokenBudget = options.maxTokens ?? 2048;
  if (provider === 'groq') {
    tokenBudget = Math.min(Math.max(tokenBudget, 1024), hasImages ? 2500 : 4096);
  } else if (xaiReasoning) {
    tokenBudget = Math.max(tokenBudget, 8192);
  }
  const payload: Record<string, unknown> = {
    model: options.model,
    temperature: options.temperature ?? 0,
    max_tokens: tokenBudget,
    messages: [
      { role: 'system', content: options.system },
      { role: 'user', content: userContent },
    ],
  };
  if (provider === 'groq') payload.max_completion_tokens = tokenBudget;
  if (jsonMode) payload.response_format = { type: 'json_object' };

  const url = providerUrl(provider);
  let response: globalThis.Response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(options.timeoutMs ?? 60_000),
    });
  } catch (error) {
    if (error instanceof Error && error.name === 'TimeoutError') {
      throw new GrokError(`${provider} request timed out`, 'timeout');
    }
    throw new GrokError(error instanceof Error ? error.message : `${provider} request failed`, 'other');
  }

  if (!response.ok) {
    const bodyText = await response.text().catch(() => '');
    const code = classifyGrokHttpError(response.status, bodyText);
    const retryAfterMs = retryAfterMsFrom(bodyText, response.headers.get('retry-after'));
    throw new GrokError(
      `${provider} API error ${response.status}: ${bodyText.slice(0, 300)}`,
      code,
      response.status,
      retryAfterMs,
    );
  }

  const data = (await response.json()) as { choices?: Array<{ message?: { content?: unknown; reasoning?: unknown } }> };
  const content = completionText(data);
  const parsed = parseJsonObject(content);
  if (!parsed) {
    console.warn(`${provider} returned unparseable JSON:`, content.slice(0, 240));
    throw new GrokError(`${provider} returned no parseable JSON`, 'parse');
  }
  return parsed;
}

function retryDelayMs(error: unknown, attempt: number): number {
  if (error instanceof GrokError && error.retryAfterMs) {
    return Math.min(45_000, Math.max(400, error.retryAfterMs));
  }
  if (error instanceof GrokError && error.code === 'rate_limit') {
    return Math.min(12_000, 1_500 * (attempt + 1));
  }
  return Math.min(2_000, 350 * (attempt + 1));
}

async function tryProvider(
  provider: Provider,
  options: GrokChatOptions,
  hasImages: boolean,
): Promise<Record<string, unknown>> {
  const models = providerModels(provider, hasImages, options.model);
  let lastError: unknown;
  for (let m = 0; m < models.length; m++) {
    const model = models[m];
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        return await chatCompletions(provider, {
          ...options,
          model,
          // Second attempt drops json_object in case the model emitted valid
          // JSON that the response_format wrapper then mangled.
          jsonMode: attempt === 0,
        });
      } catch (error) {
        lastError = error;
        if (error instanceof GrokError && (error.code === 'no_key' || error.code === 'invalid_key' || error.code === 'quota')) {
          throw error;
        }
        const reason = error instanceof GrokError ? error.code : 'error';
        const hasMoreModels = m < models.length - 1;
        // A 429 on gemini-3.6-flash still has quota on lite/latest — switch
        // models immediately instead of sitting out the retry-after.
        // A 404 model_not_found (retired Groq IDs) is the same: try the next ID.
        if (error instanceof GrokError && (error.code === 'rate_limit' || error.code === 'model') && hasMoreModels) {
          console.warn(`${provider}/${model} ${reason}; trying next model.`);
          break;
        }
        // 413 TPM "request too large" will 413 again at the same size — skip retry.
        if (error instanceof GrokError && error.status === 413) {
          console.warn(`${provider}/${model} payload too large; trying next.`);
          break;
        }
        if (isTransientGrokError(error) && attempt === 0) {
          console.warn(`${provider}/${model} ${reason}; retrying.`);
          await sleep(retryDelayMs(error, attempt));
          continue;
        }
        console.warn(`${provider}/${model} unavailable (${reason}); trying next.`);
        break;
      }
    }
  }
  throw lastError;
}

const COOLDOWN_QUOTA_MS = 30 * 60_000;
const providerCooldownUntil = new Map<Provider, number>();

function isCooling(provider: Provider): boolean {
  return Date.now() < (providerCooldownUntil.get(provider) || 0);
}

function coolDown(provider: Provider, error: unknown): void {
  if (!(error instanceof GrokError)) return;
  let ms = 0;
  if (error.code === 'quota' || error.code === 'invalid_key') ms = COOLDOWN_QUOTA_MS;
  else if (error.code === 'rate_limit' && error.status !== 413) {
    ms = Math.max(error.retryAfterMs || 15_000, 8_000);
  }
  if (!ms) return;
  const prev = providerCooldownUntil.get(provider) || 0;
  const until = Date.now() + ms;
  if (until > prev) providerCooldownUntil.set(provider, until);
}

/**
 * Single chat completion that must return a JSON object. Uses the vision model
 * automatically when images are supplied. Walks xAI → Groq → Gemini, retries
 * transient 503/parse failures, and makes a second pass over the chain so a
 * later provider outage can fall back to one that has recovered.
 *
 * Providers that return a hard quota / invalid key are skipped for 30 minutes
 * so a handwritten paper does not spend every page on a dead xAI key.
 */
export async function grokChatJson(options: GrokChatOptions): Promise<Record<string, unknown>> {
  const chain: Provider[] = [];
  if (grokApiKey()) chain.push('xai');
  if (groqApiKey()) chain.push('groq');
  if (geminiApiKey()) chain.push('gemini');
  if (chain.length === 0) throw new GrokError('XAI_API_KEY is not set', 'no_key');

  const hasImages = Array.isArray(options.images) && options.images.length > 0;
  let lastError: unknown;

  for (let round = 0; round < 2; round++) {
    if (chain.every((p) => isCooling(p))) {
      const soonest = Math.min(...chain.map((p) => providerCooldownUntil.get(p) || Date.now()));
      const wait = Math.min(45_000, Math.max(0, soonest - Date.now()));
      if (wait > 0) {
        console.warn(`all providers cooling; waiting ${Math.round(wait)}ms`);
        await sleep(wait);
      }
    }
    for (let i = 0; i < chain.length; i++) {
      const provider = chain[i];
      if (isCooling(provider)) {
        console.warn(`${provider} cooling down after a previous failure; skipping.`);
        continue;
      }
      try {
        return await tryProvider(provider, options, hasImages);
      } catch (error) {
        lastError = error;
        coolDown(provider, error);
        const next = chain.slice(i + 1).find((p) => !isCooling(p))
          || (round === 0 ? chain.find((p, idx) => idx !== i && !isCooling(p)) : undefined);
        if (canFallback(error) && (next || round === 0)) {
          const reason = error instanceof GrokError ? error.code : 'error';
          console.warn(`${provider} unavailable (${reason})${next ? `; falling back to ${next}` : '; retrying chain'}.`);
          continue;
        }
        throw error;
      }
    }
    if (round === 0 && lastError && isTransientGrokError(lastError)) {
      await sleep(retryDelayMs(lastError, round + 1));
    }
  }
  throw lastError ?? new GrokError('All AI providers are temporarily unavailable. Please try again in a moment.', 'rate_limit');
}
