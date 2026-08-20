/**
 * Minimal grading client over the OpenAI-compatible chat/completions REST API.
 *
 * Primary provider is xAI Grok (XAI_API_KEY); if that call fails for any reason
 * (no key, invalid key, rate limit, model or timeout), the exact same request is
 * retried against Groq (GROQ_API_KEY) as a fallback. Both providers speak the
 * same wire format, so no SDK and no per-call branching in the callers.
 */

const GROK_URL = 'https://api.x.ai/v1/chat/completions';
const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';

export function grokApiKey(): string {
  return (process.env.XAI_API_KEY || process.env.GROK_API_KEY || '').trim();
}

/**
 * Groq fallback key for grading (OpenAI-compatible, api.groq.com). Prefers a
 * dedicated GROQ_GRADING_API_KEY so grading can use a different key than the
 * shared GROQ_API_KEY (Ask-AI / paper parsing); falls back to GROQ_API_KEY.
 */
export function groqApiKey(): string {
  return (process.env.GROQ_GRADING_API_KEY || process.env.GROQ_API_KEY || '').trim();
}

/** Grading works if EITHER provider is configured. */
export function grokEnabled(): boolean {
  return grokApiKey().length > 0 || groqApiKey().length > 0;
}

export function grokTextModel(): string {
  return (process.env.XAI_GRADING_MODEL || 'grok-4').trim();
}

export function grokVisionModel(): string {
  return (process.env.XAI_VISION_MODEL || 'grok-2-vision-1212').trim();
}

/** Groq text model — override with GROQ_GRADING_MODEL. */
export function groqTextModel(): string {
  return (process.env.GROQ_GRADING_MODEL || 'llama-3.3-70b-versatile').trim();
}

/** Groq vision model (must support image input) — override with GROQ_VISION_MODEL. */
export function groqVisionModel(): string {
  return (process.env.GROQ_VISION_MODEL || 'meta-llama/llama-4-scout-17b-16e-instruct').trim();
}

export type GrokErrorCode = 'no_key' | 'invalid_key' | 'rate_limit' | 'model' | 'timeout' | 'other';

export class GrokError extends Error {
  code: GrokErrorCode;
  status?: number;
  constructor(message: string, code: GrokErrorCode, status?: number) {
    super(message);
    this.name = 'GrokError';
    this.code = code;
    this.status = status;
  }
}

/** Human-facing message for a grading failure, safe to show a student. */
export function grokErrorMessage(error: unknown): string {
  if (error instanceof GrokError) {
    switch (error.code) {
      case 'no_key':
        return 'AI grading is not configured yet. Add XAI_API_KEY (Grok) and/or GROQ_API_KEY in OA-backend/.env.';
      case 'invalid_key':
        return 'The Grok API key is invalid or expired. Update XAI_API_KEY in OA-backend/.env.';
      case 'rate_limit':
        return 'Grading is rate-limited right now. Please try again in a moment.';
      case 'model':
        return 'The configured Grok model is unavailable for this account. Check XAI_GRADING_MODEL / XAI_VISION_MODEL.';
      case 'timeout':
        return 'Grading timed out. Please try again.';
      default:
        return error.message || 'Grading failed. Please try again.';
    }
  }
  return error instanceof Error ? error.message : 'Grading failed. Please try again.';
}

/** Robust JSON-object extraction from a model completion. */
export function parseJsonObject(raw: string): Record<string, unknown> | null {
  const text = (raw || '').trim();
  if (!text) return null;
  const attempts = [text];
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start >= 0 && end > start) attempts.push(text.slice(start, end + 1));
  for (const candidate of attempts) {
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // try next candidate
    }
  }
  return null;
}

export interface GrokImage {
  /** base64 (no data: prefix) */
  base64: string;
  mimeType: string;
}

interface GrokChatOptions {
  system: string;
  user: string;
  images?: GrokImage[];
  model?: string;
  temperature?: number;
  maxTokens?: number;
  timeoutMs?: number;
}

interface Provider {
  name: 'xai' | 'groq';
  url: string;
  apiKey: string;
  model: string;
}

/** One completion against a single OpenAI-compatible provider. Throws GrokError. */
async function completeOnce(
  provider: Provider,
  userContent: unknown,
  options: GrokChatOptions,
): Promise<Record<string, unknown>> {
  const payload: Record<string, unknown> = {
    model: provider.model,
    temperature: options.temperature ?? 0,
    max_tokens: options.maxTokens ?? 2048,
    messages: [
      { role: 'system', content: options.system },
      { role: 'user', content: userContent },
    ],
  };

  let response: globalThis.Response;
  try {
    response = await fetch(provider.url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${provider.apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(options.timeoutMs ?? 60_000),
    });
  } catch (error) {
    if (error instanceof Error && error.name === 'TimeoutError') {
      throw new GrokError(`${provider.name} request timed out`, 'timeout');
    }
    throw new GrokError(error instanceof Error ? error.message : `${provider.name} request failed`, 'other');
  }

  if (!response.ok) {
    const bodyText = await response.text().catch(() => '');
    const lowered = bodyText.toLowerCase();
    // A bad key can surface as HTTP 400 with a key-ish body, so inspect the body
    // to tell key vs model apart rather than trusting the status alone.
    const looksLikeKey = /api key|api_key|incorrect key|authenticat|unauthor|invalid.token|credential/.test(lowered);
    const looksLikeModel = /model/.test(lowered);
    const code: GrokErrorCode =
      response.status === 401 || response.status === 403 ? 'invalid_key'
        : response.status === 429 ? 'rate_limit'
        : looksLikeKey ? 'invalid_key'
        : response.status === 404 || looksLikeModel ? 'model'
        : response.status === 400 ? 'model'
        : 'other';
    throw new GrokError(`${provider.name} API error ${response.status}: ${bodyText.slice(0, 300)}`, code, response.status);
  }

  const data = (await response.json()) as { choices?: Array<{ message?: { content?: string } }> };
  const content = data.choices?.[0]?.message?.content || '';
  const parsed = parseJsonObject(content);
  if (!parsed) throw new GrokError(`${provider.name} returned no parseable JSON`, 'other');
  return parsed;
}

/**
 * Single chat completion that must return a JSON object. Uses the vision model
 * automatically when images are supplied. Tries xAI Grok first, then falls back
 * to Groq if Grok fails for any reason. Throws GrokError with a typed code when
 * every configured provider fails.
 */
export async function grokChatJson(options: GrokChatOptions): Promise<Record<string, unknown>> {
  const hasImages = Array.isArray(options.images) && options.images.length > 0;

  const userContent: unknown = hasImages
    ? [
        { type: 'text', text: options.user },
        ...options.images!.map((image) => ({
          type: 'image_url',
          image_url: { url: `data:${image.mimeType};base64,${image.base64}` },
        })),
      ]
    : options.user;

  // Priority order: xAI Grok (primary), then Groq (fallback). A caller-supplied
  // options.model only overrides the primary's model.
  const providers: Provider[] = [];
  const xaiKey = grokApiKey();
  if (xaiKey) {
    providers.push({
      name: 'xai',
      url: GROK_URL,
      apiKey: xaiKey,
      model: options.model || (hasImages ? grokVisionModel() : grokTextModel()),
    });
  }
  const groqKey = groqApiKey();
  if (groqKey) {
    providers.push({
      name: 'groq',
      url: GROQ_URL,
      apiKey: groqKey,
      model: hasImages ? groqVisionModel() : groqTextModel(),
    });
  }

  if (providers.length === 0) {
    throw new GrokError('No grading key set (XAI_API_KEY or GROQ_API_KEY)', 'no_key');
  }

  let lastError: GrokError = new GrokError('Grading failed', 'other');
  for (const provider of providers) {
    try {
      return await completeOnce(provider, userContent, options);
    } catch (error) {
      lastError = error instanceof GrokError ? error : new GrokError(String(error), 'other');
      // Try the next provider (if any). If this was the last one, we throw below.
    }
  }
  throw lastError;
}
