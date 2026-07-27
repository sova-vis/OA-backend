/**
 * Minimal xAI Grok client over the OpenAI-compatible REST endpoint.
 *
 * One XAI_API_KEY drives both written grading (text model) and handwritten
 * OCR/grading (vision model). Called via fetch so it works with any current
 * Grok model without an SDK dependency.
 */

const GROK_URL = 'https://api.x.ai/v1/chat/completions';

export function grokApiKey(): string {
  return (process.env.XAI_API_KEY || process.env.GROK_API_KEY || '').trim();
}

export function grokEnabled(): boolean {
  return grokApiKey().length > 0;
}

export function grokTextModel(): string {
  return (process.env.XAI_GRADING_MODEL || 'grok-4').trim();
}

export function grokVisionModel(): string {
  return (process.env.XAI_VISION_MODEL || 'grok-2-vision-1212').trim();
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
        return 'AI grading is not configured yet. Add a valid XAI_API_KEY (your Grok key) in OA-backend/.env.';
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

/**
 * Single chat completion that must return a JSON object. Uses the vision model
 * automatically when images are supplied. Throws GrokError with a typed code.
 */
export async function grokChatJson(options: GrokChatOptions): Promise<Record<string, unknown>> {
  const apiKey = grokApiKey();
  if (!apiKey) throw new GrokError('XAI_API_KEY is not set', 'no_key');

  const hasImages = Array.isArray(options.images) && options.images.length > 0;
  const model = options.model || (hasImages ? grokVisionModel() : grokTextModel());

  const userContent: unknown = hasImages
    ? [
        { type: 'text', text: options.user },
        ...options.images!.map((image) => ({
          type: 'image_url',
          image_url: { url: `data:${image.mimeType};base64,${image.base64}` },
        })),
      ]
    : options.user;

  const payload: Record<string, unknown> = {
    model,
    temperature: options.temperature ?? 0,
    max_tokens: options.maxTokens ?? 2048,
    messages: [
      { role: 'system', content: options.system },
      { role: 'user', content: userContent },
    ],
  };

  let response: globalThis.Response;
  try {
    response = await fetch(GROK_URL, {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(options.timeoutMs ?? 60_000),
    });
  } catch (error) {
    if (error instanceof Error && error.name === 'TimeoutError') {
      throw new GrokError('Grok request timed out', 'timeout');
    }
    throw new GrokError(error instanceof Error ? error.message : 'Grok request failed', 'other');
  }

  if (!response.ok) {
    const bodyText = await response.text().catch(() => '');
    const lowered = bodyText.toLowerCase();
    // xAI returns HTTP 400 for a bad key ("Incorrect API key provided"), so the
    // status alone is ambiguous — inspect the body to tell key vs model apart.
    const looksLikeKey = /api key|api_key|incorrect key|authenticat|unauthor|invalid.token|credential/.test(lowered);
    const looksLikeModel = /model/.test(lowered);
    const code: GrokErrorCode =
      response.status === 401 || response.status === 403 ? 'invalid_key'
        : response.status === 429 ? 'rate_limit'
        : looksLikeKey ? 'invalid_key'
        : response.status === 404 || looksLikeModel ? 'model'
        : response.status === 400 ? 'model'
        : 'other';
    throw new GrokError(`Grok API error ${response.status}: ${bodyText.slice(0, 300)}`, code, response.status);
  }

  const data = (await response.json()) as { choices?: Array<{ message?: { content?: string } }> };
  const content = data.choices?.[0]?.message?.content || '';
  const parsed = parseJsonObject(content);
  if (!parsed) throw new GrokError('Grok returned no parseable JSON', 'other');
  return parsed;
}
