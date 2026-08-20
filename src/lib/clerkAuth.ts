import { Request, Response, NextFunction } from 'express';
import { supabase } from './supabase';

export interface AuthenticatedRequest extends Request {
  auth?: {
    userId: string;
    clerkId: string;
    token: string;
    claims?: Record<string, unknown>;
  };
}

const issuer = process.env.CLERK_ISSUER?.replace(/\/$/, '');
const audience = process.env.CLERK_AUDIENCE;
const JWKS_TTL_MS = 10 * 60 * 1000;
const JWKS_TIMEOUT_MS = 15_000;
const JWKS_ATTEMPTS = 3;

type JwksCache = {
  verifier: Awaited<ReturnType<typeof import('jose')['createLocalJWKSet']>>;
  fetchedAt: number;
};

let joseModulePromise: Promise<typeof import('jose')> | null = null;
let jwksCache: JwksCache | null = null;
let jwksInflight: Promise<JwksCache> | null = null;

function loadJoseModule(): Promise<typeof import('jose')> {
  if (!joseModulePromise) {
    // Keep true dynamic ESM import at runtime in CJS builds.
    joseModulePromise = Function('return import("jose")')() as Promise<typeof import('jose')>;
  }
  return joseModulePromise;
}

function getPublicKeyFromEnv() {
  const raw = process.env.CLERK_JWT_KEY;
  if (!raw) return null;
  const normalized = (raw.includes('\\n') ? raw.replace(/\\n/g, '\n') : raw)
    .trim()
    .replace(/^"|"$/g, '');

  // If CLERK_JWT_KEY is not an SPKI PEM, we'll fall back to JWKS verification.
  if (!normalized.includes('BEGIN PUBLIC KEY')) {
    return null;
  }

  // Validate the key has enough content (a real RSA SPKI key is 300+ chars).
  // Short/malformed keys cause DOMException: Invalid keyData errors.
  const keyBody = normalized
    .replace(/-----BEGIN PUBLIC KEY-----/, '')
    .replace(/-----END PUBLIC KEY-----/, '')
    .replace(/\s/g, '');
  if (keyBody.length < 100) {
    return null;
  }

  return normalized;
}

function getBearerToken(authHeader?: string) {
  if (!authHeader?.startsWith('Bearer ')) return null;
  return authHeader.slice(7).trim();
}

function errorCode(error: unknown): string {
  if (typeof error === 'object' && error && 'code' in error && typeof (error as { code?: unknown }).code === 'string') {
    return (error as { code: string }).code;
  }
  return '';
}

async function fetchJwksDocument(): Promise<{ keys: Record<string, unknown>[] }> {
  if (!issuer) {
    throw new Error('Missing Clerk verifier config: set CLERK_JWT_KEY or CLERK_ISSUER');
  }

  const url = `${issuer}/.well-known/jwks.json`;
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= JWKS_ATTEMPTS; attempt += 1) {
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        cache: 'no-store',
        signal: AbortSignal.timeout(JWKS_TIMEOUT_MS),
      });
      const text = await response.text();
      if (!response.ok) {
        throw new Error(`Clerk JWKS HTTP ${response.status}`);
      }

      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch {
        throw new Error('Clerk JWKS response was not JSON');
      }

      const keys = (parsed as { keys?: unknown }).keys;
      if (!Array.isArray(keys) || keys.length === 0) {
        throw new Error('Clerk JWKS response had no keys');
      }

      return parsed as { keys: Record<string, unknown>[] };
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      if (attempt < JWKS_ATTEMPTS) {
        await new Promise((resolve) => setTimeout(resolve, 250 * attempt));
      }
    }
  }

  throw lastError ?? new Error('Failed to fetch Clerk JWKS');
}

async function getJwksVerifier() {
  const fresh = jwksCache && Date.now() - jwksCache.fetchedAt < JWKS_TTL_MS;
  if (fresh && jwksCache) return jwksCache.verifier;

  const pending = jwksInflight ?? (jwksInflight = (async () => {
    const { createLocalJWKSet } = await loadJoseModule();
    const body = await fetchJwksDocument();
    const verifier = createLocalJWKSet(body as Parameters<typeof createLocalJWKSet>[0]);
    const entry: JwksCache = { verifier, fetchedAt: Date.now() };
    jwksCache = entry;
    return entry;
  })().finally(() => {
    jwksInflight = null;
  }));

  try {
    return (await pending).verifier;
  } catch (error) {
    if (jwksCache) {
      console.warn('Clerk JWKS refresh failed; using cached keys.', errorCode(error) || error);
      return jwksCache.verifier;
    }
    throw error;
  }
}

/** Prefetch Clerk JWKS so the first authenticated request is not a network round-trip. */
export async function warmupClerkVerifier(): Promise<void> {
  if (!issuer && !getPublicKeyFromEnv()) return;
  try {
    if (issuer) {
      await getJwksVerifier();
      console.log('Clerk JWKS verifier ready');
    }
  } catch (error) {
    console.warn('Clerk JWKS warmup failed; auth will retry on the first request.', error);
  }
}

async function verifyClerkJwt(token: string) {
  const { importSPKI, jwtVerify } = await loadJoseModule();
  const verifyOpts = {
    ...(issuer ? { issuer } : {}),
    ...(audience ? { audience } : {}),
    clockTolerance: 30,
  };

  const publicKey = getPublicKeyFromEnv();
  if (publicKey) {
    try {
      const key = await importSPKI(publicKey, 'RS256');
      return await jwtVerify(token, key, { ...verifyOpts, algorithms: ['RS256'] });
    } catch (error) {
      const code = errorCode(error);
      // Expired / claim mismatches will fail the same way on JWKS — don't pay another round-trip.
      if (code === 'ERR_JWT_EXPIRED' || code === 'ERR_JWT_CLAIM_VALIDATION_FAILED') {
        throw error;
      }
      if (!issuer) throw error;
    }
  }

  if (!issuer) {
    throw new Error('Missing Clerk verifier config: set CLERK_JWT_KEY or CLERK_ISSUER');
  }

  const jwks = await getJwksVerifier();
  return jwtVerify(token, jwks, verifyOpts);
}

/**
 * Middleware to verify Clerk JWT tokens in backend API
 * Add this middleware to protected routes:
 * router.get('/route', clerkAuth, handler)
 */
export async function clerkAuth(
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
) {
  try {
    const token = getBearerToken(req.headers.authorization);

    if (!token) {
      return res.status(401).json({ error: 'Unauthorized - No token' });
    }

    const verified = await verifyClerkJwt(token);
    const clerkId = verified.payload.sub;

    if (!clerkId || typeof clerkId !== 'string') {
      return res.status(401).json({ error: 'Unauthorized - Invalid token subject' });
    }

    req.auth = {
      userId: clerkId,
      clerkId,
      token,
      claims: verified.payload as Record<string, unknown>,
    };

    next();
  } catch (error) {
    console.error('Clerk auth error:', errorCode(error) || error);
    return res.status(401).json({ error: 'Unauthorized - Invalid token' });
  }
}

/**
 * Optional: Middleware to check if user has specific role
 * Usage: router.get('/admin', clerkAuth, requireRole('admin'), handler)
 */
export function requireRole(requiredRole: string) {
  return async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    try {
      if (!req.auth?.clerkId) {
        return res.status(401).json({ error: 'Unauthorized' });
      }

      const { data: profile, error } = await supabase
        .from('profiles')
        .select('role')
        .eq('clerk_id', req.auth.clerkId)
        .single();

      if (error) {
        console.error('Role lookup error:', error);
        return res.status(500).json({ error: 'Server error' });
      }

      if (!profile || profile.role !== requiredRole) {
        return res.status(403).json({ error: 'Forbidden - Insufficient permissions' });
      }

      next();
    } catch (error) {
      console.error('Role check error:', error);
      return res.status(500).json({ error: 'Server error' });
    }
  };
}
