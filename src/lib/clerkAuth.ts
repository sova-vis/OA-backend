import { Request, Response, NextFunction } from 'express';
import { supabase } from './supabase';

/**
 * Authentication middleware — verifies Supabase Auth (GoTrue) access tokens.
 *
 * NOTE: file/export names are kept as `clerkAuth` for backward compatibility
 * (33 call sites import it) — this project migrated OFF Clerk onto self-hosted
 * Supabase Auth. Supabase user tokens are HS256, signed with the project's
 * JWT secret (SUPABASE_JWT_SECRET), and carry:
 *   sub   -> the user's UUID  (we keep storing this in profiles.clerk_id)
 *   email -> the user's email
 *   role  -> "authenticated" (the app role lives in profiles.role, not here)
 *   user_metadata.full_name -> display name (flattened to `full_name` below)
 */

export interface AuthenticatedRequest extends Request {
  auth?: {
    userId: string;
    clerkId: string; // = Supabase user UUID (column name kept for compatibility)
    token: string;
    claims?: Record<string, unknown>;
  };
}

const JWT_SECRET = (process.env.SUPABASE_JWT_SECRET || '').trim();
const JWT_AUD = (process.env.SUPABASE_JWT_AUD || 'authenticated').trim();

let joseModulePromise: Promise<typeof import('jose')> | null = null;
function loadJoseModule(): Promise<typeof import('jose')> {
  if (!joseModulePromise) {
    // Keep a true dynamic ESM import at runtime in CJS builds.
    joseModulePromise = Function('return import("jose")')() as Promise<typeof import('jose')>;
  }
  return joseModulePromise;
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

async function verifySupabaseJwt(token: string) {
  if (!JWT_SECRET) {
    throw new Error('Missing SUPABASE_JWT_SECRET — cannot verify Supabase Auth tokens');
  }
  const { jwtVerify } = await loadJoseModule();
  const key = new TextEncoder().encode(JWT_SECRET);
  return jwtVerify(token, key, {
    algorithms: ['HS256'],
    audience: JWT_AUD,
    clockTolerance: 30,
  });
}

/** No-op kept so index.ts's warmup call still resolves (nothing to prefetch for HS256). */
export async function warmupClerkVerifier(): Promise<void> {
  if (!JWT_SECRET) {
    console.warn('SUPABASE_JWT_SECRET is not set — authenticated routes will reject every request.');
    return;
  }
  console.log('Supabase Auth verifier ready (HS256).');
}

/**
 * Flatten the useful bits of a Supabase token so downstream claim readers
 * (which look for top-level `email` / `full_name` / `name`) keep working.
 */
function normalizeClaims(payload: Record<string, unknown>): Record<string, unknown> {
  const meta = (payload.user_metadata as Record<string, unknown> | undefined) || {};
  const fullName =
    (typeof meta.full_name === 'string' && meta.full_name.trim()) ? meta.full_name.trim()
    : (typeof meta.name === 'string' && meta.name.trim()) ? meta.name.trim()
    : undefined;
  return {
    ...payload,
    ...(fullName ? { full_name: fullName, name: fullName } : {}),
  };
}

/**
 * Middleware to verify the Supabase Auth JWT on protected routes:
 *   router.get('/route', clerkAuth, handler)
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

    const verified = await verifySupabaseJwt(token);
    const userId = verified.payload.sub;
    if (!userId || typeof userId !== 'string') {
      return res.status(401).json({ error: 'Unauthorized - Invalid token subject' });
    }

    req.auth = {
      userId,
      clerkId: userId,
      token,
      claims: normalizeClaims(verified.payload as Record<string, unknown>),
    };

    next();
  } catch (error) {
    console.error('Auth error:', errorCode(error) || error);
    return res.status(401).json({ error: 'Unauthorized - Invalid token' });
  }
}

/** Alias for clarity in new code; identical to clerkAuth. */
export const requireAuth = clerkAuth;

/**
 * Middleware to require a specific app role (from profiles.role).
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
