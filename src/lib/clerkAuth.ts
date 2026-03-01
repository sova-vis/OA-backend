import { Request, Response, NextFunction } from 'express';
import { supabase } from './supabase';

export interface AuthenticatedRequest extends Request {
  auth?: {
    userId: string;
    clerkId: string;
    token: string;
  };
}

const issuer = process.env.CLERK_ISSUER?.replace(/\/$/, '');
const audience = process.env.CLERK_AUDIENCE;
let cachedJwks: ReturnType<(typeof import('jose'))['createRemoteJWKSet']> | null = null;

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

async function verifyClerkJwt(token: string) {
  const { createRemoteJWKSet, importSPKI, jwtVerify } = await import('jose');
  const publicKey = getPublicKeyFromEnv();

  if (publicKey) {
    try {
      const key = await importSPKI(publicKey, 'RS256');
      return jwtVerify(token, key, {
        ...(issuer ? { issuer } : {}),
        ...(audience ? { audience } : {}),
        algorithms: ['RS256'],
      });
    } catch {
      // Invalid key — silently fall back to JWKS if issuer is configured.
      if (!issuer) {
        throw new Error('CLERK_JWT_KEY is invalid and no CLERK_ISSUER configured for JWKS fallback');
      }
    }
  }

  if (!issuer) {
    throw new Error('Missing Clerk verifier config: set CLERK_JWT_KEY or CLERK_ISSUER');
  }

  if (!cachedJwks) {
    cachedJwks = createRemoteJWKSet(new URL(`${issuer}/.well-known/jwks.json`));
  }

  const jwks = cachedJwks;
  if (!jwks) {
    throw new Error('Unable to initialize Clerk JWKS verifier');
  }

  return jwtVerify(token, jwks, {
    issuer,
    ...(audience ? { audience } : {}),
  });
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
    };

    next();
  } catch (error) {
    console.error('Clerk auth error:', error);
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
