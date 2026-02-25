import { Request, Response, NextFunction } from 'express';
import { createRemoteJWKSet, importSPKI, jwtVerify } from 'jose';
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

function getPublicKeyFromEnv() {
  const raw = process.env.CLERK_JWT_KEY;
  if (!raw) return null;
  return raw.includes('\\n') ? raw.replace(/\\n/g, '\n') : raw;
}

function getBearerToken(authHeader?: string) {
  if (!authHeader?.startsWith('Bearer ')) return null;
  return authHeader.slice(7).trim();
}

async function verifyClerkJwt(token: string) {
  const publicKey = getPublicKeyFromEnv();

  if (publicKey) {
    const key = await importSPKI(publicKey, 'RS256');
    return jwtVerify(token, key, {
      ...(issuer ? { issuer } : {}),
      ...(audience ? { audience } : {}),
      algorithms: ['RS256'],
    });
  }

  if (!issuer) {
    throw new Error('Missing Clerk verifier config: set CLERK_JWT_KEY or CLERK_ISSUER');
  }

  const jwks = createRemoteJWKSet(new URL(`${issuer}/.well-known/jwks.json`));
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
