import { Response, NextFunction } from 'express';
import { AuthenticatedRequest } from './clerkAuth';

/**
 * Minimal in-memory per-user (or per-IP) fixed-window rate limiter, used to cap
 * spend on the paid-AI routes (F-04). Deliberately generous so it never trips
 * normal use — it only stops abuse. For multi-instance deployments this should
 * later move to a shared store (Redis), but a single Railway instance is fine.
 */

interface Bucket { count: number; resetAt: number }
const buckets = new Map<string, Bucket>();

// bound memory: drop expired buckets periodically
const cleanup = setInterval(() => {
  const now = Date.now();
  for (const [key, bucket] of Array.from(buckets.entries())) {
    if (now >= bucket.resetAt) buckets.delete(key);
  }
}, 60_000);
cleanup.unref?.();

export function rateLimit(opts: { windowMs: number; max: number; name?: string }) {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    const who = req.auth?.clerkId || req.ip || 'anon';
    const key = `${opts.name || 'default'}|${who}`;
    const now = Date.now();
    let bucket = buckets.get(key);
    if (!bucket || now >= bucket.resetAt) {
      bucket = { count: 0, resetAt: now + opts.windowMs };
      buckets.set(key, bucket);
    }
    bucket.count += 1;
    if (bucket.count > opts.max) {
      const retryAfter = Math.ceil((bucket.resetAt - now) / 1000);
      res.setHeader('Retry-After', String(retryAfter));
      return res.status(429).json({ error: `Too many requests — please wait ${retryAfter}s and try again.` });
    }
    next();
  };
}
