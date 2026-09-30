/**
 * Teacher Portal v1 — role & school-scope guards.
 *
 * The Sept 2026 spec defines a 4-tier hierarchy above the student:
 *   owner (platform)  ->  school_admin  ->  teacher  ->  student
 * ('admin' is the legacy name for the platform owner and is treated as 'owner'.)
 *
 * Every teacher/admin route MUST sit behind one of these guards — the spec is
 * emphatic that limits, scope and visibility are enforced server-side, never in
 * the UI. Guards are STRICT: a platform owner does not silently pass a teacher
 * guard (owners never grade; they act cross-school via their own routes or the
 * explicit, audited impersonation flow). Runs AFTER clerkAuth (needs req.auth).
 */
import { Response, NextFunction } from 'express';
import { supabase } from './supabase';
import { AuthenticatedRequest } from './clerkAuth';

export type AppRole = 'student' | 'teacher' | 'admin' | 'owner' | 'school_admin';

export interface ActorProfile {
  clerkId: string;
  role: AppRole;
  schoolId: string | null;
}

/** Request with the resolved actor attached by a guard / loadActor. */
export interface ActorRequest extends AuthenticatedRequest {
  actor?: ActorProfile;
}

/** Owner and the legacy 'admin' are the same platform-owner tier. */
export function isOwner(role: AppRole): boolean {
  return role === 'owner' || role === 'admin';
}

/** Fetch the caller's role + school from their profile. */
export async function getActor(clerkId: string): Promise<ActorProfile | null> {
  const { data, error } = await supabase
    .from('profiles')
    .select('role, school_id')
    .eq('clerk_id', clerkId)
    .maybeSingle();
  if (error || !data) return null;
  const row = data as { role?: string; school_id?: string | null };
  return {
    clerkId,
    role: (row.role as AppRole) ?? 'student',
    schoolId: row.school_id ?? null,
  };
}

/**
 * Non-enforcing: resolve the actor and attach it to req.actor, then continue.
 * Use on routes that branch on role rather than reject.
 */
export async function loadActor(req: ActorRequest, res: Response, next: NextFunction) {
  if (!req.auth?.clerkId) return res.status(401).json({ error: 'Unauthorized' });
  req.actor = (await getActor(req.auth.clerkId)) ?? undefined;
  next();
}

/**
 * Enforcing guard factory. Passes only if the caller's role is in `roles`.
 * Attaches req.actor on success so handlers can read schoolId without a re-query.
 */
export function requireRoles(...roles: AppRole[]) {
  return async (req: ActorRequest, res: Response, next: NextFunction) => {
    try {
      if (!req.auth?.clerkId) return res.status(401).json({ error: 'Unauthorized' });
      const actor = await getActor(req.auth.clerkId);
      if (!actor) return res.status(403).json({ error: 'Forbidden - no profile' });
      if (!roles.includes(actor.role)) {
        return res.status(403).json({ error: 'Forbidden - insufficient role' });
      }
      req.actor = actor;
      next();
    } catch (err) {
      console.error('requireRoles error:', err);
      return res.status(500).json({ error: 'Server error' });
    }
  };
}

/** Platform owner only (spec §3). */
export const requireOwner = requireRoles('owner', 'admin');
/** School admin only — an account manager, never a grader (spec §4). */
export const requireSchoolAdmin = requireRoles('school_admin');
/** Teacher only — full control inside their own classes (spec §5). */
export const requireTeacher = requireRoles('teacher');

/**
 * The school a school_admin / teacher acts within. Returns null for an owner
 * (owner is cross-school and must pass an explicit school_id in the request).
 * Guards should reject a school_admin/teacher whose schoolId is null.
 */
export async function resolveActorSchool(actor: ActorProfile): Promise<string | null> {
  if (isOwner(actor.role)) return null;
  return actor.schoolId;
}

/**
 * Guard: require the caller to be a school_admin or teacher WITH a school, and
 * attach that schoolId. Rejects an unassigned account (403) rather than leaking
 * cross-school data.
 */
export function requireSchoolScope(...roles: AppRole[]) {
  const allowed = roles.length ? roles : (['school_admin', 'teacher'] as AppRole[]);
  return async (req: ActorRequest, res: Response, next: NextFunction) => {
    try {
      if (!req.auth?.clerkId) return res.status(401).json({ error: 'Unauthorized' });
      const actor = await getActor(req.auth.clerkId);
      if (!actor) return res.status(403).json({ error: 'Forbidden - no profile' });
      if (!allowed.includes(actor.role)) {
        return res.status(403).json({ error: 'Forbidden - insufficient role' });
      }
      if (!actor.schoolId) {
        return res.status(403).json({ error: 'Forbidden - no school assigned' });
      }
      req.actor = actor;
      next();
    } catch (err) {
      console.error('requireSchoolScope error:', err);
      return res.status(500).json({ error: 'Server error' });
    }
  };
}
