/**
 * Audit log writer (spec §3.4 owner monitoring, §4.2 school-admin oversight,
 * §5.5 mark overrides). Best-effort and never throws into the caller — an audit
 * failure must not break the action being audited.
 */
import { supabase } from './supabase';

export interface AuditEntry {
  actorClerkId?: string | null;
  actorRole?: string | null;
  action: string;              // e.g. 'school.create', 'teacher.deactivate'
  targetType?: string | null;  // e.g. 'school', 'profile'
  targetId?: string | null;
  before?: unknown;
  after?: unknown;
  schoolId?: string | null;
}

export async function logAudit(e: AuditEntry): Promise<void> {
  try {
    await supabase.from('audit_log').insert({
      actor_clerk_id: e.actorClerkId ?? null,
      actor_role: e.actorRole ?? null,
      action: e.action,
      target_type: e.targetType ?? null,
      target_id: e.targetId ?? null,
      before: (e.before ?? null) as never,
      after: (e.after ?? null) as never,
      school_id: e.schoolId ?? null,
    });
  } catch (err) {
    console.error('[audit] write failed:', err);
  }
}
