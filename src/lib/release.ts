import { supabase } from './supabase';

/**
 * Result-release helpers (spec §11). Results are invisible to students until
 * explicitly released; automatic release fires once every item in a submission
 * has been reviewed or auto-approved (§11.1). MCQ and AI marks alike respect the
 * gate.
 */

const REVIEWED_STATUSES = ['approved', 'overridden', 'auto_approved'];

/** A submission is releasable when every one of its marks has been reviewed. */
export async function isSubmissionFullyReviewed(submissionId: string): Promise<boolean> {
  const { data: marks } = await supabase.from('submission_marks').select('status').eq('submission_id', submissionId);
  const rows = (marks ?? []) as { status: string }[];
  if (rows.length === 0) return false;
  return rows.every((m) => REVIEWED_STATUSES.includes(m.status));
}

export async function releaseSubmission(submissionId: string, actorClerkId: string): Promise<void> {
  const now = new Date().toISOString();
  await supabase.from('submissions').update({ released_at: now, updated_at: now }).eq('id', submissionId);
  await supabase.from('activity_log').insert({ actor_clerk_id: actorClerkId, event_type: 'release', target_type: 'submission', target_id: submissionId, detail: {} });
}

/**
 * Auto-release any fully-reviewed, not-yet-released submissions when the
 * assignment has auto_release on. Safe to call after any review action.
 */
export async function maybeAutoRelease(assignmentId: string, actorClerkId: string): Promise<number> {
  const { data: assignment } = await supabase.from('assignments').select('auto_release').eq('id', assignmentId).maybeSingle();
  if (!assignment || !(assignment as { auto_release: boolean }).auto_release) return 0;

  const { data: subs } = await supabase
    .from('submissions')
    .select('id, released_at, status')
    .eq('assignment_id', assignmentId)
    .in('status', ['submitted', 'late']);

  let released = 0;
  for (const s of (subs ?? []) as { id: string; released_at: string | null }[]) {
    if (s.released_at) continue;
    if (await isSubmissionFullyReviewed(s.id)) {
      await releaseSubmission(s.id, actorClerkId);
      released += 1;
    }
  }
  return released;
}
