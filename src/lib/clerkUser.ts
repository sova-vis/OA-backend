import { supabase } from './supabase';

/**
 * Identity helpers for Supabase Auth. (File name kept for import compatibility.)
 * Email/name now come from the verified access token (Supabase puts `email` at
 * the top level and the display name in `user_metadata.full_name`), so there is
 * no admin-API round-trip any more — callers pass what the token already gave us.
 */

/**
 * Ensures a minimal student profile row exists so teachers can see who requested
 * to join (name/email), even before the student finishes onboarding. Never
 * downgrades an existing role or clears data already captured.
 */
export async function ensureStudentProfile(
  clerkId: string,
  email?: string | null,
  fullName?: string | null,
): Promise<void> {
  const { data: existing } = await supabase
    .from('profiles')
    .select('clerk_id, email, full_name')
    .eq('clerk_id', clerkId)
    .maybeSingle();
  const row = existing as { email?: string | null; full_name?: string | null } | null;
  if (row && row.email && row.full_name) return; // already identified

  const cleanEmail = (email ?? '').trim() || null;
  const cleanName = (fullName ?? '').trim() || null;
  if (!cleanEmail && !cleanName && row) return;

  if (row) {
    const update: Record<string, unknown> = {};
    if (!row.email && cleanEmail) update.email = cleanEmail;
    if (!row.full_name && cleanName) update.full_name = cleanName;
    if (Object.keys(update).length) await supabase.from('profiles').update(update).eq('clerk_id', clerkId);
  } else {
    await supabase.from('profiles').insert({
      clerk_id: clerkId,
      email: cleanEmail,
      full_name: cleanName,
      role: 'student',
      onboarding_complete: false,
    });
  }
}

/**
 * A student who joins a class via a link is a student by definition and their
 * level/subject are known from the class — so we skip the onboarding survey
 * entirely: mark them onboarded and merge the class subject into their subjects
 * (never clobbering subjects they already picked).
 */
export async function completeJoinedStudentOnboarding(clerkId: string, subject: string | null): Promise<void> {
  const { data } = await supabase.from('profiles').select('selected_subjects, role').eq('clerk_id', clerkId).maybeSingle();
  const row = data as { selected_subjects?: string[] | null; role?: string } | null;
  const subjects = new Set<string>((row?.selected_subjects ?? []).filter(Boolean));
  if (subject && subject.trim()) subjects.add(subject.trim());
  const patch = { onboarding_complete: true, role: row?.role || 'student', selected_subjects: Array.from(subjects) };
  if (row) await supabase.from('profiles').update(patch).eq('clerk_id', clerkId);
  else await supabase.from('profiles').insert({ clerk_id: clerkId, ...patch });
}
