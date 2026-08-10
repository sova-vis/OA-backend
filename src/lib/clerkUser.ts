import { supabase } from './supabase';

/**
 * Fetches a Clerk user's basic identity (primary email + name) via the Clerk
 * Admin API. Clerk's default session token often omits these claims, so we read
 * them straight from the API when we need to know who a user is.
 */
export async function fetchClerkUserBasics(clerkId: string): Promise<{ email: string | null; fullName: string | null }> {
  const key = process.env.CLERK_SECRET_KEY;
  if (!key) return { email: null, fullName: null };
  try {
    const res = await fetch(`https://api.clerk.com/v1/users/${clerkId}`, { headers: { Authorization: `Bearer ${key}` } });
    if (!res.ok) return { email: null, fullName: null };
    const u = (await res.json()) as {
      primary_email_address_id?: string;
      email_addresses?: { id: string; email_address: string }[];
      first_name?: string | null;
      last_name?: string | null;
      username?: string | null;
    };
    const emails = Array.isArray(u.email_addresses) ? u.email_addresses : [];
    const primary = emails.find((e) => e.id === u.primary_email_address_id) || emails[0];
    const email = primary?.email_address ?? null;
    const first = (u.first_name ?? '').trim();
    const last = (u.last_name ?? '').trim();
    const fullName = [first, last].filter(Boolean).join(' ') || (u.username ?? null);
    return { email, fullName };
  } catch {
    return { email: null, fullName: null };
  }
}

/**
 * Ensures a minimal student profile row exists so teachers can see who requested
 * to join (name/email), even before the student finishes onboarding. Never
 * downgrades an existing role or clears data already captured.
 */
export async function ensureStudentProfile(clerkId: string): Promise<void> {
  const { data: existing } = await supabase.from('profiles').select('clerk_id, email, full_name').eq('clerk_id', clerkId).maybeSingle();
  const row = existing as { email?: string | null; full_name?: string | null } | null;
  if (row && row.email && row.full_name) return; // already identified

  const { email, fullName } = await fetchClerkUserBasics(clerkId);
  if (!email && !fullName) return;

  if (row) {
    const update: Record<string, unknown> = {};
    if (!row.email && email) update.email = email;
    if (!row.full_name && fullName) update.full_name = fullName;
    if (Object.keys(update).length) await supabase.from('profiles').update(update).eq('clerk_id', clerkId);
  } else {
    await supabase.from('profiles').insert({ clerk_id: clerkId, email, full_name: fullName, role: 'student', onboarding_complete: false });
  }
}
