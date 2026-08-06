import { supabase } from './supabase';

/**
 * Teacher-portal shared helpers for institutions and class join codes.
 *
 * Institution stubs (spec §1.2): a school exists in data before it signs up.
 * When a teacher has no institution yet, we mint an `unclaimed` stub so every
 * class can carry a non-null institution_id and later roll in with no migration
 * once a school claims the stub (§1.3 / §14.2).
 */

export interface ProfileLike {
  clerk_id: string;
  full_name?: string | null;
  school_name?: string | null;
  institution_id?: string | null;
}

function normalizeName(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, ' ');
}

/**
 * Resolve (or create) the institution a teacher's classes belong to.
 *
 * - If the profile already has institution_id, return it.
 * - Else if the teacher gave a school name that matches an existing institution,
 *   we do NOT auto-join (§1.1 edge case) — a matching name still gets its own
 *   stub; joining an existing school is an explicit opt-in handled in Phase 1.
 * - Else create an `unclaimed` stub named after the school (or the teacher).
 *
 * The resolved id is written back onto the profile so this runs once per teacher.
 */
export async function ensureInstitutionForTeacher(profile: ProfileLike): Promise<string> {
  if (profile.institution_id) {
    return profile.institution_id;
  }

  const schoolName = (profile.school_name || '').trim();
  const stubName = schoolName || `${(profile.full_name || 'Teacher').trim()}'s School`;

  const { data: created, error: createError } = await supabase
    .from('institutions')
    .insert({
      name: stubName,
      name_normalized: normalizeName(stubName),
      status: 'unclaimed',
    })
    .select('id')
    .single();

  if (createError) {
    throw createError;
  }

  const institutionId = created.id as string;

  const { error: updateError } = await supabase
    .from('profiles')
    .update({ institution_id: institutionId, updated_at: new Date().toISOString() })
    .eq('clerk_id', profile.clerk_id);

  if (updateError) {
    throw updateError;
  }

  return institutionId;
}

// Ambiguous characters (0/O, 1/I) are excluded so codes read cleanly aloud and
// off a projector (§3.2).
const JOIN_CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';

function randomJoinCode(length = 6): string {
  let code = '';
  for (let i = 0; i < length; i += 1) {
    code += JOIN_CODE_ALPHABET[Math.floor(Math.random() * JOIN_CODE_ALPHABET.length)];
  }
  return code;
}

/**
 * Generate a 6-char join code that is unique among currently-enabled codes.
 * Stored uppercase; entry is normalised uppercase so it is case-insensitive.
 */
export async function generateUniqueJoinCode(): Promise<string> {
  for (let attempt = 0; attempt < 12; attempt += 1) {
    const code = randomJoinCode();
    const { data, error } = await supabase
      .from('classes')
      .select('id')
      .eq('join_code', code)
      .eq('join_enabled', true)
      .maybeSingle();

    if (error) {
      throw error;
    }
    if (!data) {
      return code;
    }
  }
  // Astronomically unlikely with a 32^6 space; widen rather than loop forever.
  return randomJoinCode(8);
}

export function normalizeJoinCode(input: string): string {
  return input.trim().toUpperCase();
}
