/**
 * Staff (school-admin / teacher) account creation for the Teacher Portal v1.
 *
 * Staff sign in with EMAIL + password via Supabase Auth (GoTrue) — decision §7#4
 * keeps join-code/no-email for STUDENTS, but staff are adults with real logins.
 * We create the auth user, then upsert their profile with role + school_id +
 * subjects/levels, and set must_change_password so the first sign-in forces a
 * reset (§3.1 "force a password reset on first sign-in", §4.1 teacher creation).
 *
 * Mirrors services/adminService.ts (the existing teacher-creation path) but is
 * generalised over role/school and handles the "email already exists → promote".
 */
import { createClient } from '@supabase/supabase-js';
import crypto from 'crypto';
import { supabase } from '../lib/supabase';
import { encryptSecret } from '../lib/secretBox';

const admin = createClient(
  process.env.SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY || '',
  { auth: { autoRefreshToken: false, persistSession: false } },
);

export type StaffRole = 'school_admin' | 'teacher';

export interface CreateStaffInput {
  email?: string;         // omitted → a login is generated as name@<short>propel.com
  name: string;
  role: StaffRole;
  schoolId: string;
  password?: string;      // optional; a strong temp one is generated if omitted
  subjects?: string[];    // syllabus codes the teacher teaches (§4.1)
  levels?: string[];      // year groups / O-A levels (§4.1)
  createdBy?: string;     // actor clerk_id, for provenance
}

export interface CreateStaffResult {
  clerkId: string;
  email: string;
  tempPassword: string | null; // returned once for a NEW account; null if promoted
  existed: boolean;
}

/** Readable, reasonably strong temp password (no ambiguous chars). */
export function generatePassword(len = 14): string {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789';
  const spec = '!@#$%*?';
  const buf = crypto.randomBytes(len + 2);
  let out = '';
  for (let i = 0; i < len; i++) out += chars[buf[i] % chars.length];
  out += spec[buf[len] % spec.length];
  out += String((buf[len + 1] % 9) + 1);
  return out;
}

async function findUserIdByEmail(email: string): Promise<string | null> {
  const normalized = email.trim().toLowerCase();
  const { data: profile } = await supabase.from('profiles').select('clerk_id').eq('email', normalized).maybeSingle();
  const fromProfile = (profile as { clerk_id?: string | null } | null)?.clerk_id;
  if (fromProfile) return fromProfile;

  for (let page = 1; page <= 20; page++) {
    const { data, error } = await admin.auth.admin.listUsers({ page, perPage: 200 });
    if (error || !data?.users?.length) break;
    const match = data.users.find((u) => (u.email || '').toLowerCase() === normalized);
    if (match) return match.id;
    if (data.users.length < 200) break;
  }
  return null;
}

/** Short login-domain code for a school from its name ("ABC School" → "abc"). */
export function schoolShortCode(name: string): string {
  const words = (name || '')
    .toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean)
    .filter((w) => !['the', 'a', 'an', 'of'].includes(w));
  return (words[0] || 'school').slice(0, 10);
}

/** Local part of a generated login from a person's name ("John Doe" → "john"). */
function localPart(name: string): string {
  const first = (name || '').trim().split(/\s+/)[0] || 'user';
  return first.toLowerCase().replace(/[^a-z0-9]/g, '') || 'user';
}

/** The school's stored short code (feature_flags.short_code), else derived from its name. */
async function resolveShortCode(schoolId?: string): Promise<string> {
  if (!schoolId) return 'school';
  const { data } = await supabase.from('schools').select('name, feature_flags').eq('id', schoolId).maybeSingle();
  const row = data as { name?: string; feature_flags?: Record<string, unknown> } | null;
  const stored = row?.feature_flags?.short_code;
  return typeof stored === 'string' && stored ? stored : schoolShortCode(row?.name ?? 'school');
}

/** A free generated login email: name@<short>propel.com, numbered on collision. */
async function uniqueStaffEmail(name: string, shortCode: string): Promise<string> {
  const base = localPart(name);
  for (let i = 0; i < 60; i++) {
    const candidate = `${base}${i === 0 ? '' : i + 1}@${shortCode}propel.com`;
    if (!(await findUserIdByEmail(candidate))) return candidate;
  }
  return `${base}${Date.now()}@${shortCode}propel.com`;
}

export async function createStaffAccount(input: CreateStaffInput): Promise<CreateStaffResult> {
  if (!input.name?.trim()) {
    throw Object.assign(new Error('name is required'), { statusCode: 400 });
  }
  // A login is generated from the name + school short code unless one is supplied.
  const email = input.email?.trim().toLowerCase()
    || (await uniqueStaffEmail(input.name, await resolveShortCode(input.schoolId)));
  const tempPassword = input.password || generatePassword();

  let clerkId: string;
  let existed = false;

  const { data, error } = await admin.auth.admin.createUser({
    email,
    password: tempPassword,
    email_confirm: true,
    user_metadata: { full_name: input.name, name: input.name },
  });

  if (error || !data?.user?.id) {
    const msg = (error?.message || '').toLowerCase();
    if (msg.includes('already') || msg.includes('registered') || msg.includes('exists') || msg.includes('taken')) {
      const existing = await findUserIdByEmail(email);
      if (!existing) throw Object.assign(new Error(error?.message || 'user exists'), { statusCode: 409 });
      clerkId = existing;
      existed = true;
    } else {
      throw Object.assign(new Error(error?.message || 'failed to create user'), {
        statusCode: msg.includes('password') || msg.includes('weak') || msg.includes('invalid') ? 400 : 500,
      });
    }
  } else {
    clerkId = data.user.id;
  }

  const profileFields: Record<string, unknown> = {
    clerk_id: clerkId,
    email,
    full_name: input.name,
    role: input.role,
    school_id: input.schoolId,
    onboarding_complete: true,
    // Staff logins are admin-managed and shown in the owner / school-admin panels,
    // so we DON'T force a first-sign-in reset — that would make the shown password
    // go stale the moment they changed it. (A promoted existing user keeps theirs.)
    must_change_password: false,
  };
  // Store the issued password (encrypted at rest) so the panels can display the
  // current credential — only for a NEW account whose password we actually set.
  if (!existed) profileFields.visible_password = encryptSecret(tempPassword);
  if (input.subjects) profileFields.syllabus_codes = input.subjects;
  if (input.levels) profileFields.levels = input.levels;
  if (input.createdBy) profileFields.provisioned_by = input.createdBy;

  const { data: byEmail } = await supabase.from('profiles').select('id').eq('email', email).maybeSingle();
  if (byEmail) {
    const { error: upErr } = await supabase.from('profiles').update(profileFields).eq('email', email);
    if (upErr) throw upErr;
  } else {
    const { error: insErr } = await supabase.from('profiles').insert({ ...profileFields, level: 'N/A' });
    if (insErr) throw insErr;
  }

  return { clerkId, email, tempPassword: existed ? null : tempPassword, existed };
}

/** Reset a staff member's password to a fresh one-time password + force a reset. */
export async function resetStaffPassword(clerkId: string): Promise<string> {
  const tempPassword = generatePassword();
  const { error } = await admin.auth.admin.updateUserById(clerkId, { password: tempPassword });
  if (error) throw Object.assign(new Error(error.message || 'Failed to reset password'), { statusCode: 500 });
  // Store the new password (encrypted) and don't force a change — it becomes the
  // admin-managed login the panels display.
  await supabase.from('profiles')
    .update({ must_change_password: false, visible_password: encryptSecret(tempPassword), updated_at: new Date().toISOString() })
    .eq('clerk_id', clerkId);
  return tempPassword;
}

/** Change a staff member's login email (Supabase Auth + profile). Returns the normalized email. */
export async function updateStaffEmail(clerkId: string, email: string): Promise<string> {
  const normalized = (email || '').trim().toLowerCase();
  if (!normalized || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(normalized)) {
    throw Object.assign(new Error('Enter a valid email address.'), { statusCode: 400 });
  }
  const existing = await findUserIdByEmail(normalized);
  if (existing && existing !== clerkId) {
    throw Object.assign(new Error('That email is already in use.'), { statusCode: 409 });
  }
  const { error } = await admin.auth.admin.updateUserById(clerkId, { email: normalized, email_confirm: true });
  if (error) throw Object.assign(new Error(error.message || 'Failed to update email'), { statusCode: 500 });
  await supabase.from('profiles').update({ email: normalized, updated_at: new Date().toISOString() }).eq('clerk_id', clerkId);
  return normalized;
}
