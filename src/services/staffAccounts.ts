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

const admin = createClient(
  process.env.SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY || '',
  { auth: { autoRefreshToken: false, persistSession: false } },
);

export type StaffRole = 'school_admin' | 'teacher';

export interface CreateStaffInput {
  email: string;
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

export async function createStaffAccount(input: CreateStaffInput): Promise<CreateStaffResult> {
  const email = input.email.trim().toLowerCase();
  if (!email || !input.name?.trim()) {
    throw Object.assign(new Error('email and name are required'), { statusCode: 400 });
  }
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
    // Only force a reset for a freshly issued temp password, not for an existing
    // user we're promoting (they keep their own password).
    must_change_password: !existed,
  };
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
