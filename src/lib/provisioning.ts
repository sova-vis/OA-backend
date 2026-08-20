import bcrypt from 'bcrypt';
import crypto from 'crypto';
import { supabase } from './supabase';

/**
 * Teacher-provisioned student accounts (spec §4.3). Generates readable
 * usernames and one-time passwords; the plaintext password is returned once for
 * the printable credential sheet and never stored (only its bcrypt hash is).
 */

const PASSWORD_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789';

export function generatePassword(length = 8): string {
  // CSPRNG — these become one-time login passwords and synthetic ids, so the
  // sequence must not be predictable from observed outputs (Math.random isn't).
  let out = '';
  for (let i = 0; i < length; i += 1) {
    out += PASSWORD_ALPHABET[crypto.randomInt(PASSWORD_ALPHABET.length)];
  }
  return out;
}

function slugifyName(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]+/g, '.')
    .replace(/^\.+|\.+$/g, '')
    .slice(0, 20) || 'student';
}

/** Build a username from a name that is unique (case-insensitive) in profiles. */
export async function generateUniqueUsername(name: string): Promise<string> {
  const base = slugifyName(name);
  for (let attempt = 0; attempt < 50; attempt += 1) {
    const candidate = attempt === 0 ? base : `${base}${attempt + 1}`;
    const { data, error } = await supabase
      .from('profiles')
      .select('id')
      .ilike('username', candidate)
      .maybeSingle();
    if (error) throw error;
    if (!data) return candidate;
  }
  // Fall back to a random suffix rather than loop forever.
  return `${base}.${generatePassword(4).toLowerCase()}`;
}

export async function hashPassword(plaintext: string): Promise<string> {
  return bcrypt.hash(plaintext, 10);
}

/** Synthetic clerk_id for accounts with no Clerk identity (§4.3). */
export function syntheticClerkId(): string {
  return `prov_${generatePassword(20)}`;
}
