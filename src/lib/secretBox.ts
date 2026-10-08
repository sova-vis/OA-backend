/**
 * App-level reversible encryption for PROVISIONED staff credentials.
 *
 * School-admin / teacher accounts are issued (and reset) with a password the
 * operator must be able to hand out, so the owner + school-admin panels show it.
 * We store it encrypted-at-rest (AES-256-GCM) rather than as plaintext, so a raw
 * DB dump alone does not reveal logins — the key lives only in the app env.
 *
 * This is explicitly NOT a replacement for hashing real user secrets. It is used
 * only for admin-managed, admin-visible logins the operator already controls.
 * The key is derived from CREDENTIAL_ENC_KEY when set, else the service-role key,
 * so no new env var is strictly required; rotating that key makes previously
 * stored values undecryptable (they then simply read back as "reset to reveal").
 */
import crypto from 'crypto';

const KEY = crypto
  .createHash('sha256')
  .update(
    process.env.CREDENTIAL_ENC_KEY
      || process.env.SUPABASE_SERVICE_ROLE_KEY
      || process.env.SUPABASE_KEY
      || 'propel-dev-credential-key',
  )
  .digest(); // 32 bytes for AES-256

/** Encrypt a short secret to a self-describing `v1:iv:tag:ciphertext` string. */
export function encryptSecret(plain: string): string {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', KEY, iv);
  const enc = Buffer.concat([cipher.update(plain, 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();
  return `v1:${iv.toString('base64')}:${tag.toString('base64')}:${enc.toString('base64')}`;
}

/** Decrypt a value produced by encryptSecret; null for empty/invalid/undecryptable. */
export function decryptSecret(stored: string | null | undefined): string | null {
  if (!stored || typeof stored !== 'string') return null;
  const parts = stored.split(':');
  if (parts.length !== 4 || parts[0] !== 'v1') return null;
  try {
    const iv = Buffer.from(parts[1], 'base64');
    const tag = Buffer.from(parts[2], 'base64');
    const enc = Buffer.from(parts[3], 'base64');
    const decipher = crypto.createDecipheriv('aes-256-gcm', KEY, iv);
    decipher.setAuthTag(tag);
    const dec = Buffer.concat([decipher.update(enc), decipher.final()]);
    return dec.toString('utf8');
  } catch {
    return null;
  }
}
