/**
 * Consistent display-name resolution: prefer the real name, then the local part
 * of the email (before @), then a role-appropriate fallback. Avoids ever showing
 * a raw "User"/"Teacher"/"Student" when we at least have an email.
 */
export function displayName(
  fullName: string | null | undefined,
  email: string | null | undefined,
  fallback: 'Teacher' | 'Student' | 'User' = 'User'
): string {
  const name = (fullName || '').trim();
  if (name && name.toLowerCase() !== 'user') return name;
  const local = (email || '').split('@')[0]?.trim();
  if (local) return local;
  return fallback;
}
