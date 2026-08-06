// Promote an existing signed-in account to the "teacher" role, in both Clerk
// (public_metadata.role) and the Supabase profiles table. Mirrors
// promote-user-admin.js. Usage:
//   node ./scripts/promote-user-teacher.js <email>
// The person must have signed in at least once first so their Clerk user exists.
const path = require('path');
const dotenv = require('dotenv');
const { createClient } = require('@supabase/supabase-js');

dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

const CLERK_SECRET_KEY = process.env.CLERK_SECRET_KEY || '';
const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

const targetEmail = (process.argv[2] || '').trim().toLowerCase();

if (!targetEmail) {
  console.error('Usage: node ./scripts/promote-user-teacher.js <email>');
  process.exit(1);
}
if (!CLERK_SECRET_KEY) {
  console.error('Missing CLERK_SECRET_KEY in OA-backend/.env');
  process.exit(1);
}
if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in OA-backend/.env');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { autoRefreshToken: false, persistSession: false },
});

function getPrimaryEmail(user) {
  if (!user || !Array.isArray(user.email_addresses)) return null;
  const primary = user.email_addresses.find((row) => row.id === user.primary_email_address_id) || user.email_addresses[0];
  return primary && primary.email_address ? String(primary.email_address).toLowerCase() : null;
}

async function findClerkUserByEmail(email) {
  const query = new URLSearchParams({ email_address: [email].join(',') });
  const response = await fetch(`https://api.clerk.com/v1/users?${query.toString()}`, {
    headers: { Authorization: `Bearer ${CLERK_SECRET_KEY}` },
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.errors?.[0]?.message || 'Failed to query Clerk users');
  if (!Array.isArray(payload)) return null;
  return payload.find((user) => getPrimaryEmail(user) === email) || null;
}

async function setClerkTeacherRole(userId) {
  const response = await fetch(`https://api.clerk.com/v1/users/${userId}/metadata`, {
    method: 'PATCH',
    headers: { Authorization: `Bearer ${CLERK_SECRET_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ public_metadata: { role: 'teacher' } }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.errors?.[0]?.message || 'Failed to set Clerk metadata');
  return payload;
}

async function upsertTeacherProfile(clerkUser) {
  const clerkId = clerkUser.id;
  const fullName = `${clerkUser.first_name || ''} ${clerkUser.last_name || ''}`.trim() || 'Teacher';

  const { data: existing } = await supabase.from('profiles').select('id').eq('clerk_id', clerkId).maybeSingle();
  const base = { clerk_id: clerkId, email: targetEmail, full_name: fullName, role: 'teacher', onboarding_complete: true };

  if (existing) {
    const { data, error } = await supabase.from('profiles').update(base).eq('clerk_id', clerkId).select('*').single();
    if (error) throw error;
    return data;
  }
  const { data, error } = await supabase.from('profiles').insert(base).select('*').single();
  if (error) throw error;
  return data;
}

async function main() {
  console.log(`Promoting ${targetEmail} to teacher...`);
  const clerkUser = await findClerkUserByEmail(targetEmail);
  if (!clerkUser) throw new Error('No Clerk user found for this email. Sign in once at http://localhost:3000 first, then retry.');
  await setClerkTeacherRole(clerkUser.id);
  console.log(`Updated Clerk metadata role=teacher for ${clerkUser.id}`);
  const profile = await upsertTeacherProfile(clerkUser);
  console.log(`Updated Supabase profile role=teacher for clerk_id=${profile.clerk_id}`);
  console.log('Done. Sign out and back in — you will land on the teacher portal.');
}

main().catch((error) => {
  console.error('Failed:', error.message || error);
  process.exit(1);
});
