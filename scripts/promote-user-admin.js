const path = require('path');
const dotenv = require('dotenv');
const { createClient } = require('@supabase/supabase-js');

dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

const CLERK_SECRET_KEY = process.env.CLERK_SECRET_KEY || '';
const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

const targetEmail = (process.argv[2] || '').trim().toLowerCase();

if (!targetEmail) {
  console.error('Usage: node ./scripts/promote-user-admin.js <email>');
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
  const primary = user.email_addresses.find((row) => row.id === user.primary_email_address_id)
    || user.email_addresses[0];
  return primary && primary.email_address ? String(primary.email_address).toLowerCase() : null;
}

async function findClerkUserByEmail(email) {
  const query = new URLSearchParams({ email_address: [email].join(',') });
  const response = await fetch(`https://api.clerk.com/v1/users?${query.toString()}`, {
    headers: { Authorization: `Bearer ${CLERK_SECRET_KEY}` },
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload?.errors?.[0]?.message || 'Failed to query Clerk users');
  }

  if (!Array.isArray(payload)) {
    return null;
  }

  return payload.find((user) => getPrimaryEmail(user) === email) || null;
}

async function setClerkAdminRole(userId) {
  const response = await fetch(`https://api.clerk.com/v1/users/${userId}/metadata`, {
    method: 'PATCH',
    headers: {
      Authorization: `Bearer ${CLERK_SECRET_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ public_metadata: { role: 'admin' } }),
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload?.errors?.[0]?.message || 'Failed to set Clerk metadata');
  }

  return payload;
}

async function upsertSupabaseAdminProfile(clerkUser) {
  const clerkId = clerkUser.id;
  const firstName = clerkUser.first_name || '';
  const lastName = clerkUser.last_name || '';
  const fullName = `${firstName} ${lastName}`.trim() || 'Admin';

  const { data: byEmail, error: lookupError } = await supabase
    .from('profiles')
    .select('*')
    .eq('email', targetEmail)
    .maybeSingle();

  if (lookupError) {
    throw lookupError;
  }

  if (byEmail) {
    const { data: updated, error: updateError } = await supabase
      .from('profiles')
      .update({
        clerk_id: clerkId,
        full_name: fullName,
        role: 'admin',
        onboarding_complete: true,
      })
      .eq('email', targetEmail)
      .select('*')
      .single();

    if (updateError) {
      throw updateError;
    }

    return updated;
  }

  const { data: created, error: createError } = await supabase
    .from('profiles')
    .insert({
      clerk_id: clerkId,
      email: targetEmail,
      full_name: fullName,
      role: 'admin',
      onboarding_complete: true,
    })
    .select('*')
    .single();

  if (createError) {
    throw createError;
  }

  return created;
}

async function main() {
  console.log(`Promoting ${targetEmail} to admin...`);

  const clerkUser = await findClerkUserByEmail(targetEmail);
  if (!clerkUser) {
    throw new Error('No Clerk user found for this email. Sign in once first (Google or password), then retry.');
  }

  await setClerkAdminRole(clerkUser.id);
  console.log(`Updated Clerk metadata role=admin for ${clerkUser.id}`);

  const profile = await upsertSupabaseAdminProfile(clerkUser);
  console.log(`Updated Supabase profile role=admin for clerk_id=${profile.clerk_id}`);

  console.log('Done. You can now log in and will be redirected to admin dashboard.');
}

main().catch((error) => {
  console.error('Failed:', error.message || error);
  process.exit(1);
});
