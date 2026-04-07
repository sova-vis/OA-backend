const path = require('path');
const dotenv = require('dotenv');
const { createClient } = require('@supabase/supabase-js');

dotenv.config({ path: path.resolve(__dirname, '..', '.env') });

const ADMIN_EMAIL = process.env.ADMIN_BOOTSTRAP_EMAIL || 'sovavis2025@gmail.com';
const ADMIN_PASSWORD = process.env.ADMIN_BOOTSTRAP_PASSWORD || 'ChangeThisAdminPassword123!';
const ADMIN_NAME = process.env.ADMIN_BOOTSTRAP_NAME || 'Admin';
const CLERK_SECRET_KEY = process.env.CLERK_SECRET_KEY || '';
const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

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

function splitName(name) {
  const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return { first_name: 'Admin', last_name: '' };
  if (parts.length === 1) return { first_name: parts[0], last_name: '' };
  return { first_name: parts[0], last_name: parts.slice(1).join(' ') };
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

  if (Array.isArray(payload) && payload.length > 0) {
    return payload[0];
  }

  return null;
}

async function createClerkAdminUser() {
  const { first_name, last_name } = splitName(ADMIN_NAME);

  const response = await fetch('https://api.clerk.com/v1/users', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${CLERK_SECRET_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      email_address: [ADMIN_EMAIL],
      password: ADMIN_PASSWORD,
      first_name,
      last_name: last_name || undefined,
      public_metadata: { role: 'admin' },
    }),
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload?.errors?.[0]?.long_message || payload?.errors?.[0]?.message || 'Failed to create Clerk admin user');
  }

  return payload;
}

async function updateClerkAdminMetadata(userId) {
  const response = await fetch(`https://api.clerk.com/v1/users/${userId}/metadata`, {
    method: 'PATCH',
    headers: {
      Authorization: `Bearer ${CLERK_SECRET_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      public_metadata: { role: 'admin' },
    }),
  });

  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload?.errors?.[0]?.message || 'Failed to update Clerk admin metadata');
  }

  return payload;
}

async function upsertSupabaseProfile(clerkId) {
  const { data: byEmail, error: lookupError } = await supabase
    .from('profiles')
    .select('*')
    .eq('email', ADMIN_EMAIL)
    .maybeSingle();

  if (lookupError) {
    throw lookupError;
  }

  if (byEmail) {
    const { data: updated, error: updateError } = await supabase
      .from('profiles')
      .update({
        clerk_id: clerkId,
        full_name: ADMIN_NAME,
        role: 'admin',
        onboarding_complete: true,
      })
      .eq('email', ADMIN_EMAIL)
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
      email: ADMIN_EMAIL,
      full_name: ADMIN_NAME,
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
  console.log(`Ensuring admin account exists in Clerk and Supabase for ${ADMIN_EMAIL} ...`);

  let clerkUser = await findClerkUserByEmail(ADMIN_EMAIL);

  if (!clerkUser) {
    console.log('Admin user not found in Clerk. Creating...');
    clerkUser = await createClerkAdminUser();
    console.log(`Created Clerk admin user: ${clerkUser.id}`);
  } else {
    console.log(`Found existing Clerk user: ${clerkUser.id}`);
  }

  await updateClerkAdminMetadata(clerkUser.id);
  console.log('Ensured Clerk metadata role=admin');

  const profile = await upsertSupabaseProfile(clerkUser.id);
  console.log(`Upserted Supabase profile for clerk_id=${profile.clerk_id}`);

  console.log('Admin bootstrap completed successfully.');
}

main().catch((error) => {
  console.error('Admin bootstrap failed:', error.message || error);
  process.exit(1);
});
