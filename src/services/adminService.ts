import { createClient } from '@supabase/supabase-js';

// Initialize Supabase Admin Client
// REQUIRES separate environment variables for admin privileges
if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
    console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables");
}

const supabaseAdmin = createClient(
    process.env.SUPABASE_URL || '',
    process.env.SUPABASE_SERVICE_ROLE_KEY || '',
    {
        auth: {
            autoRefreshToken: false,
            persistSession: false
        }
    }
);

function splitName(name: string) {
    const parts = name.trim().split(/\s+/).filter(Boolean);
    if (parts.length === 0) {
        return { firstName: 'Teacher', lastName: '' };
    }
    if (parts.length === 1) {
        return { firstName: parts[0], lastName: '' };
    }
    return {
        firstName: parts[0],
        lastName: parts.slice(1).join(' '),
    };
}

async function createClerkTeacher(email: string, password: string, name: string) {
    const secretKey = process.env.CLERK_SECRET_KEY;

    if (!secretKey) {
        throw new Error('CLERK_SECRET_KEY is not configured');
    }

    const { firstName, lastName } = splitName(name);

    const response = await fetch('https://api.clerk.com/v1/users', {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${secretKey}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            email_address: [email],
            password,
            first_name: firstName,
            last_name: lastName || undefined,
            public_metadata: { role: 'teacher' },
        }),
    });

    const payload = await response.json() as any;

    if (!response.ok) {
        const message = payload?.errors?.[0]?.long_message || payload?.errors?.[0]?.message || 'Failed to create Clerk user';
        throw new Error(message);
    }

    return payload;
}

export const createTeacherAccount = async (email: string, password: string, name: string) => {
    const clerkUser = await createClerkTeacher(email, password, name);

    const { error: profileError } = await supabaseAdmin
        .from('profiles')
        .upsert({
            clerk_id: clerkUser.id,
            email,
            full_name: name,
            role: 'teacher',
            level: 'N/A',
            onboarding_complete: true,
        }, {
            onConflict: 'clerk_id',
        });

    if (profileError) {
        throw profileError;
    }

    return {
        id: clerkUser.id,
    };
};
