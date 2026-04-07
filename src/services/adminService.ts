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

class ServiceError extends Error {
    statusCode: number;

    constructor(message: string, statusCode = 500) {
        super(message);
        this.name = 'ServiceError';
        this.statusCode = statusCode;
    }
}

function getClerkSecretKey(): string {
    const raw = process.env.CLERK_SECRET_KEY || process.env.CLERK_API_KEY || '';
    const normalized = raw.trim().replace(/^['\"]+|['\"]+$/g, '');
    if (!normalized) {
        throw new ServiceError('Clerk secret key is not configured. Set CLERK_SECRET_KEY (or CLERK_API_KEY).', 500);
    }
    return normalized;
}

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
    const secretKey = getClerkSecretKey();

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
        const lowered = String(message).toLowerCase();
        if (lowered.includes('breach') || lowered.includes('password') || lowered.includes('invalid')) {
            throw new ServiceError(message, 400);
        }
        throw new ServiceError(message, 500);
    }

    return payload;
}

async function getClerkUserByEmail(email: string) {
    const secretKey = getClerkSecretKey();

    const query = new URLSearchParams({ email_address: email });
    const response = await fetch(`https://api.clerk.com/v1/users?${query.toString()}`, {
        method: 'GET',
        headers: {
            Authorization: `Bearer ${secretKey}`,
        },
    });

    const payload = await response.json() as any;
    if (!response.ok) {
        const message = payload?.errors?.[0]?.long_message || payload?.errors?.[0]?.message || 'Failed to query Clerk user';
        throw new ServiceError(message, 500);
    }

    if (Array.isArray(payload) && payload.length > 0) {
        return payload[0];
    }

    return null;
}

async function updateClerkUserRoleToTeacher(userId: string) {
    const secretKey = getClerkSecretKey();

    const response = await fetch(`https://api.clerk.com/v1/users/${userId}/metadata`, {
        method: 'PATCH',
        headers: {
            Authorization: `Bearer ${secretKey}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            public_metadata: { role: 'teacher' },
        }),
    });

    const payload = await response.json() as any;
    if (!response.ok) {
        const message = payload?.errors?.[0]?.long_message || payload?.errors?.[0]?.message || 'Failed to update Clerk user metadata';
        throw new ServiceError(message, 500);
    }

    return payload;
}

async function upsertTeacherProfileNoConstraint(email: string, name: string, clerkUserId: string) {
    const normalizedEmail = email.trim().toLowerCase();

    const { data: byEmail, error: lookupError } = await supabaseAdmin
        .from('profiles')
        .select('*')
        .eq('email', normalizedEmail)
        .maybeSingle();

    if (lookupError) {
        throw lookupError;
    }

    if (byEmail) {
        const { error: updateError } = await supabaseAdmin
            .from('profiles')
            .update({
                clerk_id: clerkUserId,
                full_name: name,
                role: 'teacher',
                onboarding_complete: true,
            })
            .eq('email', normalizedEmail);

        if (updateError) {
            throw updateError;
        }

        return;
    }

    const { error: createError } = await supabaseAdmin
        .from('profiles')
        .insert({
            clerk_id: clerkUserId,
            email: normalizedEmail,
            full_name: name,
            role: 'teacher',
            level: 'N/A',
            onboarding_complete: true,
        });

    if (createError) {
        throw createError;
    }
}

export const createTeacherAccount = async (email: string, password: string, name: string) => {
    let clerkUser: any;

    try {
        clerkUser = await createClerkTeacher(email, password, name);
    } catch (error: any) {
        const message = String(error?.message || 'Failed to create Clerk user').toLowerCase();
        if (message.includes('already') || message.includes('exists') || message.includes('taken')) {
            const existingUser = await getClerkUserByEmail(email.trim().toLowerCase());
            if (!existingUser?.id) {
                throw error;
            }
            clerkUser = existingUser;
        } else {
            throw error;
        }
    }

    if (!clerkUser?.id) {
        throw new ServiceError('Failed to resolve Clerk teacher user', 500);
    }

    await updateClerkUserRoleToTeacher(clerkUser.id);
    await upsertTeacherProfileNoConstraint(email, name, clerkUser.id);

    return {
        id: clerkUser.id,
    };
};
