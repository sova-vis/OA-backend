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

// Create the Supabase Auth (GoTrue) user for a new teacher and return its UUID.
// Replaces the former Clerk admin-API call. The app role itself lives in
// profiles.role (set by upsertTeacherProfileNoConstraint below), not in auth
// metadata, so we only seed display-name metadata here.
async function createSupabaseTeacher(email: string, password: string, name: string): Promise<string> {
    const { firstName, lastName } = splitName(name);
    const { data, error } = await supabaseAdmin.auth.admin.createUser({
        email: email.trim().toLowerCase(),
        password,
        email_confirm: true,
        user_metadata: {
            full_name: name,
            name,
            first_name: firstName,
            last_name: lastName || undefined,
        },
    });

    if (error || !data?.user?.id) {
        const message = error?.message || 'Failed to create user';
        const lowered = message.toLowerCase();
        // Let the caller catch "already registered" to promote the existing user.
        if (lowered.includes('already') || lowered.includes('exists') || lowered.includes('registered') || lowered.includes('taken')) {
            throw new ServiceError(message, 409);
        }
        if (lowered.includes('password') || lowered.includes('weak') || lowered.includes('breach') || lowered.includes('invalid')) {
            throw new ServiceError(message, 400);
        }
        throw new ServiceError(message, 500);
    }

    return data.user.id;
}

// Resolve an existing auth user's UUID by email, for the "email already exists →
// promote them to teacher" path. Prefers the profiles row (which stores the auth
// uid as clerk_id); falls back to paging the auth user list.
async function findSupabaseUserIdByEmail(email: string): Promise<string | null> {
    const normalizedEmail = email.trim().toLowerCase();

    const { data: profile } = await supabaseAdmin
        .from('profiles')
        .select('clerk_id')
        .eq('email', normalizedEmail)
        .maybeSingle();
    const fromProfile = (profile as { clerk_id?: string | null } | null)?.clerk_id;
    if (fromProfile) {
        return fromProfile;
    }

    for (let page = 1; page <= 20; page++) {
        const { data, error } = await supabaseAdmin.auth.admin.listUsers({ page, perPage: 200 });
        if (error || !data?.users?.length) {
            break;
        }
        const match = data.users.find((u) => (u.email || '').toLowerCase() === normalizedEmail);
        if (match) {
            return match.id;
        }
        if (data.users.length < 200) {
            break;
        }
    }

    return null;
}

async function upsertTeacherProfileNoConstraint(email: string, name: string, userId: string) {
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
                clerk_id: userId,
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
            clerk_id: userId,
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
    let userId: string;

    try {
        userId = await createSupabaseTeacher(email, password, name);
    } catch (error: any) {
        const message = String(error?.message || '').toLowerCase();
        if (message.includes('already') || message.includes('exists') || message.includes('registered') || message.includes('taken')) {
            const existingId = await findSupabaseUserIdByEmail(email);
            if (!existingId) {
                throw error;
            }
            userId = existingId;
        } else {
            throw error;
        }
    }

    await upsertTeacherProfileNoConstraint(email, name, userId);

    return {
        id: userId,
    };
};
