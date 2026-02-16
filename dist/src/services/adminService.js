"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createTeacherAccount = void 0;
const supabase_js_1 = require("@supabase/supabase-js");
// Initialize Supabase Admin Client
// REQUIRES separate environment variables for admin privileges
if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
    console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY environment variables");
}
const supabaseAdmin = (0, supabase_js_1.createClient)(process.env.SUPABASE_URL || '', process.env.SUPABASE_SERVICE_ROLE_KEY || '', {
    auth: {
        autoRefreshToken: false,
        persistSession: false
    }
});
const createTeacherAccount = async (email, password, name) => {
    // 1. Create Auth User
    const { data: authData, error: authError } = await supabaseAdmin.auth.admin.createUser({
        email,
        password,
        email_confirm: true,
        user_metadata: { full_name: name, role: 'teacher' }
    });
    if (authError)
        throw authError;
    if (!authData.user)
        throw new Error("Failed to create user");
    // 2. Create Profile Entry
    // Note: If you have a Trigger on auth.users -> public.profiles, this might be redundant or require an update instead.
    // Assuming manual profile creation or update:
    const { error: profileError } = await supabaseAdmin
        .from('profiles')
        .insert([
        {
            id: authData.user.id,
            full_name: name,
            role: 'teacher',
            level: 'N/A', // Teachers don't have levels usually
            onboarding_complete: true
        }
    ]);
    // If insert fails (e.g. trigger already created it), try update
    if (profileError) {
        // Check if duplicate key error, then update
        if (profileError.code === '23505') {
            const { error: updateError } = await supabaseAdmin
                .from('profiles')
                .update({ role: 'teacher', full_name: name, onboarding_complete: true })
                .eq('id', authData.user.id);
            if (updateError)
                throw updateError;
        }
        else {
            throw profileError;
        }
    }
    return authData.user;
};
exports.createTeacherAccount = createTeacherAccount;
