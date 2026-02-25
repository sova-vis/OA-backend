import { Router, Request, Response } from 'express';
import { supabase } from './lib/supabase';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';

const router = Router();

type UserRole = 'student' | 'teacher' | 'admin';

function parseRole(role: unknown): UserRole {
  if (role === 'teacher' || role === 'admin') {
    return role;
  }
  return 'student';
}

/**
 * GET /auth/profile
 * Get current user's profile (requires auth)
 */
router.get('/profile', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    if (!req.auth?.clerkId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const { data: profile, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('clerk_id', req.auth.clerkId)
      .single();

    if (error && error.code !== 'PGRST116') {
      throw error;
    }

    if (!profile) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    return res.json(profile);
  } catch (err) {
    console.error('Error fetching profile:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * POST /auth/sync-profile
 * Ensure current Clerk user has a profile row in Supabase
 */
router.post('/sync-profile', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;

    console.log('🔄 Sync-profile request received');
    console.log('👤 Clerk ID:', clerkId);
    console.log('📦 Request body:', req.body);

    if (!clerkId) {
      console.error('❌ Missing Clerk ID');
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const body = (req.body ?? {}) as {
      email?: string;
      full_name?: string;
      role?: string;
    };

    const incomingRole = parseRole(body.role);
    const fullName = (body.full_name || '').trim() || 'User';
    const email = (body.email || '').trim() || null;

    console.log('📝 Parsed data:', { clerkId, email, fullName, role: incomingRole });

    const { data: existing, error: fetchError } = await supabase
      .from('profiles')
      .select('*')
      .eq('clerk_id', clerkId)
      .maybeSingle();

    if (fetchError) {
      console.error('❌ Error checking existing profile:', fetchError);
      return res.status(500).json({ error: 'Failed to fetch profile' });
    }

    if (existing) {
      console.log('✅ Profile exists, updating:', existing.id);
      const { data: updated, error: updateError } = await supabase
        .from('profiles')
        .update({
          email: email ?? existing.email,
          full_name: fullName || existing.full_name,
        })
        .eq('clerk_id', clerkId)
        .select('*')
        .single();

      if (updateError) {
        console.error('❌ Error updating existing profile:', updateError);
        return res.status(500).json({ error: 'Failed to update profile' });
      }

      console.log('✅ Profile updated successfully');
      return res.json(updated);
    }

    console.log('🆕 Creating new profile for:', clerkId);
    const { data: created, error: createError } = await supabase
      .from('profiles')
      .insert({
        clerk_id: clerkId,
        email,
        full_name: fullName,
        role: incomingRole,
      })
      .select('*')
      .single();

    if (createError) {
      console.error('❌ Error creating profile:', createError);
      return res.status(500).json({ error: 'Failed to create profile' });
    }

    console.log('✅ Profile created successfully:', created.id);
    return res.status(201).json(created);
  } catch (error) {
    console.error('❌ Sync profile error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * DEPRECATED: Signup is now handled by Clerk
 * This endpoint is kept for backward compatibility
 */
router.post('/signup', async (req: Request, res: Response) => {
  return res.status(410).json({
    error: 'Signup is now handled by Clerk',
    message: 'Use /sign-up on the frontend',
  });
});

/**
 * DEPRECATED: Login is now handled by Clerk
 * This endpoint is kept for backward compatibility
 */
router.post('/login', async (req: Request, res: Response) => {
  return res.status(410).json({
    error: 'Login is now handled by Clerk',
    message: 'Use /sign-in on the frontend',
  });
});

/**
 * GET /auth/health
 * Check if auth service is running (no auth required)
 */
router.get('/health', (req: Request, res: Response) => {
  return res.json({
    status: 'ok',
    auth: 'Clerk',
    database: 'Supabase (profiles only)',
  });
});

export default router;
