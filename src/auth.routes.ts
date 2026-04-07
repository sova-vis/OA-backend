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

function extractClaimString(claims: Record<string, unknown> | undefined, keys: string[]): string | null {
  if (!claims) return null;

  for (const key of keys) {
    const value = claims[key];
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }

  return null;
}

function extractRoleFromClaims(claims: Record<string, unknown> | undefined): UserRole {
  const directRole = extractClaimString(claims, ['role']);
  if (directRole) return parseRole(directRole);

  const metadata = claims?.public_metadata as Record<string, unknown> | undefined;
  const metadataRole = metadata && typeof metadata.role === 'string' ? metadata.role : null;
  return parseRole(metadataRole);
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
      const claims = req.auth.claims;
      const claimEmail = extractClaimString(claims, ['email', 'email_address']);
      const claimFullName =
        extractClaimString(claims, ['name', 'full_name']) ||
        [extractClaimString(claims, ['first_name']), extractClaimString(claims, ['last_name'])]
          .filter(Boolean)
          .join(' ')
          .trim() ||
        'User';
      const claimRole = extractRoleFromClaims(claims);

      try {
        if (claimEmail) {
          // Allow pre-seeded admin records to bind to the current Clerk user on first login.
          const { data: byEmail, error: byEmailError } = await supabase
            .from('profiles')
            .select('*')
            .eq('email', claimEmail)
            .maybeSingle();

          if (byEmailError) {
            throw byEmailError;
          }

          if (byEmail) {
            const { data: rebound, error: reboundError } = await supabase
              .from('profiles')
              .update({
                clerk_id: req.auth.clerkId,
                full_name: byEmail.full_name || claimFullName,
                role: byEmail.role || claimRole,
              })
              .eq('email', claimEmail)
              .select('*')
              .single();

            if (reboundError) {
              throw reboundError;
            }

            return res.json(rebound);
          }
        }

        const { data: created, error: createError } = await supabase
          .from('profiles')
          .insert({
            clerk_id: req.auth.clerkId,
            email: claimEmail,
            full_name: claimFullName,
            role: claimRole,
          })
          .select('*')
          .single();

        if (createError) {
          throw createError;
        }

        return res.status(201).json(created);
      } catch (provisionError) {
        console.warn('Profile auto-provision did not complete in /auth/profile, falling back to sync-profile path:', provisionError);
        return res.status(404).json({ error: 'Profile not found' });
      }
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
          role: existing.role || incomingRole,
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

    if (email) {
      const { data: existingByEmail, error: emailLookupError } = await supabase
        .from('profiles')
        .select('*')
        .eq('email', email)
        .maybeSingle();

      if (emailLookupError) {
        console.error('❌ Error checking profile by email:', emailLookupError);
        return res.status(500).json({ error: 'Failed to fetch profile by email' });
      }

      if (existingByEmail) {
        console.log('🔗 Binding existing email profile to clerk_id:', existingByEmail.id);
        const { data: bound, error: bindError } = await supabase
          .from('profiles')
          .update({
            clerk_id: clerkId,
            full_name: fullName || existingByEmail.full_name,
            role: existingByEmail.role || incomingRole,
          })
          .eq('email', email)
          .select('*')
          .single();

        if (bindError) {
          console.error('❌ Error binding profile by email:', bindError);
          return res.status(500).json({ error: 'Failed to bind profile' });
        }

        console.log('✅ Profile bound by email successfully');
        return res.json(bound);
      }
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
