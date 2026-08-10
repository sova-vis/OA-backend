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

router.patch('/profile', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    if (!req.auth?.clerkId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const body = (req.body ?? {}) as { selected_subjects?: unknown };
    const payload: Record<string, unknown> = {};

    if (Array.isArray(body.selected_subjects)) {
      payload.selected_subjects = body.selected_subjects
        .filter((item) => typeof item === 'string')
        .map((item) => item.trim())
        .filter(Boolean)
        .slice(0, 30);
    }

    if (Object.keys(payload).length === 0) {
      return res.status(400).json({ error: 'No valid profile fields provided' });
    }

    const { data: updated, error } = await supabase
      .from('profiles')
      .update(payload)
      .eq('clerk_id', req.auth.clerkId)
      .select('*')
      .single();

    if (error) {
      throw error;
    }

    return res.json(updated);
  } catch (err) {
    console.error('Error updating profile:', err);
    return res.status(500).json({ error: 'Failed to update profile' });
  }
});

/**
 * POST /auth/select-role
 * One-time role choice at onboarding: student or teacher. Only permitted while
 * the profile has not completed onboarding — after that, role changes are
 * admin-only (§15.1). Never allows selecting 'admin'.
 */
router.post('/select-role', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const role = req.body?.role;
    if (role !== 'student' && role !== 'teacher') {
      return res.status(400).json({ error: "role must be 'student' or 'teacher'" });
    }

    const email = extractClaimString(req.auth?.claims, ['email', 'email_address', 'primary_email_address']);
    const fullName =
      extractClaimString(req.auth?.claims, ['name', 'full_name']) ||
      [extractClaimString(req.auth?.claims, ['first_name']), extractClaimString(req.auth?.claims, ['last_name'])]
        .filter(Boolean)
        .join(' ')
        .trim() ||
      'User';

    const { data: existing, error: fetchError } = await supabase.from('profiles').select('*').eq('clerk_id', clerkId).maybeSingle();
    if (fetchError) throw fetchError;

    // Guard against self-escalation: once onboarding is done, only an admin can
    // change a role.
    if (existing && existing.onboarding_complete) {
      return res.status(403).json({ error: 'Your role is already set. Contact an administrator to change it.' });
    }

    if (existing) {
      const { data: updated, error } = await supabase
        .from('profiles')
        .update({ role, onboarding_complete: true, full_name: existing.full_name || fullName })
        .eq('clerk_id', clerkId)
        .select('*')
        .single();
      if (error) throw error;
      return res.json(updated);
    }

    const { data: created, error } = await supabase
      .from('profiles')
      .insert({ clerk_id: clerkId, email, full_name: fullName, role, onboarding_complete: true })
      .select('*')
      .single();
    if (error) throw error;
    return res.status(201).json(created);
  } catch (err) {
    console.error('Select-role error:', err);
    return res.status(500).json({ error: 'Failed to set role' });
  }
});

/**
 * POST /auth/complete-onboarding
 * Save the onboarding survey (role + collected fields) in one call and mark
 * onboarding complete. One-time: only permitted while onboarding_complete=false,
 * so it cannot be used to self-escalate a role later (§15.1).
 */
router.post('/complete-onboarding', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const b = (req.body ?? {}) as Record<string, unknown>;
    const role = b.role === 'teacher' ? 'teacher' : b.role === 'student' ? 'student' : null;
    if (!role) return res.status(400).json({ error: "role must be 'student' or 'teacher'" });

    const email = extractClaimString(req.auth?.claims, ['email', 'email_address', 'primary_email_address']);
    const fullName = typeof b.full_name === 'string' && b.full_name.trim() ? b.full_name.trim() : 'User';

    const { data: existing } = await supabase.from('profiles').select('onboarding_complete').eq('clerk_id', clerkId).maybeSingle();
    if (existing && (existing as { onboarding_complete?: boolean }).onboarding_complete) {
      return res.status(403).json({ error: 'Onboarding already completed.' });
    }

    const common: Record<string, unknown> = {
      clerk_id: clerkId,
      email,
      full_name: fullName,
      role,
      onboarding_complete: true,
    };
    if (typeof b.photo_url === 'string' && b.photo_url) common.photo_url = b.photo_url;

    if (role === 'student') {
      if (typeof b.level === 'string') common.level = b.level;
      if (Array.isArray(b.selected_subjects)) common.selected_subjects = b.selected_subjects.filter((s) => typeof s === 'string').slice(0, 20);
      if (typeof b.exam_session === 'string') common.exam_session = b.exam_session;
      if (typeof b.target_grade === 'string') common.target_grade = b.target_grade;
      if (typeof b.study_days === 'string') common.study_days = b.study_days;
      if (b.subject_confidence && typeof b.subject_confidence === 'object') common.subject_confidence = b.subject_confidence;
    } else {
      if (Number.isFinite(Number(b.experience_years))) common.experience_years = Number(b.experience_years);
      if (typeof b.school_name === 'string') common.school_name = b.school_name.trim() || null;
      if (['private', 'public', 'none'].includes(b.school_type as string)) common.school_type = b.school_type;
      if (typeof b.level === 'string') common.level = b.level; // levels taught, optional
      if (Array.isArray(b.selected_subjects)) common.selected_subjects = b.selected_subjects.filter((s) => typeof s === 'string').slice(0, 20);
    }

    // Manual update-or-insert (avoids depending on an ON CONFLICT target, which
    // isn't reliably present on this pre-existing profiles table).
    const { data: existingRow } = await supabase.from('profiles').select('id').eq('clerk_id', clerkId).maybeSingle();
    const result = existingRow
      ? await supabase.from('profiles').update(common).eq('clerk_id', clerkId).select('*').single()
      : await supabase.from('profiles').insert(common).select('*').single();
    if (result.error) throw result.error;
    return res.json(result.data);
  } catch (err) {
    console.error('Complete-onboarding error:', err);
    return res.status(500).json({ error: 'Failed to complete onboarding' });
  }
});

/**
 * POST /auth/delete-account
 * Self-service account deletion. Requires the caller to re-type their own email.
 * Removes all their Propel data and the Clerk user. When a student deletes,
 * their teachers are notified that an enrolled student left.
 */
async function deleteClerkUser(clerkId: string): Promise<void> {
  const key = process.env.CLERK_SECRET_KEY;
  if (!key) return;
  try {
    await fetch(`https://api.clerk.com/v1/users/${clerkId}`, { method: 'DELETE', headers: { Authorization: `Bearer ${key}` } });
  } catch {
    /* best-effort */
  }
}

router.post('/delete-account', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    // Resolve the account email from the stored profile first — Clerk's default
    // session token often omits the email claim, so relying on it alone made the
    // check always fail. Accept the DB email or (as a fallback) the JWT claim.
    const { data: profile } = await supabase.from('profiles').select('role, full_name, email').eq('clerk_id', clerkId).maybeSingle();
    const claimEmail = (extractClaimString(req.auth?.claims, ['email', 'email_address', 'primary_email_address']) || '').toLowerCase();
    const profileEmail = ((profile as { email?: string } | null)?.email || '').toLowerCase();
    const accountEmail = profileEmail || claimEmail;
    const typed = String(req.body?.email_confirmation ?? '').trim().toLowerCase();
    if (!accountEmail || typed !== accountEmail) {
      return res.status(400).json({ error: 'The email you typed does not match your account email.' });
    }

    const role = (profile as { role?: string } | null)?.role ?? 'student';
    const fullName = (profile as { full_name?: string } | null)?.full_name || accountEmail.split('@')[0];

    if (role === 'teacher') {
      // Classes cascade to co-teachers, enrolments, assignments, submissions.
      await supabase.from('classes').delete().eq('owner_clerk_id', clerkId);
      await supabase.from('class_co_teachers').delete().eq('teacher_clerk_id', clerkId);
      await supabase.from('custom_questions').delete().eq('owner_clerk_id', clerkId);
      await supabase.from('comment_bank').delete().eq('owner_clerk_id', clerkId);
      await supabase.from('scope_grants').delete().eq('user_clerk_id', clerkId);
    } else {
      // Notify each class owner that an enrolled student deleted their account.
      const { data: enr } = await supabase.from('class_enrollments').select('class_id').eq('student_clerk_id', clerkId).in('status', ['active', 'pending']);
      const classIds = ((enr ?? []) as { class_id: string }[]).map((e) => e.class_id);
      if (classIds.length > 0) {
        const { data: classes } = await supabase.from('classes').select('id, name, owner_clerk_id').in('id', classIds);
        for (const c of (classes ?? []) as { id: string; name: string; owner_clerk_id: string }[]) {
          await supabase.from('notifications').insert({
            recipient_clerk_id: c.owner_clerk_id,
            type: 'account_deleted',
            class_id: c.id,
            body: `${fullName}, enrolled in ${c.name}, has deleted their Propel account.`,
          });
        }
      }
      // Submissions cascade to answers + marks.
      await supabase.from('submissions').delete().eq('student_clerk_id', clerkId);
      await supabase.from('class_enrollments').delete().eq('student_clerk_id', clerkId);
    }

    await supabase.from('notifications').delete().eq('recipient_clerk_id', clerkId);
    await supabase.from('profiles').delete().eq('clerk_id', clerkId);
    await deleteClerkUser(clerkId);

    return res.json({ ok: true });
  } catch (err) {
    console.error('Delete-account error:', err);
    return res.status(500).json({ error: 'Failed to delete account' });
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

    // SECURITY (F-01, F-02): never trust the client for identity or privilege.
    // Bind by the Clerk-VERIFIED email claim (not body.email), and never accept a
    // role from the request — new profiles are always students; promotion is
    // admin-only via the requireRole('admin')-guarded admin routes.
    const fullName = (body.full_name || '').trim() || 'User';
    const email = extractClaimString(req.auth?.claims, ['email', 'email_address', 'primary_email_address']);

    console.log('📝 Parsed data:', { clerkId, email, fullName });

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
          role: existing.role || 'student',
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
            role: existingByEmail.role || 'student',
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
        role: 'student',
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
