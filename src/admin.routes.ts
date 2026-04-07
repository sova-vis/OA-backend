import { Router, Request, Response } from 'express';
import { createTeacherAccount } from './services/adminService';
import { AuthenticatedRequest, clerkAuth, requireRole } from './lib/clerkAuth';
import { supabase } from './lib/supabase';

const router = Router();

// Deprecated legacy endpoint kept for backward compatibility
router.post('/login', (req: Request, res: Response) => {
  return res.status(410).json({
    error: 'Admin login is now handled by Clerk',
    message: 'Use /sign-in and ensure your profile role is admin',
  });
});

// Add teacher using Supabase Admin
router.post('/add-teacher', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  const { email, password, name } = req.body;
  if (!email || !password || !name) return res.status(400).json({ error: 'Missing fields' });

  try {
    const user = await createTeacherAccount(email, password, name);
    return res.json({ message: 'Teacher created successfully', userId: user.id });
  } catch (error: any) {
    console.error("Create teacher error:", error);
    const statusCode = typeof error?.statusCode === 'number' ? error.statusCode : 500;
    return res.status(statusCode).json({ error: error.message || 'Failed to create teacher' });
  }
});

router.get('/teachers', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const { data: teachers, error: teacherError } = await supabase
      .from('profiles')
      .select('clerk_id, full_name, email, role')
      .eq('role', 'teacher')
      .order('full_name', { ascending: true });

    if (teacherError) {
      throw teacherError;
    }

    const teacherIds = (teachers ?? []).map((teacher) => teacher.clerk_id);

    let details: Array<Record<string, unknown>> = [];
    if (teacherIds.length > 0) {
      const { data, error } = await supabase
        .from('teacher_profiles')
        .select('clerk_id, headline, bio, subjects, availability, meeting_provider, is_active, updated_at')
        .in('clerk_id', teacherIds);

      if (error) {
        throw error;
      }

      details = data ?? [];
    }

    const detailsById = new Map<string, Record<string, unknown>>();
    for (const row of details) {
      const key = typeof row.clerk_id === 'string' ? row.clerk_id : '';
      if (key) {
        detailsById.set(key, row);
      }
    }

    return res.json({
      teachers: (teachers ?? []).map((teacher) => {
        const detail = detailsById.get(teacher.clerk_id);
        return {
          ...teacher,
          headline: detail?.headline ?? null,
          bio: detail?.bio ?? null,
          subjects: Array.isArray(detail?.subjects) ? detail.subjects : [],
          availability: Array.isArray(detail?.availability) ? detail.availability : [],
          meeting_provider: typeof detail?.meeting_provider === 'string' ? detail.meeting_provider : 'google_meet',
          is_active: typeof detail?.is_active === 'boolean' ? detail.is_active : true,
        };
      }),
    });
  } catch (error: any) {
    console.error('Failed to list teachers:', error);
    return res.status(500).json({ error: error.message || 'Failed to list teachers' });
  }
});

router.get('/users', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const { data, error } = await supabase
      .from('profiles')
      .select('*');

    if (error) {
      throw error;
    }

    return res.json({ users: data ?? [] });
  } catch (error: any) {
    console.error('Failed to list users:', error);
    return res.status(500).json({ error: error.message || 'Failed to list users' });
  }
});

router.get('/meetings', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const { data: meetings, error: meetingsError } = await supabase
      .from('mentoring_meetings')
      .select('*')
      .order('requested_at', { ascending: false });

    if (meetingsError) {
      throw meetingsError;
    }

    const participantIds = new Set<string>();
    for (const meeting of meetings ?? []) {
      if (meeting.student_clerk_id) participantIds.add(meeting.student_clerk_id);
      if (meeting.teacher_clerk_id) participantIds.add(meeting.teacher_clerk_id);
    }

    let participantsById = new Map<string, { clerk_id: string; full_name: string | null; email: string | null; role: string | null }>();
    if (participantIds.size > 0) {
      const { data: participants, error: participantsError } = await supabase
        .from('profiles')
        .select('clerk_id, full_name, email, role')
        .in('clerk_id', Array.from(participantIds));

      if (participantsError) {
        throw participantsError;
      }

      participantsById = new Map(
        (participants ?? []).map((row) => [
          row.clerk_id,
          {
            clerk_id: row.clerk_id,
            full_name: row.full_name ?? null,
            email: row.email ?? null,
            role: row.role ?? null,
          },
        ])
      );
    }

    return res.json({
      meetings: (meetings ?? []).map((meeting) => ({
        ...meeting,
        student_profile: participantsById.get(meeting.student_clerk_id) ?? null,
        teacher_profile: participantsById.get(meeting.teacher_clerk_id) ?? null,
      })),
    });
  } catch (error: any) {
    console.error('Failed to list meetings:', error);
    return res.status(500).json({ error: error.message || 'Failed to list meetings' });
  }
});

router.patch('/teacher-profile/:clerkId', clerkAuth, requireRole('admin'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const teacherClerkId = req.params.clerkId;
    const body = (req.body ?? {}) as {
      headline?: string;
      bio?: string;
      subjects?: string[];
      availability?: Array<{ day: string; start: string; end: string }>;
      meeting_provider?: string;
      is_active?: boolean;
    };

    const payload: Record<string, unknown> = {
      clerk_id: teacherClerkId,
      updated_at: new Date().toISOString(),
    };

    if (typeof body.headline === 'string') payload.headline = body.headline.trim();
    if (typeof body.bio === 'string') payload.bio = body.bio.trim();
    if (Array.isArray(body.subjects)) {
      payload.subjects = body.subjects
        .filter((subject) => typeof subject === 'string')
        .map((subject) => subject.trim())
        .filter(Boolean);
    }
    if (Array.isArray(body.availability)) payload.availability = body.availability;
    if (typeof body.meeting_provider === 'string') payload.meeting_provider = body.meeting_provider.trim() || 'google_meet';
    if (typeof body.is_active === 'boolean') payload.is_active = body.is_active;

    const { data: existing, error: existingError } = await supabase
      .from('teacher_profiles')
      .select('*')
      .eq('clerk_id', teacherClerkId)
      .maybeSingle();

    if (existingError) {
      throw existingError;
    }

    if (existing) {
      const { data: updated, error: updateError } = await supabase
        .from('teacher_profiles')
        .update(payload)
        .eq('clerk_id', teacherClerkId)
        .select('*')
        .single();

      if (updateError) {
        throw updateError;
      }

      return res.json({ teacher_profile: updated });
    }

    const { data: created, error: createError } = await supabase
      .from('teacher_profiles')
      .insert(payload)
      .select('*')
      .single();

    if (createError) {
      throw createError;
    }

    return res.json({ teacher_profile: created });
  } catch (error: any) {
    console.error('Failed to update teacher profile:', error);
    return res.status(500).json({ error: error.message || 'Failed to update teacher profile' });
  }
});

// Update user profile (admin only)
// Update user profile (admin only) - Placeholder for future implementation using Supabase
router.put('/update-profile/:id', clerkAuth, requireRole('admin'), (req: Request, res: Response) => {
  return res.status(501).json({ error: 'Not implemented yet' });
});

export default router;
