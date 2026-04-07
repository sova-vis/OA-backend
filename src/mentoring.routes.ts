import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth, requireRole } from './lib/clerkAuth';
import { supabase } from './lib/supabase';

const router = Router();

type Role = 'student' | 'teacher' | 'admin';

interface ProfileRow {
  clerk_id: string;
  full_name: string | null;
  email: string | null;
  role: Role;
}

function isValidUrl(value: string | null | undefined): boolean {
  if (!value) return false;
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'https:' || parsed.protocol === 'http:';
  } catch {
    return false;
  }
}

function parseUtcDate(value: unknown): Date | null {
  if (typeof value !== 'string') return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date;
}

async function getProfileByClerkId(clerkId: string): Promise<ProfileRow | null> {
  const { data, error } = await supabase
    .from('profiles')
    .select('clerk_id, full_name, email, role')
    .eq('clerk_id', clerkId)
    .maybeSingle();

  if (error) {
    throw error;
  }

  return (data as ProfileRow | null) ?? null;
}

async function ensureConversation(studentClerkId: string, teacherClerkId: string) {
  const { data: existing, error: existingError } = await supabase
    .from('mentoring_conversations')
    .select('id, student_clerk_id, teacher_clerk_id, updated_at')
    .eq('student_clerk_id', studentClerkId)
    .eq('teacher_clerk_id', teacherClerkId)
    .maybeSingle();

  if (existingError) {
    throw existingError;
  }

  if (existing?.id) {
    return existing;
  }

  const { data: created, error: createError } = await supabase
    .from('mentoring_conversations')
    .insert({
      student_clerk_id: studentClerkId,
      teacher_clerk_id: teacherClerkId,
    })
    .select('id, student_clerk_id, teacher_clerk_id, updated_at')
    .single();

  if (createError) {
    throw createError;
  }

  return created;
}

async function hasTeacherConflict(teacherClerkId: string, start: Date, end: Date, ignoreMeetingId?: string) {
  let query = supabase
    .from('mentoring_meetings')
    .select('id, start_time, end_time, status')
    .eq('teacher_clerk_id', teacherClerkId)
    .in('status', ['accepted', 'scheduled'])
    .lt('start_time', end.toISOString())
    .gt('end_time', start.toISOString());

  if (ignoreMeetingId) {
    query = query.neq('id', ignoreMeetingId);
  }

  const { data, error } = await query;

  if (error) {
    throw error;
  }

  return Array.isArray(data) && data.length > 0;
}

router.get('/teachers', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const callerId = req.auth?.clerkId;
    if (!callerId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const caller = await getProfileByClerkId(callerId);
    if (!caller) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    if (caller.role !== 'student' && caller.role !== 'admin') {
      return res.status(403).json({ error: 'Only students or admins can fetch teachers' });
    }

    const { data: teachers, error: teacherError } = await supabase
      .from('profiles')
      .select('clerk_id, full_name, email, role')
      .eq('role', 'teacher')
      .order('full_name', { ascending: true });

    if (teacherError) {
      throw teacherError;
    }

    const teacherIds = (teachers ?? []).map((teacher) => teacher.clerk_id);

    let teacherProfiles: Array<Record<string, unknown>> = [];
    if (teacherIds.length > 0) {
      const { data, error } = await supabase
        .from('teacher_profiles')
        .select('clerk_id, headline, bio, subjects, availability, meeting_provider, is_active, updated_at')
        .in('clerk_id', teacherIds);

      if (error) {
        throw error;
      }

      teacherProfiles = data ?? [];
    }

    const byTeacherId = new Map<string, Record<string, unknown>>();
    for (const row of teacherProfiles) {
      const key = typeof row.clerk_id === 'string' ? row.clerk_id : '';
      if (key) byTeacherId.set(key, row);
    }

    return res.json({
      teachers: (teachers ?? []).map((teacher) => {
        const details = byTeacherId.get(teacher.clerk_id);
        return {
          clerk_id: teacher.clerk_id,
          full_name: teacher.full_name,
          email: teacher.email,
          role: teacher.role,
          headline: details?.headline ?? null,
          bio: details?.bio ?? null,
          subjects: Array.isArray(details?.subjects) ? details?.subjects : [],
          availability: Array.isArray(details?.availability) ? details?.availability : [],
          meeting_provider: typeof details?.meeting_provider === 'string' ? details.meeting_provider : 'google_meet',
          is_active: typeof details?.is_active === 'boolean' ? details.is_active : true,
        };
      }),
    });
  } catch (error: any) {
    console.error('Failed to fetch teachers:', error);
    return res.status(500).json({ error: error.message || 'Failed to fetch teachers' });
  }
});

router.get('/students', clerkAuth, requireRole('admin'), async (_req: AuthenticatedRequest, res: Response) => {
  try {
    const { data, error } = await supabase
      .from('profiles')
      .select('clerk_id, full_name, email, role, selected_subjects, created_at')
      .eq('role', 'student')
      .order('created_at', { ascending: false });

    if (error) {
      throw error;
    }

    return res.json({ students: data ?? [] });
  } catch (error: any) {
    console.error('Failed to fetch students:', error);
    return res.status(500).json({ error: error.message || 'Failed to fetch students' });
  }
});

router.post('/meetings/request', clerkAuth, requireRole('student'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const studentClerkId = req.auth?.clerkId;
    if (!studentClerkId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const body = (req.body ?? {}) as {
      teacher_clerk_id?: string;
      agenda?: string;
      note_from_student?: string;
      preferred_start_time?: string;
      preferred_end_time?: string;
    };

    if (!body.teacher_clerk_id || !body.agenda?.trim()) {
      return res.status(400).json({ error: 'teacher_clerk_id and agenda are required' });
    }

    const teacher = await getProfileByClerkId(body.teacher_clerk_id);
    if (!teacher || teacher.role !== 'teacher') {
      return res.status(400).json({ error: 'Teacher not found' });
    }

    const preferredStart = parseUtcDate(body.preferred_start_time);
    const preferredEnd = parseUtcDate(body.preferred_end_time);

    if ((preferredStart && !preferredEnd) || (!preferredStart && preferredEnd)) {
      return res.status(400).json({ error: 'Both preferred_start_time and preferred_end_time are required together' });
    }

    if (preferredStart && preferredEnd && preferredStart >= preferredEnd) {
      return res.status(400).json({ error: 'preferred_end_time must be after preferred_start_time' });
    }

    const meetingPayload: Record<string, unknown> = {
      student_clerk_id: studentClerkId,
      teacher_clerk_id: body.teacher_clerk_id,
      agenda: body.agenda.trim(),
      note_from_student: body.note_from_student?.trim() || null,
      status: 'pending',
    };

    if (preferredStart && preferredEnd) {
      meetingPayload.start_time = preferredStart.toISOString();
      meetingPayload.end_time = preferredEnd.toISOString();
    }

    const { data, error } = await supabase
      .from('mentoring_meetings')
      .insert(meetingPayload)
      .select('*')
      .single();

    if (error) {
      throw error;
    }

    await ensureConversation(studentClerkId, body.teacher_clerk_id);

    return res.status(201).json({ meeting: data });
  } catch (error: any) {
    console.error('Failed to request meeting:', error);
    return res.status(500).json({ error: error.message || 'Failed to request meeting' });
  }
});

router.get('/meetings', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const profile = await getProfileByClerkId(clerkId);
    if (!profile) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    const scope = (req.query.scope as string | undefined) || 'mine';

    let query = supabase
      .from('mentoring_meetings')
      .select('*')
      .order('requested_at', { ascending: false });

    if (profile.role === 'teacher') {
      query = scope === 'requested' ? query.eq('student_clerk_id', clerkId) : query.eq('teacher_clerk_id', clerkId);
    } else if (profile.role === 'student') {
      query = query.eq('student_clerk_id', clerkId);
    }

    if (profile.role === 'admin' && scope === 'mine') {
      // Admin can inspect all meeting records when scope=mine.
    }

    const { data: meetings, error } = await query;

    if (error) {
      throw error;
    }

    const participantIds = new Set<string>();
    for (const meeting of meetings ?? []) {
      if (meeting.student_clerk_id) participantIds.add(meeting.student_clerk_id);
      if (meeting.teacher_clerk_id) participantIds.add(meeting.teacher_clerk_id);
    }

    let participantsById = new Map<string, ProfileRow>();
    if (participantIds.size > 0) {
      const { data: participants, error: participantError } = await supabase
        .from('profiles')
        .select('clerk_id, full_name, email, role')
        .in('clerk_id', Array.from(participantIds));

      if (participantError) {
        throw participantError;
      }

      participantsById = new Map((participants ?? []).map((row) => [row.clerk_id, row as ProfileRow]));
    }

    return res.json({
      meetings: (meetings ?? []).map((meeting) => ({
        ...meeting,
        student_profile: participantsById.get(meeting.student_clerk_id) ?? null,
        teacher_profile: participantsById.get(meeting.teacher_clerk_id) ?? null,
      })),
    });
  } catch (error: any) {
    console.error('Failed to fetch meetings:', error);
    return res.status(500).json({ error: error.message || 'Failed to fetch meetings' });
  }
});

router.patch('/meetings/:meetingId', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const callerId = req.auth?.clerkId;
    if (!callerId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const caller = await getProfileByClerkId(callerId);
    if (!caller) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    const meetingId = req.params.meetingId;

    const { data: existing, error: existingError } = await supabase
      .from('mentoring_meetings')
      .select('*')
      .eq('id', meetingId)
      .maybeSingle();

    if (existingError) {
      throw existingError;
    }

    if (!existing) {
      return res.status(404).json({ error: 'Meeting not found' });
    }

    const isTeacherParticipant = caller.role === 'teacher' && existing.teacher_clerk_id === callerId;
    const isStudentParticipant = caller.role === 'student' && existing.student_clerk_id === callerId;
    const isAdmin = caller.role === 'admin';

    if (!isTeacherParticipant && !isStudentParticipant && !isAdmin) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    const body = (req.body ?? {}) as {
      status?: string;
      start_time?: string;
      end_time?: string;
      meeting_link?: string;
      provider?: string;
      teacher_notes?: string;
      note_from_student?: string;
    };

    const updatePayload: Record<string, unknown> = {};

    if (typeof body.status === 'string') {
      const allowedStatuses = ['pending', 'accepted', 'scheduled', 'completed', 'cancelled', 'declined'];
      if (!allowedStatuses.includes(body.status)) {
        return res.status(400).json({ error: 'Invalid status value' });
      }

      if (body.status === 'scheduled') {
        const hasIncomingStartAndEnd = typeof body.start_time === 'string' && typeof body.end_time === 'string';
        const hasExistingStartAndEnd = Boolean(existing.start_time && existing.end_time);
        if (!hasIncomingStartAndEnd && !hasExistingStartAndEnd) {
          return res.status(400).json({ error: 'start_time and end_time are required before scheduling' });
        }
      }

      updatePayload.status = body.status;
    }

    if (typeof body.start_time === 'string' || typeof body.end_time === 'string') {
      const parsedStart = parseUtcDate(body.start_time);
      const parsedEnd = parseUtcDate(body.end_time);

      if (!parsedStart || !parsedEnd) {
        return res.status(400).json({ error: 'Valid start_time and end_time are required together' });
      }

      if (parsedStart >= parsedEnd) {
        return res.status(400).json({ error: 'end_time must be after start_time' });
      }

      const conflict = await hasTeacherConflict(existing.teacher_clerk_id, parsedStart, parsedEnd, meetingId);
      if (conflict) {
        return res.status(409).json({ error: 'Teacher already has a meeting in that time range' });
      }

      updatePayload.start_time = parsedStart.toISOString();
      updatePayload.end_time = parsedEnd.toISOString();
    }

    if (typeof body.meeting_link === 'string') {
      const trimmed = body.meeting_link.trim();
      if (trimmed && !isValidUrl(trimmed)) {
        return res.status(400).json({ error: 'meeting_link must be a valid URL' });
      }
      updatePayload.meeting_link = trimmed || null;
    }

    if (typeof body.provider === 'string') {
      updatePayload.provider = body.provider.trim() || null;
    }

    if (typeof body.teacher_notes === 'string' && (isTeacherParticipant || isAdmin)) {
      updatePayload.teacher_notes = body.teacher_notes.trim();
    }

    if (typeof body.note_from_student === 'string' && (isStudentParticipant || isAdmin)) {
      updatePayload.note_from_student = body.note_from_student.trim();
    }

    if (Object.keys(updatePayload).length === 0) {
      return res.status(400).json({ error: 'No valid fields provided for update' });
    }

    updatePayload.updated_at = new Date().toISOString();

    const { data: updated, error: updateError } = await supabase
      .from('mentoring_meetings')
      .update(updatePayload)
      .eq('id', meetingId)
      .select('*')
      .single();

    if (updateError) {
      throw updateError;
    }

    return res.json({ meeting: updated });
  } catch (error: any) {
    console.error('Failed to update meeting:', error);
    return res.status(500).json({ error: error.message || 'Failed to update meeting' });
  }
});

router.delete('/meetings/:meetingId', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const callerId = req.auth?.clerkId;
    if (!callerId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const caller = await getProfileByClerkId(callerId);
    if (!caller) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    const meetingId = req.params.meetingId;
    const { data: existing, error: existingError } = await supabase
      .from('mentoring_meetings')
      .select('id, student_clerk_id, teacher_clerk_id')
      .eq('id', meetingId)
      .maybeSingle();

    if (existingError) {
      throw existingError;
    }

    if (!existing) {
      return res.status(404).json({ error: 'Meeting not found' });
    }

    const isTeacherParticipant = caller.role === 'teacher' && existing.teacher_clerk_id === callerId;
    const isStudentParticipant = caller.role === 'student' && existing.student_clerk_id === callerId;
    const isAdmin = caller.role === 'admin';

    if (!isTeacherParticipant && !isStudentParticipant && !isAdmin) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    const { error: deleteError } = await supabase
      .from('mentoring_meetings')
      .delete()
      .eq('id', meetingId);

    if (deleteError) {
      throw deleteError;
    }

    return res.json({ success: true });
  } catch (error: any) {
    console.error('Failed to delete meeting:', error);
    return res.status(500).json({ error: error.message || 'Failed to delete meeting' });
  }
});

router.get('/conversations', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const profile = await getProfileByClerkId(clerkId);
    if (!profile) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    const column = profile.role === 'teacher' ? 'teacher_clerk_id' : 'student_clerk_id';

    const { data: conversations, error } = await supabase
      .from('mentoring_conversations')
      .select('id, student_clerk_id, teacher_clerk_id, created_at, updated_at')
      .eq(column, clerkId)
      .eq('is_deleted', false)
      .order('updated_at', { ascending: false });

    if (error) {
      throw error;
    }

    const participantIds = new Set<string>();
    for (const conversation of conversations ?? []) {
      participantIds.add(conversation.student_clerk_id);
      participantIds.add(conversation.teacher_clerk_id);
    }

    let profiles = new Map<string, ProfileRow>();
    if (participantIds.size > 0) {
      const { data: rows, error: profileError } = await supabase
        .from('profiles')
        .select('clerk_id, full_name, email, role')
        .in('clerk_id', Array.from(participantIds));

      if (profileError) {
        throw profileError;
      }

      profiles = new Map((rows ?? []).map((row) => [row.clerk_id, row as ProfileRow]));
    }

    return res.json({
      conversations: (conversations ?? []).map((conversation) => ({
        ...conversation,
        student_profile: profiles.get(conversation.student_clerk_id) ?? null,
        teacher_profile: profiles.get(conversation.teacher_clerk_id) ?? null,
      })),
    });
  } catch (error: any) {
    console.error('Failed to fetch conversations:', error);
    return res.status(500).json({ error: error.message || 'Failed to fetch conversations' });
  }
});

router.post('/conversations', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const callerId = req.auth?.clerkId;
    if (!callerId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const caller = await getProfileByClerkId(callerId);
    if (!caller) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    const body = (req.body ?? {}) as { partner_clerk_id?: string };
    if (!body.partner_clerk_id) {
      return res.status(400).json({ error: 'partner_clerk_id is required' });
    }

    const partner = await getProfileByClerkId(body.partner_clerk_id);
    if (!partner) {
      return res.status(404).json({ error: 'Partner not found' });
    }

    let studentClerkId = '';
    let teacherClerkId = '';

    if (caller.role === 'student' && partner.role === 'teacher') {
      studentClerkId = callerId;
      teacherClerkId = body.partner_clerk_id;
    } else if (caller.role === 'teacher' && partner.role === 'student') {
      studentClerkId = body.partner_clerk_id;
      teacherClerkId = callerId;
    } else {
      return res.status(400).json({ error: 'Conversations are only allowed between one student and one teacher' });
    }

    const conversation = await ensureConversation(studentClerkId, teacherClerkId);
    return res.status(201).json({ conversation });
  } catch (error: any) {
    console.error('Failed to create conversation:', error);
    return res.status(500).json({ error: error.message || 'Failed to create conversation' });
  }
});

router.get('/conversations/:conversationId/messages', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const callerId = req.auth?.clerkId;
    if (!callerId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const conversationId = req.params.conversationId;

    const { data: conversation, error: conversationError } = await supabase
      .from('mentoring_conversations')
      .select('id, student_clerk_id, teacher_clerk_id')
      .eq('id', conversationId)
      .eq('is_deleted', false)
      .maybeSingle();

    if (conversationError) {
      throw conversationError;
    }

    if (!conversation) {
      return res.status(404).json({ error: 'Conversation not found' });
    }

    if (conversation.student_clerk_id !== callerId && conversation.teacher_clerk_id !== callerId) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    const { data: messages, error: messageError } = await supabase
      .from('mentoring_messages')
      .select('*')
      .eq('conversation_id', conversationId)
      .eq('is_deleted', false)
      .order('created_at', { ascending: true });

    if (messageError) {
      throw messageError;
    }

    return res.json({ messages: messages ?? [] });
  } catch (error: any) {
    console.error('Failed to fetch messages:', error);
    return res.status(500).json({ error: error.message || 'Failed to fetch messages' });
  }
});

router.post('/conversations/:conversationId/messages', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const callerId = req.auth?.clerkId;
    if (!callerId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const caller = await getProfileByClerkId(callerId);
    if (!caller) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    if (caller.role !== 'student' && caller.role !== 'teacher') {
      return res.status(403).json({ error: 'Only students and teachers can send messages' });
    }

    const conversationId = req.params.conversationId;

    const { data: conversation, error: conversationError } = await supabase
      .from('mentoring_conversations')
      .select('id, student_clerk_id, teacher_clerk_id')
      .eq('id', conversationId)
      .eq('is_deleted', false)
      .maybeSingle();

    if (conversationError) {
      throw conversationError;
    }

    if (!conversation) {
      return res.status(404).json({ error: 'Conversation not found' });
    }

    if (conversation.student_clerk_id !== callerId && conversation.teacher_clerk_id !== callerId) {
      return res.status(403).json({ error: 'Forbidden' });
    }

    const body = (req.body ?? {}) as { body?: string };
    const text = body.body?.trim();
    if (!text) {
      return res.status(400).json({ error: 'Message body is required' });
    }

    const { data: created, error: createError } = await supabase
      .from('mentoring_messages')
      .insert({
        conversation_id: conversationId,
        sender_clerk_id: callerId,
        sender_role: caller.role,
        body: text,
      })
      .select('*')
      .single();

    if (createError) {
      throw createError;
    }

    await supabase
      .from('mentoring_conversations')
      .update({ updated_at: new Date().toISOString() })
      .eq('id', conversationId);

    return res.status(201).json({ message: created });
  } catch (error: any) {
    console.error('Failed to send message:', error);
    return res.status(500).json({ error: error.message || 'Failed to send message' });
  }
});

router.delete('/conversations/:conversationId/messages/:messageId', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const callerId = req.auth?.clerkId;
    if (!callerId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    const caller = await getProfileByClerkId(callerId);
    if (!caller) {
      return res.status(404).json({ error: 'Profile not found' });
    }

    const { conversationId, messageId } = req.params;

    const { data: conversation, error: conversationError } = await supabase
      .from('mentoring_conversations')
      .select('id, student_clerk_id, teacher_clerk_id')
      .eq('id', conversationId)
      .eq('is_deleted', false)
      .maybeSingle();

    if (conversationError) {
      throw conversationError;
    }

    if (!conversation) {
      return res.status(404).json({ error: 'Conversation not found' });
    }

    const isParticipant = conversation.student_clerk_id === callerId || conversation.teacher_clerk_id === callerId;
    if (!isParticipant && caller.role !== 'admin') {
      return res.status(403).json({ error: 'Forbidden' });
    }

    const { data: message, error: messageError } = await supabase
      .from('mentoring_messages')
      .select('*')
      .eq('id', messageId)
      .eq('conversation_id', conversationId)
      .maybeSingle();

    if (messageError) {
      throw messageError;
    }

    if (!message) {
      return res.status(404).json({ error: 'Message not found' });
    }

    if (caller.role !== 'admin' && message.sender_clerk_id !== callerId) {
      return res.status(403).json({ error: 'You can only delete your own messages' });
    }

    const { error: deleteError } = await supabase
      .from('mentoring_messages')
      .update({ is_deleted: true })
      .eq('id', messageId)
      .eq('conversation_id', conversationId);

    if (deleteError) {
      throw deleteError;
    }

    await supabase
      .from('mentoring_conversations')
      .update({ updated_at: new Date().toISOString() })
      .eq('id', conversationId);

    return res.json({ success: true });
  } catch (error: any) {
    console.error('Failed to delete message:', error);
    return res.status(500).json({ error: error.message || 'Failed to delete message' });
  }
});

router.patch('/teacher-profile', clerkAuth, requireRole('teacher'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const teacherClerkId = req.auth?.clerkId;
    if (!teacherClerkId) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

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
    if (Array.isArray(body.subjects)) payload.subjects = body.subjects.filter((item) => typeof item === 'string').map((item) => item.trim()).filter(Boolean);
    if (Array.isArray(body.availability)) payload.availability = body.availability;
    if (typeof body.meeting_provider === 'string') payload.meeting_provider = body.meeting_provider.trim() || 'google_meet';
    if (typeof body.is_active === 'boolean') payload.is_active = body.is_active;

    const { data, error } = await supabase
      .from('teacher_profiles')
      .upsert(payload, { onConflict: 'clerk_id' })
      .select('*')
      .single();

    if (error) {
      throw error;
    }

    return res.json({ teacher_profile: data });
  } catch (error: any) {
    console.error('Failed to update teacher profile:', error);
    return res.status(500).json({ error: error.message || 'Failed to update teacher profile' });
  }
});

export default router;
