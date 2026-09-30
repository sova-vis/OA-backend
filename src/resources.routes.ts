/**
 * Teacher notes & resources (spec §5.9). Mounted at /resources.
 *
 * A teacher attaches a small file (base64, within the 2 MB request limit) or an
 * external link to one of their classes, tags it to a topic, and publishes it.
 * Enrolled students read the published ones. Total per-teacher storage is capped
 * by their school's storage_cap_mb (a default applies to standalone teachers).
 *
 * Access is checked in-handler: teachers via resolveClassAccess, students via an
 * active class enrolment.
 */
import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import { resolveClassAccess } from './lib/portalAccess';

const router = Router();
router.use(clerkAuth);

// Never select `data` (the base64 blob) in a list — it's served on demand.
const LIST_COLS = 'id, class_id, title, topic, kind, url, mime, size_bytes, published, created_at, updated_at';
const DEFAULT_CAP_MB = 50;

async function teacherCanManage(classId: string, clerkId: string): Promise<boolean> {
  const access = await resolveClassAccess(classId, clerkId);
  return !!access && (access.isOwner || !!access.isCoTeacher);
}

async function studentEnrolled(classId: string, clerkId: string): Promise<boolean> {
  const { data } = await supabase
    .from('class_enrollments').select('id')
    .eq('class_id', classId).eq('student_clerk_id', clerkId).eq('status', 'active').maybeSingle();
  return !!data;
}

async function storageCapBytes(clerkId: string): Promise<number> {
  const { data: prof } = await supabase.from('profiles').select('school_id').eq('clerk_id', clerkId).maybeSingle();
  const schoolId = (prof as { school_id?: string | null } | null)?.school_id;
  let capMb = DEFAULT_CAP_MB;
  if (schoolId) {
    const { data: lim } = await supabase.from('school_limits').select('storage_cap_mb').eq('school_id', schoolId).maybeSingle();
    const v = (lim as { storage_cap_mb?: number | null } | null)?.storage_cap_mb;
    if (v && v > 0) capMb = v;
  }
  return capMb * 1024 * 1024;
}

async function usedBytes(clerkId: string): Promise<number> {
  const { data } = await supabase.from('teacher_resources').select('size_bytes').eq('owner_clerk_id', clerkId);
  return ((data ?? []) as { size_bytes: number }[]).reduce((s, r) => s + (Number(r.size_bytes) || 0), 0);
}

// GET /resources?class_id= — the teacher's resources for a class + storage usage.
router.get('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const classId = String(req.query.class_id ?? '');
    if (!classId) return res.status(400).json({ error: 'class_id is required' });
    if (!(await teacherCanManage(classId, req.auth!.clerkId))) return res.status(403).json({ error: 'No access to this class' });
    const [{ data }, cap, used] = await Promise.all([
      supabase.from('teacher_resources').select(LIST_COLS).eq('class_id', classId).order('created_at', { ascending: false }),
      storageCapBytes(req.auth!.clerkId), usedBytes(req.auth!.clerkId),
    ]);
    return res.json({ resources: data ?? [], storage: { used_bytes: used, cap_bytes: cap } });
  } catch (err) {
    console.error('GET /resources', err);
    return res.status(500).json({ error: 'Failed to load resources' });
  }
});

// GET /resources/published?class_id= — published resources for an enrolled student.
router.get('/published', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const classId = String(req.query.class_id ?? '');
    if (!classId) return res.status(400).json({ error: 'class_id is required' });
    if (!(await studentEnrolled(classId, req.auth!.clerkId))) return res.status(403).json({ error: 'Not enrolled in this class' });
    const { data } = await supabase
      .from('teacher_resources').select(LIST_COLS)
      .eq('class_id', classId).eq('published', true).order('created_at', { ascending: false });
    return res.json({ resources: data ?? [] });
  } catch (err) {
    console.error('GET /resources/published', err);
    return res.status(500).json({ error: 'Failed to load resources' });
  }
});

// POST /resources — add a note/resource (file or link), enforcing the storage cap.
router.post('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const b = req.body || {};
    const classId = String(b.class_id ?? '');
    const title = String(b.title ?? '').trim().slice(0, 200);
    const kind = b.kind === 'link' ? 'link' : 'file';
    if (!classId || !title) return res.status(400).json({ error: 'class_id and title are required' });
    if (!(await teacherCanManage(classId, clerkId))) return res.status(403).json({ error: 'No access to this class' });

    let url: string | null = null, data: string | null = null, mime: string | null = null, size = 0;
    if (kind === 'link') {
      url = String(b.url ?? '').trim().slice(0, 2000);
      if (!/^https?:\/\//i.test(url)) return res.status(400).json({ error: 'A valid http(s) link is required' });
    } else {
      data = String(b.data ?? '');
      if (!data.startsWith('data:')) return res.status(400).json({ error: 'A file is required' });
      mime = (data.slice(5, data.indexOf(';')) || 'application/octet-stream').slice(0, 100);
      const b64 = data.slice(data.indexOf(',') + 1);
      size = Math.floor((b64.length * 3) / 4);
      const [cap, used] = await Promise.all([storageCapBytes(clerkId), usedBytes(clerkId)]);
      if (used + size > cap) {
        return res.status(413).json({ error: `Storage cap reached (${Math.round(cap / 1048576)} MB). Delete a resource, or add it as a link instead.` });
      }
    }

    const row = {
      owner_clerk_id: clerkId, class_id: classId, title,
      topic: b.topic ? String(b.topic).trim().slice(0, 120) : null,
      kind, url, data, mime, size_bytes: size,
      published: b.published === false ? false : true,
    };
    const { data: created, error } = await supabase.from('teacher_resources').insert(row).select(LIST_COLS).single();
    if (error) throw error;
    return res.status(201).json({ resource: created });
  } catch (err) {
    console.error('POST /resources', err);
    return res.status(500).json({ error: 'Failed to add resource' });
  }
});

// PATCH /resources/:id — rename / retag / publish toggle.
router.patch('/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { data: existing } = await supabase.from('teacher_resources').select('id, class_id').eq('id', req.params.id).maybeSingle();
    const ex = existing as { class_id?: string } | null;
    if (!ex?.class_id) return res.status(404).json({ error: 'Resource not found' });
    if (!(await teacherCanManage(ex.class_id, req.auth!.clerkId))) return res.status(403).json({ error: 'No access' });
    const patch: Record<string, unknown> = { updated_at: new Date().toISOString() };
    if (typeof req.body?.title === 'string') patch.title = req.body.title.trim().slice(0, 200);
    if (typeof req.body?.topic === 'string') patch.topic = req.body.topic.trim().slice(0, 120) || null;
    if (typeof req.body?.published === 'boolean') patch.published = req.body.published;
    const { data: updated, error } = await supabase.from('teacher_resources').update(patch).eq('id', req.params.id).select(LIST_COLS).single();
    if (error) throw error;
    return res.json({ resource: updated });
  } catch (err) {
    console.error('PATCH /resources/:id', err);
    return res.status(500).json({ error: 'Failed to update resource' });
  }
});

// DELETE /resources/:id
router.delete('/:id', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { data: existing } = await supabase.from('teacher_resources').select('id, class_id').eq('id', req.params.id).maybeSingle();
    const ex = existing as { class_id?: string } | null;
    if (!ex?.class_id) return res.status(404).json({ error: 'Resource not found' });
    if (!(await teacherCanManage(ex.class_id, req.auth!.clerkId))) return res.status(403).json({ error: 'No access' });
    const { error } = await supabase.from('teacher_resources').delete().eq('id', req.params.id);
    if (error) throw error;
    return res.json({ ok: true });
  } catch (err) {
    console.error('DELETE /resources/:id', err);
    return res.status(500).json({ error: 'Failed to delete resource' });
  }
});

// GET /resources/:id/file — stream a stored file (a teacher who manages it, or an
// enrolled student if it's published).
router.get('/:id/file', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const { data } = await supabase.from('teacher_resources')
      .select('class_id, kind, data, mime, published').eq('id', req.params.id).maybeSingle();
    const r = data as { class_id: string; kind: string; data: string | null; mime: string | null; published: boolean } | null;
    if (!r || r.kind !== 'file' || !r.data) return res.status(404).json({ error: 'File not found' });
    const clerkId = req.auth!.clerkId;
    const canView = (await teacherCanManage(r.class_id, clerkId)) || (r.published && (await studentEnrolled(r.class_id, clerkId)));
    if (!canView) return res.status(403).json({ error: 'No access' });
    const comma = r.data.indexOf(',');
    const buf = Buffer.from(comma >= 0 ? r.data.slice(comma + 1) : r.data, 'base64');
    res.setHeader('Content-Type', r.mime || 'application/octet-stream');
    res.setHeader('Cache-Control', 'private, max-age=300');
    return res.send(buf);
  } catch (err) {
    console.error('GET /resources/:id/file', err);
    return res.status(500).json({ error: 'Failed to load file' });
  }
});

export default router;
