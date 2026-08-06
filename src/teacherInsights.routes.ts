import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import { resolveClassAccess } from './lib/portalAccess';

/**
 * Teacher Portal — dashboard (spec feature group 2) and insights (§12).
 * Aggregates over submissions/marks. Mounted at /teacher-insights.
 */

const router = Router();
router.use(clerkAuth);

interface Criterion { marks_available: number; marks_awarded: number; awarded: boolean; }

function markScore(m: Record<string, unknown>): { earned: number; available: number } {
  const criteria = (m.final_criteria ?? m.ai_criteria ?? []) as Criterion[];
  const earned = m.final_score != null ? Number(m.final_score) : m.ai_score != null ? Number(m.ai_score) : criteria.reduce((s, c) => s + (Number(c.marks_awarded) || 0), 0);
  const available = Number(m.ai_marks_available ?? criteria.reduce((s, c) => s + Number(c.marks_available || 0), 0));
  return { earned, available };
}

const REVIEWED = ['approved', 'overridden', 'auto_approved'];

async function accessibleClasses(clerkId: string) {
  const [{ data: owned }, { data: co }] = await Promise.all([
    supabase.from('classes').select('id, name, subject, syllabus_code, level, archived_at').eq('owner_clerk_id', clerkId),
    supabase.from('class_co_teachers').select('class_id').eq('teacher_clerk_id', clerkId),
  ]);
  const coIds = ((co ?? []) as { class_id: string }[]).map((c) => c.class_id);
  let coClasses: Record<string, unknown>[] = [];
  if (coIds.length > 0) {
    const { data } = await supabase.from('classes').select('id, name, subject, syllabus_code, level, archived_at').in('id', coIds);
    coClasses = (data as Record<string, unknown>[]) ?? [];
  }
  const seen = new Set<string>();
  return [...((owned as Record<string, unknown>[]) ?? []), ...coClasses].filter((c) => {
    if (c.archived_at || seen.has(c.id as string)) return false;
    seen.add(c.id as string);
    return true;
  });
}

async function getThreshold(clerkId: string): Promise<number> {
  const { data } = await supabase.from('profiles').select('auto_approve_threshold').eq('clerk_id', clerkId).maybeSingle();
  return Number((data as { auto_approve_threshold?: number } | null)?.auto_approve_threshold ?? 0.85);
}

// ---------------------------------------------------------------------------
// GET /teacher-insights/dashboard — triage surface (§2.1–2.4).
// ---------------------------------------------------------------------------
router.get('/dashboard', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const classes = await accessibleClasses(clerkId);
    const classIds = classes.map((c) => c.id as string);
    if (classIds.length === 0) {
      return res.json({ action_queue: { low_confidence: 0, ready_bulk: 0, ocr_failed: 0, reviewed_unreleased: 0 }, active_assignments: [], class_pulse: [], needs_attention: [] });
    }

    const threshold = await getThreshold(clerkId);

    const [{ data: assignments }, { data: enrollments }] = await Promise.all([
      supabase.from('assignments').select('id, class_id, title, status, deadline_at, target_all').in('class_id', classIds).eq('status', 'published'),
      supabase.from('class_enrollments').select('class_id, student_clerk_id').in('class_id', classIds).eq('status', 'active'),
    ]);
    const asgRows = (assignments ?? []) as Record<string, unknown>[];
    const asgIds = asgRows.map((a) => a.id as string);
    const enrollRows = (enrollments ?? []) as { class_id: string; student_clerk_id: string }[];

    const activeByClass = new Map<string, string[]>();
    for (const e of enrollRows) {
      const list = activeByClass.get(e.class_id) ?? [];
      list.push(e.student_clerk_id);
      activeByClass.set(e.class_id, list);
    }

    let submissions: Record<string, unknown>[] = [];
    let marks: Record<string, unknown>[] = [];
    let aqs: Record<string, unknown>[] = [];
    if (asgIds.length > 0) {
      const [{ data: subs }, { data: aqRows }] = await Promise.all([
        supabase.from('submissions').select('id, assignment_id, student_clerk_id, status, submitted_at, released_at, total_score, total_marks').in('assignment_id', asgIds),
        supabase.from('assignment_questions').select('id, assignment_id, snapshot').in('assignment_id', asgIds),
      ]);
      submissions = (subs as Record<string, unknown>[]) ?? [];
      aqs = (aqRows as Record<string, unknown>[]) ?? [];
      const subIds = submissions.map((s) => s.id as string);
      if (subIds.length > 0) {
        const { data: mk } = await supabase.from('submission_marks').select('submission_id, assignment_question_id, status, confidence, ai_criteria, final_criteria, ai_score, final_score, ai_marks_available').in('submission_id', subIds);
        marks = (mk as Record<string, unknown>[]) ?? [];
      }
    }

    // §2.1 action queue.
    const action_queue = {
      low_confidence: marks.filter((m) => m.status === 'needs_review' && (m.confidence === null || Number(m.confidence) < threshold)).length,
      ready_bulk: marks.filter((m) => m.status === 'needs_review' && m.confidence !== null && Number(m.confidence) >= threshold).length,
      ocr_failed: marks.filter((m) => m.status === 'ocr_failed').length,
      reviewed_unreleased: 0,
    };
    // Fully-reviewed-but-unreleased submissions.
    const marksBySub = new Map<string, Record<string, unknown>[]>();
    for (const m of marks) {
      const list = marksBySub.get(m.submission_id as string) ?? [];
      list.push(m);
      marksBySub.set(m.submission_id as string, list);
    }
    for (const s of submissions) {
      if (s.released_at) continue;
      const ms = marksBySub.get(s.id as string) ?? [];
      if (ms.length > 0 && ms.every((m) => REVIEWED.includes(m.status as string))) action_queue.reviewed_unreleased += 1;
    }

    // §2.2 active assignments (max 5).
    const classNameById = new Map(classes.map((c) => [c.id as string, c.name as string]));
    const subsByAsg = new Map<string, Record<string, unknown>[]>();
    for (const s of submissions) {
      const list = subsByAsg.get(s.assignment_id as string) ?? [];
      list.push(s);
      subsByAsg.set(s.assignment_id as string, list);
    }
    const active_assignments = asgRows
      .map((a) => {
        const subs = subsByAsg.get(a.id as string) ?? [];
        const total = a.target_all ? (activeByClass.get(a.class_id as string) ?? []).length : subs.length;
        const submitted = subs.filter((s) => s.status === 'submitted' || s.status === 'late').length;
        const late = subs.filter((s) => s.status === 'late').length;
        return { id: a.id, title: a.title, class_name: classNameById.get(a.class_id as string) ?? 'Class', submitted, total, late, deadline_at: a.deadline_at };
      })
      .sort((x, y) => (String(y.deadline_at ?? '')).localeCompare(String(x.deadline_at ?? '')))
      .slice(0, 5);

    // §2.3 class pulse (per class): 30-day avg + weakest topics.
    const thirtyDaysAgo = Date.now() - 30 * 24 * 60 * 60 * 1000;
    const topicByAq = new Map<string, string>();
    for (const q of aqs) topicByAq.set(q.id as string, String((q.snapshot as Record<string, unknown>)?.topic ?? 'General'));
    const asgClass = new Map(asgRows.map((a) => [a.id as string, a.class_id as string]));
    const subClass = new Map(submissions.map((s) => [s.id as string, asgClass.get(s.assignment_id as string)]));

    const class_pulse = classes.map((c) => {
      const clsSubs = submissions.filter((s) => subClass.get(s.id as string) === c.id && (s.status === 'submitted' || s.status === 'late'));
      const recent = clsSubs.filter((s) => s.submitted_at && new Date(s.submitted_at as string).getTime() >= thirtyDaysAgo && Number(s.total_marks) > 0);
      const avg = recent.length > 0 ? Math.round((recent.reduce((sum, s) => sum + Number(s.total_score) / Number(s.total_marks), 0) / recent.length) * 100) : null;

      // Weakest topics from this class's marks.
      const clsSubIds = new Set(clsSubs.map((s) => s.id));
      const byTopic = new Map<string, { earned: number; available: number }>();
      for (const m of marks) {
        if (!clsSubIds.has(m.submission_id)) continue;
        const topic = topicByAq.get(m.assignment_question_id as string) || 'General';
        const { earned, available } = markScore(m);
        const cur = byTopic.get(topic) ?? { earned: 0, available: 0 };
        cur.earned += earned;
        cur.available += available;
        byTopic.set(topic, cur);
      }
      const weakest = Array.from(byTopic, ([topic, v]) => ({ topic, mastery: v.available > 0 ? Math.round((v.earned / v.available) * 100) : 0 }))
        .filter((t) => t.topic !== 'General')
        .sort((a, b) => a.mastery - b.mastery)
        .slice(0, 3);

      return { class_id: c.id, class_name: c.name, subject: c.subject, avg, weakest_topics: weakest, topics_attempted: byTopic.size };
    });

    // §2.4 students needing attention (computable triggers: inactive, missed).
    const now = Date.now();
    const sevenDaysAgo = now - 7 * 24 * 60 * 60 * 1000;
    const lastSubByStudent = new Map<string, number>();
    for (const s of submissions) {
      if (!s.submitted_at) continue;
      const t = new Date(s.submitted_at as string).getTime();
      const prev = lastSubByStudent.get(s.student_clerk_id as string) ?? 0;
      if (t > prev) lastSubByStudent.set(s.student_clerk_id as string, t);
    }
    // Missed = past-deadline assignments a targeted student didn't submit.
    const missedByStudent = new Map<string, number>();
    for (const a of asgRows) {
      if (!a.deadline_at || new Date(a.deadline_at as string).getTime() > now) continue;
      const targeted = a.target_all ? (activeByClass.get(a.class_id as string) ?? []) : (subsByAsg.get(a.id as string) ?? []).map((s) => s.student_clerk_id as string);
      const submittedIds = new Set((subsByAsg.get(a.id as string) ?? []).filter((s) => s.status === 'submitted' || s.status === 'late').map((s) => s.student_clerk_id));
      for (const st of targeted) if (!submittedIds.has(st)) missedByStudent.set(st, (missedByStudent.get(st) ?? 0) + 1);
    }

    const allStudentIds = Array.from(new Set(enrollRows.map((e) => e.student_clerk_id)));
    const needsRaw: { clerk_id: string; reason: string }[] = [];
    for (const id of allStudentIds) {
      const missed = missedByStudent.get(id) ?? 0;
      const last = lastSubByStudent.get(id);
      if (missed >= 2) needsRaw.push({ clerk_id: id, reason: `Missed ${missed} assignments` });
      else if (!last || last < sevenDaysAgo) needsRaw.push({ clerk_id: id, reason: 'No practice in 7+ days' });
    }
    const attentionIds = needsRaw.slice(0, 5).map((n) => n.clerk_id);
    const nameMap = new Map<string, { name: string; class_id: string }>();
    if (attentionIds.length > 0) {
      const { data: profs } = await supabase.from('profiles').select('clerk_id, full_name, email').in('clerk_id', attentionIds);
      for (const p of (profs ?? []) as { clerk_id: string; full_name: string | null; email: string | null }[]) {
        const cls = enrollRows.find((e) => e.student_clerk_id === p.clerk_id)?.class_id ?? '';
        nameMap.set(p.clerk_id, { name: p.full_name || p.email || 'Student', class_id: cls });
      }
    }
    const needs_attention = needsRaw.slice(0, 5).map((n) => ({
      clerk_id: n.clerk_id,
      name: nameMap.get(n.clerk_id)?.name ?? 'Student',
      class_id: nameMap.get(n.clerk_id)?.class_id ?? '',
      reason: n.reason,
    }));

    return res.json({ action_queue, active_assignments, class_pulse, needs_attention });
  } catch (err) {
    console.error('Dashboard error:', err);
    return res.status(500).json({ error: 'Failed to load dashboard' });
  }
});

// ---------------------------------------------------------------------------
// Class analytics helpers — gather marks + topics for a class.
// ---------------------------------------------------------------------------
async function gatherClassMarks(classId: string) {
  const { data: assignments } = await supabase.from('assignments').select('id, title').eq('class_id', classId).eq('status', 'published');
  const asgIds = ((assignments ?? []) as { id: string }[]).map((a) => a.id);
  if (asgIds.length === 0) return { marks: [], aqs: [], submissions: [], assignments: (assignments ?? []) as Record<string, unknown>[] };
  const [{ data: subs }, { data: aqs }] = await Promise.all([
    supabase.from('submissions').select('id, assignment_id, student_clerk_id, status, submitted_at, total_score, total_marks').in('assignment_id', asgIds),
    supabase.from('assignment_questions').select('id, assignment_id, order_index, snapshot').in('assignment_id', asgIds),
  ]);
  const submissions = (subs as Record<string, unknown>[]) ?? [];
  const subIds = submissions.map((s) => s.id as string);
  let marks: Record<string, unknown>[] = [];
  if (subIds.length > 0) {
    const { data: mk } = await supabase.from('submission_marks').select('submission_id, assignment_question_id, ai_criteria, final_criteria, ai_score, final_score, ai_marks_available').in('submission_id', subIds);
    marks = (mk as Record<string, unknown>[]) ?? [];
  }
  return { marks, aqs: (aqs as Record<string, unknown>[]) ?? [], submissions, assignments: (assignments ?? []) as Record<string, unknown>[] };
}

// GET /teacher-insights/class/:classId/heatmap — students × topics (§12.1).
router.get('/class/:classId/heatmap', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.classId, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    const { marks, aqs, submissions } = await gatherClassMarks(req.params.classId);

    const topicByAq = new Map<string, string>();
    for (const q of aqs) topicByAq.set(q.id as string, String((q.snapshot as Record<string, unknown>)?.topic ?? 'General'));
    const studentBySub = new Map(submissions.map((s) => [s.id as string, s.student_clerk_id as string]));

    // cell[student][topic] = {earned, available}
    const grid = new Map<string, Map<string, { earned: number; available: number }>>();
    const topics = new Set<string>();
    for (const m of marks) {
      const student = studentBySub.get(m.submission_id as string);
      if (!student) continue;
      const topic = topicByAq.get(m.assignment_question_id as string) || 'General';
      topics.add(topic);
      const row = grid.get(student) ?? new Map();
      const cell = row.get(topic) ?? { earned: 0, available: 0 };
      const sc = markScore(m);
      cell.earned += sc.earned;
      cell.available += sc.available;
      row.set(topic, cell);
      grid.set(student, row);
    }

    const studentIds = Array.from(grid.keys());
    const nameMap = new Map<string, string>();
    if (studentIds.length > 0) {
      const { data: profs } = await supabase.from('profiles').select('clerk_id, full_name, email').in('clerk_id', studentIds);
      for (const p of (profs ?? []) as { clerk_id: string; full_name: string | null; email: string | null }[]) nameMap.set(p.clerk_id, p.full_name || p.email || 'Student');
    }

    const topicList = Array.from(topics).sort();
    const rows = studentIds.map((s) => ({
      student: nameMap.get(s) ?? 'Student',
      cells: topicList.map((t) => {
        const cell = grid.get(s)?.get(t);
        return cell && cell.available > 0 ? Math.round((cell.earned / cell.available) * 100) : null;
      }),
    }));
    // Class average row.
    const classAvg = topicList.map((t, i) => {
      const vals = rows.map((r) => r.cells[i]).filter((v): v is number => v !== null);
      return vals.length > 0 ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : null;
    });

    return res.json({ topics: topicList, rows, class_average: classAvg });
  } catch (err) {
    console.error('Heatmap error:', err);
    return res.status(500).json({ error: 'Failed to load heatmap' });
  }
});

// GET /teacher-insights/class/:classId/difficulty — per-question % (§12.3).
router.get('/class/:classId/difficulty', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.classId, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    const { marks, aqs } = await gatherClassMarks(req.params.classId);

    const metaByAq = new Map<string, { topic: string; number: string }>();
    for (const q of aqs) {
      const snap = (q.snapshot as Record<string, unknown>) || {};
      metaByAq.set(q.id as string, { topic: String(snap.topic ?? 'General'), number: String(snap.question_number ?? q.order_index ?? '') });
    }

    const byAq = new Map<string, { earned: number; available: number }>();
    for (const m of marks) {
      const cur = byAq.get(m.assignment_question_id as string) ?? { earned: 0, available: 0 };
      const sc = markScore(m);
      cur.earned += sc.earned;
      cur.available += sc.available;
      byAq.set(m.assignment_question_id as string, cur);
    }

    const questions = Array.from(byAq, ([aqId, v]) => ({
      topic: metaByAq.get(aqId)?.topic ?? 'General',
      number: metaByAq.get(aqId)?.number ?? '',
      pct: v.available > 0 ? Math.round((v.earned / v.available) * 100) : 0,
    })).sort((a, b) => a.pct - b.pct);

    return res.json({ questions });
  } catch (err) {
    console.error('Difficulty error:', err);
    return res.status(500).json({ error: 'Failed to load question difficulty' });
  }
});

// GET /teacher-insights/class/:classId/student/:clerkId/mastery — per-topic
// mastery for one student (feeds the student profile §4.2 and report §13.2).
router.get('/class/:classId/student/:clerkId/mastery', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.classId, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    const { marks, aqs, submissions } = await gatherClassMarks(req.params.classId);
    const mySubIds = new Set(submissions.filter((s) => s.student_clerk_id === req.params.clerkId).map((s) => s.id));
    const topicByAq = new Map<string, string>();
    for (const q of aqs) topicByAq.set(q.id as string, String((q.snapshot as Record<string, unknown>)?.topic ?? 'General'));

    const byTopic = new Map<string, { earned: number; available: number }>();
    for (const m of marks) {
      if (!mySubIds.has(m.submission_id)) continue;
      const topic = topicByAq.get(m.assignment_question_id as string) || 'General';
      const sc = markScore(m);
      const cur = byTopic.get(topic) ?? { earned: 0, available: 0 };
      cur.earned += sc.earned;
      cur.available += sc.available;
      byTopic.set(topic, cur);
    }
    const mastery = Array.from(byTopic, ([topic, v]) => ({ topic, mastery: v.available > 0 ? Math.round((v.earned / v.available) * 100) : 0 }))
      .sort((a, b) => a.mastery - b.mastery);
    return res.json({ mastery });
  } catch (err) {
    console.error('Mastery error:', err);
    return res.status(500).json({ error: 'Failed to load mastery' });
  }
});

// GET /teacher-insights/export/csv?class_id=&from=&to= — raw marks CSV (§13.3/§13.4).
router.get('/export/csv', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const classId = typeof req.query.class_id === 'string' ? req.query.class_id : '';
    const access = await resolveClassAccess(classId, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });
    const from = typeof req.query.from === 'string' ? new Date(req.query.from) : null;
    const to = typeof req.query.to === 'string' ? new Date(req.query.to) : null;

    const { data: assignments } = await supabase.from('assignments').select('id, title').eq('class_id', classId).eq('status', 'published');
    const asgMap = new Map(((assignments ?? []) as { id: string; title: string }[]).map((a) => [a.id, a.title]));
    const asgIds = Array.from(asgMap.keys());
    if (asgIds.length === 0) return res.type('text/csv').send('student,assignment,syllabus,date,marks_earned,marks_available,percentage,grade\n');

    const { data: subs } = await supabase
      .from('submissions')
      .select('assignment_id, student_clerk_id, submitted_at, total_score, total_marks, status, released_at')
      .in('assignment_id', asgIds)
      .in('status', ['submitted', 'late']);

    const rows = ((subs ?? []) as Record<string, unknown>[]).filter((s) => {
      if (!s.submitted_at) return false;
      const d = new Date(s.submitted_at as string);
      if (from && d < from) return false;
      if (to && d > to) return false;
      return true;
    });

    const studentIds = Array.from(new Set(rows.map((r) => r.student_clerk_id as string)));
    const nameMap = new Map<string, string>();
    if (studentIds.length > 0) {
      const { data: profs } = await supabase.from('profiles').select('clerk_id, full_name, email').in('clerk_id', studentIds);
      for (const p of (profs ?? []) as { clerk_id: string; full_name: string | null; email: string | null }[]) nameMap.set(p.clerk_id, p.full_name || p.email || 'Student');
    }
    const { data: klass } = await supabase.from('classes').select('syllabus_code').eq('id', classId).maybeSingle();
    const syllabus = (klass as { syllabus_code?: string } | null)?.syllabus_code ?? '';

    const esc = (v: string | number) => `"${String(v).replace(/"/g, '""')}"`;
    const lines = ['student,assignment,syllabus,date,marks_earned,marks_available,percentage,grade'];
    for (const r of rows) {
      const pct = Number(r.total_marks) > 0 ? Math.round((Number(r.total_score) / Number(r.total_marks)) * 100) : 0;
      lines.push(
        [
          esc(nameMap.get(r.student_clerk_id as string) ?? 'Student'),
          esc(asgMap.get(r.assignment_id as string) ?? 'Assignment'),
          esc(syllabus),
          esc(new Date(r.submitted_at as string).toISOString().slice(0, 10)),
          Number(r.total_score) || 0,
          Number(r.total_marks) || 0,
          pct,
          '',
        ].join(',')
      );
    }

    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', `attachment; filename="class-export.csv"`);
    return res.send(lines.join('\n'));
  } catch (err) {
    console.error('CSV export error:', err);
    return res.status(500).json({ error: 'Failed to export CSV' });
  }
});

// GET /teacher-insights/class/:classId/student/:clerkId/progress — chronological
// scores, distinguishing full papers from topical sets (§12.4).
router.get('/class/:classId/student/:clerkId/progress', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.classId, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });

    const { data: assignments } = await supabase.from('assignments').select('id, title, source_mode').eq('class_id', req.params.classId);
    const asgMap = new Map(((assignments ?? []) as Record<string, unknown>[]).map((a) => [a.id as string, a]));
    const asgIds = Array.from(asgMap.keys());
    if (asgIds.length === 0) return res.json({ points: [] });

    const { data: subs } = await supabase
      .from('submissions')
      .select('assignment_id, submitted_at, total_score, total_marks, status')
      .eq('student_clerk_id', req.params.clerkId)
      .in('assignment_id', asgIds)
      .in('status', ['submitted', 'late']);

    const points = ((subs ?? []) as Record<string, unknown>[])
      .filter((s) => s.submitted_at && Number(s.total_marks) > 0)
      .map((s) => {
        const a = asgMap.get(s.assignment_id as string);
        return {
          date: s.submitted_at,
          title: (a?.title as string) ?? 'Assignment',
          pct: Math.round((Number(s.total_score) / Number(s.total_marks)) * 100),
          is_full_paper: a?.source_mode === 'full_paper',
        };
      })
      .sort((x, y) => String(x.date).localeCompare(String(y.date)));

    return res.json({ points });
  } catch (err) {
    console.error('Progress error:', err);
    return res.status(500).json({ error: 'Failed to load progress' });
  }
});

// GET /teacher-insights/class/:classId/student/:clerkId/predicted — predicted
// grade with the minimum-evidence rule (§12.6).
router.get('/class/:classId/student/:clerkId/predicted', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const access = await resolveClassAccess(req.params.classId, req.auth!.clerkId);
    if (!access) return res.status(404).json({ error: 'Class not found' });

    const { data: klass } = await supabase.from('classes').select('syllabus_code, exam_series').eq('id', req.params.classId).maybeSingle();
    const syllabusCode = (klass as { syllabus_code?: string } | null)?.syllabus_code ?? '';
    const series = (klass as { exam_series?: string } | null)?.exam_series ?? null;

    const { data: assignments } = await supabase.from('assignments').select('id').eq('class_id', req.params.classId).eq('status', 'published');
    const asgIds = ((assignments ?? []) as { id: string }[]).map((a) => a.id);
    if (asgIds.length === 0) return res.json({ enough_data: false, reason: 'not enough data yet' });

    const { data: subs } = await supabase
      .from('submissions')
      .select('id, assignment_id, submitted_at, total_score, total_marks, status')
      .eq('student_clerk_id', req.params.clerkId)
      .in('assignment_id', asgIds)
      .in('status', ['submitted', 'late']);
    const done = ((subs ?? []) as Record<string, unknown>[]).filter((s) => Number(s.total_marks) > 0);

    // §12.6 minimum-evidence rule: default 4 assignments covering 3+ topics.
    const subIds = done.map((s) => s.id as string);
    let topicsCovered = new Set<string>();
    if (subIds.length > 0) {
      const { data: marks } = await supabase.from('submission_marks').select('submission_id, assignment_question_id').in('submission_id', subIds);
      const aqIds = Array.from(new Set(((marks ?? []) as { assignment_question_id: string }[]).map((m) => m.assignment_question_id)));
      if (aqIds.length > 0) {
        const { data: aqs } = await supabase.from('assignment_questions').select('id, snapshot').in('id', aqIds);
        for (const q of (aqs ?? []) as Record<string, unknown>[]) topicsCovered.add(String((q.snapshot as Record<string, unknown>)?.topic ?? 'General'));
      }
    }

    if (done.length < 4 || topicsCovered.size < 3) {
      return res.json({ enough_data: false, reason: 'not enough data yet', completed: done.length, topics: topicsCovered.size });
    }

    const rollingPct = Math.round((done.reduce((s, x) => s + Number(x.total_score) / Number(x.total_marks), 0) / done.length) * 100);

    // Grade boundaries are data (§12.6). Match the most specific session available.
    const { data: boundaries } = await supabase
      .from('grade_boundaries')
      .select('grade, min_percent, series, year')
      .eq('syllabus_code', syllabusCode)
      .order('min_percent', { ascending: false });
    const rows = ((boundaries ?? []) as { grade: string; min_percent: number; series: string | null; year: number | null }[])
      .filter((b) => !series || !b.series || b.series === series);

    if (rows.length === 0) {
      return res.json({ enough_data: true, rolling_pct: rollingPct, grade: null, reason: 'grade boundaries not configured for this syllabus' });
    }

    let grade = rows[rows.length - 1].grade;
    let nextBoundary: number | null = null;
    for (const b of rows) {
      if (rollingPct >= Number(b.min_percent)) {
        grade = b.grade;
        break;
      }
      nextBoundary = Number(b.min_percent);
    }
    const marksToNext = nextBoundary !== null ? Math.max(0, nextBoundary - rollingPct) : null;
    const sessionUsed = rows[0].series && rows[0].year ? `${rows[0].series} ${rows[0].year}` : 'stored boundaries';

    return res.json({ enough_data: true, rolling_pct: rollingPct, grade, marks_to_next_pct: marksToNext, session_used: sessionUsed });
  } catch (err) {
    console.error('Predicted grade error:', err);
    return res.status(500).json({ error: 'Failed to compute predicted grade' });
  }
});

export default router;
