import { Router, Response } from 'express';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { supabase } from './lib/supabase';

/**
 * Propel — exam datesheet (spec §3). Resolves a student's exam dates for their
 * declared session, filtered to their subjects. Tier 1 rows are 'confirmed';
 * Tier 2 rows are 'predicted' (rendered distinctly). No data → 'unavailable'.
 */

const router = Router();
router.use(clerkAuth);

router.get('/', async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const scope = req.query.scope === 'classroom' ? 'classroom' : 'personal';

    const { data: profile } = await supabase
      .from('profiles')
      .select('exam_session, selected_subjects, level')
      .eq('clerk_id', clerkId)
      .maybeSingle();

    let session = (profile as { exam_session?: string } | null)?.exam_session || null;

    // Personal scope → the student's own declared subjects. Classroom scope →
    // the subjects of the classes they've joined (what their teachers teach).
    let subjectNames: string[];
    if (scope === 'classroom') {
      const { data: enr } = await supabase.from('class_enrollments').select('class_id').eq('student_clerk_id', clerkId).eq('status', 'active');
      const classIds = ((enr ?? []) as { class_id: string }[]).map((e) => e.class_id);
      if (classIds.length === 0) subjectNames = [];
      else {
        const { data: classes } = await supabase.from('classes').select('subject').in('id', classIds);
        subjectNames = Array.from(new Set(((classes ?? []) as { subject: string }[]).map((c) => c.subject)));
      }
    } else {
      subjectNames = (profile as { selected_subjects?: string[] } | null)?.selected_subjects || [];
    }
    const subjects = subjectNames.map((s) => s.toLowerCase());

    if (!subjects.length) return res.json({ status: 'no_subjects', session, papers: [], next_exam: null });

    // If the student hasn't declared a session (e.g. joined via class link), we
    // still show their subjects' timetable — falling back to the nearest upcoming
    // session in the data rather than a blank "unavailable".
    const sessionUnset = !session || /not sure/i.test(session);
    let tq = supabase.from('exam_timetable').select('*');
    if (!sessionUnset) tq = tq.eq('exam_session', session);
    const { data: rows } = await tq;

    const subjectMatch = (r: Record<string, unknown>) => {
      const subj = String(r.subject || '').toLowerCase();
      // Exact match, or one is a whole-word prefix of the other — avoids
      // "Mathematics" pulling in "Additional Mathematics".
      return subjects.some((s) => subj === s || subj.startsWith(`${s} `) || s.startsWith(`${subj} `));
    };
    let matched = ((rows ?? []) as Record<string, unknown>[]).filter(subjectMatch);

    // No session declared → pick the session with the nearest upcoming paper.
    if (sessionUnset && matched.length) {
      const todayStr = new Date().toISOString().slice(0, 10);
      const rank = (sess: string) => {
        const dates = matched.filter((r) => r.exam_session === sess).map((r) => String(r.exam_date || r.date_start || '')).filter(Boolean).sort();
        const future = dates.find((d) => d >= todayStr);
        return future || dates[0] || '9999';
      };
      const sessions = Array.from(new Set(matched.map((r) => String(r.exam_session))));
      session = sessions.sort((a, b) => rank(a).localeCompare(rank(b)))[0] || session;
      matched = matched.filter((r) => r.exam_session === session);
    }

    // Filter to the student's subjects (case-insensitive, partial match either way).
    const papers = matched
      .map((r) => ({
        subject: r.subject,
        component: r.component,
        syllabus_code: r.syllabus_code,
        date: r.exam_date,
        date_start: r.date_start,
        date_end: r.date_end,
        session_slot: r.session_slot,
        duration: r.duration,
        status: r.status,
      }))
      .sort((a, b) => {
        const da = String(a.date || a.date_start || '');
        const db = String(b.date || b.date_start || '');
        return da.localeCompare(db);
      });

    if (papers.length === 0) return res.json({ status: 'unavailable', session, papers: [], next_exam: null });

    const anyPredicted = papers.some((p) => p.status === 'predicted');
    const today = new Date();
    // Next exam = first paper whose (start) date is today or later.
    const upcoming = papers.filter((p) => {
      const d = p.date || p.date_start;
      return d ? new Date(String(d)) >= new Date(today.toDateString()) : false;
    });
    const next = upcoming[0] || papers[0];
    const nextDate = next?.date || next?.date_start;
    const daysRemaining = nextDate ? Math.ceil((new Date(String(nextDate)).getTime() - today.getTime()) / (1000 * 60 * 60 * 24)) : null;

    return res.json({
      status: anyPredicted ? 'predicted' : 'confirmed',
      session,
      next_exam: next ? { ...next, days_remaining: daysRemaining } : null,
      papers,
    });
  } catch (err) {
    console.error('Datesheet error:', err);
    return res.status(500).json({ error: 'Failed to load datesheet' });
  }
});

export default router;
