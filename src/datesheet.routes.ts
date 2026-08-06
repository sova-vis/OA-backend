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
    const { data: profile } = await supabase
      .from('profiles')
      .select('exam_session, selected_subjects, level')
      .eq('clerk_id', clerkId)
      .maybeSingle();

    const session = (profile as { exam_session?: string } | null)?.exam_session || null;
    const subjects = ((profile as { selected_subjects?: string[] } | null)?.selected_subjects || []).map((s) => s.toLowerCase());

    if (!subjects.length) return res.json({ status: 'no_subjects', session, papers: [], next_exam: null });
    if (!session || /not sure/i.test(session)) return res.json({ status: 'unavailable', session, papers: [], next_exam: null });

    const { data: rows } = await supabase
      .from('exam_timetable')
      .select('*')
      .eq('exam_session', session);

    // Filter to the student's subjects (case-insensitive, partial match either way).
    const papers = ((rows ?? []) as Record<string, unknown>[])
      .filter((r) => {
        const subj = String(r.subject || '').toLowerCase();
        // Exact match, or one is a whole-word prefix of the other — avoids
        // "Mathematics" pulling in "Additional Mathematics".
        return subjects.some((s) => subj === s || subj.startsWith(`${s} `) || s.startsWith(`${subj} `));
      })
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
