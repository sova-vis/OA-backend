/**
 * Thin client for the OA/QA grading engine (`/oa-level/evaluate`). Used to mark
 * a single mark-scheme criterion: we pass the criterion text as the "mark
 * scheme" and read back how well the student's answer satisfies it (§8.3).
 *
 * The engine returns a holistic 0–1 score, NOT a calibrated confidence
 * (Appendix A blocking dependency). We derive a HEURISTIC confidence from the
 * score's distance from the 0.5 decision boundary and flag it as uncalibrated;
 * it must be replaced when the engine emits real calibrated confidence.
 *
 * All failures are swallowed (returns null) so marking degrades gracefully to
 * a review scaffold when the engine is down.
 */

const base = (
  process.env.OA_GRADING_SERVICE_URL ||
  process.env.QA_GRADING_SERVICE_URL ||
  'http://127.0.0.1:8001'
).replace(/\/$/, '');

export interface CriterionEval {
  score: number; // 0..1
  feedback: string;
}

export async function evaluateCriterion(input: {
  question: string;
  studentAnswer: string;
  criterionText: string;
  subject: string | null;
}): Promise<CriterionEval | null> {
  try {
    const res = await fetch(`${base}/oa-level/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: input.question,
        student_answer: input.studentAnswer,
        marking_scheme_answer: input.criterionText,
        subject: input.subject ?? undefined,
      }),
      signal: AbortSignal.timeout(25000),
    });
    if (!res.ok) return null;
    const data = (await res.json()) as { score?: unknown; feedback?: unknown };
    const raw = Number(data.score);
    const score = Number.isFinite(raw) ? Math.max(0, Math.min(1, raw)) : 0;
    return { score, feedback: typeof data.feedback === 'string' ? data.feedback : '' };
  } catch {
    return null;
  }
}

/**
 * Heuristic confidence from a holistic score: certain at the extremes, unsure
 * near 0.5. UNCALIBRATED — placeholder until the engine returns real confidence.
 */
export function heuristicConfidence(score: number): number {
  return Math.round(Math.min(1, Math.abs(score - 0.5) * 2) * 100) / 100;
}

/** A criterion is awarded when the answer clears this satisfaction threshold. */
export const CRITERION_AWARD_THRESHOLD = 0.6;
