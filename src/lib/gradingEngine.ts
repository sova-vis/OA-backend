/**
 * Marks a single mark-scheme criterion: we pass the criterion text as the "mark
 * scheme" and read back how well the student's answer satisfies it (§8.3).
 *
 * Primary marker is the OA/QA grading engine (`/oa-level/evaluate`). When that
 * service is down/unconfigured we fall back to the Grok LLM (the same key that
 * powers OCR/handwriting), so theory auto-marking still works out of the box
 * instead of leaving every answer as "No AI mark".
 *
 * The score is a holistic 0–1, NOT a calibrated confidence (Appendix A blocking
 * dependency). We derive a HEURISTIC confidence from the score's distance from
 * the 0.5 decision boundary and flag it as uncalibrated.
 *
 * Returns null only when BOTH markers are unavailable, so marking degrades
 * gracefully to a review scaffold.
 */
import { grokChatJson, grokEnabled } from './grok';

const base = (
  process.env.OA_GRADING_SERVICE_URL ||
  process.env.QA_GRADING_SERVICE_URL ||
  'http://127.0.0.1:8001'
).replace(/\/$/, '');

export interface CriterionEval {
  score: number; // 0..1
  feedback: string;
}

interface CriterionInput {
  question: string;
  studentAnswer: string;
  criterionText: string;
  subject: string | null;
}

export async function evaluateCriterion(input: CriterionInput): Promise<CriterionEval | null> {
  const fromEngine = await evaluateWithEngine(input);
  if (fromEngine) return fromEngine;
  // Engine down/unconfigured → LLM fallback so marking still happens.
  return evaluateWithGrok(input);
}

async function evaluateWithEngine(input: CriterionInput): Promise<CriterionEval | null> {
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

async function evaluateWithGrok(input: CriterionInput): Promise<CriterionEval | null> {
  if (!grokEnabled()) return null;
  const system =
    'You are a strict but fair Cambridge O/A Level examiner marking ONE mark-scheme point. ' +
    'Decide how well the student answer satisfies the given marking point. ' +
    'Reply ONLY with JSON: {"score": <0..1>, "feedback": "<one short sentence>"}. ' +
    'score 1 = the point is fully and correctly made; 0 = absent or wrong; ' +
    'use values in between for partial credit. Judge meaning, not wording.';
  const user =
    `SUBJECT: ${input.subject ?? 'General'}\n` +
    `QUESTION: ${input.question || '(not provided)'}\n` +
    `MARKING POINT (award if satisfied): ${input.criterionText}\n` +
    `STUDENT ANSWER: ${input.studentAnswer || '(blank)'}`;
  try {
    const parsed = await grokChatJson({ system, user, temperature: 0, maxTokens: 400, timeoutMs: 45000 });
    const raw = Number((parsed as { score?: unknown }).score);
    const score = Number.isFinite(raw) ? Math.max(0, Math.min(1, raw)) : 0;
    const feedback = typeof (parsed as { feedback?: unknown }).feedback === 'string' ? String((parsed as { feedback?: unknown }).feedback) : '';
    return { score, feedback };
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
