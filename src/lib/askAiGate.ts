/**
 * §6.3 Ask-AI metering. Runs on /rag BEFORE requirePro.
 *
 * A classroom student whose school still has Ask-AI allowance is granted access
 * via the school pool: we set `req.aiViaSchool` (which requirePro honours) and
 * meter each SUCCESSFUL generation (POST /query, /explain-mcq, /ask-image)
 * against the school. When the pool is spent the request falls through to
 * requirePro (personal Pro / the discounted upsell). Fail-safe: any error just
 * passes through unchanged, so Ask-AI is never harmed by a metering hiccup.
 */
import { Response, NextFunction } from 'express';
import { AuthenticatedRequest } from './clerkAuth';
import { primaryStudentSchoolId } from './schoolOffer';
import { askAiUsage, recordAskAi } from './quota';

export type AskAiRequest = AuthenticatedRequest & { aiViaSchool?: boolean };

export async function schoolAskAiGate(req: AskAiRequest, res: Response, next: NextFunction) {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return next();
    const schoolId = await primaryStudentSchoolId(clerkId);
    if (!schoolId) return next();

    const { used, allowance } = await askAiUsage(schoolId);
    if (allowance > 0 && used < allowance) {
      req.aiViaSchool = true;
      // Count only a successful AI generation against the pool.
      if (req.method === 'POST') {
        res.on('finish', () => {
          if (res.statusCode < 300) void recordAskAi({ schoolId, studentClerkId: clerkId }).catch(() => {});
        });
      }
    }
  } catch (e) {
    console.error('[askai gate]', e);
  }
  next();
}
