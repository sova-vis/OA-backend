/**
 * Propel billing API — /billing
 *
 * Live now (no payment provider needed):
 *   GET  /billing/status       → current access state for the frontend
 *   POST /billing/start-trial  → start the 10-day free trial (no card)
 *   POST /billing/cancel       → turn off auto-renew (keeps access until period end)
 *   POST /billing/tick         → renewal engine, secret-protected (cron calls this)
 *
 * Stubs until Safepay is connected (deliberately inert — they never charge):
 *   POST /billing/checkout     → returns 501 with a friendly "coming soon" message
 *   POST /billing/webhook      → accepts pings but does not change state yet
 */
import { Router, Request, Response } from 'express';
import { clerkAuth, AuthenticatedRequest } from './lib/clerkAuth';
import { supabase } from './lib/supabase';
import {
  ensureBilling,
  computeAccess,
  startTrialForUser,
  tick,
  BILLING_ENFORCED,
  TRIAL_DAYS,
  GRACE_DAYS,
  PRICE_PKR_MONTHLY,
  PRICE_PKR_ANNUAL,
} from './lib/entitlements';

const router = Router();

/** What the frontend needs to render trial banners / gates / upgrade prompts. */
router.get('/status', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const row = await ensureBilling(clerkId);
    const access = computeAccess(row);
    res.json({
      enforced: BILLING_ENFORCED,
      trialDays: TRIAL_DAYS,
      graceDays: GRACE_DAYS,
      price: { monthlyPkr: PRICE_PKR_MONTHLY, annualPkr: PRICE_PKR_ANNUAL },
      ...access,
    });
  } catch (error) {
    console.error('GET /billing/status error:', error);
    res.status(500).json({ error: 'server_error' });
  }
});

/** Start the free trial. One tap, no card. Idempotent + soft device de-dupe. */
router.post('/start-trial', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const raw = (req.body && typeof req.body.deviceHash === 'string') ? req.body.deviceHash : '';
    const deviceHash = raw ? raw.slice(0, 128) : undefined;
    const result = await startTrialForUser(clerkId, deviceHash);
    res.json({ enforced: BILLING_ENFORCED, ...result });
  } catch (error) {
    console.error('POST /billing/start-trial error:', error);
    res.status(500).json({ error: 'server_error' });
  }
});

/** Turn off auto-renew. Access continues until the current period ends. */
router.post('/cancel', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth!.clerkId;
    const row = await ensureBilling(clerkId);
    if (row.status === 'active') {
      await supabase
        .from('student_billing')
        .update({ status: 'canceled', auto_renew: false, updated_at: new Date().toISOString() })
        .eq('clerk_id', clerkId);
    }
    const fresh = await ensureBilling(clerkId);
    res.json({ ok: true, enforced: BILLING_ENFORCED, ...computeAccess(fresh) });
  } catch (error) {
    console.error('POST /billing/cancel error:', error);
    res.status(500).json({ error: 'server_error' });
  }
});

/**
 * STUB — Safepay not connected yet. When it is, this will create a Safepay
 * checkout/subscription and return a redirect URL. Until then it stays inert so
 * nothing can accidentally charge. The frontend shows the message gracefully.
 */
router.post('/checkout', clerkAuth, async (_req: AuthenticatedRequest, res: Response) => {
  res.status(501).json({
    error: 'payments_not_connected',
    message: 'Payments will be enabled once Safepay is connected. Your free trial works right now.',
  });
});

/**
 * STUB — Safepay webhook receiver. Will verify the signature and, on a successful
 * charge, set status='active', current_period_end=+1 period, store the token/sub
 * id, and write a payments row. For now it just acknowledges so provider test
 * pings don't error. It NEVER grants access on its own yet.
 */
router.post('/webhook', async (_req: Request, res: Response) => {
  // TODO(safepay): verify signature, then apply the paid period + write payments row.
  res.status(200).json({ received: true, note: 'stub — not applying state until Safepay is wired' });
});

/**
 * Renewal engine. A scheduler (Railway Cron or an external cron) POSTs here with
 * the shared secret header. Advances trial/period/grace states from timestamps.
 */
router.post('/tick', async (req: Request, res: Response) => {
  const secret = (process.env.BILLING_TICK_SECRET || '').trim();
  const provided = String(req.headers['x-cron-secret'] || '');
  if (!secret || provided !== secret) {
    return res.status(403).json({ error: 'forbidden' });
  }
  try {
    const summary = await tick();
    res.json({ ok: true, ranAt: new Date().toISOString(), ...summary });
  } catch (error) {
    console.error('POST /billing/tick error:', error);
    res.status(500).json({ error: 'server_error' });
  }
});

export default router;
