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
import { SAFEPAY_CONFIGURED, createCheckout, verifyWebhook } from './lib/safepay';

const router = Router();

// Base URL of the student app (for post-payment redirects back into the app).
function appBaseUrl(): string {
  const raw = (process.env.APP_URL || process.env.FRONTEND_URL || 'https://oalevels.vercel.app')
    .split(',')[0].trim();
  return raw.replace(/\/+$/, '');
}
// Public base URL of THIS backend (what Safepay redirects the browser to). Prefer
// SELF_PUBLIC_URL; otherwise derive from the incoming request.
function selfBaseUrl(req: Request): string {
  const configured = (process.env.SELF_PUBLIC_URL || '').trim();
  if (configured) return configured.replace(/\/+$/, '');
  const proto = String(req.headers['x-forwarded-proto'] || req.protocol || 'https').split(',')[0].trim();
  return `${proto}://${req.get('host')}`;
}

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
 * Start a Safepay hosted checkout for the monthly (or annual) plan and return the
 * redirect URL. One-time payment — works for cards + JazzCash + Easypaisa. The
 * webhook activates Pro; the tracker token is stored so the webhook can resolve
 * the user. Falls back to 501 until the Safepay keys are configured.
 */
router.post('/checkout', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  if (!SAFEPAY_CONFIGURED) {
    return res.status(501).json({
      error: 'payments_not_connected',
      message: 'Payments aren’t connected yet. Your free trial works right now.',
    });
  }
  try {
    const clerkId = req.auth!.clerkId;
    const plan: 'monthly' | 'annual' = req.body?.plan === 'annual' ? 'annual' : 'monthly';
    const amount = plan === 'annual' ? (PRICE_PKR_ANNUAL ?? PRICE_PKR_MONTHLY * 12) : PRICE_PKR_MONTHLY;
    const orderId = `propel:${clerkId}:${plan}:${Date.now()}`;
    const base = selfBaseUrl(req);
    const { token, url } = await createCheckout({
      amountPkr: amount,
      orderId,
      redirectUrl: `${base}/billing/return`,
      cancelUrl: `${base}/billing/cancel-return`,
    });
    // Map tracker token -> user so the webhook can resolve who paid (belt-and-
    // suspenders alongside the order_id encoding).
    await ensureBilling(clerkId);
    await supabase.from('payments').insert({
      clerk_id: clerkId, provider: 'safepay', provider_tx_id: token,
      amount_pkr: amount, currency: 'PKR', status: 'pending', plan,
    });
    res.json({ ok: true, redirectUrl: url });
  } catch (error) {
    console.error('POST /billing/checkout error:', error);
    res.status(502).json({ error: 'checkout_failed', message: 'Could not start checkout. Please try again.' });
  }
});

// Safepay redirects the buyer's browser (form POST) to redirect_url / cancel_url
// after checkout. We 302 them back into the app. Activation is done by the webhook
// (source of truth), so these are just UX landings — no auth, no state change.
router.all('/return', (_req: Request, res: Response) => {
  res.redirect(302, `${appBaseUrl()}/student/dashboard?billing=success`);
});
router.all('/cancel-return', (_req: Request, res: Response) => {
  res.redirect(302, `${appBaseUrl()}/student/dashboard?billing=canceled`);
});

function pickField(obj: Record<string, unknown> | undefined | null, keys: string[]): string | null {
  if (!obj || typeof obj !== 'object') return null;
  for (const k of keys) {
    const v = obj[k];
    if (typeof v === 'string' && v) return v;
  }
  return null;
}

/** Set a student to paid-active for one period and mark their payment succeeded. */
async function activatePaid(clerkId: string, plan: 'monthly' | 'annual', token?: string): Promise<void> {
  const now = new Date();
  const end = new Date(now);
  if (plan === 'annual') end.setFullYear(end.getFullYear() + 1);
  else end.setMonth(end.getMonth() + 1);
  await ensureBilling(clerkId);
  await supabase.from('student_billing').update({
    status: 'active', plan, provider: 'safepay',
    current_period_end: end.toISOString(), last_payment_at: now.toISOString(),
    updated_at: now.toISOString(),
  }).eq('clerk_id', clerkId);
  if (token) {
    await supabase.from('payments').update({
      status: 'succeeded', period_start: now.toISOString(), period_end: end.toISOString(),
    }).eq('provider_tx_id', token);
  }
}

/** Parse a verified Safepay webhook and, on success, activate the payer. */
async function handleSafepayWebhook(body: Record<string, unknown> | undefined): Promise<void> {
  const b = (body || {}) as Record<string, unknown>;
  const data = (b.data && typeof b.data === 'object' ? b.data : {}) as Record<string, unknown>;
  const type = String(b.type || b.event || data.type || '').toLowerCase();
  const state = String(pickField(data, ['state', 'status', 'tracker_state', 'payment_state']) || '').toUpperCase();
  const token = pickField(data, ['tracker', 'token', 'tracker_token'])
    || pickField(data.tracker as Record<string, unknown>, ['token']);
  const orderId = pickField(data, ['order_id', 'orderId', 'reference']);

  let clerkId: string | null = null;
  let plan: 'monthly' | 'annual' = 'monthly';
  if (orderId && orderId.startsWith('propel:')) {
    const parts = orderId.split(':');
    clerkId = parts[1] || null;
    plan = parts[2] === 'annual' ? 'annual' : 'monthly';
  }
  if (!clerkId && token) {
    const p = await supabase.from('payments').select('clerk_id, plan')
      .eq('provider_tx_id', token).order('created_at', { ascending: false }).limit(1).maybeSingle();
    if (p.data && typeof p.data.clerk_id === 'string') {
      clerkId = p.data.clerk_id;
      plan = p.data.plan === 'annual' ? 'annual' : 'monthly';
    }
  }

  const isSuccess = /paid|complete|succeed|captured/i.test(type)
    || /PAID|COMPLET|SUCCE|CAPTURED|ENDED/.test(state);

  if (!clerkId) { console.warn('[safepay webhook] unresolved user (token=%s order=%s)', token, orderId); return; }
  if (!isSuccess) { console.log('[safepay webhook] ignored non-success type=%s state=%s', type, state); return; }

  await activatePaid(clerkId, plan, token || undefined);
  console.log('[safepay webhook] activated Pro for %s (%s)', clerkId, plan);
}

/**
 * Safepay webhook receiver. Verifies the HMAC-SHA512 signature, then on a
 * successful payment sets the student 'active' (current_period_end = +1 period)
 * and marks the payment succeeded. Logs the payload (sandbox) so the exact
 * success/identifier fields can be confirmed from a real event, then tightened.
 */
router.post('/webhook', async (req: Request, res: Response) => {
  const valid = verifyWebhook({ body: req.body, headers: req.headers });
  try {
    console.log('[safepay webhook] valid=%s body=%s', valid, JSON.stringify(req.body).slice(0, 2000));
  } catch { /* ignore log errors */ }
  if (!valid) return res.status(400).json({ error: 'invalid_signature' });
  try {
    await handleSafepayWebhook(req.body as Record<string, unknown>);
  } catch (error) {
    console.error('[safepay webhook] handler error:', error);
  }
  // Always 200 after a valid signature so Safepay doesn't retry-storm; we logged.
  res.status(200).json({ received: true });
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
