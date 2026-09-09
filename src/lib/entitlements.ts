/**
 * Propel entitlements — the "is this student allowed to use Pro features?" brain.
 *
 * DESIGN: a master switch (BILLING_ENFORCED) gates the whole thing. While it is
 * OFF (the default), `requirePro` is a pass-through no-op and NOTHING changes for
 * any user — the trial/paywall code is dormant but fully testable. Flip the env
 * var to true (once Safepay is live and existing users are handled) to turn it on.
 *
 * Access is always driven by student_billing (the source of truth), never by the
 * UI. Every paid endpoint calls requirePro so a user can't bypass the paywall by
 * hitting the API directly.
 */
import { Response, NextFunction } from 'express';
import { supabase } from './supabase';
import { AuthenticatedRequest } from './clerkAuth';

const DAY_MS = 86_400_000;

function intEnv(name: string, fallback: number): number {
  const raw = (process.env[name] || '').trim();
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}
function boolEnv(name: string, fallback: boolean): boolean {
  const raw = (process.env[name] || '').trim().toLowerCase();
  if (raw === 'true' || raw === '1' || raw === 'yes') return true;
  if (raw === 'false' || raw === '0' || raw === 'no') return false;
  return fallback;
}

/** Master switch. Default OFF → paywall dormant, app behaves exactly as before. */
export const BILLING_ENFORCED = boolEnv('BILLING_ENFORCED', false);
/** Free-trial length in days. */
export const TRIAL_DAYS = intEnv('BILLING_TRIAL_DAYS', 10);
/** Grace window after a paid month ends before access is cut (the "remind for N days" period). */
export const GRACE_DAYS = intEnv('BILLING_GRACE_DAYS', 2);
/** How many days before expiry to start nudging the user to renew. */
export const RENEWAL_REMINDER_DAYS = intEnv('BILLING_REMINDER_DAYS', 3);
/** Soft one-trial-per-device check on top of one-trial-per-Google-account. */
export const TRIAL_DEVICE_CHECK = boolEnv('BILLING_TRIAL_DEVICE_CHECK', true);
/** Display prices (PKR). Charging isn't wired until Safepay; these drive the UI copy. */
export const PRICE_PKR_MONTHLY = intEnv('BILLING_PRICE_MONTHLY_PKR', 999);
export const PRICE_PKR_ANNUAL = (process.env.BILLING_PRICE_ANNUAL_PKR || '').trim()
  ? intEnv('BILLING_PRICE_ANNUAL_PKR', 0)
  : null;

export type BillingStatus =
  | 'free' | 'trialing' | 'active' | 'past_due' | 'expired' | 'canceled';

export interface BillingRow {
  clerk_id: string;
  status: BillingStatus | string;
  trial_started_at: string | null;
  trial_ends_at: string | null;
  current_period_end: string | null;
  plan: string | null;
  provider: string | null;
  provider_customer: string | null;
  provider_sub_id: string | null;
  auto_renew: boolean;
  last_payment_at: string | null;
  renewal_reminder_sent_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface AccessInfo {
  status: BillingStatus | string;
  isPro: boolean;              // true = full access right now (real state, ignores the master switch)
  trialAvailable: boolean;     // never started a trial → can start one
  daysLeft: number | null;     // days until current access (trial or paid) lapses
  autoRenew: boolean;
  plan: string | null;
  currentPeriodEnd: string | null;
  trialEndsAt: string | null;
}

/**
 * Pure access computation from a billing row. Reflects the TRUE billing state and
 * intentionally ignores BILLING_ENFORCED — enforcement is applied separately (in
 * requirePro on the server, and in the usePro hook on the client) so testing shows
 * the real state.
 */
export function computeAccess(row: BillingRow): AccessInfo {
  const now = Date.now();
  const status = (row.status || 'free') as BillingStatus;
  let isPro = false;
  let accessEnd: number | null = null;

  if (status === 'trialing' && row.trial_ends_at) {
    accessEnd = Date.parse(row.trial_ends_at);
    isPro = now < accessEnd;
  } else if ((status === 'active' || status === 'canceled') && row.current_period_end) {
    accessEnd = Date.parse(row.current_period_end);
    isPro = now < accessEnd;
  } else if (status === 'past_due' && row.current_period_end) {
    // Still full access during the grace window after the paid month ended.
    accessEnd = Date.parse(row.current_period_end) + GRACE_DAYS * DAY_MS;
    isPro = now < accessEnd;
  }

  const trialAvailable = status === 'free' && !row.trial_started_at;
  const daysLeft = accessEnd != null ? Math.max(0, Math.ceil((accessEnd - now) / DAY_MS)) : null;

  return {
    status,
    isPro,
    trialAvailable,
    daysLeft,
    autoRenew: !!row.auto_renew,
    plan: row.plan ?? null,
    currentPeriodEnd: row.current_period_end ?? null,
    trialEndsAt: row.trial_ends_at ?? null,
  };
}

/** Fetch the student's billing row, lazily creating a default 'free' row. */
export async function ensureBilling(clerkId: string): Promise<BillingRow> {
  const existing = await supabase
    .from('student_billing')
    .select('*')
    .eq('clerk_id', clerkId)
    .maybeSingle();

  if (existing.data) return existing.data as BillingRow;

  const created = await supabase
    .from('student_billing')
    .insert({ clerk_id: clerkId, status: 'free' })
    .select('*')
    .single();

  if (created.error) {
    // Race: a concurrent request created it first — just read it back.
    const again = await supabase
      .from('student_billing')
      .select('*')
      .eq('clerk_id', clerkId)
      .maybeSingle();
    if (again.data) return again.data as BillingRow;
    throw created.error;
  }
  return created.data as BillingRow;
}

/**
 * Express middleware: allow only Pro (or trialing) students through.
 * Mirrors requireRole() in clerkAuth.ts. MUST run after clerkAuth (needs req.auth).
 */
export async function requirePro(req: AuthenticatedRequest, res: Response, next: NextFunction) {
  // Master switch off → paywall dormant, everyone passes. App is unchanged.
  if (!BILLING_ENFORCED) return next();

  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const row = await ensureBilling(clerkId);
    const access = computeAccess(row);
    if (access.isPro) return next();

    return res.status(402).json({
      error: 'pro_required',
      status: access.status,
      trialAvailable: access.trialAvailable,
    });
  } catch (error) {
    // Fail OPEN on an unexpected error: a transient DB blip must never lock a
    // paying student out mid-study. The gap is logged; abuse risk is negligible.
    console.error('requirePro error (failing open):', error);
    return next();
  }
}

/**
 * Start the 10-day trial. Idempotent and one-per-account-ever. Optional device
 * hash adds a soft "this device already trialed" block.
 */
export async function startTrialForUser(
  clerkId: string,
  deviceHash?: string,
): Promise<{ started: boolean; reason?: string; access: AccessInfo }> {
  const row = await ensureBilling(clerkId);

  // Already used a trial, or already entitled some other way → no-op.
  if (row.trial_started_at || row.status !== 'free') {
    return {
      started: false,
      reason: row.trial_started_at ? 'trial_already_used' : 'not_eligible',
      access: computeAccess(row),
    };
  }

  // Soft device check: block if THIS device already ran a trial for another account.
  if (TRIAL_DEVICE_CHECK && deviceHash) {
    const dev = await supabase
      .from('trial_devices')
      .select('clerk_id')
      .eq('device_hash', deviceHash)
      .neq('clerk_id', clerkId)
      .limit(1)
      .maybeSingle();
    if (dev.data) {
      return { started: false, reason: 'device_used', access: computeAccess(row) };
    }
  }

  const now = new Date();
  const ends = new Date(now.getTime() + TRIAL_DAYS * DAY_MS);
  const updated = await supabase
    .from('student_billing')
    .update({
      status: 'trialing',
      trial_started_at: now.toISOString(),
      trial_ends_at: ends.toISOString(),
      updated_at: now.toISOString(),
    })
    .eq('clerk_id', clerkId)
    .select('*')
    .single();

  if (deviceHash) {
    await supabase
      .from('trial_devices')
      .upsert({ device_hash: deviceHash, clerk_id: clerkId }, { onConflict: 'device_hash,clerk_id' });
  }

  return { started: true, access: computeAccess((updated.data ?? row) as BillingRow) };
}

/**
 * The renewal engine — run on a schedule (Railway Cron / external cron hitting
 * POST /billing/tick). Advances billing states purely from timestamps:
 *   trialing  → expired   when the trial ends
 *   active    → past_due  when the paid month ends (enters the grace window)
 *   canceled  → expired   at period end (they turned off auto-renew)
 *   past_due  → expired   once the grace window is over
 * Safe to run any time; only mutates student_billing.
 */
export async function tick(): Promise<{
  trialsExpired: number;
  movedToPastDue: number;
  canceledExpired: number;
  graceExpired: number;
}> {
  const now = new Date();
  const nowIso = now.toISOString();
  const graceCutoffIso = new Date(now.getTime() - GRACE_DAYS * DAY_MS).toISOString();

  const trials = await supabase
    .from('student_billing')
    .update({ status: 'expired', updated_at: nowIso })
    .eq('status', 'trialing')
    .lt('trial_ends_at', nowIso)
    .select('clerk_id');

  const pastDue = await supabase
    .from('student_billing')
    .update({ status: 'past_due', updated_at: nowIso })
    .eq('status', 'active')
    .lt('current_period_end', nowIso)
    .select('clerk_id');

  const canceled = await supabase
    .from('student_billing')
    .update({ status: 'expired', updated_at: nowIso })
    .eq('status', 'canceled')
    .lt('current_period_end', nowIso)
    .select('clerk_id');

  const grace = await supabase
    .from('student_billing')
    .update({ status: 'expired', updated_at: nowIso })
    .eq('status', 'past_due')
    .lt('current_period_end', graceCutoffIso)
    .select('clerk_id');

  return {
    trialsExpired: trials.data?.length ?? 0,
    movedToPastDue: pastDue.data?.length ?? 0,
    canceledExpired: canceled.data?.length ?? 0,
    graceExpired: grace.data?.length ?? 0,
  };
}
