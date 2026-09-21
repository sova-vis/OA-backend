/**
 * Shared "Pro is active" notification — used by both the Safepay webhook and the
 * admin manual activation. Best-effort: never throws, no-op until a mail provider
 * is configured (see lib/mailer.ts).
 */
import { supabase } from './supabase';
import { sendEmail, emailProvider } from './mailer';
import { proWelcomeEmail } from './emails/proWelcome';
import { proPendingEmail } from './emails/proPending';

const SUPPORT_EMAIL = (process.env.SUPPORT_EMAIL || 'sovavis2025@gmail.com').trim();

export function appBaseUrl(): string {
  const raw = (process.env.APP_URL || process.env.FRONTEND_URL || 'https://oalevels.vercel.app')
    .split(',')[0].trim();
  return raw.replace(/\/+$/, '');
}

export async function sendProWelcome(clerkId: string, opts: {
  plan: 'monthly' | 'annual' | 'manual';
  periodEndIso: string;
  amountPkr?: number | null;
  paymentMethod?: string;
}): Promise<void> {
  const provider = emailProvider();
  if (provider === 'none' || provider === 'disabled') return; // no provider wired yet
  try {
    const prof = await supabase.from('profiles').select('email, full_name').eq('clerk_id', clerkId).maybeSingle();
    const to = typeof prof.data?.email === 'string' ? prof.data.email.trim() : '';
    if (!to) { console.warn('[pro welcome] no email on profile for %s — skipped', clerkId); return; }
    const { subject, html, text } = proWelcomeEmail({
      name: (prof.data?.full_name as string | null) ?? null,
      plan: opts.plan,
      periodEndIso: opts.periodEndIso,
      amountPkr: opts.amountPkr ?? null,
      appUrl: appBaseUrl(),
      supportEmail: SUPPORT_EMAIL,
      paymentMethod: opts.paymentMethod,
    });
    const result = await sendEmail({ to, subject, html, text });
    if (result.ok) console.log('[pro welcome] sent to %s via %s (id=%s)', to, result.provider, result.id || '—');
    else console.warn('[pro welcome] NOT sent to %s: %s', to, result.error || (result.skipped ? 'skipped' : 'unknown'));
  } catch (error) {
    console.warn('[pro welcome] error:', (error as Error)?.message || error);
  }
}

/** "We've received your payment — verifying" email, sent when a student confirms they've paid. */
export async function sendProPending(clerkId: string, opts: {
  amountPkr?: number | null;
  promoCode?: string | null;
}): Promise<void> {
  const provider = emailProvider();
  if (provider === 'none' || provider === 'disabled') return;
  try {
    const prof = await supabase.from('profiles').select('email, full_name').eq('clerk_id', clerkId).maybeSingle();
    const to = typeof prof.data?.email === 'string' ? prof.data.email.trim() : '';
    if (!to) { console.warn('[pro pending] no email on profile for %s — skipped', clerkId); return; }
    const { subject, html, text } = proPendingEmail({
      name: (prof.data?.full_name as string | null) ?? null,
      amountPkr: opts.amountPkr ?? null,
      promoCode: opts.promoCode ?? null,
      supportEmail: SUPPORT_EMAIL,
    });
    const result = await sendEmail({ to, subject, html, text });
    if (result.ok) console.log('[pro pending] sent to %s via %s (id=%s)', to, result.provider, result.id || '—');
    else console.warn('[pro pending] NOT sent to %s: %s', to, result.error || (result.skipped ? 'skipped' : 'unknown'));
  } catch (error) {
    console.warn('[pro pending] error:', (error as Error)?.message || error);
  }
}
