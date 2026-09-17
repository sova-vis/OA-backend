/**
 * Propel transactional email — provider-agnostic and best-effort.
 *
 * Priority:  Resend (HTTPS API, no extra dependency)  →  SMTP via nodemailer
 * (Gmail app-password or any host)  →  no-op. `sendEmail` NEVER throws: a mail
 * failure must never break a payment/activation flow.
 *
 * Turn it on by setting env on the backend (Railway) — no code change:
 *   Perfect  : RESEND_API_KEY=…            EMAIL_FROM="Propel <noreply@propelcambridge.com>"
 *   Right now: GMAIL_USER=…  GMAIL_APP_PASSWORD=…   EMAIL_FROM="Propel <you@gmail.com>"
 *   Any SMTP : SMTP_HOST=… SMTP_PORT=… SMTP_USER=… SMTP_PASS=… [SMTP_SECURE=true]
 *   Kill sw. : EMAILS_ENABLED=false
 */

export type SendResult = { ok: boolean; provider: string; id?: string; skipped?: boolean; error?: string };

const DEFAULT_FROM = 'Propel <noreply@propelcambridge.com>';

function fromAddress(): string {
  return (process.env.EMAIL_FROM || DEFAULT_FROM).trim();
}
function emailsDisabled(): boolean {
  const v = (process.env.EMAILS_ENABLED || '').trim().toLowerCase();
  return v === 'false' || v === '0' || v === 'no';
}
function resendKey(): string {
  return (process.env.RESEND_API_KEY || '').trim();
}

type Smtp = { host: string; port: number; secure: boolean; user: string; pass: string };
function smtpConfig(): Smtp | null {
  // Gmail shortcut: just an address + app password.
  const gUser = (process.env.GMAIL_USER || '').trim();
  const gPass = (process.env.GMAIL_APP_PASSWORD || '').trim();
  if (gUser && gPass) return { host: 'smtp.gmail.com', port: 465, secure: true, user: gUser, pass: gPass };

  const host = (process.env.SMTP_HOST || '').trim();
  const user = (process.env.SMTP_USER || '').trim();
  const pass = (process.env.SMTP_PASS || '').trim();
  if (host && user && pass) {
    const port = Number.parseInt(process.env.SMTP_PORT || '587', 10) || 587;
    const secure = (process.env.SMTP_SECURE || '').trim().toLowerCase() === 'true' || port === 465;
    return { host, port, secure, user, pass };
  }
  return null;
}

/** Which provider will actually be used (for /health + diagnostics). */
export function emailProvider(): 'disabled' | 'resend' | 'smtp' | 'none' {
  if (emailsDisabled()) return 'disabled';
  if (resendKey()) return 'resend';
  if (smtpConfig()) return 'smtp';
  return 'none';
}
export function emailConfigured(): boolean {
  const p = emailProvider();
  return p === 'resend' || p === 'smtp';
}

/** Send one email. Resolves with a result object; never rejects. */
export async function sendEmail(opts: {
  to: string;
  subject: string;
  html: string;
  text?: string;
  from?: string;
}): Promise<SendResult> {
  const from = (opts.from || fromAddress()).trim();
  if (emailsDisabled()) return { ok: false, provider: 'disabled', skipped: true };

  try {
    const key = resendKey();
    if (key) {
      const res = await fetch('https://api.resend.com/emails', {
        method: 'POST',
        headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ from, to: [opts.to], subject: opts.subject, html: opts.html, text: opts.text }),
        signal: AbortSignal.timeout(15_000),
      });
      const body = (await res.json().catch(() => ({}))) as { id?: string; message?: string };
      if (!res.ok) return { ok: false, provider: 'resend', error: `HTTP ${res.status}: ${body?.message || JSON.stringify(body).slice(0, 200)}` };
      return { ok: true, provider: 'resend', id: body?.id };
    }

    const smtp = smtpConfig();
    if (smtp) {
      // Lazy require so the Resend path carries no dependency, and a missing
      // nodemailer install degrades to a logged skip instead of a crash.
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const nodemailer = require('nodemailer') as typeof import('nodemailer');
      const transport = nodemailer.createTransport({
        host: smtp.host, port: smtp.port, secure: smtp.secure,
        auth: { user: smtp.user, pass: smtp.pass },
      });
      const info = await transport.sendMail({ from, to: opts.to, subject: opts.subject, html: opts.html, text: opts.text });
      return { ok: true, provider: 'smtp', id: info?.messageId };
    }

    return { ok: false, provider: 'none', skipped: true };
  } catch (error) {
    return { ok: false, provider: emailProvider(), error: String((error as Error)?.message || error) };
  }
}
