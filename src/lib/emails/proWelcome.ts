/**
 * "Welcome to Propel Pro" — the confirmation email sent when a subscription goes
 * active. Table-based, fully inline-styled HTML for broad email-client support
 * (Gmail, Outlook, Apple Mail), with a plain-text alternative. Brand: crimson
 * #A8123C on cream #FAF6F0, serif display (Georgia fallback for Fraunces).
 */

export interface ProWelcomeInput {
  name?: string | null;
  plan: 'monthly' | 'annual' | 'manual';
  periodEndIso: string;
  amountPkr?: number | null;
  appUrl: string;
  supportEmail: string;
  instagramUrl?: string;
  /** Optional "Payment method" receipt row (e.g. "Safepay"); omitted if unset. */
  paymentMethod?: string;
}

const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

function formatDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}
function money(pkr?: number | null): string | null {
  if (typeof pkr !== 'number' || !Number.isFinite(pkr) || pkr <= 0) return null;
  return `Rs ${Math.round(pkr).toLocaleString('en-US')}`;
}
function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function proWelcomeEmail(input: ProWelcomeInput): { subject: string; html: string; text: string } {
  const firstName = (input.name || '').trim().split(/\s+/)[0] || '';
  const hi = firstName ? `Hi ${esc(firstName)},` : 'Hi there,';
  const planLabel = input.plan === 'annual' ? 'Annual' : input.plan === 'monthly' ? 'Monthly' : '';
  const validUntil = formatDate(input.periodEndIso);
  const amount = money(input.amountPkr);
  const dashboard = `${input.appUrl.replace(/\/+$/, '')}/student/dashboard`;
  const support = input.supportEmail;
  const instagram = input.instagramUrl || 'https://www.instagram.com/propelcambridge/';

  const subject = 'Welcome to Propel Pro 🎉';
  const preheader = 'Your payment went through — Propel Pro is now active on your account.';

  const perks = [
    'Instant marking against the official Cambridge mark scheme',
    'Every mark explained, point by point',
    'Full topical practice and complete past papers',
    'Progress tracking across your weak topics',
  ];

  // ---- receipt rows ----
  const rows: Array<[string, string]> = [['Plan', planLabel ? `${planLabel} — Propel Pro` : 'Propel Pro']];
  if (amount) rows.push(['Amount', amount]);
  if (validUntil) rows.push(['Valid until', validUntil]);
  if (input.paymentMethod) rows.push(['Payment method', input.paymentMethod]);
  const receiptRows = rows
    .map(
      ([k, v], i) => `
        <tr>
          <td style="padding:10px 0;${i ? 'border-top:1px solid #EFE9E1;' : ''}font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#9A8D83;">${esc(k)}</td>
          <td align="right" style="padding:10px 0;${i ? 'border-top:1px solid #EFE9E1;' : ''}font-family:Arial,Helvetica,sans-serif;font-size:13px;font-weight:bold;color:#1C1714;">${esc(v)}</td>
        </tr>`,
    )
    .join('');

  const perkItems = perks
    .map(
      (p) => `
        <tr>
          <td valign="top" style="padding:6px 10px 6px 0;font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.5;color:#A8123C;font-weight:bold;">✓</td>
          <td valign="top" style="padding:6px 0;font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.5;color:#3A322C;">${esc(p)}</td>
        </tr>`,
    )
    .join('');

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="x-apple-disable-message-reformatting">
<title>${subject}</title>
</head>
<body style="margin:0;padding:0;background:#FAF6F0;">
<div style="display:none;max-height:0;overflow:hidden;opacity:0;color:#FAF6F0;font-size:1px;line-height:1px;">${esc(preheader)}</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#FAF6F0;">
  <tr>
    <td align="center" style="padding:28px 16px;">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="width:600px;max-width:100%;background:#FFFFFF;border:1px solid #E8E2D9;border-radius:18px;overflow:hidden;">
        <!-- brand header -->
        <tr>
          <td style="background:#A8123C;padding:26px 32px;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td style="font-family:Georgia,'Times New Roman',serif;font-size:24px;font-weight:bold;letter-spacing:-0.5px;color:#FAF6F0;">Propel</td>
                <td align="right">
                  <span style="display:inline-block;background:rgba(255,255,255,0.16);color:#FFFFFF;font-family:Arial,Helvetica,sans-serif;font-size:11px;font-weight:bold;letter-spacing:1.5px;padding:5px 11px;border-radius:999px;">PRO</span>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        <!-- hero -->
        <tr>
          <td style="padding:36px 32px 8px 32px;">
            <h1 style="margin:0;font-family:Georgia,'Times New Roman',serif;font-size:28px;line-height:1.25;color:#1C1714;font-weight:normal;">You’re Propel&nbsp;Pro now 🎉</h1>
          </td>
        </tr>
        <tr>
          <td style="padding:8px 32px 0 32px;">
            <p style="margin:0 0 14px 0;font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.6;color:#4A413B;">${hi}</p>
            <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.6;color:#4A413B;">Your payment’s gone through and <strong>Propel&nbsp;Pro is active</strong> on your account. Here’s everything you’ve just unlocked:</p>
          </td>
        </tr>
        <!-- perks -->
        <tr>
          <td style="padding:18px 32px 6px 32px;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0">${perkItems}</table>
          </td>
        </tr>
        <!-- CTA -->
        <tr>
          <td style="padding:22px 32px 6px 32px;">
            <table role="presentation" cellpadding="0" cellspacing="0">
              <tr>
                <td align="center" bgcolor="#A8123C" style="border-radius:12px;">
                  <a href="${esc(dashboard)}" target="_blank" style="display:inline-block;padding:14px 30px;font-family:Arial,Helvetica,sans-serif;font-size:15px;font-weight:bold;color:#FFFFFF;text-decoration:none;border-radius:12px;">Start practising →</a>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        <!-- receipt -->
        <tr>
          <td style="padding:20px 32px 8px 32px;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#FAF6F0;border:1px solid #EFE9E1;border-radius:12px;">
              <tr><td style="padding:6px 18px 0 18px;font-family:Arial,Helvetica,sans-serif;font-size:11px;font-weight:bold;letter-spacing:1.2px;color:#A8123C;">SUBSCRIPTION</td></tr>
              <tr>
                <td style="padding:2px 18px 8px 18px;">
                  <table role="presentation" width="100%" cellpadding="0" cellspacing="0">${receiptRows}</table>
                </td>
              </tr>
            </table>
          </td>
        </tr>
        <!-- support -->
        <tr>
          <td style="padding:16px 32px 30px 32px;">
            <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:1.6;color:#6B5F57;">
              Questions about your subscription? Just reply to this email or reach us at
              <a href="mailto:${esc(support)}" style="color:#A8123C;text-decoration:none;">${esc(support)}</a>
              or on Instagram <a href="${esc(instagram)}" target="_blank" style="color:#A8123C;text-decoration:none;">@propelcambridge</a>.
            </p>
          </td>
        </tr>
        <!-- footer -->
        <tr>
          <td style="background:#1C1714;padding:18px 32px;">
            <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:12px;line-height:1.6;color:rgba(255,255,255,0.55);">
              © 2026 Propel Cambridge · You’re receiving this because you subscribed to Propel Pro.
            </p>
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>
</body>
</html>`;

  const textLines = [
    'Welcome to Propel Pro',
    '',
    firstName ? `Hi ${firstName},` : 'Hi there,',
    '',
    'Your payment has gone through and Propel Pro is now active on your account. You’ve unlocked:',
    ...perks.map((p) => `  • ${p}`),
    '',
    `Start practising: ${dashboard}`,
    '',
    'Subscription',
    ...rows.map(([k, v]) => `  ${k}: ${v}`),
    '',
    `Questions? Email ${support} or message @propelcambridge on Instagram.`,
    '',
    '© 2026 Propel Cambridge',
  ];

  return { subject, html, text: textLines.join('\n') };
}
