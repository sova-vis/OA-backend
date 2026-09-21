/**
 * "We've received your payment — verifying" email, sent when a student confirms
 * they've paid (raises a manual Pro request). Table-based inline-styled HTML for
 * broad client support, plus a plain-text alternative. Brand: crimson on cream.
 */

export interface ProPendingInput {
  name?: string | null;
  amountPkr?: number | null;
  promoCode?: string | null;
  supportEmail: string;
  instagramUrl?: string;
}

function money(pkr?: number | null): string | null {
  if (typeof pkr !== 'number' || !Number.isFinite(pkr) || pkr <= 0) return null;
  return `Rs ${Math.round(pkr).toLocaleString('en-US')}`;
}
function esc(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function proPendingEmail(input: ProPendingInput): { subject: string; html: string; text: string } {
  const firstName = (input.name || '').trim().split(/\s+/)[0] || '';
  const hi = firstName ? `Hi ${esc(firstName)},` : 'Hi there,';
  const amount = money(input.amountPkr);
  const support = input.supportEmail;
  const instagram = input.instagramUrl || 'https://www.instagram.com/propelcambridge/';

  const subject = 'We’ve received your Propel Pro payment ⏳';
  const preheader = 'Thanks! We’re verifying your payment and will activate Pro shortly.';

  const rows: Array<[string, string]> = [['Plan', 'Propel Pro — 30 days']];
  if (amount) rows.push(['Amount', amount]);
  if (input.promoCode) rows.push(['Promo code', input.promoCode]);
  rows.push(['Status', 'Verifying your payment']);

  const receiptRows = rows
    .map(([k, v], i) => `
        <tr>
          <td style="padding:10px 0;${i ? 'border-top:1px solid #EFE9E1;' : ''}font-family:Arial,Helvetica,sans-serif;font-size:13px;color:#9A8D83;">${esc(k)}</td>
          <td align="right" style="padding:10px 0;${i ? 'border-top:1px solid #EFE9E1;' : ''}font-family:Arial,Helvetica,sans-serif;font-size:13px;font-weight:bold;color:#1C1714;">${esc(v)}</td>
        </tr>`)
    .join('');

  const steps = [
    'We verify your payment (usually within a short while).',
    'We activate 30 days of Propel Pro on your account.',
    'You get a confirmation email — and you’re all set.',
  ];
  const stepItems = steps
    .map((s, i) => `
        <tr>
          <td valign="top" style="padding:6px 12px 6px 0;font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.5;color:#A8123C;font-weight:bold;">${i + 1}.</td>
          <td valign="top" style="padding:6px 0;font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.5;color:#3A322C;">${esc(s)}</td>
        </tr>`)
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
        <tr>
          <td style="background:#A8123C;padding:26px 32px;">
            <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td style="font-family:Georgia,'Times New Roman',serif;font-size:24px;font-weight:bold;letter-spacing:-0.5px;color:#FAF6F0;">Propel</td>
                <td align="right"><span style="display:inline-block;background:rgba(255,255,255,0.16);color:#FFFFFF;font-family:Arial,Helvetica,sans-serif;font-size:11px;font-weight:bold;letter-spacing:1.5px;padding:5px 11px;border-radius:999px;">PENDING</span></td>
              </tr>
            </table>
          </td>
        </tr>
        <tr><td style="padding:36px 32px 8px 32px;">
          <h1 style="margin:0;font-family:Georgia,'Times New Roman',serif;font-size:27px;line-height:1.25;color:#1C1714;font-weight:normal;">Thanks — we’re verifying your payment ⏳</h1>
        </td></tr>
        <tr><td style="padding:8px 32px 0 32px;">
          <p style="margin:0 0 14px 0;font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.6;color:#4A413B;">${hi}</p>
          <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.6;color:#4A413B;">We’ve received your request to upgrade to <strong>Propel Pro</strong> and we’re verifying your payment now. As soon as it’s confirmed, we’ll activate your account and email you again — you’ll be all set.</p>
        </td></tr>
        <tr><td style="padding:18px 32px 6px 32px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0">${stepItems}</table>
        </td></tr>
        <tr><td style="padding:14px 32px 8px 32px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#FAF6F0;border:1px solid #EFE9E1;border-radius:12px;">
            <tr><td style="padding:6px 18px 0 18px;font-family:Arial,Helvetica,sans-serif;font-size:11px;font-weight:bold;letter-spacing:1.2px;color:#A8123C;">YOUR REQUEST</td></tr>
            <tr><td style="padding:2px 18px 8px 18px;"><table role="presentation" width="100%" cellpadding="0" cellspacing="0">${receiptRows}</table></td></tr>
          </table>
        </td></tr>
        <tr><td style="padding:16px 32px 30px 32px;">
          <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:13px;line-height:1.6;color:#6B5F57;">
            Didn’t pay yet, or need help? Just reply to this email or reach us at
            <a href="mailto:${esc(support)}" style="color:#A8123C;text-decoration:none;">${esc(support)}</a>
            or on Instagram <a href="${esc(instagram)}" target="_blank" style="color:#A8123C;text-decoration:none;">@propelcambridge</a>.
          </p>
        </td></tr>
        <tr><td style="background:#1C1714;padding:18px 32px;">
          <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:12px;line-height:1.6;color:rgba(255,255,255,0.55);">© 2026 Propel Cambridge · You’re receiving this because you requested Propel Pro.</p>
        </td></tr>
      </table>
    </td>
  </tr>
</table>
</body>
</html>`;

  const textLines = [
    'We’ve received your Propel Pro payment',
    '',
    firstName ? `Hi ${firstName},` : 'Hi there,',
    '',
    'We’ve received your request to upgrade to Propel Pro and we’re verifying your payment now. As soon as it’s confirmed we’ll activate your account and email you again.',
    '',
    'Your request',
    ...rows.map(([k, v]) => `  ${k}: ${v}`),
    '',
    `Questions? Email ${support} or message @propelcambridge on Instagram.`,
    '',
    '© 2026 Propel Cambridge',
  ];

  return { subject, html, text: textLines.join('\n') };
}
