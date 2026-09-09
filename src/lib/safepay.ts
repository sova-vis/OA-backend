/**
 * Safepay integration helper (Pakistan gateway). Wraps the official @sfpy/node-sdk.
 *
 * Keys come from env ONLY (never commit them):
 *   SAFEPAY_ENV            'sandbox' | 'production'   (default 'sandbox')
 *   SAFEPAY_API_KEY        the "Public key" (sec_...) — identifies the merchant
 *   SAFEPAY_V1_SECRET      the "Secret key" (hex) — signs/verifies transactions
 *   SAFEPAY_WEBHOOK_SECRET from Developer -> Endpoints (per registered webhook)
 *
 * One-time checkout works for cards + JazzCash + Easypaisa. Subscriptions
 * (card auto-renew) need a Plan created in the Safepay dashboard (planId) — added
 * later; the reminder + re-pay flow covers wallet users regardless.
 */
import { Safepay } from '@sfpy/node-sdk';

const ENV = (process.env.SAFEPAY_ENV || 'sandbox').trim().toLowerCase();
const API_KEY = (process.env.SAFEPAY_API_KEY || '').trim();
const V1_SECRET = (process.env.SAFEPAY_V1_SECRET || '').trim();
const WEBHOOK_SECRET = (process.env.SAFEPAY_WEBHOOK_SECRET || '').trim();

/** True once the keys needed to START a checkout are present. */
export const SAFEPAY_CONFIGURED = !!(API_KEY && V1_SECRET);
/** True once we can also verify incoming webhooks. */
export const SAFEPAY_WEBHOOK_READY = !!WEBHOOK_SECRET;
export const SAFEPAY_ENV = ENV;

let client: Safepay | null = null;
function sp(): Safepay {
  if (!client) {
    client = new Safepay({
      environment: ENV as never, // 'sandbox' | 'production'
      apiKey: API_KEY,
      v1Secret: V1_SECRET,
      webhookSecret: WEBHOOK_SECRET,
    });
  }
  return client;
}

export interface CheckoutParams {
  amountPkr: number;
  orderId: string;
  redirectUrl: string;
  cancelUrl: string;
}

/**
 * Create a one-time hosted-checkout session. Returns the tracker `token` (store it
 * so the webhook can map back to the user) and the `url` to redirect the buyer to.
 *
 * NOTE on amount units: Safepay's /order/v1/init takes the amount in the currency's
 * MAJOR unit for PKR (rupees), e.g. 999 = Rs 999. Verify on the sandbox checkout
 * page on first run and adjust if it ever shows ×100.
 */
export async function createCheckout(params: CheckoutParams): Promise<{ token: string; url: string }> {
  const { token } = await sp().payments.create({ amount: params.amountPkr, currency: 'PKR' });
  const url = sp().checkout.create({
    token,
    orderId: params.orderId,
    redirectUrl: params.redirectUrl,
    cancelUrl: params.cancelUrl,
    source: 'custom',
    webhooks: true,
  });
  return { token, url };
}

/** Verify a Safepay webhook (HMAC-SHA512 of body.data with the webhook secret). */
export function verifyWebhook(req: { body?: unknown; headers?: unknown }): boolean {
  if (!WEBHOOK_SECRET) return false;
  try {
    return sp().verify.webhook({ body: req.body as never, headers: req.headers as never });
  } catch (error) {
    console.error('[safepay] webhook verify threw:', error);
    return false;
  }
}
