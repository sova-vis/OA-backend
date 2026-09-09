-- Propel billing: 10-day free trial + Pro subscription (provider-agnostic, Safepay-ready).
--
-- IMPORTANT: creating these tables changes NOTHING for existing users. Access is
-- only ever restricted when the backend runs with BILLING_ENFORCED=true; while the
-- flag is off (the default) the requirePro middleware is a pass-through no-op.
--
-- Fully idempotent — the migration runner (scripts/run-migrations.js) re-applies
-- every .sql file on each run, so everything here uses IF NOT EXISTS.

-- One billing record per student. Keyed by clerk_id (= the Supabase user UUID),
-- matching profiles.clerk_id so it follows the account across devices/domains.
-- NOTE: no FK to profiles(clerk_id) on purpose — the live profiles table has no
-- unique constraint on clerk_id (it predates these migrations), so an FK can't
-- reference it. clerk_id is this table's own primary key, which is all we need;
-- orphan cleanup on account deletion is handled in the app layer instead.
create table if not exists public.student_billing (
  clerk_id                 text primary key,
  status                   text not null default 'free',   -- free | trialing | active | past_due | expired | canceled
  trial_started_at         timestamptz,
  trial_ends_at            timestamptz,
  current_period_end       timestamptz,                    -- when the current PAID month/year lapses
  plan                     text,                           -- 'monthly' | 'annual'
  provider                 text default 'safepay',
  provider_customer        text,                           -- gateway customer / saved-card token id
  provider_sub_id          text,                           -- gateway subscription id (card auto-renew)
  auto_renew               boolean not null default false, -- true only for saved-card users
  last_payment_at          timestamptz,
  renewal_reminder_sent_at timestamptz,                    -- so the renewal nudge isn't sent twice
  created_at               timestamptz not null default now(),
  updated_at               timestamptz not null default now()
);

create index if not exists idx_student_billing_status      on public.student_billing(status);
create index if not exists idx_student_billing_period_end  on public.student_billing(current_period_end);
create index if not exists idx_student_billing_trial_ends  on public.student_billing(trial_ends_at);

-- Payment audit / receipts. Append-only; never the source of truth for access
-- (that's student_billing), but the bookkeeping record of every charge.
create table if not exists public.payments (
  id             uuid primary key default gen_random_uuid(),
  clerk_id       text not null,
  provider       text default 'safepay',
  provider_tx_id text,
  amount_pkr     integer,
  currency       text default 'PKR',
  status         text,                              -- succeeded | failed | refunded
  method         text,                              -- card | jazzcash | easypaisa | bank
  plan           text,
  period_start   timestamptz,
  period_end     timestamptz,
  created_at     timestamptz not null default now()
);

create index if not exists idx_payments_clerk   on public.payments(clerk_id);
create index if not exists idx_payments_created  on public.payments(created_at);

-- One-trial-per-device signal (SOFT anti-abuse). The STRONG signal is that login
-- is Google-only, so it's already one trial per verified Google account. This adds
-- a second layer: a device that already burned a trial won't hand a brand-new
-- account a fresh 10 days. Best-effort only (a determined user can reset it).
create table if not exists public.trial_devices (
  device_hash text not null,
  clerk_id    text not null,
  created_at  timestamptz not null default now(),
  primary key (device_hash, clerk_id)
);

create index if not exists idx_trial_devices_hash on public.trial_devices(device_hash);

-- Reload PostgREST's schema cache so the new tables are queryable immediately.
notify pgrst, 'reload schema';
