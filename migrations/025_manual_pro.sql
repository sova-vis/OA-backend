-- 025_manual_pro.sql
-- Manual Pro activation flow (temporary, while Safepay live keys are pending).
-- Students submit a manual payment request (they pay via a QR shown on-page);
-- an admin reviews it and activates a 30-day Pro period. Promo codes swap the QR.
-- Additive and idempotent (safe to re-run).

-- ---------------------------------------------------------------------------
-- Manual payment requests raised by students after they click "I've paid".
-- ---------------------------------------------------------------------------
create table if not exists public.pro_requests (
  id uuid primary key default gen_random_uuid(),
  clerk_id text not null,
  name text,
  email text not null,
  phone text,
  cardholder_name text,          -- name on the card/account they paid from
  promo_code text,               -- promo code applied (if any), uppercased
  plan text not null default 'manual',
  amount_pkr integer,            -- amount shown to the student
  status text not null default 'pending',   -- pending | approved | rejected
  note text,                     -- optional admin note
  created_at timestamptz not null default now(),
  reviewed_at timestamptz,
  reviewed_by text               -- admin clerk_id who actioned it
);
create index if not exists idx_pro_requests_status on public.pro_requests(status);
create index if not exists idx_pro_requests_clerk on public.pro_requests(clerk_id);
create index if not exists idx_pro_requests_created on public.pro_requests(created_at desc);

-- ---------------------------------------------------------------------------
-- Promo codes, each with its own QR image (base64 data URL) + display note.
-- ---------------------------------------------------------------------------
create table if not exists public.promo_codes (
  id uuid primary key default gen_random_uuid(),
  code text not null unique,     -- stored UPPERCASE
  label text,                    -- admin-facing name, e.g. "Eid 20% off"
  note text,                     -- shown to the student on the pay page
  amount_pkr integer,            -- optional discounted price to display
  qr_image text,                 -- base64 data URL of the promo-specific QR
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create index if not exists idx_promo_codes_code on public.promo_codes(code);

-- ---------------------------------------------------------------------------
-- Single-row general payment config: the default QR + payee details the admin
-- sets, shown when no (or an invalid) promo code is applied.
-- ---------------------------------------------------------------------------
create table if not exists public.pay_config (
  id smallint primary key default 1 check (id = 1),
  qr_image text,                 -- base64 data URL of the general QR
  payee_name text,               -- account / card holder name
  account_number text,           -- IBAN / account / wallet number
  bank_name text,
  instructions text,             -- free-text instructions shown to the student
  amount_pkr integer,            -- base price to display
  updated_at timestamptz not null default now(),
  updated_by text
);
insert into public.pay_config (id) values (1) on conflict (id) do nothing;

notify pgrst, 'reload schema';
