-- 033_staff_visible_password.sql
-- Store the admin-issued login password for provisioned staff (school-admin /
-- teacher) accounts, ENCRYPTED AT REST (AES-256-GCM — see src/lib/secretBox.ts),
-- so the owner + school-admin panels can display the current credential to hand
-- out. This is only for admin-managed, admin-visible logins the operator already
-- controls; real end-user secrets stay one-way hashed in GoTrue. The column is
-- NULL for existing accounts until their next create or password reset.
alter table public.profiles
  add column if not exists visible_password text;

notify pgrst, 'reload schema';
