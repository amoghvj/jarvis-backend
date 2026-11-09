-- Additive migration: persistent, revocable login sessions.
-- Safe to run more than once. Does not touch existing tables.
--
-- Why it is needed: the existing schema has no place to store sessions. A
-- signed stateless cookie cannot be revoked on logout, and in-memory sessions
-- do not work on Vercel serverless. Only a SHA-256 hash of the session token
-- is stored, never the token itself.

create table if not exists public.sessions (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references public.users(id) on delete cascade,
    token_hash  char(64) not null unique,
    created_at  timestamptz not null default now(),
    expires_at  timestamptz not null
);

create index if not exists idx_sessions_user_id on public.sessions (user_id);
create index if not exists idx_sessions_expires_at on public.sessions (expires_at);

-- Supabase exposes the public schema through its REST API. Enable RLS with no
-- policies so only the backend's direct database role can read this table.
alter table public.sessions enable row level security;

