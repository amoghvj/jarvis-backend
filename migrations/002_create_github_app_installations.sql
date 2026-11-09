-- Additive migration: GitHub App installations and short-lived pending installation flows.
-- Safe to run more than once. Does not modify existing tables.

create table if not exists public.github_app_installations (
    id                   uuid primary key default gen_random_uuid(),
    user_id              uuid not null references public.users(id) on delete cascade,
    installation_id      bigint not null unique,
    github_account_id    bigint not null,
    github_account_login text not null,
    github_account_type  text not null,
    repository_selection text,
    permissions          jsonb not null default '{}'::jsonb,
    suspended_at         timestamptz,
    created_at           timestamptz not null default now(),
    updated_at           timestamptz not null default now()
);

create index if not exists idx_github_app_installations_user_id
    on public.github_app_installations (user_id);

create index if not exists idx_github_app_installations_account_id
    on public.github_app_installations (github_account_id);

alter table public.github_app_installations enable row level security;

-- Short-lived, single-use pending installation / OAuth state records.
-- Only SHA-256 hashes of flow/state tokens are stored.
create table if not exists public.github_app_pending_installations (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid not null references public.users(id) on delete cascade,
    installation_id bigint,
    setup_action    text,
    stage           text not null default 'oauth',
    state_hash      char(64) not null unique,
    consumed_at     timestamptz,
    expires_at      timestamptz not null,
    created_at      timestamptz not null default now()
);

create index if not exists idx_github_app_pending_user_id
    on public.github_app_pending_installations (user_id);

create index if not exists idx_github_app_pending_expires_at
    on public.github_app_pending_installations (expires_at);

alter table public.github_app_pending_installations enable row level security;

