-- Test-only bootstrap: the pre-existing production tables that have no migration file in this repo
-- (introspected from the live schema, read-only). It lets the automated tests run against a
-- throwaway local PostgreSQL instead of the shared Supabase database.
-- Deliberately includes the LEGACY global UNIQUE (project_id) so migration 003 is tested for real.

create table if not exists public.users (
    id              uuid primary key default gen_random_uuid(),
    github_id       bigint not null unique,
    github_username text not null,
    email           text,
    avatar_url      text,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create table if not exists public.api_keys (
    id           uuid primary key default gen_random_uuid(),
    user_id      uuid not null unique references public.users(id) on delete cascade,
    key_hash     char(64) not null unique,
    key_prefix   text not null,
    created_at   timestamptz not null default now(),
    last_used_at timestamptz
);

create table if not exists public.projects (
    id         uuid primary key default gen_random_uuid(),
    user_id    uuid not null references public.users(id) on delete cascade,
    project_id text not null,
    name       text not null,
    repository text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint projects_project_id_key unique (project_id)
);
create index if not exists idx_projects_user_id on public.projects (user_id);

create table if not exists public.incidents (
    id            uuid primary key default gen_random_uuid(),
    project_id    uuid not null references public.projects(id) on delete cascade,
    title         text not null,
    status        text not null default 'open',
    severity      text,
    incident_data jsonb not null default '{}'::jsonb,
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);
create index if not exists idx_incidents_project_id on public.incidents (project_id);
create index if not exists idx_incidents_status on public.incidents (status);
create index if not exists idx_incidents_created_at on public.incidents (created_at desc);


-- The live tables have RLS enabled (verified read-only against the production schema).
alter table public.users enable row level security;
alter table public.api_keys enable row level security;
alter table public.projects enable row level security;
alter table public.incidents enable row level security;
