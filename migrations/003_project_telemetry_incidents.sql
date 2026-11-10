-- Migration 003: project-scoped telemetry, repository discovery and incident intelligence.
--
-- NOT applied automatically. Review it, then apply manually (see README, "Applying migration 003").
--
-- Properties
--   * Non-destructive: drops no table or column, deletes and rewrites no rows. The only thing it drops is the
--     old global UNIQUE (project_id) on projects, and only after its replacement UNIQUE (user_id, project_id)
--     exists (the new rule is strictly weaker, so any data valid before is valid after).
--   * Idempotent: safe to run any number of times. Statements that touch EXISTING tables (projects, incidents)
--     are guarded, so a repeat run takes no table locks and changes nothing.
--   * Fails clearly and early if its prerequisites are missing (see the first block), before changing anything.
--   * No BEGIN/COMMIT on purpose: run it as ONE transaction (psql --single-transaction, or the Supabase SQL
--     editor, which sends the script as a single request). If it fails part-way, nothing is applied.
--
-- Prerequisites: public.users, public.projects, public.incidents (existing application tables) and
-- public.github_app_installations (migration 002, with its UNIQUE installation_id).
-- Run it as the table owner (the `postgres` role in Supabase).
--
-- What it does
--   1. projects: replaces the global UNIQUE (project_id) with UNIQUE (user_id, project_id).
--   2. incidents: adds grouping/lifecycle columns, a status check and lookup indexes.
--   3. telemetry_events: persisted SDK events (idempotent on project_id + event_id; classification is
--      constrained to normal_behaviour / abnormal_behaviour).
--   4. project_repositories: repositories discovered from telemetry, with explicit verification state.
--   5. incident_history: recorded lifecycle transitions and severity escalations.
-- Every new table has RLS enabled with no policies (only the backend's direct DB role, which bypasses RLS,
-- can read it), exactly like the existing tables.
--
-- Note: CREATE INDEX here is not CONCURRENTLY (that cannot run inside a transaction). The affected tables are
-- new or small at the time of the first run; apply during a quiet period.

-- 0. Prerequisites ---------------------------------------------------------------------------------
do $$
declare
    missing text;
begin
    select string_agg(t, ', ' order by t) into missing
    from unnest(array['users', 'projects', 'incidents', 'github_app_installations']) as t
    where to_regclass('public.' || t) is null;
    if missing is not null then
        raise exception 'Migration 003 prerequisites are missing: public.% (apply the earlier migrations first)', missing;
    end if;

    -- project_repositories.installation_id references github_app_installations(installation_id), which
    -- therefore must be unique.
    if not exists (
        select 1 from pg_constraint
        where conrelid = 'public.github_app_installations'::regclass
          and contype in ('u', 'p')
          and conkey = array[(select attnum from pg_attribute
                              where attrelid = 'public.github_app_installations'::regclass
                                and attname = 'installation_id')]
    ) then
        raise exception 'public.github_app_installations.installation_id must be UNIQUE (migration 002)';
    end if;
end $$;

-- 1. Per-user project IDs ------------------------------------------------------------------------
do $$
declare
    old_name text;
begin
    if not exists (
        select 1 from pg_constraint
        where conrelid = 'public.projects'::regclass and conname = 'projects_user_id_project_id_key'
    ) then
        alter table public.projects
            add constraint projects_user_id_project_id_key unique (user_id, project_id);
    end if;
    -- Drop the old single-column unique constraint on project_id (found by definition, not by name, so a
    -- renamed constraint is still handled). Only after the replacement exists, so uniqueness is never absent.
    for old_name in
        select conname from pg_constraint
        where conrelid = 'public.projects'::regclass
          and contype = 'u'
          and conkey = array[(select attnum from pg_attribute
                              where attrelid = 'public.projects'::regclass and attname = 'project_id')]
    loop
        execute format('alter table public.projects drop constraint %I', old_name);
    end loop;
end $$;

-- 2. Incident grouping and lifecycle columns -----------------------------------------------------
do $$
begin
    if (
        select count(*) from information_schema.columns
        where table_schema = 'public' and table_name = 'incidents'
          and column_name in ('fingerprint', 'service_id', 'environment', 'first_seen_at', 'last_seen_at',
                              'event_count', 'acknowledged_at', 'resolved_at')
    ) < 8 then
        alter table public.incidents
            add column if not exists fingerprint     text,
            add column if not exists service_id      text,
            add column if not exists environment     text,
            add column if not exists first_seen_at   timestamptz,
            add column if not exists last_seen_at    timestamptz,
            add column if not exists event_count     integer not null default 0,
            add column if not exists acknowledged_at timestamptz,
            add column if not exists resolved_at     timestamptz;
    end if;
    if not exists (
        select 1 from pg_constraint
        where conrelid = 'public.incidents'::regclass and conname = 'incidents_status_check'
    ) then
        -- NOT VALID: enforced for all new/updated rows without scanning (or failing on) old rows.
        alter table public.incidents
            add constraint incidents_status_check
            check (status in ('open', 'acknowledged', 'resolved')) not valid;
    end if;
end $$;

create index if not exists idx_incidents_project_fingerprint
    on public.incidents (project_id, fingerprint, status, last_seen_at desc);
create index if not exists idx_incidents_project_last_seen
    on public.incidents (project_id, last_seen_at desc);

-- 3. Telemetry events ----------------------------------------------------------------------------
create table if not exists public.telemetry_events (
    id               uuid primary key default gen_random_uuid(),
    project_id       uuid not null references public.projects(id) on delete cascade,
    event_id         text not null,
    schema_version   text,
    event_type       text not null,
    classification   text not null,
    event_timestamp  timestamptz not null,
    received_at      timestamptz not null default now(),
    application      jsonb not null default '{}'::jsonb,
    service_id       text,
    environment      text,
    instance_id      text,
    runtime          jsonb not null default '{}'::jsonb,
    severity         text,
    message          text,
    logger_name      text,
    error_type       text,
    stack_trace      text,
    source_file      text,
    source_function  text,
    source_line      integer,
    trace_id         text,
    span_id          text,
    request_id       text,
    thread_name      text,
    attributes       jsonb not null default '{}'::jsonb,
    payload          jsonb not null,
    incident_id      uuid references public.incidents(id) on delete set null,
    constraint telemetry_events_project_event_key unique (project_id, event_id),
    constraint telemetry_events_classification_check
        check (classification in ('normal_behaviour', 'abnormal_behaviour'))
);

-- If the table already existed from an earlier draft of this migration it lacks the check above (CREATE TABLE IF
-- NOT EXISTS skips it). Add it NOT VALID so it applies to all new rows without failing on legacy ones; run
-- `alter table public.telemetry_events validate constraint telemetry_events_classification_check;` afterwards.
do $$
begin
    if not exists (
        select 1 from pg_constraint
        where conrelid = 'public.telemetry_events'::regclass and conname = 'telemetry_events_classification_check'
    ) then
        alter table public.telemetry_events
            add constraint telemetry_events_classification_check
            check (classification in ('normal_behaviour', 'abnormal_behaviour')) not valid;
    end if;
end $$;

create index if not exists idx_telemetry_events_project_time
    on public.telemetry_events (project_id, event_timestamp desc, id desc);
create index if not exists idx_telemetry_events_incident
    on public.telemetry_events (incident_id, event_timestamp, id) where incident_id is not null;
create index if not exists idx_telemetry_events_trace
    on public.telemetry_events (project_id, trace_id, event_timestamp) where trace_id is not null;
create index if not exists idx_telemetry_events_request
    on public.telemetry_events (project_id, request_id, event_timestamp) where request_id is not null;
create index if not exists idx_telemetry_events_service
    on public.telemetry_events (project_id, service_id, environment, event_timestamp desc);
create index if not exists idx_telemetry_events_severity
    on public.telemetry_events (project_id, severity, event_timestamp desc);
create index if not exists idx_telemetry_events_type
    on public.telemetry_events (project_id, event_type, event_timestamp desc);
create index if not exists idx_telemetry_events_error_type
    on public.telemetry_events (project_id, error_type, event_timestamp desc) where error_type is not null;

alter table public.telemetry_events enable row level security;

-- 4. Repositories discovered from telemetry ------------------------------------------------------
-- service_id is '' (not NULL) when unknown so the uniqueness rule below also covers that case.
-- Rows are UNVERIFIED metadata until the repository is found in one of the owning user's GitHub App
-- installations. installation_id is only ever set by that verification, never copied from telemetry.
create table if not exists public.project_repositories (
    id                    uuid primary key default gen_random_uuid(),
    project_id            uuid not null references public.projects(id) on delete cascade,
    service_id            text not null default '',
    repository_url        text not null,
    owner                 text,
    repo                  text,
    installation_id       bigint references public.github_app_installations(installation_id) on delete set null,
    default_branch        text,
    verification_status   text not null default 'unverified'
        check (verification_status in ('unverified', 'verified', 'unsupported_url')),
    verification_note     text,
    discovery_source      text not null default 'startup_event',
    first_seen_at         timestamptz not null default now(),
    last_seen_at          timestamptz not null default now(),
    last_verification_at  timestamptz,
    verified_at           timestamptz,
    created_at            timestamptz not null default now(),
    updated_at            timestamptz not null default now(),
    constraint project_repositories_unique unique (project_id, service_id, repository_url),
    -- A verified row must at least name its owner and repo. installation_id is intentionally NOT
    -- required here: deleting a GitHub App installation sets it to NULL (the existing disconnect
    -- flow must keep working), and every read path re-checks the installation instead of trusting
    -- this column (see verified_repositories / list_project_repositories).
    constraint project_repositories_verified_complete check (
        verification_status <> 'verified' or (owner is not null and repo is not null)
    )
);

create index if not exists idx_project_repositories_project
    on public.project_repositories (project_id, verification_status);
create index if not exists idx_project_repositories_installation
    on public.project_repositories (installation_id) where installation_id is not null;

alter table public.project_repositories enable row level security;

-- 5. Incident history ----------------------------------------------------------------------------
create table if not exists public.incident_history (
    id             uuid primary key default gen_random_uuid(),
    incident_id    uuid not null references public.incidents(id) on delete cascade,
    project_id     uuid not null references public.projects(id) on delete cascade,
    change_type    text not null check (change_type in ('created', 'status_changed', 'severity_escalated')),
    from_status    text,
    to_status      text,
    from_severity  text,
    to_severity    text,
    actor_user_id  uuid references public.users(id) on delete set null,
    note           text,
    -- clock_timestamp() (not now()) so rows written in one transaction still differ; seq is the
    -- deterministic tiebreaker for display order.
    created_at     timestamptz not null default clock_timestamp(),
    seq            bigint generated always as identity
);

create index if not exists idx_incident_history_incident
    on public.incident_history (incident_id, created_at, seq);
create index if not exists idx_incident_history_project
    on public.incident_history (project_id, created_at desc);

alter table public.incident_history enable row level security;
