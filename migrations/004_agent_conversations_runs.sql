-- Migration 004: Persistent project-scoped conversational investigation agent
-- Adds conversations, conversation_messages, agent_runs, and agent_run_steps.
-- Fully idempotent and safe to re-run on existing databases.
--
-- Apply manually in the Supabase SQL Editor (or via psql against DATABASE_URL).

-- ============================================================================
-- 1. CONVERSATIONS
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    project_id      UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    title           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_message_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_project_last_msg
    ON public.conversations (user_id, project_id, last_message_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_project_id
    ON public.conversations (project_id);

ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- 2. CONVERSATION MESSAGES
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.conversation_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
    sequence_number BIGINT NOT NULL,
    role            TEXT NOT NULL,
    message_type    TEXT NOT NULL,
    content         TEXT,
    tool_calls      JSONB,
    tool_call_id    TEXT,
    tool_name       TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    run_id          UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT conversation_messages_id_conversation_unique
        UNIQUE (id, conversation_id),
    CONSTRAINT conversation_messages_sequence_unique
        UNIQUE (conversation_id, sequence_number),
    CONSTRAINT conversation_messages_sequence_positive_check
        CHECK (sequence_number >= 1),
    CONSTRAINT conversation_messages_role_check
        CHECK (role IN ('user', 'assistant', 'tool')),
    CONSTRAINT conversation_messages_type_check
        CHECK (message_type IN ('text', 'tool_call', 'tool_result'))
);

-- Ensure (id, conversation_id) unique constraint exists if table was created earlier.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'conversation_messages_id_conversation_unique'
          AND conrelid = 'public.conversation_messages'::regclass
    ) THEN
        ALTER TABLE public.conversation_messages
            ADD CONSTRAINT conversation_messages_id_conversation_unique
            UNIQUE (id, conversation_id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_conversation_messages_conv_seq
    ON public.conversation_messages (conversation_id, sequence_number ASC);

CREATE INDEX IF NOT EXISTS idx_conversation_messages_run_id
    ON public.conversation_messages (run_id)
    WHERE run_id IS NOT NULL;

ALTER TABLE public.conversation_messages ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- 3. AGENT RUNS (Durable background execution & lease queue)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.agent_runs (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id   UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
    user_message_id   UUID NOT NULL,
    status            TEXT NOT NULL DEFAULT 'queued',
    error_message     TEXT,
    client_request_id TEXT,
    worker_id         TEXT,
    claimed_at        TIMESTAMPTZ,
    lease_expires_at  TIMESTAMPTZ,
    heartbeat_at      TIMESTAMPTZ,
    attempt_count     INTEGER NOT NULL DEFAULT 0,
    max_attempts      INTEGER NOT NULL DEFAULT 3,
    next_retry_at     TIMESTAMPTZ,
    timeout_at        TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT agent_runs_user_message_conversation_fkey
        FOREIGN KEY (user_message_id, conversation_id)
        REFERENCES public.conversation_messages(id, conversation_id) ON DELETE CASCADE,
    CONSTRAINT agent_runs_status_check
        CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT agent_runs_attempt_count_check
        CHECK (attempt_count >= 0),
    CONSTRAINT agent_runs_max_attempts_check
        CHECK (max_attempts >= 1)
);

-- Ensure composite FK (user_message_id, conversation_id) -> conversation_messages(id, conversation_id)
-- exists and remove any redundant single-column user_message_id FK if table was created earlier.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'agent_runs_user_message_conversation_fkey'
          AND conrelid = 'public.agent_runs'::regclass
    ) THEN
        ALTER TABLE public.agent_runs
            ADD CONSTRAINT agent_runs_user_message_conversation_fkey
            FOREIGN KEY (user_message_id, conversation_id)
            REFERENCES public.conversation_messages(id, conversation_id) ON DELETE CASCADE;
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'agent_runs_user_message_id_fkey'
          AND conrelid = 'public.agent_runs'::regclass
    ) THEN
        ALTER TABLE public.agent_runs
            DROP CONSTRAINT agent_runs_user_message_id_fkey;
    END IF;
END $$;

-- Link conversation_messages.run_id -> agent_runs.id ON DELETE SET NULL idempotently.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'conversation_messages_run_id_fkey'
          AND conrelid = 'public.conversation_messages'::regclass
    ) THEN
        ALTER TABLE public.conversation_messages
            ADD CONSTRAINT conversation_messages_run_id_fkey
            FOREIGN KEY (run_id) REFERENCES public.agent_runs(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Enforce that agent_runs.user_message_id must reference a user text message (never assistant/tool).
CREATE OR REPLACE FUNCTION public.enforce_agent_run_user_message()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    msg_role TEXT;
    msg_type TEXT;
BEGIN
    SELECT role, message_type
      INTO msg_role, msg_type
      FROM public.conversation_messages
     WHERE id = NEW.user_message_id
       AND conversation_id = NEW.conversation_id;

    -- If no row matched (id, conversation_id), allow the composite foreign key
    -- constraint (agent_runs_user_message_conversation_fkey) to raise ForeignKeyViolation.
    IF FOUND AND (msg_role <> 'user' OR msg_type <> 'text') THEN
        RAISE EXCEPTION 'agent_runs.user_message_id (%) must reference a user text message (got role=%, message_type=%)',
            NEW.user_message_id, msg_role, msg_type
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_agent_runs_enforce_user_message'
          AND tgrelid = 'public.agent_runs'::regclass
    ) THEN
        CREATE TRIGGER trg_agent_runs_enforce_user_message
            BEFORE INSERT OR UPDATE OF user_message_id, conversation_id
            ON public.agent_runs
            FOR EACH ROW
            EXECUTE FUNCTION public.enforce_agent_run_user_message();
    END IF;
END $$;

-- Enforce at most one active (queued or running) run per conversation at a time.
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runs_one_active_per_conversation
    ON public.agent_runs (conversation_id)
    WHERE status IN ('queued', 'running');

-- Enforce idempotent message submission per conversation when client_request_id is provided.
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runs_conv_client_request_id
    ON public.agent_runs (conversation_id, client_request_id)
    WHERE client_request_id IS NOT NULL;

-- Index for worker claiming queued or lease-expired runs.
CREATE INDEX IF NOT EXISTS idx_agent_runs_claimable
    ON public.agent_runs (status, next_retry_at, lease_expires_at, created_at ASC)
    WHERE status IN ('queued', 'running');

CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation_created
    ON public.agent_runs (conversation_id, created_at DESC);

ALTER TABLE public.agent_runs ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- 4. AGENT RUN STEPS (Structured live progress events)
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.agent_run_steps (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id         UUID NOT NULL REFERENCES public.agent_runs(id) ON DELETE CASCADE,
    step_number    BIGINT NOT NULL,
    step_type      TEXT NOT NULL,
    tool_name      TEXT,
    status         TEXT NOT NULL,
    input_summary  JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT agent_run_steps_run_step_unique
        UNIQUE (run_id, step_number),
    CONSTRAINT agent_run_steps_step_positive_check
        CHECK (step_number >= 1),
    CONSTRAINT agent_run_steps_step_type_check
        CHECK (step_type IN (
            'model_started',
            'tool_started',
            'tool_completed',
            'tool_failed',
            'model_completed',
            'run_completed',
            'run_failed'
        )),
    CONSTRAINT agent_run_steps_status_check
        CHECK (status IN ('in_progress', 'completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS idx_agent_run_steps_run_step
    ON public.agent_run_steps (run_id, step_number ASC);

ALTER TABLE public.agent_run_steps ENABLE ROW LEVEL SECURITY;
