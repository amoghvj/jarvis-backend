-- Migration 005: Enforce non-null worker_id and lease_expires_at invariant on running agent_runs.
-- Safely remediates any pre-existing malformed running rows before adding the CHECK constraint.
-- Fully idempotent and safe to run on existing production databases.
--
-- Apply manually in the Supabase SQL Editor (or via psql against DATABASE_URL).

DO $$
DECLARE
    rec RECORD;
    next_step_num BIGINT;
BEGIN
    -- 1. Recoverable malformed running rows (missing lease_expires_at or worker_id while
    --    overall timeout has not elapsed and retry budget remains): return cleanly to queued.
    UPDATE public.agent_runs
       SET status = 'queued',
           worker_id = NULL,
           lease_expires_at = NULL,
           next_retry_at = NULL,
           updated_at = clock_timestamp()
     WHERE status = 'running'
       AND (lease_expires_at IS NULL OR worker_id IS NULL)
       AND (timeout_at IS NULL OR timeout_at > clock_timestamp())
       AND attempt_count < max_attempts;

    -- 2. Exhausted or timed-out malformed running rows: transition to terminal failed
    --    with cleared lease fields and record an audit step.
    FOR rec IN
        UPDATE public.agent_runs
           SET status = 'failed',
               error_message = CASE
                   WHEN timeout_at IS NOT NULL AND timeout_at <= clock_timestamp()
                       THEN 'Investigation run timed out before completion.'
                   ELSE COALESCE(
                       error_message,
                       'Investigation run failed during migration remediation due to missing worker lease.'
                   )
               END,
               worker_id = NULL,
               lease_expires_at = NULL,
               next_retry_at = NULL,
               completed_at = COALESCE(completed_at, clock_timestamp()),
               updated_at = clock_timestamp()
         WHERE status = 'running'
           AND (lease_expires_at IS NULL OR worker_id IS NULL)
        RETURNING id, error_message
    LOOP
        SELECT COALESCE(MAX(step_number), 0) + 1
          INTO next_step_num
          FROM public.agent_run_steps
         WHERE run_id = rec.id;

        INSERT INTO public.agent_run_steps (
            run_id,
            step_number,
            step_type,
            status,
            input_summary,
            output_summary,
            created_at
        )
        VALUES (
            rec.id,
            next_step_num,
            'run_failed',
            'failed',
            '{}'::jsonb,
            jsonb_build_object('message', rec.error_message),
            clock_timestamp()
        );
    END LOOP;

    -- 3. Enforce invariant: every running run must have non-null lease_expires_at and worker_id.
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint
         WHERE conname = 'agent_runs_running_requires_lease_check'
           AND conrelid = 'public.agent_runs'::regclass
    ) THEN
        ALTER TABLE public.agent_runs
            ADD CONSTRAINT agent_runs_running_requires_lease_check
            CHECK (
                status <> 'running'
                OR (
                    lease_expires_at IS NOT NULL
                    AND worker_id IS NOT NULL
                )
            );
    END IF;
END $$;

