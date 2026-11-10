"""Authenticated routes for project-scoped investigation conversations, messages, and runs."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status

from app import repositories, repositories_agent
from app.agent.worker import trigger_background_run
from app.config import Settings, get_settings
from app.database import get_db
from app.dependencies.authentication import get_current_user, require_trusted_origin
from app.schemas import (
    AgentRunStatusOut,
    ConversationCreateIn,
    ConversationDeleteOut,
    ConversationDetailOut,
    ConversationListOut,
    ConversationMessageCreateIn,
    ConversationMessageSubmitOut,
    ConversationSummaryOut,
)

MAX_USER_MESSAGE_CHARS = 16000
MAX_CONVERSATION_TITLE_CHARS = 200

router = APIRouter(
    prefix="/projects/{project_uuid}/conversations",
    tags=["conversations"],
)


def _require_project(db, project_uuid: UUID, user_id: UUID) -> dict:
    row = repositories.get_project_for_user(db, user_id, project_uuid)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
    return row


def _require_conversation(
    db,
    *,
    user_id: UUID,
    project_uuid: UUID,
    conversation_id: UUID,
) -> dict:
    _require_project(db, project_uuid, user_id)
    conv = repositories_agent.get_conversation(
        db,
        user_id=user_id,
        project_id=project_uuid,
        conversation_id=conversation_id,
    )
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return conv


@router.get("", response_model=ConversationListOut)
def list_project_conversations(
    project_uuid: UUID,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
) -> ConversationListOut:
    _require_project(db, project_uuid, user["id"])
    rows, has_more = repositories_agent.list_conversations(
        db,
        user_id=user["id"],
        project_id=project_uuid,
        limit=limit,
        offset=offset,
    )
    return ConversationListOut(
        conversations=rows,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


@router.post(
    "",
    response_model=ConversationSummaryOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_trusted_origin)],
)
def create_project_conversation(
    project_uuid: UUID,
    body: Optional[ConversationCreateIn] = Body(default=None),
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
) -> ConversationSummaryOut:
    _require_project(db, project_uuid, user["id"])
    title: Optional[str] = None
    if body is not None and body.title is not None:
        stripped = body.title.strip()
        if len(stripped) > MAX_CONVERSATION_TITLE_CHARS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"title must be at most {MAX_CONVERSATION_TITLE_CHARS} characters",
            )
        title = stripped or None

    row = repositories_agent.create_conversation(
        db,
        user_id=user["id"],
        project_id=project_uuid,
        title=title,
    )
    return ConversationSummaryOut(**row)


@router.get("/{conversation_id}", response_model=ConversationDetailOut)
def get_project_conversation(
    project_uuid: UUID,
    conversation_id: UUID,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
) -> ConversationDetailOut:
    conv = _require_conversation(
        db,
        user_id=user["id"],
        project_uuid=project_uuid,
        conversation_id=conversation_id,
    )
    messages, has_more = repositories_agent.list_visible_messages(
        db,
        conversation_id=conversation_id,
        limit=limit,
        offset=offset,
    )
    return ConversationDetailOut(
        conversation=conv,
        messages=messages,
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


@router.delete(
    "/{conversation_id}",
    response_model=ConversationDeleteOut,
    dependencies=[Depends(require_trusted_origin)],
)
def delete_project_conversation(
    project_uuid: UUID,
    conversation_id: UUID,
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
) -> ConversationDeleteOut:
    _require_project(db, project_uuid, user["id"])
    deleted = repositories_agent.delete_conversation(
        db,
        user_id=user["id"],
        project_id=project_uuid,
        conversation_id=conversation_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
    return ConversationDeleteOut(
        status="deleted",
        conversation_id=conversation_id,
    )


@router.post(
    "/{conversation_id}/messages",
    response_model=ConversationMessageSubmitOut,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_trusted_origin)],
)
def submit_conversation_message(
    request: Request,
    project_uuid: UUID,
    conversation_id: UUID,
    body: ConversationMessageCreateIn,
    user: dict = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    db=Depends(get_db),
) -> ConversationMessageSubmitOut:
    _require_conversation(
        db,
        user_id=user["id"],
        project_uuid=project_uuid,
        conversation_id=conversation_id,
    )
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="content must not be empty",
        )
    if len(content) > MAX_USER_MESSAGE_CHARS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"content must be at most {MAX_USER_MESSAGE_CHARS} characters",
        )

    client_request_id: Optional[str] = None
    if body.client_request_id is not None:
        req_id = body.client_request_id.strip()
        if len(req_id) > 128:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="client_request_id must be at most 128 characters",
            )
        client_request_id = req_id or None

    try:
        user_msg, run, deduplicated = repositories_agent.submit_user_message_and_create_run(
            db,
            user_id=user["id"],
            project_id=project_uuid,
            conversation_id=conversation_id,
            content=content,
            client_request_id=client_request_id,
            max_attempts=settings.agent_max_retries + 1,
            timeout_seconds=settings.agent_run_timeout_seconds,
        )
    except repositories_agent.ActiveRunConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "active_run_in_progress",
                "message": "An investigation run is already in progress for this conversation.",
                "active_run_id": str(exc.active_run_id),
            },
        )

    # Only spawn a separate DB-connection thread when get_db is not overridden by an
    # uncommitted rollback test fixture.
    is_test_override = get_db in request.app.dependency_overrides
    if not is_test_override and not deduplicated:
        trigger_background_run(run["id"], enabled=True)

    return ConversationMessageSubmitOut(
        run_id=run["id"],
        conversation_id=conversation_id,
        user_message_id=user_msg["id"],
        status=run["status"],
        deduplicated=deduplicated,
        created_at=run["created_at"],
    )


@router.get("/{conversation_id}/runs/{run_id}", response_model=AgentRunStatusOut)
def get_conversation_run_status(
    request: Request,
    project_uuid: UUID,
    conversation_id: UUID,
    run_id: UUID,
    after_step: int = Query(
        default=0,
        ge=0,
        description="Cursor: only return run steps with step_number > after_step",
    ),
    limit: int = Query(default=100, ge=1, le=500),
    user: dict = Depends(get_current_user),
    db=Depends(get_db),
) -> AgentRunStatusOut:
    _require_conversation(
        db,
        user_id=user["id"],
        project_uuid=project_uuid,
        conversation_id=conversation_id,
    )

    # Ensure any timed-out or exhausted lease runs transition cleanly to failed.
    repositories_agent.expire_stale_runs(
        db,
        conversation_id=conversation_id,
        run_id=run_id,
    )

    run = repositories_agent.get_run(
        db,
        conversation_id=conversation_id,
        run_id=run_id,
    )
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found",
        )

    # Poll-assisted recovery for serverless environments: if the run is still queued
    # or its worker lease expired while still eligible for retry, trigger background execution.
    is_test_override = get_db in request.app.dependency_overrides
    if not is_test_override and run["status"] in ("queued", "running"):
        trigger_background_run(run_id, enabled=True)

    steps, latest_step = repositories_agent.list_run_steps(
        db,
        run_id=run_id,
        after_step=after_step,
        limit=limit,
    )
    assistant_messages = repositories_agent.list_assistant_messages_for_run(
        db,
        conversation_id=conversation_id,
        run_id=run_id,
    )

    return AgentRunStatusOut(
        run_id=run["id"],
        conversation_id=run["conversation_id"],
        user_message_id=run["user_message_id"],
        status=run["status"],
        error_message=run["error_message"],
        attempt_count=run["attempt_count"],
        max_attempts=run["max_attempts"],
        created_at=run["created_at"],
        started_at=run["started_at"],
        completed_at=run["completed_at"],
        updated_at=run["updated_at"],
        steps=steps,
        latest_step_number=latest_step,
        assistant_messages=assistant_messages,
    )
