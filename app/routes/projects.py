"""Project, incident and repository-mapping endpoints for the logged-in user.

Authentication is the session cookie; state-changing requests also require a trusted Origin.
The owner is always the session user. A project that exists but belongs to someone else is
indistinguishable from one that does not exist (404).
"""

from typing import Optional
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app import repositories_telemetry as telemetry_repo
from app.database import get_db
from app.dependencies.authentication import get_current_user, require_trusted_origin
from app.schemas import (
    IncidentListOut,
    IncidentOut,
    IncidentStatusIn,
    ProjectCreateIn,
    ProjectListOut,
    ProjectOut,
    ProjectRepositoryListOut,
    ProjectRepositoryOut,
    ProjectUpdateIn,
)
from app.services import incident_service, project_service, repository_discovery

router = APIRouter(prefix="/projects", tags=["projects"])

NO_STORE = {"Cache-Control": "no-store"}
MAX_PAGE = 100


def _owned_project(conn: psycopg.Connection, user: dict, project_uuid: UUID) -> dict:
    project = project_service.get_project(conn, user["id"], project_uuid)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _repository_out(row: dict) -> ProjectRepositoryOut:
    return ProjectRepositoryOut(**{**row, "service_id": row["service_id"] or None})


@router.get("", response_model=ProjectListOut, summary="List the current user's projects")
def list_projects(user: dict = Depends(get_current_user), conn: psycopg.Connection = Depends(get_db)):
    return {"projects": project_service.list_projects(conn, user["id"])}


@router.post(
    "",
    response_model=ProjectOut,
    status_code=201,
    dependencies=[Depends(require_trusted_origin)],
    summary="Create a project (project_id is unique per user)",
)
def create_project(
    body: ProjectCreateIn,
    user: dict = Depends(get_current_user),
    conn: psycopg.Connection = Depends(get_db),
):
    try:
        return project_service.create_project(conn, user["id"], body.project_id, body.name)
    except project_service.InvalidProject as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except project_service.DuplicateProject:
        raise HTTPException(status_code=409, detail="You already have a project with this project_id")


@router.get("/{project_uuid}", response_model=ProjectOut, summary="Project details")
def get_project(
    project_uuid: UUID, user: dict = Depends(get_current_user), conn: psycopg.Connection = Depends(get_db)
):
    return _owned_project(conn, user, project_uuid)


@router.patch(
    "/{project_uuid}",
    response_model=ProjectOut,
    dependencies=[Depends(require_trusted_origin)],
    summary="Update the project's display name",
)
def update_project(
    project_uuid: UUID,
    body: ProjectUpdateIn,
    user: dict = Depends(get_current_user),
    conn: psycopg.Connection = Depends(get_db),
):
    try:
        updated = project_service.rename_project(conn, user["id"], project_uuid, body.name)
    except project_service.InvalidProject as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if updated is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return updated


@router.get("/{project_uuid}/incidents", response_model=IncidentListOut, summary="List incidents")
def list_incidents(
    project_uuid: UUID,
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=25, ge=1, le=MAX_PAGE),
    offset: int = Query(default=0, ge=0, le=100_000),
    user: dict = Depends(get_current_user),
    conn: psycopg.Connection = Depends(get_db),
):
    _owned_project(conn, user, project_uuid)
    if status is not None and status not in incident_service.STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of: {', '.join(incident_service.STATUSES)}")
    rows = telemetry_repo.list_incidents(conn, project_uuid, status, limit + 1, offset)
    return {
        "incidents": [IncidentOut(**r) for r in rows[:limit]],
        "limit": limit,
        "offset": offset,
        "has_more": len(rows) > limit,
    }


@router.post(
    "/{project_uuid}/incidents/{incident_id}/status",
    response_model=IncidentOut,
    dependencies=[Depends(require_trusted_origin)],
    summary="Acknowledge, resolve or reopen an incident (never automatic)",
)
def set_incident_status(
    project_uuid: UUID,
    incident_id: UUID,
    body: IncidentStatusIn,
    user: dict = Depends(get_current_user),
    conn: psycopg.Connection = Depends(get_db),
):
    _owned_project(conn, user, project_uuid)
    try:
        return incident_service.change_status(conn, project_uuid, incident_id, body.status, user["id"], body.note)
    except incident_service.IncidentNotFound:
        raise HTTPException(status_code=404, detail="Incident not found")
    except incident_service.InvalidTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get(
    "/{project_uuid}/repositories",
    response_model=ProjectRepositoryListOut,
    summary="Repositories discovered from telemetry, with explicit verification status",
)
def list_repositories(
    project_uuid: UUID, user: dict = Depends(get_current_user), conn: psycopg.Connection = Depends(get_db)
):
    _owned_project(conn, user, project_uuid)
    rows = telemetry_repo.list_project_repositories(conn, project_uuid)
    return JSONResponse(
        content=ProjectRepositoryListOut(repositories=[_repository_out(r) for r in rows]).model_dump(mode="json"),
        headers=NO_STORE,
    )


@router.post(
    "/{project_uuid}/repositories/verify",
    response_model=ProjectRepositoryListOut,
    dependencies=[Depends(require_trusted_origin)],
    summary="Re-check unverified repositories against your GitHub App installations",
)
def verify_repositories(
    project_uuid: UUID, user: dict = Depends(get_current_user), conn: psycopg.Connection = Depends(get_db)
):
    _owned_project(conn, user, project_uuid)
    repository_discovery.verify_pending(conn, user["id"], project_uuid)
    rows = telemetry_repo.list_project_repositories(conn, project_uuid)
    return JSONResponse(
        content=ProjectRepositoryListOut(repositories=[_repository_out(r) for r in rows]).model_dump(mode="json"),
        headers=NO_STORE,
    )
