"""Telemetry ingestion endpoint (API-key authenticated, used by the SDKs)."""

from typing import Any, Callable, Coroutine, Dict, List

import psycopg
from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from app.config import get_settings
from app.database import get_db
from app.dependencies.authentication import get_api_key_account
from app.models import EventOutcome, EventsResponse
from app.services import project_service, repository_discovery, telemetry_service
from app.services.api_key_service import ApiKeyAccount


class PayloadLimitedRoute(APIRoute):
    """Reject oversized bodies before they are parsed."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            limit = telemetry_service.MAX_PAYLOAD_BYTES
            declared = request.headers.get("content-length", "")
            too_big = declared.isdigit() and int(declared) > limit
            if not too_big:
                too_big = len(await request.body()) > limit
            if too_big:
                return JSONResponse(
                    status_code=413, content={"detail": f"Payload exceeds the {limit} byte limit"}
                )
            return await original(request)

        return handler


router = APIRouter(prefix="/v1", route_class=PayloadLimitedRoute)


@router.post("/events", response_model=EventsResponse, response_model_exclude_none=True)
def receive_events(
    body: Dict[str, Any] = Body(...),
    account: ApiKeyAccount = Depends(get_api_key_account),
    conn: psycopg.Connection = Depends(get_db),
):
    raw_events = body.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise HTTPException(status_code=422, detail="'events' must be a non-empty list")
    if len(raw_events) > telemetry_service.MAX_BATCH_EVENTS:
        raise HTTPException(
            status_code=413,
            detail=f"A batch may contain at most {telemetry_service.MAX_BATCH_EVENTS} events",
        )

    valid, rejected = telemetry_service.parse_events(raw_events)
    if not valid:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "No valid events in the batch",
                "rejected": [{"index": r.index, "event_id": r.event_id, "error": r.error} for r in rejected],
            },
        )

    # Ownership is decided by the API key's account, never by anything the client claims.
    try:
        project_map = project_service.authorize_projects(
            conn, account.user_id, {event.application.project_id for _, event, _ in valid}
        )
    except project_service.UnknownProject as exc:
        raise HTTPException(status_code=404, detail=f"Unknown project_id '{exc.project_id}'")
    except project_service.ProjectForbidden:
        raise HTTPException(status_code=403, detail="project_id does not belong to this account")

    ingested = telemetry_service.ingest_events(
        conn, valid, project_map, get_settings().incident_window_seconds
    )
    repository_discovery.process_startup_events(conn, account.user_id, ingested.new_startup_events)

    results: List[telemetry_service.EventResult] = sorted(
        ingested.results + rejected, key=lambda r: r.index
    )
    accepted, duplicates = ingested.count("accepted"), ingested.count("duplicate")
    if not duplicates and not rejected:
        return EventsResponse(
            status="success", message="Events received successfully", received=len(raw_events)
        )
    return EventsResponse(
        status="partial" if rejected else "success",
        message="Some events were rejected" if rejected else "Events received successfully",
        received=len(raw_events),
        accepted=accepted,
        duplicates=duplicates,
        rejected=len(rejected),
        results=[
            EventOutcome(index=r.index, event_id=r.event_id, status=r.status, error=r.error)
            for r in results
            if r.status != "accepted"
        ],
    )
