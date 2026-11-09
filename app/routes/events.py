import json

import psycopg
from fastapi import APIRouter, Depends, HTTPException

from app.database import get_db
from app.dependencies.authentication import get_api_key_account
from app.models import EventBatch, EventsResponse
from app.services import project_service
from app.services.api_key_service import ApiKeyAccount

router = APIRouter(prefix="/v1")


@router.post("/events", response_model=EventsResponse)
def receive_events(
    batch: EventBatch,
    account: ApiKeyAccount = Depends(get_api_key_account),
    conn: psycopg.Connection = Depends(get_db),
) -> EventsResponse:
    try:
        project_service.authorize_projects(
            conn, account.user_id, {e.application.project_id for e in batch.events}
        )
    except project_service.UnknownProject as exc:
        raise HTTPException(status_code=404, detail=f"Unknown project_id '{exc.project_id}'")
    except project_service.ProjectForbidden:
        raise HTTPException(status_code=403, detail="project_id does not belong to this account")

    print(f"\n=== Received {len(batch.events)} event(s) ===", flush=True)
    for event in batch.events:
        print(json.dumps(event.model_dump(mode="json"), indent=2), flush=True)
    return EventsResponse(
        status="success",
        message="Events received successfully",
        received=len(batch.events),
    )
