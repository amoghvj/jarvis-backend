import json

from fastapi import APIRouter

from app.models import EventBatch, EventsResponse

router = APIRouter(prefix="/v1")


@router.post("/events", response_model=EventsResponse)
async def receive_events(batch: EventBatch) -> EventsResponse:
    print(f"\n=== Received {len(batch.events)} event(s) ===", flush=True)
    for event in batch.events:
        print(json.dumps(event.model_dump(mode="json"), indent=2), flush=True)
    return EventsResponse(
        status="success",
        message="Events received successfully",
        received=len(batch.events),
    )

