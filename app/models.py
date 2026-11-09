from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    # Tolerate extra fields so SDKs can evolve without breaking ingestion.
    model_config = ConfigDict(extra="allow")


class Runtime(_Base):
    language: Optional[str] = None
    version: Optional[str] = None


class Application(_Base):
    project_id: str
    service_id: Optional[str] = None
    repository: Optional[str] = None
    environment: Optional[str] = None
    instance_id: Optional[str] = None
    runtime: Optional[Runtime] = None


class Source(_Base):
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    line_number: Optional[int] = None
    module: Optional[str] = None


class Details(_Base):
    severity: Optional[str] = None
    message: Optional[str] = None
    logger_name: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    source: Optional[Source] = None


class Execution(_Base):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    request_id: Optional[str] = None
    thread_name: Optional[str] = None


class Event(_Base):
    schema_version: str
    event_id: str
    timestamp: datetime
    type: str
    application: Application
    details: Details
    execution: Optional[Execution] = None
    attributes: Optional[Dict[str, Any]] = None


class EventBatch(_Base):
    events: List[Event] = Field(..., min_length=1)


class EventsResponse(BaseModel):
    status: str
    message: str
    received: int

