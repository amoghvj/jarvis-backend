"""Response models for auth and API key endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class UserOut(BaseModel):
    id: UUID
    github_id: int
    github_username: str
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: datetime


class LoginSuccess(BaseModel):
    status: str = "success"
    message: str = "Logged in with GitHub"
    new_account: bool
    user: UserOut


class MessageOut(BaseModel):
    status: str
    message: str


class RotatedKeyOut(BaseModel):
    api_key: str
    key_prefix: str
    created_at: datetime
    message: str = "Store this key now. It will not be shown again."


class ApiKeyStatusOut(BaseModel):
    active: bool
    key_prefix: Optional[str] = None
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class GitHubAppInstallationOut(BaseModel):
    installation_id: int
    account_id: int
    account_login: str
    account_type: str
    repository_selection: Optional[str] = None
    permissions: Dict[str, Any] = {}
    suspended_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class GitHubIntegrationStatusOut(BaseModel):
    connected: bool
    installations: List[GitHubAppInstallationOut]


class GitHubDisconnectOut(BaseModel):
    status: str
    message: str
    installation_id: int




class ProjectOut(BaseModel):
    id: UUID
    project_id: str
    name: str
    created_at: datetime
    updated_at: datetime


class ProjectListOut(BaseModel):
    projects: List[ProjectOut]


class ProjectCreateIn(BaseModel):
    """Only the project ID and display name come from the client; the owner is the session user."""

    model_config = {"extra": "forbid"}

    project_id: str
    name: Optional[str] = None


class ProjectUpdateIn(BaseModel):
    model_config = {"extra": "forbid"}

    name: str


class IncidentOut(BaseModel):
    id: UUID
    project_id: UUID
    title: str
    status: str
    severity: Optional[str] = None
    service_id: Optional[str] = None
    environment: Optional[str] = None
    event_count: int
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class IncidentListOut(BaseModel):
    incidents: List[IncidentOut]
    limit: int
    offset: int
    has_more: bool


class IncidentStatusIn(BaseModel):
    model_config = {"extra": "forbid"}

    status: str
    note: Optional[str] = None


class ProjectRepositoryOut(BaseModel):
    id: UUID
    service_id: Optional[str] = None
    repository_url: str
    owner: Optional[str] = None
    repo: Optional[str] = None
    installation_id: Optional[int] = None
    default_branch: Optional[str] = None
    verification_status: str
    verification_note: Optional[str] = None
    discovery_source: str
    first_seen_at: datetime
    last_seen_at: datetime
    last_verification_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None


class ProjectRepositoryListOut(BaseModel):
    repositories: List[ProjectRepositoryOut]


# ============================================================================
# Conversational Investigation Agent Schemas
# ============================================================================


class ConversationCreateIn(BaseModel):
    model_config = {"extra": "forbid"}

    title: Optional[str] = None


class ConversationSummaryOut(BaseModel):
    id: UUID
    project_id: UUID
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime


class ConversationListOut(BaseModel):
    conversations: List[ConversationSummaryOut]
    limit: int
    offset: int
    has_more: bool


class ConversationMessageOut(BaseModel):
    id: UUID
    conversation_id: UUID
    sequence_number: int
    role: str
    message_type: str
    content: Optional[str] = None
    metadata: Dict[str, Any] = {}
    run_id: Optional[UUID] = None
    created_at: datetime


class ConversationDetailOut(BaseModel):
    conversation: ConversationSummaryOut
    messages: List[ConversationMessageOut]
    limit: int
    offset: int
    has_more: bool


class ConversationDeleteOut(BaseModel):
    status: str = "deleted"
    conversation_id: UUID


class ConversationMessageCreateIn(BaseModel):
    model_config = {"extra": "forbid"}

    content: str
    client_request_id: Optional[str] = None


class ConversationMessageSubmitOut(BaseModel):
    run_id: UUID
    conversation_id: UUID
    user_message_id: UUID
    status: str
    deduplicated: bool = False
    created_at: datetime


class AgentRunStepOut(BaseModel):
    id: UUID
    run_id: UUID
    step_number: int
    step_type: str
    tool_name: Optional[str] = None
    status: str
    input_summary: Dict[str, Any] = {}
    output_summary: Dict[str, Any] = {}
    created_at: datetime


class AgentRunStatusOut(BaseModel):
    run_id: UUID
    conversation_id: UUID
    user_message_id: UUID
    status: str
    error_message: Optional[str] = None
    attempt_count: int = 0
    max_attempts: int = 3
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime
    steps: List[AgentRunStepOut]
    latest_step_number: int
    assistant_messages: List[ConversationMessageOut] = []

