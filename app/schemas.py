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
