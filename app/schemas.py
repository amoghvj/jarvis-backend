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


