"""Response models for auth and API key endpoints."""

from datetime import datetime
from typing import Optional
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

