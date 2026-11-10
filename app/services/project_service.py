"""Project creation, retrieval and ownership checks.

`project_id` (external) is chosen by the user, unique per user, and is what SDKs send.
`id` (internal UUID) is what relationships and authorization use. The owner always comes from the
authenticated session or API key, never from the request.
"""

import re
from typing import Dict, Iterable, List, Optional
from uuid import UUID

import psycopg

from app import repositories

PROJECT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}")
MAX_NAME_LENGTH = 100


class UnknownProject(Exception):
    def __init__(self, project_id: str):
        self.project_id = project_id


class ProjectForbidden(Exception):
    pass


class DuplicateProject(Exception):
    """The user already has a project with this project_id."""


class InvalidProject(ValueError):
    """Project ID or name failed validation (message is safe to return)."""


def validate_project_id(value: str) -> str:
    value = value.strip() if isinstance(value, str) else ""
    if not PROJECT_ID_RE.fullmatch(value):
        raise InvalidProject(
            "project_id must be 3-64 characters: letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return value


def validate_name(value: Optional[str], default: str) -> str:
    name = (value if value is not None else default).strip()
    if not name or len(name) > MAX_NAME_LENGTH or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise InvalidProject(f"name must be 1-{MAX_NAME_LENGTH} printable characters")
    return name


def authorize_projects(
    conn: psycopg.Connection, user_id: UUID, project_ids: Iterable[str]
) -> Dict[str, UUID]:
    """Resolve external project IDs to internal UUIDs, all-or-nothing, for `user_id`.

    Projects are never auto-created. A project ID that the caller does not own raises
    ProjectForbidden when it exists for another account and UnknownProject otherwise.
    """
    wanted = set(project_ids)
    owned = repositories.get_user_projects_by_external_ids(conn, user_id, wanted)
    missing = wanted - set(owned)
    elsewhere = repositories.get_external_ids_owned_by_others(conn, user_id, missing) if missing else set()
    for project_id in sorted(missing):
        if project_id in elsewhere:
            raise ProjectForbidden()
        raise UnknownProject(project_id)
    return owned


def create_project(
    conn: psycopg.Connection, user_id: UUID, project_id: str, name: Optional[str] = None
) -> dict:
    project_id = validate_project_id(project_id)
    name = validate_name(name, project_id)
    with conn.transaction():
        row = repositories.insert_project(conn, user_id, project_id, name)
    if row is None:
        raise DuplicateProject()
    return row


def list_projects(conn: psycopg.Connection, user_id: UUID) -> List[dict]:
    return repositories.list_projects_for_user(conn, user_id)


def get_project(conn: psycopg.Connection, user_id: UUID, project_uuid: UUID) -> Optional[dict]:
    return repositories.get_project_for_user(conn, user_id, project_uuid)


def rename_project(
    conn: psycopg.Connection, user_id: UUID, project_uuid: UUID, name: str
) -> Optional[dict]:
    name = validate_name(name, "")
    with conn.transaction():
        return repositories.update_project_name(conn, user_id, project_uuid, name)
