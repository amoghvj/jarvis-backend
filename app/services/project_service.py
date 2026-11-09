"""Project ownership checks for telemetry ingestion."""

from typing import Iterable
from uuid import UUID

import psycopg

from app import repositories


class UnknownProject(Exception):
    def __init__(self, project_id: str):
        self.project_id = project_id


class ProjectForbidden(Exception):
    pass


def authorize_projects(conn: psycopg.Connection, user_id: UUID, project_ids: Iterable[str]) -> None:
    """Ensure every project_id exists and belongs to `user_id` (all-or-nothing).

    Projects are never auto-created: the existing ingestion logic has no project
    registration, so unknown projects are rejected.
    """
    wanted = set(project_ids)
    owners = repositories.get_project_owners(conn, wanted)
    for project_id in sorted(wanted):
        if project_id not in owners:
            raise UnknownProject(project_id)
        if owners[project_id] != user_id:
            raise ProjectForbidden()

