from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .storage.agent_repository import AgentRepositoryMixin
from .storage.audit_archive_repository import AuditArchiveRepositoryMixin
from .storage.copilot_repository import CopilotRepositoryMixin
from .storage.evaluation_repository import EvaluationRepositoryMixin
from .storage.event_repository import EventRepositoryMixin
from .storage.memory_repository import MemoryRepositoryMixin
from .storage.notification_repository import NotificationRepositoryMixin
from .storage.proposal_repository import ProposalRepositoryMixin
from .storage.runtime_repository import RuntimeRepositoryMixin
from .storage.schema import REPOSITORY_SCHEMA_SQL
from .storage.trigger_repository import TriggerRepositoryMixin


class SQLiteRepository(
    EventRepositoryMixin,
    RuntimeRepositoryMixin,
    ProposalRepositoryMixin,
    CopilotRepositoryMixin,
    TriggerRepositoryMixin,
    AgentRepositoryMixin,
    MemoryRepositoryMixin,
    EvaluationRepositoryMixin,
    AuditArchiveRepositoryMixin,
    NotificationRepositoryMixin,
):
    """SQLite-backed repository composed from focused storage mixins."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(REPOSITORY_SCHEMA_SQL)
