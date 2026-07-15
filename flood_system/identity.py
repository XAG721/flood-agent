from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping

from .response_workflow.models import OperatorRole


class IdentityAssertionError(PermissionError):
    pass


ASSURANCE_RANK = {"aal1": 1, "aal2": 2, "aal3": 3}
NONCE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


@dataclass(frozen=True)
class OperatorIdentity:
    operator_id: str
    operator_role: OperatorRole
    assurance_level: str
    terminal_id: str
    issued_at: datetime
    nonce: str

    def require_assurance(self, minimum: str) -> None:
        if ASSURANCE_RANK.get(self.assurance_level, 0) < ASSURANCE_RANK.get(minimum, 99):
            raise IdentityAssertionError(
                f"{minimum} authentication is required; current assertion is {self.assurance_level}"
            )


def development_identity_from_headers(headers: Mapping[str, str]) -> OperatorIdentity | None:
    """Resolve explicitly enabled unsigned demo headers; never available in production."""
    environment = os.getenv("FLOOD_ENVIRONMENT", "development").strip().lower()
    enabled = os.getenv("FLOOD_ALLOW_DEV_IDENTITY_HEADERS", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled or environment in {"production", "prod"}:
        return None
    operator_id = str(headers.get("x-operator-id", "")).strip()
    role_value = str(headers.get("x-operator-role", "")).strip()
    terminal_id = str(headers.get("x-operator-terminal", "")).strip()
    if not operator_id or not role_value or not terminal_id:
        return None
    try:
        role = OperatorRole(role_value)
    except ValueError as exc:
        raise IdentityAssertionError("development identity contains an unknown operator role") from exc
    assurance = "aal2" if role in {OperatorRole.REVIEWER, OperatorRole.COMMANDER, OperatorRole.ADMIN} else "aal1"
    return OperatorIdentity(
        operator_id=operator_id,
        operator_role=role,
        assurance_level=assurance,
        terminal_id=terminal_id,
        issued_at=datetime.now(timezone.utc),
        nonce=f"development-{int(time.time() * 1000)}",
    )


class TrustedIdentityVerifier:
    """Verify short-lived, signed identity assertions supplied by a trusted authentication gateway."""

    HEADER_MAP = {
        "operator_id": "x-identity-operator-id",
        "operator_role": "x-identity-operator-role",
        "assurance_level": "x-identity-assurance-level",
        "terminal_id": "x-identity-terminal-id",
        "timestamp": "x-identity-timestamp",
        "nonce": "x-identity-nonce",
        "signature": "x-identity-signature",
    }

    def __init__(self, repository, secret: str, *, max_age_seconds: int = 120) -> None:
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("trusted identity secret must contain at least 32 bytes")
        self.repository = repository
        self._secret = secret.encode("utf-8")
        self.max_age_seconds = max(30, int(max_age_seconds))

    @classmethod
    def from_environment(cls, repository) -> "TrustedIdentityVerifier":
        configured = os.getenv("FLOOD_TRUSTED_IDENTITY_SECRET", "").strip()
        environment = os.getenv("FLOOD_ENVIRONMENT", "development").strip().lower()
        if not configured and environment in {"production", "prod"}:
            raise ValueError("FLOOD_TRUSTED_IDENTITY_SECRET is required in production")
        secret = configured or "local-development-identity-secret-change-before-production-v1"
        max_age = int(os.getenv("FLOOD_IDENTITY_ASSERTION_MAX_AGE_SECONDS", "120"))
        return cls(repository, secret, max_age_seconds=max_age)

    def verify(self, headers: Mapping[str, str], *, method: str, path: str) -> OperatorIdentity:
        values: dict[str, str] = {}
        for field, header in self.HEADER_MAP.items():
            value = headers.get(header) or headers.get(header.title())
            if value is None or not str(value).strip():
                raise IdentityAssertionError(f"missing trusted identity header: {header}")
            values[field] = str(value).strip()
        try:
            role = OperatorRole(values["operator_role"])
        except ValueError as exc:
            raise IdentityAssertionError("trusted identity contains an unknown operator role") from exc
        assurance = values["assurance_level"].lower()
        if assurance not in ASSURANCE_RANK:
            raise IdentityAssertionError("trusted identity contains an invalid assurance level")
        if not NONCE_PATTERN.fullmatch(values["nonce"]):
            raise IdentityAssertionError("trusted identity nonce is invalid")
        try:
            timestamp = int(values["timestamp"])
        except ValueError as exc:
            raise IdentityAssertionError("trusted identity timestamp is invalid") from exc
        now = int(time.time())
        if abs(now - timestamp) > self.max_age_seconds:
            raise IdentityAssertionError("trusted identity assertion has expired or is not yet valid")
        expected = self.sign(
            secret=self._secret,
            operator_id=values["operator_id"],
            operator_role=role.value,
            assurance_level=assurance,
            terminal_id=values["terminal_id"],
            timestamp=timestamp,
            nonce=values["nonce"],
            method=method,
            path=path,
        )
        if not hmac.compare_digest(expected, values["signature"].lower()):
            raise IdentityAssertionError("trusted identity signature verification failed")
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.max_age_seconds)
        if not self.repository.consume_identity_nonce(values["nonce"], expires_at=expires_at):
            raise IdentityAssertionError("trusted identity assertion was already used")
        return OperatorIdentity(
            operator_id=values["operator_id"],
            operator_role=role,
            assurance_level=assurance,
            terminal_id=values["terminal_id"],
            issued_at=datetime.fromtimestamp(timestamp, timezone.utc),
            nonce=values["nonce"],
        )

    @classmethod
    def build_headers(
        cls,
        *,
        secret: str,
        operator_id: str,
        operator_role: OperatorRole,
        assurance_level: str,
        terminal_id: str,
        nonce: str,
        method: str,
        path: str,
        timestamp: int | None = None,
    ) -> dict[str, str]:
        issued = int(time.time()) if timestamp is None else timestamp
        signature = cls.sign(
            secret=secret.encode("utf-8"), operator_id=operator_id, operator_role=operator_role.value,
            assurance_level=assurance_level, terminal_id=terminal_id, timestamp=issued,
            nonce=nonce, method=method, path=path,
        )
        return {
            "X-Identity-Operator-Id": operator_id,
            "X-Identity-Operator-Role": operator_role.value,
            "X-Identity-Assurance-Level": assurance_level,
            "X-Identity-Terminal-Id": terminal_id,
            "X-Identity-Timestamp": str(issued),
            "X-Identity-Nonce": nonce,
            "X-Identity-Signature": signature,
        }

    @staticmethod
    def sign(
        *, secret: bytes, operator_id: str, operator_role: str, assurance_level: str,
        terminal_id: str, timestamp: int, nonce: str, method: str, path: str,
    ) -> str:
        canonical = "\n".join(
            (operator_id, operator_role, assurance_level.lower(), terminal_id, str(timestamp), nonce, method.upper(), path)
        )
        return hmac.new(secret, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
