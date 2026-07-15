from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


class DataProtectionError(ValueError):
    pass


class BackupManifestError(ValueError):
    pass


class BackupManifestSigner:
    def __init__(self, secret: str) -> None:
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("backup manifest secret must contain at least 32 bytes")
        self._secret = secret.encode("utf-8")

    @classmethod
    def from_environment(cls) -> "BackupManifestSigner":
        configured = os.getenv("FLOOD_BACKUP_MANIFEST_SECRET", "").strip()
        environment = os.getenv("FLOOD_ENVIRONMENT", "development").strip().lower()
        if not configured and environment in {"production", "prod"}:
            raise BackupManifestError("FLOOD_BACKUP_MANIFEST_SECRET is required in production")
        return cls(configured or "local-development-backup-manifest-secret-change-in-production-v1")

    def sign(self, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hmac.new(self._secret, canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify(self, payload: dict[str, Any], signature: str) -> None:
        if not signature or not hmac.compare_digest(self.sign(payload), signature.lower()):
            raise BackupManifestError("backup manifest signature verification failed")


class DataProtector:
    """Authenticated encryption for repository payloads with plaintext migration support."""

    ENVELOPE_VERSION = 1

    def __init__(self, key: bytes, decryption_keys: list[bytes] | tuple[bytes, ...] = ()) -> None:
        self._key = key
        self.key_id = self.key_identifier(key)
        self._keys: dict[str, bytes] = {}
        self._fernets: dict[str, Fernet] = {}
        for candidate in (key, *decryption_keys):
            candidate_id = self.key_identifier(candidate)
            if candidate_id in self._keys:
                continue
            self._keys[candidate_id] = candidate
            self._fernets[candidate_id] = Fernet(candidate)
        self._fernet = self._fernets[self.key_id]

    @classmethod
    def for_database(cls, db_path: Path) -> "DataProtector":
        configured = os.getenv("FLOOD_DATA_ENCRYPTION_KEY", "").strip()
        if configured:
            active_key = configured.encode("ascii")
            previous = cls._environment_decryption_keys()
            return cls(active_key, previous)
        if os.getenv("FLOOD_ENVIRONMENT", "development").strip().lower() in {"production", "prod"}:
            raise DataProtectionError("FLOOD_DATA_ENCRYPTION_KEY is required in production")
        key_path = db_path.with_suffix(f"{db_path.suffix}.key")
        if key_path.exists():
            key = key_path.read_bytes().strip()
        else:
            key = Fernet.generate_key()
            key_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with key_path.open("xb") as handle:
                    handle.write(key + b"\n")
                try:
                    key_path.chmod(0o600)
                except OSError:
                    pass
            except FileExistsError:
                key = key_path.read_bytes().strip()
        return cls(key, cls._environment_decryption_keys())

    @staticmethod
    def _environment_decryption_keys() -> list[bytes]:
        raw = os.getenv("FLOOD_DATA_DECRYPTION_KEYS", "")
        return [item.strip().encode("ascii") for item in raw.split(",") if item.strip()]

    @staticmethod
    def key_identifier(key: bytes) -> str:
        return hashlib.sha256(key).hexdigest()[:16]

    def has_key(self, key_id: str) -> bool:
        return key_id in self._keys

    def with_active_key(self, new_key: bytes) -> "DataProtector":
        previous_keys = [self._key, *[key for key_id, key in self._keys.items() if key_id != self.key_id]]
        return DataProtector(new_key, previous_keys)

    def encrypt_payload(self, raw_json: str) -> str:
        token = self._fernet.encrypt(raw_json.encode("utf-8")).decode("ascii")
        return json.dumps(
            {
                "protected": True,
                "version": self.ENVELOPE_VERSION,
                "algorithm": "fernet-aes128-cbc-hmac-sha256",
                "key_id": self.key_id,
                "ciphertext": token,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def decrypt_payload(self, stored_payload: str) -> str:
        try:
            envelope = json.loads(stored_payload)
        except json.JSONDecodeError as exc:
            raise DataProtectionError("stored payload is not valid JSON") from exc
        if not isinstance(envelope, dict) or envelope.get("protected") is not True:
            return stored_payload
        envelope_key_id = str(envelope.get("key_id", ""))
        fernet = self._fernets.get(envelope_key_id)
        if fernet is None:
            raise DataProtectionError("encrypted payload key identifier is not present in the active key ring")
        try:
            return fernet.decrypt(str(envelope["ciphertext"]).encode("ascii")).decode("utf-8")
        except (InvalidToken, KeyError, UnicodeDecodeError) as exc:
            raise DataProtectionError("encrypted payload authentication failed") from exc

    @staticmethod
    def is_encrypted_payload(stored_payload: str) -> bool:
        try:
            envelope = json.loads(stored_payload)
        except json.JSONDecodeError:
            return False
        return isinstance(envelope, dict) and envelope.get("protected") is True

    @staticmethod
    def payload_key_id(stored_payload: str) -> str | None:
        try:
            envelope = json.loads(stored_payload)
        except json.JSONDecodeError:
            return None
        if isinstance(envelope, dict) and envelope.get("protected") is True:
            return str(envelope.get("key_id", "")) or None
        return None


@dataclass
class PrivacySanitizationReport:
    sanitized_payload: dict[str, Any]
    redaction_count: int
    redaction_types: dict[str, int] = field(default_factory=dict)
    source_digest: str = ""
    sanitized_digest: str = ""

    def audit_details(self) -> dict[str, Any]:
        return {
            "redaction_count": self.redaction_count,
            "redaction_types": dict(self.redaction_types),
            "source_digest": self.source_digest,
            "sanitized_digest": self.sanitized_digest,
        }


class OutboundPrivacyFilter:
    """Remove personal identifiers before a payload crosses the external model boundary."""

    SENSITIVE_KEYS = {
        "phone", "mobile", "telephone", "contact_phone", "id_card", "identity_number",
        "resident_name", "contact_name", "person_name", "home_address", "resident_address",
        "special_population_notes", "emergency_contacts",
    }
    PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
    ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

    @classmethod
    def sanitize(cls, payload: dict[str, Any]) -> PrivacySanitizationReport:
        counts: dict[str, int] = {}

        def count(kind: str) -> None:
            counts[kind] = counts.get(kind, 0) + 1

        def clean(value: Any, key: str = "") -> Any:
            normalized = key.lower()
            if normalized in cls.SENSITIVE_KEYS:
                count(f"field:{normalized}")
                return f"[REDACTED:{normalized}]"
            if isinstance(value, dict):
                return {str(item_key): clean(item_value, str(item_key)) for item_key, item_value in value.items()}
            if isinstance(value, list):
                return [clean(item, key) for item in value]
            if isinstance(value, str):
                cleaned, phone_count = cls.PHONE_PATTERN.subn("[REDACTED:phone]", value)
                if phone_count:
                    counts["pattern:phone"] = counts.get("pattern:phone", 0) + phone_count
                cleaned, card_count = cls.ID_CARD_PATTERN.subn("[REDACTED:id_card]", cleaned)
                if card_count:
                    counts["pattern:id_card"] = counts.get("pattern:id_card", 0) + card_count
                cleaned, email_count = cls.EMAIL_PATTERN.subn("[REDACTED:email]", cleaned)
                if email_count:
                    counts["pattern:email"] = counts.get("pattern:email", 0) + email_count
                return cleaned
            return value

        source_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        sanitized = clean(payload)
        sanitized_json = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return PrivacySanitizationReport(
            sanitized_payload=sanitized,
            redaction_count=sum(counts.values()),
            redaction_types=counts,
            source_digest=hashlib.sha256(source_json.encode("utf-8")).hexdigest(),
            sanitized_digest=hashlib.sha256(sanitized_json.encode("utf-8")).hexdigest(),
        )
