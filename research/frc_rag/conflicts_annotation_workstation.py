"""Local-only workstation for completing frozen CONFLICTS review batches."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from research.frc_rag.conflicts_annotation_operations import (
    BATCH_SCHEMA_VERSION,
    BATCH_SUBMISSION_SCHEMA_VERSION,
    REVIEWER_SLOTS,
)
from research.frc_rag.conflicts_expected_behavior import (
    ALIASES,
    ANSWER_RATINGS,
    PREFERENCES,
    RATING_FIELDS,
    TERNARY_RATINGS,
    canonical_json_sha256,
)


WORKSTATION_SCHEMA_VERSION = "frc-conflicts-annotation-workstation-v1"
RECEIPT_SCHEMA_VERSION = "frc-conflicts-annotation-batch-receipt-v1"
MANIFEST_SCHEMA_VERSION = "frc-conflicts-annotation-operations-manifest-v1"
EXPECTED_OPERATIONS_MANIFEST_SHA256 = (
    "8af745c52f02c34207559806873d67ec5649104194a590ab82f906dfc0081ae3"
)
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_IDENTITY_LENGTH = 128
MAX_RATIONALE_LENGTH = 4000
MAX_NOTES_LENGTH = 4000
ASSET_ROOT = Path(__file__).resolve().parent / "annotation_workstation"
PRIVATE_PUBLIC_KEYS = {
    "item_id",
    "occurrence",
    "paired_primary_task_id",
    "method",
    "methods",
    "score",
    "scores",
    "role_score",
    "role_scores",
    "routing",
}
METHOD_MARKERS = ("coverage_greedy_proxy", "frc_select")


class WorkstationValidationError(ValueError):
    """A user-correctable workstation request error."""

    def __init__(
        self,
        message: str,
        *,
        status: HTTPStatus = HTTPStatus.UNPROCESSABLE_ENTITY,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.details = details or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested_keys(payload: Any) -> set[str]:
    if isinstance(payload, dict):
        return set(payload) | {
            key for value in payload.values() for key in _nested_keys(value)
        }
    if isinstance(payload, list):
        return {key for value in payload for key in _nested_keys(value)}
    return set()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().upper()
    return len(value.strip()) < 3 or normalized.startswith("REPLACE_WITH")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class WorkstationSession:
    """Load one frozen public batch and persist its private reviewer draft."""

    def __init__(
        self,
        *,
        repository_root: Path,
        public_root: Path,
        draft_root: Path,
        reviewer_slot: str,
        batch_index: int,
        expected_manifest_sha256: str = EXPECTED_OPERATIONS_MANIFEST_SHA256,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.public_root = public_root.resolve()
        self.draft_root = draft_root.resolve()
        self.reviewer_slot = reviewer_slot
        self.batch_index = batch_index
        self._lock = threading.RLock()

        if reviewer_slot not in REVIEWER_SLOTS:
            raise ValueError("unknown CONFLICTS reviewer slot")
        if batch_index < 1:
            raise ValueError("batch index must be positive")
        private_cache_root = (self.repository_root / ".cache").resolve()
        if not _is_relative_to(self.draft_root, private_cache_root):
            raise ValueError("workstation drafts must remain inside the ignored .cache tree")

        manifest_path = self.public_root / "manifest.json"
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None
            or not manifest_path.is_file()
            or not hmac.compare_digest(
                _file_sha256(manifest_path), expected_manifest_sha256
            )
        ):
            raise ValueError(
                "annotation operations manifest hash differs from frozen contract"
            )
        self.manifest = _load_json(manifest_path)
        if self.manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported annotation operations manifest")
        if self.manifest.get("status") != "PREPARED_AWAITING_INDEPENDENT_HUMAN_REVIEW":
            raise ValueError("annotation operations are not frozen for independent review")
        if batch_index > int(self.manifest.get("batch_count", 0)):
            raise ValueError("batch index exceeds the frozen reviewer batch count")

        batch_id = f"{reviewer_slot.upper()}-B{batch_index:02d}"
        entries = [
            entry
            for entry in self.manifest.get("public_files", [])
            if entry.get("reviewer_slot") == reviewer_slot
            and entry.get("batch_id") == batch_id
        ]
        by_kind = {str(entry.get("kind")): entry for entry in entries}
        if set(by_kind) != {"batch", "submission_template"}:
            raise ValueError("manifest does not contain one frozen batch/template pair")

        self.batch_path = self._verified_public_path(by_kind["batch"])
        self.template_path = self._verified_public_path(
            by_kind["submission_template"]
        )
        self.batch = _load_json(self.batch_path)
        self.template = _load_json(self.template_path)
        self._validate_public_pair(batch_id)
        self.task_by_id = {
            str(task["review_task_id"]): task for task in self.batch["tasks"]
        }

        slot_directory = self.draft_root / reviewer_slot.replace("_", "-")
        self.draft_path = slot_directory / f"batch-{batch_index:02d}-submission.json"
        self.receipt_path = slot_directory / f"batch-{batch_index:02d}-receipt.json"
        if self.draft_path.exists():
            self.draft = _load_json(self.draft_path)
            self._validate_submission_shape(self.draft)
        else:
            self.draft = copy.deepcopy(self.template)
            _atomic_write_json(self.draft_path, self.draft)

    def _verified_public_path(self, entry: dict[str, Any]) -> Path:
        relative = Path(str(entry.get("path", "")))
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("annotation manifest contains an unsafe public path")
        path = (self.public_root / relative).resolve()
        if not _is_relative_to(path, self.public_root) or not path.is_file():
            raise ValueError("annotation public file is missing or outside its root")
        expected = str(entry.get("sha256", ""))
        if not hmac.compare_digest(_file_sha256(path), expected):
            raise ValueError("annotation public file hash differs from its manifest")
        return path

    def _validate_public_pair(self, batch_id: str) -> None:
        if self.batch.get("schema_version") != BATCH_SCHEMA_VERSION:
            raise ValueError("unsupported annotation batch schema")
        if self.template.get("schema_version") != BATCH_SUBMISSION_SCHEMA_VERSION:
            raise ValueError("unsupported annotation batch template schema")
        for field in ("operations_id", "package_id", "reviewer_slot", "batch_id"):
            if self.batch.get(field) != self.template.get(field):
                raise ValueError(f"annotation batch/template {field} differs")
        if self.batch.get("operations_id") != self.manifest.get("operations_id"):
            raise ValueError("annotation batch operation id differs from manifest")
        if self.batch.get("package_id") != self.manifest.get("package_id"):
            raise ValueError("annotation batch package id differs from manifest")
        if self.batch.get("reviewer_slot") != self.reviewer_slot:
            raise ValueError("annotation batch reviewer slot differs")
        if self.batch.get("batch_id") != batch_id:
            raise ValueError("annotation batch id differs")
        task_ids = [str(task.get("review_task_id", "")) for task in self.batch.get("tasks", [])]
        decision_ids = [
            str(decision.get("review_task_id", ""))
            for decision in self.template.get("decisions", [])
        ]
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("annotation batch task ids are empty or duplicated")
        if task_ids != decision_ids or self.batch.get("task_count") != len(task_ids):
            raise ValueError("annotation batch/template task order differs")
        if not all(
            [response.get("response_id") for response in task.get("responses", [])]
            == ["A", "B"]
            and all(response.get("sources") for response in task["responses"])
            for task in self.batch["tasks"]
        ):
            raise ValueError("annotation batch must contain sourced blinded A/B responses")
        public_payload = [self.batch, self.template]
        if PRIVATE_PUBLIC_KEYS & _nested_keys(public_payload):
            raise ValueError("annotation public pair leaks private routing fields")
        public_text = json.dumps(public_payload, ensure_ascii=False, sort_keys=True)
        if any(marker in public_text for marker in METHOD_MARKERS):
            raise ValueError("annotation public pair leaks retrieval method identity")

    def _validate_submission_shape(self, submission: dict[str, Any]) -> None:
        expected_top_level = {
            "schema_version",
            "operations_id",
            "package_id",
            "reviewer_slot",
            "batch_id",
            "annotator_id",
            "decisions",
        }
        if set(submission) != expected_top_level:
            raise WorkstationValidationError("submission fields differ from the frozen template")
        for field in (
            "schema_version",
            "operations_id",
            "package_id",
            "reviewer_slot",
            "batch_id",
        ):
            if submission.get(field) != self.template.get(field):
                raise WorkstationValidationError(f"submission {field} is immutable")
        identity = submission.get("annotator_id")
        if not isinstance(identity, str) or len(identity) > MAX_IDENTITY_LENGTH:
            raise WorkstationValidationError("annotator id is invalid or too long")
        decisions = submission.get("decisions")
        if not isinstance(decisions, list):
            raise WorkstationValidationError("submission decisions must be a list")
        expected_ids = [
            str(decision["review_task_id"]) for decision in self.template["decisions"]
        ]
        actual_ids = [str(decision.get("review_task_id", "")) for decision in decisions]
        if actual_ids != expected_ids:
            raise WorkstationValidationError("submission task order is immutable")
        for decision in decisions:
            self._validate_partial_decision(decision)

    def _validate_partial_decision(self, decision: dict[str, Any]) -> None:
        if set(decision) != {"review_task_id", "ratings", "preference", "notes"}:
            raise WorkstationValidationError("decision fields differ from the frozen template")
        task_id = str(decision["review_task_id"])
        task = self.task_by_id.get(task_id)
        if task is None:
            raise WorkstationValidationError("submission contains an unknown review task")
        ratings = decision.get("ratings")
        if not isinstance(ratings, dict) or set(ratings) != set(ALIASES):
            raise WorkstationValidationError("decision must contain blinded A/B ratings")
        allowed_ternary = set(TERNARY_RATINGS) | {"REQUIRED"}
        for rating in ratings.values():
            if set(rating) != set(RATING_FIELDS) | {"rationale"}:
                raise WorkstationValidationError("rating fields differ from the frozen template")
            for field in RATING_FIELDS[:3]:
                if rating.get(field) not in allowed_ternary:
                    raise WorkstationValidationError(f"invalid partial {field} rating")
            answer = rating.get("answer_correctness")
            if task.get("correct_answer") is None:
                if answer != "NOT_APPLICABLE":
                    raise WorkstationValidationError(
                        "answer correctness is fixed to NOT_APPLICABLE without gold"
                    )
            elif answer not in (set(ANSWER_RATINGS) - {"NOT_APPLICABLE"}) | {
                "REQUIRED"
            }:
                raise WorkstationValidationError("invalid partial answer correctness rating")
            rationale = rating.get("rationale")
            if not isinstance(rationale, str) or len(rationale) > MAX_RATIONALE_LENGTH:
                raise WorkstationValidationError("rating rationale is invalid or too long")
        if decision.get("preference") not in set(PREFERENCES) | {"REQUIRED"}:
            raise WorkstationValidationError("invalid partial pair preference")
        notes = decision.get("notes")
        if not isinstance(notes, str) or len(notes) > MAX_NOTES_LENGTH:
            raise WorkstationValidationError("decision notes are invalid or too long")

    def _decision_errors(self, decision: dict[str, Any]) -> list[str]:
        task = self.task_by_id[str(decision["review_task_id"])]
        errors: list[str] = []
        for alias in ALIASES:
            rating = decision["ratings"][alias]
            for field in RATING_FIELDS[:3]:
                if rating[field] not in TERNARY_RATINGS:
                    errors.append(f"{alias}.{field}")
            answer = rating["answer_correctness"]
            if task.get("correct_answer") is None:
                if answer != "NOT_APPLICABLE":
                    errors.append(f"{alias}.answer_correctness")
            elif answer not in set(ANSWER_RATINGS) - {"NOT_APPLICABLE"}:
                errors.append(f"{alias}.answer_correctness")
            rationale = str(rating["rationale"]).strip()
            if len(rationale) < 3 or rationale.startswith("REQUIRED"):
                errors.append(f"{alias}.rationale")
        if decision["preference"] not in PREFERENCES:
            errors.append("preference")
        return errors

    def progress(self) -> dict[str, Any]:
        decisions = self.draft["decisions"]
        task_status = []
        for index, decision in enumerate(decisions, start=1):
            errors = self._decision_errors(decision)
            task_status.append(
                {
                    "review_task_id": decision["review_task_id"],
                    "index": index,
                    "complete": not errors,
                    "missing_fields": errors,
                }
            )
        completed = sum(item["complete"] for item in task_status)
        identity_ready = not _is_placeholder(str(self.draft.get("annotator_id", "")))
        return {
            "completed_task_count": completed,
            "total_task_count": len(task_status),
            "identity_ready": identity_ready,
            "batch_complete": completed == len(task_status) and identity_ready,
            "task_status": task_status,
        }

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": WORKSTATION_SCHEMA_VERSION,
                "status": "FINALIZED" if self.receipt_path.exists() else "IN_PROGRESS",
                "batch": copy.deepcopy(self.batch),
                "draft": copy.deepcopy(self.draft),
                "progress": self.progress(),
                "draft_path_label": self.draft_path.name,
                "receipt": _load_json(self.receipt_path)
                if self.receipt_path.exists()
                else None,
                "privacy": {
                    "local_only": True,
                    "draft_git_ignored": True,
                    "routing_loaded": False,
                    "method_identity_loaded": False,
                },
            }

    def save(self, submission: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._validate_submission_shape(submission)
            self.draft = copy.deepcopy(submission)
            if self.receipt_path.exists():
                self.receipt_path.unlink()
            _atomic_write_json(self.draft_path, self.draft)
            return self.state()

    def finalize(self, submission: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.save(submission)
            progress = self.progress()
            if not progress["identity_ready"]:
                raise WorkstationValidationError(
                    "请先填写本角色全部批次共用的私有评审员标识。",
                    details={"code": "ANNOTATOR_ID_REQUIRED"},
                )
            incomplete = [
                item for item in progress["task_status"] if not item["complete"]
            ]
            if incomplete:
                raise WorkstationValidationError(
                    f"本批仍有 {len(incomplete)} 项未完成。",
                    details={
                        "code": "INCOMPLETE_BATCH",
                        "incomplete_count": len(incomplete),
                        "first_review_task_id": incomplete[0]["review_task_id"],
                        "missing_fields": incomplete[0]["missing_fields"],
                    },
                )
            receipt = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "status": "FINALIZED_PRIVATE_BATCH",
                "operations_id": self.draft["operations_id"],
                "package_id": self.draft["package_id"],
                "reviewer_slot": self.reviewer_slot,
                "batch_id": self.draft["batch_id"],
                "submission_sha256": canonical_json_sha256(self.draft),
                "finalized_at": datetime.now(UTC).isoformat(),
                "human_evidence_complete": False,
                "gate_2": "NO-GO/SHADOW",
            }
            _atomic_write_json(self.receipt_path, receipt)
            return self.state()

    def reopen(self) -> dict[str, Any]:
        with self._lock:
            if self.receipt_path.exists():
                self.receipt_path.unlink()
            return self.state()


def create_workstation_server(
    session: WorkstationSession,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    token: str | None = None,
) -> tuple[ThreadingHTTPServer, str]:
    """Create a token-protected loopback server and return it with its launch URL."""

    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("annotation workstation must bind to loopback")
    session_token = token or secrets.token_urlsafe(32)

    class WorkstationHandler(BaseHTTPRequestHandler):
        server_version = "CONFLICTSWorkstation/1"
        sys_version = ""

        def log_message(self, format_string: str, *args: Any) -> None:
            safe_path = urlsplit(self.path).path
            print(f"workstation {self.command} {safe_path} {args[1] if len(args) > 1 else ''}")

        def _security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")

        def _valid_host(self) -> bool:
            host_header = self.headers.get("Host", "")
            return host_header in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def _valid_origin(self) -> bool:
            origin = self.headers.get("Origin")
            if not origin:
                return True
            return origin in {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }

        def _valid_api_token(self) -> bool:
            supplied = self.headers.get("X-Review-Token", "")
            return hmac.compare_digest(supplied, session_token)

        def _valid_page_cookie(self) -> bool:
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
            except CookieError:
                return False
            supplied = cookie.get("FRCReviewSession")
            return supplied is not None and hmac.compare_digest(
                supplied.value, session_token
            )

        def _write_json_response(
            self, status: HTTPStatus, payload: dict[str, Any]
        ) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self._security_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _write_error(self, error: Exception) -> None:
            if isinstance(error, WorkstationValidationError):
                self._write_json_response(
                    error.status,
                    {"error": str(error), "details": error.details},
                )
                return
            self._write_json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "本地工作台发生内部错误。", "details": {}},
            )

        def send_error(  # type: ignore[override]
            self,
            code: int,
            message: str | None = None,
            explain: str | None = None,
        ) -> None:
            del explain
            self._write_json_response(
                HTTPStatus(code),
                {"error": message or HTTPStatus(code).phrase, "details": {}},
            )

        def _authorize_api(self) -> bool:
            if not self._valid_host() or not self._valid_origin():
                self._write_json_response(
                    HTTPStatus.FORBIDDEN,
                    {"error": "工作台拒绝非本机来源。", "details": {}},
                )
                return False
            if not self._valid_api_token():
                self._write_json_response(
                    HTTPStatus.FORBIDDEN,
                    {"error": "工作台会话令牌无效。", "details": {}},
                )
                return False
            return True

        def _read_json_body(self) -> dict[str, Any]:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("application/json"):
                raise WorkstationValidationError(
                    "请求必须使用 application/json。",
                    status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise WorkstationValidationError(
                    "请求长度无效。", status=HTTPStatus.BAD_REQUEST
                ) from exc
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise WorkstationValidationError(
                    "请求为空或超过本地工作台限制。",
                    status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                )
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise WorkstationValidationError(
                    "请求 JSON 无效。", status=HTTPStatus.BAD_REQUEST
                ) from exc
            if not isinstance(payload, dict):
                raise WorkstationValidationError(
                    "请求 JSON 必须是对象。", status=HTTPStatus.BAD_REQUEST
                )
            return payload

        def _serve_asset(
            self,
            name: str,
            content_type: str,
            *,
            establish_page_session: bool = False,
        ) -> None:
            path = ASSET_ROOT / name
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            encoded = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._security_headers()
            self.send_header("Content-Type", content_type)
            if establish_page_session:
                self.send_header(
                    "Set-Cookie",
                    "FRCReviewSession="
                    f"{session_token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=28800",
                )
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            parsed = urlsplit(self.path)
            if not self._valid_host():
                self._write_json_response(
                    HTTPStatus.FORBIDDEN,
                    {"error": "工作台拒绝无效主机。", "details": {}},
                )
                return
            if parsed.path == "/":
                supplied = parse_qs(parsed.query).get("token", [""])[0]
                query_token_valid = hmac.compare_digest(supplied, session_token)
                if not query_token_valid and not self._valid_page_cookie():
                    self._write_json_response(
                        HTTPStatus.FORBIDDEN,
                        {"error": "工作台启动链接无效。", "details": {}},
                    )
                    return
                self._serve_asset(
                    "index.html",
                    "text/html; charset=utf-8",
                    establish_page_session=query_token_valid,
                )
                return
            if parsed.path == "/styles.css":
                self._serve_asset("styles.css", "text/css; charset=utf-8")
                return
            if parsed.path == "/app.js":
                self._serve_asset("app.js", "text/javascript; charset=utf-8")
                return
            if parsed.path in {"/api/session", "/api/export"}:
                if not self._authorize_api():
                    return
                if parsed.path == "/api/session":
                    self._write_json_response(HTTPStatus.OK, session.state())
                    return
                encoded = json.dumps(
                    session.draft, ensure_ascii=False, indent=2
                ).encode("utf-8") + b"\n"
                self.send_response(HTTPStatus.OK)
                self._security_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{session.draft_path.name}"',
                )
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if parsed.path.startswith("/api/"):
                if not self._authorize_api():
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            parsed = urlsplit(self.path)
            if parsed.path != "/api/draft":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not self._authorize_api():
                return
            try:
                self._write_json_response(
                    HTTPStatus.OK, session.save(self._read_json_body())
                )
            except Exception as exc:  # noqa: BLE001 - HTTP error boundary
                self._write_error(exc)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            parsed = urlsplit(self.path)
            if parsed.path not in {"/api/finalize", "/api/reopen"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not self._authorize_api():
                return
            try:
                state = (
                    session.finalize(self._read_json_body())
                    if parsed.path == "/api/finalize"
                    else session.reopen()
                )
                self._write_json_response(HTTPStatus.OK, state)
            except Exception as exc:  # noqa: BLE001 - HTTP error boundary
                self._write_error(exc)

    server = ThreadingHTTPServer((host, port), WorkstationHandler)
    server.daemon_threads = True
    launch_url = f"http://{host}:{server.server_port}/?token={session_token}"
    return server, launch_url


def serve_workstation(
    session: WorkstationSession,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server, launch_url = create_workstation_server(session, host=host, port=port)
    print("CONFLICTS 独立盲评工作台已启动：")
    print(launch_url)
    print(f"私有草稿：{session.draft_path}")
    print("仅监听本机；按 Ctrl+C 停止。")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
