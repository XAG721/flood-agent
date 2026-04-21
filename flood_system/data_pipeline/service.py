from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..v2.models import (
    AlertSeverity,
    DatasetBuildRequest,
    DatasetFetchRequest,
    DatasetJobView,
    DatasetPipelineStatusView,
    DatasetSyncRequest,
)
from .beilin_dataset import (
    BEILIN_AREA_ID,
    build_beilin_profiles,
    compile_beilin_rag,
    fetch_beilin_sources,
    generate_beilin_observations,
    inspect_dataset_status,
    normalize_beilin_sources,
    sync_demo_db,
    validate_beilin_dataset,
)


ACTIVE_JOB_STATUSES = {"pending", "running", "cancel_requested"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "canceled"}
AUTO_RETRY_BY_ACTION = {
    "fetch_sources": 1,
    "build_dataset": 1,
    "validate_dataset": 1,
    "sync_demo_db": 1,
}
JOB_HISTORY_LIMIT = 40


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DatasetJobCanceled(RuntimeError):
    pass


class BeilinDatasetService:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        db_path: str | Path,
        add_audit_record: Callable[..., Any] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.db_path = Path(db_path).resolve()
        self._add_audit_record = add_audit_record
        self._lock = threading.Lock()
        normalized_dir = self.repo_root / "data_sources" / "beilin" / "normalized"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        self._job_state_path = normalized_dir / "dataset_job_state.json"
        self._job_history_path = normalized_dir / "dataset_job_history.json"
        self._cancel_events: dict[str, threading.Event] = {}

    def get_status(self) -> DatasetPipelineStatusView:
        payload = inspect_dataset_status(self.repo_root)
        active_job = self._load_active_job()
        if active_job is not None:
            payload["active_job"] = active_job
        payload["recent_jobs"] = [job.model_dump(mode="json") for job in self.list_jobs()]
        return DatasetPipelineStatusView.model_validate(payload)

    def get_active_job(self) -> DatasetJobView | None:
        return self._load_active_job()

    def list_jobs(self, limit: int = 12) -> list[DatasetJobView]:
        history = self._load_job_history()
        return history[:limit]

    def start_fetch_sources(self, request: DatasetFetchRequest) -> DatasetJobView:
        return self._start_job(
            action="fetch_sources",
            request_payload=request.model_dump(),
            retry_of_job_id=None,
        )

    def start_build(self, request: DatasetBuildRequest) -> DatasetJobView:
        return self._start_job(
            action="build_dataset",
            request_payload=request.model_dump(),
            retry_of_job_id=None,
        )

    def start_validate(self) -> DatasetJobView:
        return self._start_job(
            action="validate_dataset",
            request_payload={},
            retry_of_job_id=None,
        )

    def start_sync(self, request: DatasetSyncRequest) -> DatasetJobView:
        return self._start_job(
            action="sync_demo_db",
            request_payload=request.model_dump(),
            retry_of_job_id=None,
        )

    def cancel_job(self, job_id: str) -> DatasetJobView:
        with self._lock:
            job = self._find_job(job_id)
            if job is None:
                raise ValueError(f"Dataset job {job_id} does not exist.")
            if job.status not in ACTIVE_JOB_STATUSES:
                raise ValueError(f"Dataset job {job_id} is not running.")
            now = _utc_now()
            job.status = "cancel_requested"
            job.cancel_requested = True
            job.cancel_requested_at = now
            job.updated_at = now
            job.current_step = "cancel_requested"
            job.message = "Cancel requested by operator."
            self._persist_job(job)
            self._cancel_events.setdefault(job_id, threading.Event()).set()
        self._audit(
            action="dataset_job_cancel_requested",
            summary=f"Cancel requested for dataset job {job_id}.",
            details={"job_id": job_id},
            severity=AlertSeverity.WARNING,
        )
        return job

    def retry_job(self, job_id: str) -> DatasetJobView:
        with self._lock:
            job = self._find_job(job_id)
            if job is None:
                raise ValueError(f"Dataset job {job_id} does not exist.")
            if job.status not in {"failed", "canceled"}:
                raise ValueError(f"Dataset job {job_id} cannot be retried from status {job.status}.")
        retried = self._start_job(
            action=job.action,
            request_payload=job.request_payload,
            retry_of_job_id=job.job_id,
        )
        self._audit(
            action="dataset_job_retried",
            summary=f"Retried dataset job {job_id}.",
            details={"job_id": job_id, "retry_job_id": retried.job_id},
            severity=AlertSeverity.INFO,
        )
        return retried

    def _start_job(
        self,
        *,
        action: str,
        request_payload: dict[str, Any],
        retry_of_job_id: str | None,
    ) -> DatasetJobView:
        spec = self._build_job_spec(action, request_payload)
        with self._lock:
            active = self._load_active_job()
            if active and active.status in ACTIVE_JOB_STATUSES:
                return active
            now = _utc_now()
            job = DatasetJobView(
                job_id=f"dataset_job_{uuid.uuid4().hex[:12]}",
                action=action,
                status="pending",
                progress_percent=0,
                current_step="queued",
                message="Queued for background execution.",
                source_ids=spec["source_ids"],
                request_payload=request_payload,
                attempt_count=0,
                max_attempts=spec["max_attempts"],
                auto_retry_enabled=spec["max_attempts"] > 1,
                retry_count=0,
                retry_of_job_id=retry_of_job_id,
                cancel_requested=False,
                started_at=now,
                updated_at=now,
                completed_at=None,
                cancel_requested_at=None,
                canceled_at=None,
                error=None,
                result_summary=None,
            )
            self._persist_job(job)
            self._cancel_events[job.job_id] = threading.Event()
            thread = threading.Thread(
                target=self._execute_job,
                args=(job.job_id, spec["runner"], spec["max_attempts"]),
                name=f"dataset-job-{job.job_id}",
                daemon=True,
            )
            thread.start()
            return job

    def _execute_job(
        self,
        job_id: str,
        runner: Callable[[Callable[..., None], Callable[[], None]], dict[str, Any]],
        max_attempts: int,
    ) -> None:
        cancel_event = self._cancel_events.setdefault(job_id, threading.Event())

        def check_cancel() -> None:
            current = self._find_job(job_id)
            if cancel_event.is_set() or (current is not None and current.cancel_requested):
                raise DatasetJobCanceled("Dataset job canceled by operator.")

        def update(
            *,
            progress_percent: int,
            current_step: str,
            message: str,
            error: str | None = None,
            result_summary: str | None = None,
            completed: bool = False,
        ) -> None:
            check_cancel()
            with self._lock:
                state = self._find_job(job_id)
                if state is None:
                    return
                state.status = "completed" if completed and error is None else ("failed" if completed and error else state.status)
                state.progress_percent = progress_percent
                state.current_step = current_step
                state.message = message
                state.updated_at = _utc_now()
                state.error = error
                if result_summary is not None:
                    state.result_summary = result_summary
                if completed:
                    state.completed_at = _utc_now()
                self._persist_job(state)

        try:
            for attempt in range(1, max_attempts + 1):
                with self._lock:
                    state = self._find_job(job_id)
                    if state is None:
                        return
                    state.status = "running"
                    state.attempt_count = attempt
                    state.current_step = "starting" if attempt == 1 else "retrying"
                    state.message = "Background dataset task started." if attempt == 1 else f"Retry attempt {attempt} started."
                    state.updated_at = _utc_now()
                    self._persist_job(state)
                try:
                    result = runner(update, check_cancel)
                    summary = str(result.get("summary") or result.get("action") or "Dataset job completed.")
                    update(
                        progress_percent=100,
                        current_step="completed",
                        message=summary,
                        result_summary=summary,
                        completed=True,
                    )
                    return
                except DatasetJobCanceled as exc:
                    with self._lock:
                        state = self._find_job(job_id)
                        if state is None:
                            return
                        now = _utc_now()
                        state.status = "canceled"
                        state.cancel_requested = True
                        state.current_step = "canceled"
                        state.message = str(exc)
                        state.error = str(exc)
                        state.updated_at = now
                        state.completed_at = now
                        state.canceled_at = now
                        self._persist_job(state)
                    self._audit(
                        action="dataset_job_canceled",
                        summary=f"Dataset job {job_id} was canceled.",
                        details={"job_id": job_id},
                        severity=AlertSeverity.WARNING,
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    if attempt < max_attempts:
                        with self._lock:
                            state = self._find_job(job_id)
                            if state is None:
                                return
                            state.retry_count += 1
                            state.status = "running"
                            state.current_step = "retrying"
                            state.message = f"{exc}. Retrying automatically."
                            state.error = str(exc)
                            state.updated_at = _utc_now()
                            self._persist_job(state)
                        self._audit(
                            action="dataset_job_retry_scheduled",
                            summary=f"Dataset job {job_id} failed and will retry automatically.",
                            details={"job_id": job_id, "attempt": attempt, "error": str(exc)},
                            severity=AlertSeverity.WARNING,
                        )
                        time.sleep(0.2)
                        continue
                    with self._lock:
                        state = self._find_job(job_id)
                        if state is None:
                            return
                        now = _utc_now()
                        state.status = "failed"
                        state.current_step = "failed"
                        state.message = f"Dataset job failed: {exc}"
                        state.error = str(exc)
                        state.updated_at = now
                        state.completed_at = now
                        self._persist_job(state)
                    self._audit(
                        action="dataset_job_failed",
                        summary=f"Dataset background job {job_id} failed.",
                        details={"job_id": job_id, "error": str(exc)},
                        severity=AlertSeverity.WARNING,
                    )
                    return
        finally:
            self._cancel_events.pop(job_id, None)

    def _build_job_spec(
        self,
        action: str,
        request_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if action == "fetch_sources":
            request = DatasetFetchRequest.model_validate(request_payload)
            return {
                "source_ids": request.source_ids,
                "max_attempts": AUTO_RETRY_BY_ACTION[action] + 1,
                "runner": lambda update, check_cancel: self._run_fetch(request, update, check_cancel),
            }
        if action == "build_dataset":
            request = DatasetBuildRequest.model_validate(request_payload)
            return {
                "source_ids": [],
                "max_attempts": AUTO_RETRY_BY_ACTION[action] + 1,
                "runner": lambda update, check_cancel: self._run_build(request, update, check_cancel),
            }
        if action == "validate_dataset":
            return {
                "source_ids": [],
                "max_attempts": AUTO_RETRY_BY_ACTION[action] + 1,
                "runner": lambda update, check_cancel: self._run_validate(update, check_cancel),
            }
        if action == "sync_demo_db":
            request = DatasetSyncRequest.model_validate(request_payload)
            return {
                "source_ids": [],
                "max_attempts": AUTO_RETRY_BY_ACTION[action] + 1,
                "runner": lambda update, check_cancel: self._run_sync(request, update, check_cancel),
            }
        raise ValueError(f"Unsupported dataset job action: {action}")

    def _run_fetch(
        self,
        request: DatasetFetchRequest,
        update: Callable[..., None],
        check_cancel: Callable[[], None],
    ) -> dict[str, Any]:
        def report_progress(payload: dict[str, Any]) -> None:
            check_cancel()
            update(
                progress_percent=int(payload.get("progress_percent", 0)),
                current_step=str(payload.get("current_step", "fetching")),
                message=str(payload.get("message", "Fetching dataset sources.")),
            )

        result = fetch_beilin_sources(
            self.repo_root,
            download=request.download,
            source_ids=request.source_ids,
            force_refresh=request.force_refresh,
            progress=report_progress,
        )
        status = self.get_status()
        summary = (
            f"Fetched raw source caches for {status.cached_source_count} source(s)."
            if request.download
            else f"Refreshed the source registry with {status.source_count} source(s)."
        )
        self._audit(
            action="dataset_sources_fetched",
            summary=summary,
            details=result,
        )
        return {"action": "fetch_sources", "summary": summary, "details": result}

    def _run_build(
        self,
        request: DatasetBuildRequest,
        update: Callable[..., None],
        check_cancel: Callable[[], None],
    ) -> dict[str, Any]:
        check_cancel()
        update(progress_percent=5, current_step="fetch", message="Preparing source registry.")
        fetch_result = fetch_beilin_sources(
            self.repo_root,
            download=request.download,
            progress=lambda payload: (
                check_cancel(),
                update(
                    progress_percent=min(20, int(payload.get("progress_percent", 0))),
                    current_step="fetch",
                    message=str(payload.get("message", "Preparing source registry.")),
                ),
            ),
        )
        check_cancel()
        update(progress_percent=24, current_step="normalize", message="正在规范化已缓存的数据产物。")
        normalize_result = normalize_beilin_sources(self.repo_root)
        check_cancel()
        update(progress_percent=46, current_step="profiles", message="正在生成区域、道路、避难点和画像种子数据。")
        profiles_result = build_beilin_profiles(self.repo_root)
        check_cancel()
        update(progress_percent=64, current_step="observations", message="正在生成碑林区演示观测场景。")
        observations_result = generate_beilin_observations(self.repo_root)
        check_cancel()
        update(progress_percent=78, current_step="rag", message="正在编译运行期知识包。")
        rag_result = compile_beilin_rag(self.repo_root)
        sync_result = None
        if request.sync_demo_db:
            check_cancel()
            update(progress_percent=88, current_step="sync", message="正在同步演示运行库。")
            sync_result = sync_demo_db(self.repo_root, db_path=self.db_path)
        check_cancel()
        update(progress_percent=94, current_step="validate", message="正在校验生成结果。")
        validation_result = validate_beilin_dataset(self.repo_root)
        summary = (
            "已完成数据包构建并同步演示数据库。"
            if request.sync_demo_db
            else "已完成数据包构建。"
        )
        details = {
            "fetch": fetch_result,
            "normalize": normalize_result,
            "profiles": profiles_result,
            "observations": observations_result,
            "rag": rag_result,
            "sync_demo_db": sync_result,
            "validation": validation_result,
        }
        self._audit(
            action="dataset_built",
            summary="已为运行系统完成碑林区数据包构建。",
            details=details,
        )
        return {"action": "build_dataset", "summary": summary, "details": details}

    def _run_validate(
        self,
        update: Callable[..., None],
        check_cancel: Callable[[], None],
    ) -> dict[str, Any]:
        check_cancel()
        update(progress_percent=25, current_step="validation", message="正在校验引导数据、运行库和知识资产。")
        result = validate_beilin_dataset(self.repo_root)
        summary = (
            f"已校验 {result.get('entity_profile_count', 0)} 条对象画像、"
            f"{result.get('shelter_count', 0)} 个避难点和 "
            f"{result.get('road_count', 0)} 条道路。"
        )
        self._audit(
            action="dataset_validated",
            summary="已完成碑林区数据包校验。",
            details=result,
        )
        return {"action": "validate_dataset", "summary": summary, "details": result}

    def _run_sync(
        self,
        request: DatasetSyncRequest,
        update: Callable[..., None],
        check_cancel: Callable[[], None],
    ) -> dict[str, Any]:
        check_cancel()
        update(progress_percent=35, current_step="sync", message="正在将碑林区数据包同步到运行数据库。")
        result = sync_demo_db(self.repo_root, db_path=request.db_path or self.db_path)
        summary = "运行数据库已同步到最新碑林区数据包。"
        self._audit(
            action="dataset_synced",
            summary="已将碑林区数据包同步到运行期 SQLite 数据库。",
            details=result,
        )
        return {"action": "sync_demo_db", "summary": summary, "details": result}

    def _load_active_job(self) -> DatasetJobView | None:
        if not self._job_state_path.exists():
            return None
        try:
            payload = json.loads(self._job_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        job = DatasetJobView.model_validate(payload)
        if job.status not in ACTIVE_JOB_STATUSES:
            return None
        return job

    def _load_job_history(self) -> list[DatasetJobView]:
        if not self._job_history_path.exists():
            return []
        try:
            payload = json.loads(self._job_history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        jobs = [DatasetJobView.model_validate(item) for item in payload]
        jobs.sort(
            key=lambda item: item.updated_at or item.started_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return jobs[:JOB_HISTORY_LIMIT]

    def _save_job_history(self, jobs: list[DatasetJobView]) -> None:
        payload = [job.model_dump(mode="json") for job in jobs[:JOB_HISTORY_LIMIT]]
        self._job_history_path.parent.mkdir(parents=True, exist_ok=True)
        self._job_history_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _find_job(self, job_id: str) -> DatasetJobView | None:
        for job in self._load_job_history():
            if job.job_id == job_id:
                return job
        return None

    def _persist_job(self, job: DatasetJobView) -> None:
        history = [item for item in self._load_job_history() if item.job_id != job.job_id]
        history.insert(0, job)
        self._save_job_history(history)
        if job.status in ACTIVE_JOB_STATUSES:
            self._job_state_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        elif self._job_state_path.exists():
            active = self._load_active_job()
            if active is None or active.job_id == job.job_id:
                self._job_state_path.unlink(missing_ok=True)

    def _audit(
        self,
        *,
        action: str,
        summary: str,
        details: dict[str, Any],
        severity: AlertSeverity = AlertSeverity.INFO,
    ) -> None:
        if self._add_audit_record is None:
            return
        safe_details = json.loads(json.dumps(details, ensure_ascii=False, default=str))
        self._add_audit_record(
            source_type="dataset_pipeline",
            action=action,
            summary=summary,
            details=safe_details,
            severity=severity,
            event_id=BEILIN_AREA_ID,
        )
