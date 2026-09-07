from __future__ import annotations

import hashlib
import hmac
import json
import os

from ..simulation_dataset import SimulatedGatewayTimeout

from .models import (
    ApprovalRecord,
    CallbackSequenceState,
    DispatchCallbackRecord,
    DispatchCallbackRequest,
    DispatchCallbackStatus,
    LegacyAdapterCallRecord,
    LegacyMigrationRequest,
    FeatureFlagSetting,
    FeatureFlagUpdateRequest,
    OperatorRole,
    OutboxMessage,
    OutboxProcessRequest,
    OutboxStatus,
    MigrationBatchRecord,
    MigrationQuarantineItem,
    ResponseTask,
)



class DispatchWorkflowMixin:
    def list_feature_flags(self) -> list[FeatureFlagSetting]:
        return self.repository.list_feature_flags()

    def update_feature_flag(
        self, request: FeatureFlagUpdateRequest
    ) -> FeatureFlagSetting:
        self._require_role(
            request.operator_role, {OperatorRole.ADMIN}, "change a feature flag"
        )
        if not request.flag_key.startswith("feature."):
            raise ValueError("feature flag keys must start with 'feature.'")
        matching = [
            item
            for item in self.repository.list_feature_flags()
            if item.flag_key == request.flag_key
            and item.environment == request.environment
            and item.event_id == request.event_id
            and item.role == request.role
            and item.scenario == request.scenario
        ]
        setting = FeatureFlagSetting(
            flag_key=request.flag_key,
            enabled=request.enabled,
            environment=request.environment,
            event_id=request.event_id,
            role=request.role,
            scenario=request.scenario,
            version=(matching[0].version + 1) if matching else 1,
            reason=request.reason,
            updated_by=request.operator_id,
            terminal_id=request.terminal_id,
            updated_at=self._now(),
        )
        self.repository.save_feature_flag(setting)
        if request.event_id:
            self._record(
                event_id=request.event_id,
                entry_type="configuration",
                action="feature_flag_changed",
                actor_id=request.operator_id,
                actor_role=request.operator_role.value,
                terminal_id=request.terminal_id,
                after_state={
                    "flag_key": request.flag_key,
                    "enabled": request.enabled,
                    "version": setting.version,
                },
                detail={"reason": request.reason, "environment": request.environment},
            )
        return setting

    def is_feature_enabled(
        self,
        flag_key: str,
        *,
        environment: str | None = None,
        event_id: str | None = None,
        role: OperatorRole | None = None,
        scenario: str | None = None,
    ) -> bool:
        environment = environment or os.getenv("FLOOD_ENVIRONMENT", "development")
        candidates = [
            item
            for item in self.repository.list_feature_flags()
            if item.flag_key == flag_key
            and item.environment == environment
            and item.event_id in {None, event_id}
            and item.role in {None, role}
            and item.scenario in {None, scenario}
        ]
        if candidates:
            candidates.sort(
                key=lambda item: (
                    int(item.event_id is not None)
                    + int(item.role is not None)
                    + int(item.scenario is not None),
                    item.version,
                ),
                reverse=True,
            )
            return candidates[0].enabled
        return environment != "production" and flag_key in {
            "feature.new_warning_model",
            "feature.object_candidate_service",
            "feature.structured_task_draft",
            "feature.rule_engine",
            "feature.new_approval_flow",
            "feature.task_state_machine",
            "feature.simulated_dispatch",
            "feature.audit_hash_chain",
        }

    def list_outbox_messages(
        self, *, event_id: str | None = None
    ) -> list[OutboxMessage]:
        return self.repository.list_outbox_messages(event_id=event_id)

    def migration_status(self) -> dict:
        return {
            "schema_migrations": self.repository.list_schema_migrations(),
            "migration_batches": self.repository.list_migration_batches(),
            "legacy_adapter_calls": len(self.repository.list_legacy_adapter_calls()),
            "core_write_authority": "/response",
            "legacy_write_policy": "denied",
        }

    def run_legacy_migration_inventory(
        self, request: LegacyMigrationRequest
    ) -> MigrationBatchRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN},
            "run a legacy migration inventory",
        )
        legacy_events = self.repository.list_v2_events(limit=10000)
        canonical_rows = [
            item.model_dump(mode="json")
            for item in sorted(legacy_events, key=lambda item: item.event_id)
        ]
        source_raw = json.dumps(
            canonical_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        quarantine = []
        for item in legacy_events:
            raw = json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            quarantine.append(
                MigrationQuarantineItem(
                    source_id=item.event_id,
                    reason_code="MISSING_AUTHORITATIVE_WARNING_REVISION",
                    reason="旧事件缺少可验证的专业预警原始载荷、版本哈希和发布来源；不得伪造为新版正式事件。",
                    source_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                )
            )
        record = MigrationBatchRecord(
            batch_id=self._id("MIGRATION"),
            source_table="v2_events",
            target_domain="response_events",
            mapping_version=request.mapping_version,
            dry_run=request.dry_run,
            source_count=len(legacy_events),
            migrated_count=0,
            quarantined_count=len(quarantine),
            ignored_count=0,
            source_hash=hashlib.sha256(source_raw.encode("utf-8")).hexdigest(),
            status="completed_with_quarantine" if quarantine else "completed_empty",
            quarantine=quarantine,
            executed_by=request.operator_id,
            terminal_id=request.terminal_id,
            created_at=self._now(),
        )
        self.repository.save_migration_batch(record)
        return record

    def read_legacy_event(
        self,
        legacy_event_id: str,
        *,
        operator_id: str,
        terminal_id: str,
        trace_id: str,
    ) -> dict:
        legacy = self.repository.get_v2_event(legacy_event_id)
        status = "found" if legacy else "not_found"
        call = LegacyAdapterCallRecord(
            call_id=self._id("LEGACYCALL"),
            legacy_endpoint="GET /legacy/events/{event_id}",
            legacy_resource_id=legacy_event_id,
            mapping_version="legacy-v2-read-v1",
            trace_id=trace_id,
            result_status=status,
            operator_id=operator_id,
            terminal_id=terminal_id,
            created_at=self._now(),
        )
        self.repository.save_legacy_adapter_call(call)
        if legacy is None:
            raise LookupError(f"legacy event not found: {legacy_event_id}")
        return {
            "read_only": True,
            "mapping_version": call.mapping_version,
            "trace_id": trace_id,
            "legacy_event_id": legacy.event_id,
            "title": legacy.title,
            "area_id": legacy.area_id,
            "legacy_stage": legacy.current_stage.value,
            "legacy_risk_level": legacy.current_risk_level.value,
            "source_type": "legacy_simulation",
            "migration_eligibility": "quarantine_pending_authoritative_warning",
        }

    def process_outbox(self, request: OutboxProcessRequest) -> list[OutboxMessage]:
        self._require_role(
            request.operator_role,
            {OperatorRole.ADMIN, OperatorRole.LIAISON, OperatorRole.COMMANDER},
            "process the dispatch outbox",
        )
        if request.message_id:
            message = self.repository.get_outbox_message(request.message_id)
            if message is None:
                raise LookupError(f"outbox message not found: {request.message_id}")
            messages = [message]
        else:
            messages = self.repository.list_outbox_messages(
                status=OutboxStatus.PENDING, limit=request.max_messages
            )
        return [
            self._dispatch_outbox_message(
                item, simulation_scenario=request.simulation_scenario
            )
            for item in messages
        ]

    def list_dispatch_callbacks(self, message_id: str) -> list[DispatchCallbackRecord]:
        if self.repository.get_outbox_message(message_id) is None:
            raise LookupError(f"outbox message not found: {message_id}")
        return self.repository.list_dispatch_callbacks(message_id)

    def ingest_simulated_dispatch_callback(
        self, request: DispatchCallbackRequest
    ) -> DispatchCallbackRecord:
        self._require_role(
            request.operator_role,
            {OperatorRole.EXTERNAL_SERVICE},
            "submit a simulated dispatch callback",
        )
        if (
            os.getenv("FLOOD_ENVIRONMENT", "development").strip().lower()
            == "production"
        ):
            raise PermissionError("simulated callbacks are disabled in production")
        expected_token = os.getenv(
            "FLOOD_SIMULATION_MOCK_TOKEN", "simulation-only-change-me"
        )
        if not hmac.compare_digest(request.mock_token, expected_token):
            raise PermissionError("invalid simulation callback token")
        if not request.is_simulated:
            raise ValueError("callback must be explicitly marked as simulated")
        message = self.repository.get_outbox_message(request.message_id)
        if message is None:
            raise LookupError(f"outbox message not found: {request.message_id}")
        callback = self._record_dispatch_callback(
            message,
            request.model_dump(
                mode="json",
                exclude={
                    "operator_id",
                    "operator_role",
                    "terminal_id",
                    "note",
                    "expected_version",
                    "mock_token",
                },
            ),
            actor_id=request.operator_id,
            actor_role=request.operator_role.value,
            terminal_id=request.terminal_id,
        )
        current = self.repository.get_outbox_message(message.message_id) or message
        updated = current.model_copy(
            update={
                "callback_count": len(
                    self.repository.list_dispatch_callbacks(message.message_id)
                ),
                "updated_at": self._now(),
            }
        )
        self.repository.save_outbox_message(updated)
        return callback

    @staticmethod
    def _task_payload_hash(task: ResponseTask) -> str:
        payload = {
            "task_id": task.task_id,
            "event_id": task.event_id,
            "object_id": task.object_id,
            "version": task.version,
            "title": task.title,
            "action": task.action,
            "responsible_organization": task.responsible_organization,
            "responsible_role": task.responsible_role,
            "cooperate_roles": task.cooperate_roles,
            "deadlines": {
                "acknowledge": task.acknowledge_deadline_at.isoformat(),
                "start": task.start_deadline_at.isoformat()
                if task.start_deadline_at
                else None,
                "complete": task.deadline_at.isoformat(),
                "verify": task.verification_deadline_at.isoformat()
                if task.verification_deadline_at
                else None,
            },
            "required_evidence": task.required_evidence,
            "plan_basis": [item.model_dump(mode="json") for item in task.plan_basis],
            "dependencies": task.dependencies,
        }
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _evidence_package_hash(task: ResponseTask) -> str:
        if task.evidence_package_hash:
            return task.evidence_package_hash
        payload = [item.model_dump(mode="json") for item in task.source_evidence]
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_dispatch_outbox(
        self, task: ResponseTask, approval: ApprovalRecord
    ) -> OutboxMessage:
        idempotency_key = (
            f"dispatch:{task.task_id}:v{task.version}:{approval.task_payload_hash}"
        )
        existing = self.repository.get_outbox_message_by_idempotency_key(
            idempotency_key
        )
        if existing is not None:
            return existing
        now = self._now()
        message = OutboxMessage(
            message_id=self._id("OUTBOX"),
            event_id=task.event_id,
            task_id=task.task_id,
            idempotency_key=idempotency_key,
            payload_hash=approval.task_payload_hash,
            approval_id=approval.approval_id,
            task_version=task.version,
            created_at=now,
            updated_at=now,
        )
        return message

    def _dispatch_outbox_message(
        self,
        message: OutboxMessage,
        *,
        simulation_scenario: str = "normal",
    ) -> OutboxMessage:
        if message.status in {
            OutboxStatus.SENT,
            OutboxStatus.PARTIALLY_SENT,
            OutboxStatus.FAILED,
            OutboxStatus.MANUAL_TAKEOVER,
        }:
            return message
        task = self._task(message.task_id)
        if self._task_payload_hash(task) != message.payload_hash:
            failed = message.model_copy(
                update={
                    "status": OutboxStatus.FAILED,
                    "attempts": message.attempts + 1,
                    "last_error": "approved payload hash mismatch",
                    "updated_at": self._now(),
                }
            )
            self.repository.save_outbox_message(failed)
            return failed
        if (
            os.getenv("FLOOD_ENVIRONMENT", "development").strip().lower()
            == "production"
        ):
            manual = message.model_copy(
                update={
                    "status": OutboxStatus.MANUAL_TAKEOVER,
                    "attempts": message.attempts + 1,
                    "last_error": "simulation gateway disabled in production",
                    "updated_at": self._now(),
                }
            )
            self.repository.save_outbox_message(manual)
            return manual
        now = self._now()
        transport_payload = {
            "event_id": task.event_id,
            "task_id": task.task_id,
            "task_version": task.version,
            "approval_id": message.approval_id,
            "delivery_targets": [task.responsible_organization, *task.cooperate_roles],
            "approved_payload_hash": message.payload_hash,
        }
        try:
            result = self.simulation_gateway.dispatch(
                message.destination,
                transport_payload,
                scenario=simulation_scenario,
                event_time=now,
                request_id=f"SIMREQ-{message.message_id}",
                trace_id=f"SIMTRACE-{message.event_id}",
                idempotency_key=message.idempotency_key,
            )
        except SimulatedGatewayTimeout as exc:
            pending = message.model_copy(
                update={
                    "status": OutboxStatus.PENDING,
                    "attempts": message.attempts + 1,
                    "simulation_scenario": simulation_scenario,
                    "gateway_status": "timeout",
                    "last_error": str(exc),
                    "updated_at": now,
                }
            )
            self.repository.save_outbox_message(pending)
            self._record(
                event_id=message.event_id,
                task_id=message.task_id,
                entry_type="dispatch",
                action="simulated_dispatch_timeout",
                actor_id="simulated-gateway",
                actor_role=OperatorRole.EXTERNAL_SERVICE.value,
                detail={
                    "message_id": message.message_id,
                    "scenario": simulation_scenario,
                },
            )
            return pending

        gateway_status = str(result["status"])
        if gateway_status == "rejected":
            status = OutboxStatus.FAILED
        elif gateway_status == "partial_success":
            status = OutboxStatus.PARTIALLY_SENT
        else:
            status = OutboxStatus.SENT
        for callback_payload in result.get("callbacks", []):
            self._record_dispatch_callback(message, callback_payload)
        callbacks = self.repository.list_dispatch_callbacks(message.message_id)
        dispatched = message.model_copy(
            update={
                "status": status,
                "attempts": message.attempts + 1,
                "simulation_scenario": simulation_scenario,
                "gateway_status": gateway_status,
                "external_request_id": result["request_id"],
                "trace_id": result["trace_id"],
                "callback_count": len(callbacks),
                "last_error": result.get("error_code"),
                "updated_at": now,
                "sent_at": now
                if status in {OutboxStatus.SENT, OutboxStatus.PARTIALLY_SENT}
                else None,
            }
        )
        self.repository.save_outbox_message(dispatched)
        self._record(
            event_id=message.event_id,
            task_id=message.task_id,
            entry_type="dispatch",
            action=f"simulated_dispatch_{gateway_status}",
            actor_id="simulated-gateway",
            actor_role=OperatorRole.EXTERNAL_SERVICE.value,
            before_state={"status": message.status.value},
            after_state={"status": status.value},
            detail={
                "message_id": message.message_id,
                "scenario": simulation_scenario,
                "accepted_count": result["accepted_count"],
                "rejected_count": result["rejected_count"],
                "callback_count": len(callbacks),
            },
        )
        return dispatched

    def _record_dispatch_callback(
        self,
        message: OutboxMessage,
        payload: dict,
        *,
        actor_id: str = "simulated-gateway",
        actor_role: str = OperatorRole.EXTERNAL_SERVICE.value,
        terminal_id: str = "simulation-gateway",
    ) -> DispatchCallbackRecord:
        idempotency_key = str(payload["idempotency_key"])
        existing = self.repository.get_dispatch_callback_by_idempotency_key(
            idempotency_key
        )
        if existing is not None:
            self._record(
                event_id=message.event_id,
                task_id=message.task_id,
                entry_type="dispatch_callback",
                action="duplicate_dispatch_callback_ignored",
                actor_id=actor_id,
                actor_role=actor_role,
                terminal_id=terminal_id,
                detail={
                    "message_id": message.message_id,
                    "callback_id": existing.callback_id,
                    "idempotency_key": idempotency_key,
                },
            )
            return existing
        previous = [
            item
            for item in self.repository.list_dispatch_callbacks(message.message_id)
            if item.external_id == str(payload["external_id"])
        ]
        version = int(payload["version"])
        sequence_state = (
            CallbackSequenceState.OUT_OF_ORDER
            if previous and version <= max(item.version for item in previous)
            else CallbackSequenceState.IN_ORDER
        )
        callback = DispatchCallbackRecord(
            callback_id=self._id("CALLBACK"),
            message_id=message.message_id,
            event_id=message.event_id,
            task_id=message.task_id,
            external_id=str(payload["external_id"]),
            source=str(payload["source"]),
            version=version,
            event_time=payload["event_time"],
            received_time=payload.get("received_time") or self._now(),
            request_id=str(payload["request_id"]),
            trace_id=str(payload["trace_id"]),
            idempotency_key=idempotency_key,
            status=DispatchCallbackStatus(str(payload["status"])),
            error_code=payload.get("error_code"),
            sequence_state=sequence_state,
            created_at=self._now(),
        )
        saved = self.repository.save_dispatch_callback(callback)
        if saved.callback_id != callback.callback_id:
            self._record(
                event_id=message.event_id,
                task_id=message.task_id,
                entry_type="dispatch_callback",
                action="duplicate_dispatch_callback_ignored",
                actor_id=actor_id,
                actor_role=actor_role,
                terminal_id=terminal_id,
                detail={
                    "message_id": message.message_id,
                    "callback_id": saved.callback_id,
                    "idempotency_key": idempotency_key,
                    "detected_during_insert": True,
                },
            )
            return saved
        action = (
            "out_of_order_dispatch_callback_recorded"
            if saved.sequence_state == CallbackSequenceState.OUT_OF_ORDER
            else "dispatch_callback_recorded"
        )
        self._record(
            event_id=message.event_id,
            task_id=message.task_id,
            entry_type="dispatch_callback",
            action=action,
            actor_id=actor_id,
            actor_role=actor_role,
            terminal_id=terminal_id,
            detail={
                "message_id": message.message_id,
                "callback_id": saved.callback_id,
                "external_id": saved.external_id,
                "version": saved.version,
                "status": saved.status.value,
                "sequence_state": saved.sequence_state.value,
                "formal_task_state_unchanged": True,
            },
        )
        return saved
