from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from flood_system.system import FloodWarningSystem
from flood_system.v2.llm_gateway import LLMGenerationError
from flood_system.v2.models import (
    AdvisoryRequest,
    EventCreateRequest,
    ObservationBatchRequest,
    ObservationIngestItem,
    ProposalResolutionRequest,
    ProposalStatus,
    SimulationCell,
    SimulationUpdateRequest,
    V2CopilotMessageRequest,
    V2CopilotSessionRequest,
)


SMOKE_STAGE_RETRY_ATTEMPTS = max(1, int(os.getenv("FLOOD_SMOKE_STAGE_RETRY_ATTEMPTS", "5")))
SMOKE_STAGE_RETRY_DELAY_SECONDS = max(0.0, float(os.getenv("FLOOD_SMOKE_STAGE_RETRY_DELAY_SECONDS", "2")))


def sample_observations() -> list[ObservationIngestItem]:
    now = datetime.now(timezone.utc)
    return [
        ObservationIngestItem(
            observed_at=now - timedelta(minutes=20),
            source_type="monitoring_point",
            source_name="North monitoring point",
            village="Wuyuanli Village",
            rainfall_mm=33,
            water_level_m=4.3,
            road_blocked=False,
            citizen_reports=2,
        ),
        ObservationIngestItem(
            observed_at=now - timedelta(minutes=12),
            source_type="camera_alert",
            source_name="School gate camera",
            village="Wuyuanli Village",
            rainfall_mm=36,
            water_level_m=4.8,
            road_blocked=True,
            citizen_reports=5,
            notes="Ponding is rising quickly near the school gate.",
        ),
        ObservationIngestItem(
            observed_at=now - timedelta(minutes=6),
            source_type="water_level_sensor",
            source_name="Lowland water-level sensor",
            village="Jiansheli Village",
            rainfall_mm=35,
            water_level_m=5.1,
            road_blocked=True,
            citizen_reports=6,
            notes="Drainage pressure is still rising in the lowland area.",
        ),
    ]


def sample_simulation_update() -> SimulationUpdateRequest:
    return SimulationUpdateRequest(
        generated_at=datetime.now(timezone.utc),
        depth_threshold_m=0.45,
        flow_threshold_mps=1.2,
        cells=[
            SimulationCell(cell_id="grid_01", label="School perimeter", water_depth_m=1.2, flow_velocity_mps=1.5),
            SimulationCell(cell_id="grid_02", label="North community", water_depth_m=1.0, flow_velocity_mps=1.4),
            SimulationCell(cell_id="grid_03", label="Traffic corridor", water_depth_m=0.8, flow_velocity_mps=1.3),
            SimulationCell(cell_id="grid_04", label="Lowland residential cluster", water_depth_m=1.35, flow_velocity_mps=1.7),
        ],
    )


def _is_retryable_runtime_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    transient_signals = (
        "remote end closed connection without response",
        "unexpected eof while reading",
        "request_timeout",
        "timed out",
        "temporarily unavailable",
    )
    return any(signal in detail for signal in transient_signals)


def _emit_retry(stage: str, attempt: int, max_attempts: int, exc: Exception) -> None:
    payload: dict[str, Any] = {
        "status": "retrying",
        "stage": stage,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "detail": str(exc),
    }
    if isinstance(exc, LLMGenerationError):
        payload["error_type"] = "llm"
        payload["error_code"] = exc.code
    else:
        payload["error_type"] = "runtime"
        payload["exception_type"] = type(exc).__name__
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_stage(
    stage: str,
    action: Callable[[], Any],
    *,
    validate: Callable[[Any], str | None] | None = None,
) -> tuple[Any, int]:
    last_error: Exception | None = None
    for attempt in range(1, SMOKE_STAGE_RETRY_ATTEMPTS + 1):
        try:
            result = action()
            retry_reason = validate(result) if validate else None
            if retry_reason is None:
                return result, attempt
            last_error = RuntimeError(retry_reason)
        except LLMGenerationError as exc:
            if exc.code != "llm_unavailable":
                raise
            last_error = exc
        except Exception as exc:  # pragma: no cover - smoke helper
            if not _is_retryable_runtime_error(exc):
                raise
            last_error = exc

        if attempt >= SMOKE_STAGE_RETRY_ATTEMPTS:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"{stage} failed after {SMOKE_STAGE_RETRY_ATTEMPTS} attempts.")

        if last_error is not None:
            _emit_retry(stage, attempt, SMOKE_STAGE_RETRY_ATTEMPTS, last_error)
        time.sleep(SMOKE_STAGE_RETRY_DELAY_SECONDS * attempt)

    raise RuntimeError(f"{stage} failed after {SMOKE_STAGE_RETRY_ATTEMPTS} attempts.")


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    db_path = Path("data") / f"llm_e2e_smoke_{timestamp}.db"
    system = FloodWarningSystem(db_path)
    production = system.production_platform

    event = production.create_event(
        EventCreateRequest(
            area_id="beilin_10km2",
            title="LLM end-to-end smoke event",
            trigger_reason="llm_e2e_smoke",
            operator="codex",
        )
    )

    production.ingest_observations(
        event.event_id,
        ObservationBatchRequest(operator="codex", observations=sample_observations()),
    )

    advisory, advisory_attempts = _run_stage(
        "generate_advisory",
        lambda: production.generate_advisory(
            AdvisoryRequest(
                event_id=event.event_id,
                area_id="beilin_10km2",
                entity_id="school_wyl_primary",
                operator_role="commander",
            )
        ),
    )

    chat_view, copilot_attempts = _run_stage(
        "send_copilot_message",
        lambda: production.send_copilot_message(
            production.bootstrap_copilot_session(
                V2CopilotSessionRequest(event_id=event.event_id, operator_role="commander")
            ).session_id,
            V2CopilotMessageRequest(
                content="What does the current flood mean for the school right now? Give an explanation and actions."
            ),
        ),
        validate=lambda view: "copilot answer missing." if view.latest_answer is None else None,
    )

    simulation_request = sample_simulation_update()
    simulation_state: dict[str, dict[str, Any] | None] = {"result": None}

    def run_simulation_stage() -> tuple[dict[str, Any], list[Any]]:
        if simulation_state["result"] is None:
            simulation_state["result"] = production.ingest_simulation_update(event.event_id, simulation_request)
        else:
            reconciliation = production.reconcile_regional_proposals(event.event_id)
            simulation_state["result"] = {
                **simulation_state["result"],
                "risk_stage_key": reconciliation.get("risk_stage_key"),
                "queue_version": production.get_pending_regional_proposals_snapshot().queue_version,
                "llm_status": reconciliation.get("llm_status"),
                "llm_error": reconciliation.get("llm_error"),
            }

        pending = production.list_regional_proposals(event.event_id, statuses=[ProposalStatus.PENDING.value])
        return simulation_state["result"], pending

    def validate_simulation_stage(result: tuple[dict[str, Any], list[Any]]) -> str | None:
        simulation_result, pending = result
        llm_status = str(simulation_result.get("llm_status") or "ok")
        if llm_status != "ok":
            return f"simulation stage returned llm_status={llm_status}: {simulation_result.get('llm_error')}"
        if not pending:
            return "simulation stage did not produce pending regional proposals."
        return None

    (simulation_result, pending), simulation_attempts = _run_stage(
        "ingest_simulation_update",
        run_simulation_stage,
        validate=validate_simulation_stage,
    )

    approved_proposal = None
    notification_count = 0
    execution_log_count = 0
    approval_attempts = 0

    if pending:
        def approve_first_pending() -> tuple[Any, int, int]:
            latest_pending = production.list_regional_proposals(
                event.event_id, statuses=[ProposalStatus.PENDING.value]
            )
            if not latest_pending:
                raise RuntimeError("no pending regional proposals available to approve.")
            approved = production.approve_regional_proposal(
                latest_pending[0].proposal.proposal_id,
                ProposalResolutionRequest(
                    operator_id="smoke_commander",
                    operator_role="commander",
                    note="Approve the first generated regional action for smoke validation.",
                ),
            )
            return (
                approved,
                len(production.repository.list_v2_notification_drafts(event.event_id)),
                len(production.repository.list_v2_execution_logs(event.event_id)),
            )

        def validate_approval_stage(result: tuple[Any, int, int]) -> str | None:
            _, draft_count, log_count = result
            if draft_count <= 0:
                return "approval stage did not produce notification drafts."
            if log_count <= 0:
                return "approval stage did not produce execution logs."
            return None

        (approved_proposal, notification_count, execution_log_count), approval_attempts = _run_stage(
            "approve_regional_proposal",
            approve_first_pending,
            validate=validate_approval_stage,
        )

    output = {
        "db_path": str(db_path),
        "event_id": event.event_id,
        "stage_attempts": {
            "generate_advisory": advisory_attempts,
            "send_copilot_message": copilot_attempts,
            "ingest_simulation_update": simulation_attempts,
            "approve_regional_proposal": approval_attempts,
        },
        "advisory": {
            "generation_source": advisory.generation_source.value,
            "model_name": advisory.model_name,
            "answer_preview": advisory.answer[:120],
            "grounding_summary": advisory.grounding_summary,
        },
        "copilot": {
            "generation_source": chat_view.latest_answer.generation_source.value if chat_view.latest_answer else None,
            "model_name": chat_view.latest_answer.model_name if chat_view.latest_answer else None,
            "answer_preview": chat_view.latest_answer.answer[:120] if chat_view.latest_answer else None,
            "grounding_summary": chat_view.latest_answer.grounding_summary if chat_view.latest_answer else None,
        },
        "simulation": simulation_result,
        "pending_regional_action_types": [item.proposal.action_type for item in pending],
        "approved_regional_proposal": {
            "proposal_id": approved_proposal.proposal.proposal_id if approved_proposal else None,
            "action_type": approved_proposal.proposal.action_type if approved_proposal else None,
            "execution_mode": approved_proposal.proposal.execution_mode.value if approved_proposal else None,
            "model_name": approved_proposal.proposal.model_name if approved_proposal else None,
        }
        if approved_proposal
        else None,
        "notification_draft_count": notification_count,
        "execution_log_count": execution_log_count,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except LLMGenerationError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": "llm",
                    "error_code": exc.code,
                    "detail": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - smoke helper
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": "runtime",
                    "detail": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise SystemExit(1) from exc
