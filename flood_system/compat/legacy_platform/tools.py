from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from .models import (
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolFailureMode,
    ToolRetryPolicy,
    ToolSpec,
)


DEFAULT_STALE_DATA_THRESHOLD_SECONDS = 900


class ResolveTargetEntityToolInput(BaseModel):
    event_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    preferred_entity_id: str | None = None
    entity_type: str | None = None


class HazardTilesToolInput(BaseModel):
    event_id: str = Field(min_length=1)


class EntityProfileToolInput(BaseModel):
    entity_id: str = Field(min_length=1)


class ExposureSummaryToolInput(BaseModel):
    event_id: str = Field(min_length=1)
    entity_type: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class KnowledgeEvidenceToolInput(BaseModel):
    event_id: str = Field(min_length=1)
    area_id: str = Field(min_length=1)
    entity_id: str | None = None


class ImpactSynthesisToolInput(BaseModel):
    event_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)


class RouteOptionsToolInput(BaseModel):
    event_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)


class ShelterCapacityToolInput(BaseModel):
    area_id: str = Field(min_length=1)


class LiveTrafficToolInput(BaseModel):
    event_id: str = Field(min_length=1)


class PolicyConstraintsToolInput(BaseModel):
    entity_type: str = Field(min_length=1)
    risk_level: str = Field(min_length=1)


class ActionProposalToolInput(BaseModel):
    event_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)


@dataclass
class RegisteredTool:
    spec: ToolSpec
    input_model: type[BaseModel]
    executor: Callable[..., dict[str, Any]]


class ToolRegistry:
    def __init__(self, tools: list[RegisteredTool]) -> None:
        self._tools = {tool.spec.tool_name: tool for tool in tools}

    def get_spec(self, tool_name: str) -> ToolSpec:
        if tool_name not in self._tools:
            raise KeyError(tool_name)
        return self._tools[tool_name].spec

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def executors_by_name(self) -> dict[str, Callable[..., Any]]:
        def build_executor(tool_name: str) -> Callable[..., Any]:
            def _executor(**kwargs: Any) -> Any:
                return self.execute(tool_name, **kwargs).raw_output

            return _executor

        return {tool_name: build_executor(tool_name) for tool_name in self._tools}

    def execute(self, tool_name: str, **kwargs: Any) -> ToolExecutionResult:
        if tool_name not in self._tools:
            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.FAILED,
                input=dict(kwargs),
                output_summary="Tool is not registered.",
                failure_reason=ToolFailureMode.NOT_FOUND.value,
            )

        tool = self._tools[tool_name]
        started = perf_counter()
        try:
            validated = tool.input_model.model_validate(kwargs)
        except ValidationError as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.FAILED,
                input=dict(kwargs),
                output_summary="Tool input validation failed.",
                failure_reason=f"{ToolFailureMode.INVALID_INPUT.value}: {exc.errors()[0]['msg']}",
                duration_ms=int((perf_counter() - started) * 1000),
            )

        try:
            payload = tool.executor(**validated.model_dump())
            duration_ms = int((perf_counter() - started) * 1000)
            freshness = _normalize_freshness(payload.get("data_freshness_seconds"))
            timed_out = duration_ms > tool.spec.timeout_ms
            stale_budget = (
                tool.spec.staleness_budget_seconds
                if tool.spec.staleness_budget_seconds is not None
                else DEFAULT_STALE_DATA_THRESHOLD_SECONDS
            )
            stale = freshness is not None and freshness > stale_budget

            status = ToolExecutionStatus.SUCCESS
            failure_reason: str | None = None
            if timed_out:
                status = ToolExecutionStatus.TIMEOUT
                failure_reason = ToolFailureMode.TIMEOUT.value
            elif stale:
                status = ToolExecutionStatus.FAILED
                failure_reason = ToolFailureMode.STALE_DATA.value

            return ToolExecutionResult(
                tool_name=tool_name,
                status=status,
                input=validated.model_dump(mode="json"),
                output_summary=str(payload.get("output_summary", "")),
                raw_output=payload.get("raw_output"),
                failure_reason=failure_reason,
                duration_ms=duration_ms,
                timed_out=timed_out,
                data_freshness_seconds=freshness,
                stale=stale,
                parallel_group=tool.spec.parallel_group,
            )
        except Exception as exc:  # pragma: no cover - defensive wrapper
            duration_ms = int((perf_counter() - started) * 1000)
            return ToolExecutionResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.FAILED,
                input=validated.model_dump(mode="json"),
                output_summary=str(exc),
                failure_reason=_classify_failure_reason(exc).value,
                duration_ms=duration_ms,
                parallel_group=tool.spec.parallel_group,
            )


def build_v2_tools(platform) -> ToolRegistry:
    def resolve_target_entity(
        event_id: str,
        question: str,
        preferred_entity_id: str | None = None,
        entity_type: str | None = None,
    ) -> dict[str, Any]:
        resolution = platform.resolve_target_entity(
            event_id,
            question,
            preferred_entity_id=preferred_entity_id,
            entity_type=entity_type,
        )
        return {
            "raw_output": resolution,
            "output_summary": resolution.get("reason", "Resolved target entity."),
            "data_freshness_seconds": None,
        }

    def get_hazard_tiles(event_id: str) -> dict[str, Any]:
        hazard_state = platform.get_hazard_state(event_id)
        return {
            "raw_output": hazard_state.model_dump(mode="json"),
            "output_summary": f"Loaded {len(hazard_state.hazard_tiles)} hazard tiles at {hazard_state.overall_risk_level.value} risk.",
            "data_freshness_seconds": hazard_state.freshness_seconds,
        }

    def get_entity_profile(entity_id: str) -> dict[str, Any]:
        entity = platform.get_entity_profile(entity_id)
        return {
            "raw_output": entity.model_dump(mode="json"),
            "output_summary": f"Loaded profile for {entity.name}.",
            "data_freshness_seconds": None,
        }

    def get_exposure_summary(event_id: str, entity_type: str | None = None, top_k: int = 5) -> dict[str, Any]:
        exposure = platform.get_exposure_summary(event_id, entity_type=entity_type, top_k=top_k)
        return {
            "raw_output": exposure.model_dump(mode="json"),
            "output_summary": f"Loaded exposure summary with {len(exposure.affected_entities)} affected entities.",
            "data_freshness_seconds": platform.get_hazard_state(event_id).freshness_seconds,
        }

    def get_knowledge_evidence(event_id: str, area_id: str, entity_id: str | None = None) -> dict[str, Any]:
        evidence = platform.get_knowledge_evidence(event_id=event_id, area_id=area_id, entity_id=entity_id)
        return {
            "raw_output": [item.model_dump(mode="json") for item in evidence],
            "output_summary": f"Loaded {len(evidence)} knowledge evidence item(s).",
            "data_freshness_seconds": platform.get_hazard_state(event_id).freshness_seconds,
        }

    def synthesize_entity_impact(event_id: str, entity_id: str) -> dict[str, Any]:
        impact = platform.get_entity_impact(entity_id, event_id=event_id)
        return {
            "raw_output": impact.model_dump(mode="json"),
            "output_summary": f"Synthesized impact for {impact.entity.name}.",
            "data_freshness_seconds": platform.get_hazard_state(event_id).freshness_seconds,
        }

    def get_route_options(event_id: str, entity_id: str) -> dict[str, Any]:
        impact = platform.get_entity_impact(entity_id, event_id=event_id)
        route_bundle = {
            "safe_routes": [item.model_dump(mode="json") for item in impact.safe_routes],
            "blocked_routes": [item.model_dump(mode="json") for item in impact.blocked_routes],
        }
        return {
            "raw_output": route_bundle,
            "output_summary": f"Loaded {len(impact.safe_routes)} safe routes and {len(impact.blocked_routes)} blocked routes.",
            "data_freshness_seconds": platform.get_hazard_state(event_id).freshness_seconds,
        }

    def get_shelter_capacity(area_id: str) -> dict[str, Any]:
        shelters = platform.get_shelter_capacity(area_id)
        return {
            "raw_output": shelters,
            "output_summary": f"Loaded shelter capacity for {len(shelters)} shelters.",
            "data_freshness_seconds": None,
        }

    def get_live_traffic(event_id: str) -> dict[str, Any]:
        traffic = platform.get_live_traffic(event_id)
        return {
            "raw_output": traffic,
            "output_summary": f"Loaded traffic for {len(traffic)} road links.",
            "data_freshness_seconds": platform.get_hazard_state(event_id).freshness_seconds,
        }

    def get_policy_constraints(entity_type: str, risk_level: str) -> dict[str, Any]:
        constraint = platform.get_policy_constraints(entity_type, risk_level)
        return {
            "raw_output": constraint.model_dump(mode="json"),
            "output_summary": f"Loaded policy constraints for {entity_type} at {risk_level}.",
            "data_freshness_seconds": None,
        }

    def draft_action_proposal(event_id: str, entity_id: str) -> dict[str, Any]:
        proposal = platform.draft_action_proposal(event_id, entity_id)
        if proposal is None:
            return {
                "raw_output": None,
                "output_summary": "No action proposal is required for the current target.",
                "data_freshness_seconds": platform.get_hazard_state(event_id).freshness_seconds,
            }
        return {
            "raw_output": proposal.model_dump(mode="json"),
            "output_summary": f"Prepared action proposal for {proposal.entity_id or 'the active target'}.",
            "data_freshness_seconds": platform.get_hazard_state(event_id).freshness_seconds,
        }

    def create_action_proposal(event_id: str, entity_id: str) -> dict[str, Any]:
        return draft_action_proposal(event_id, entity_id)

    return ToolRegistry(
        [
            RegisteredTool(
                spec=_build_spec(
                    "resolve_target_entity",
                    "Resolve the active target entity from the current question, event context, and optional preference hints.",
                    ResolveTargetEntityToolInput,
                    timeout_ms=650,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.TIMEOUT,
                    ],
                    produces_memory_updates=True,
                ),
                input_model=ResolveTargetEntityToolInput,
                executor=resolve_target_entity,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "get_hazard_tiles",
                    "Load current hazard tiles and tile-level risk metadata for the active event.",
                    HazardTilesToolInput,
                    timeout_ms=900,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.STALE_DATA,
                        ToolFailureMode.TIMEOUT,
                    ],
                    retry_policy=ToolRetryPolicy(max_attempts=2, retryable_failure_modes=[ToolFailureMode.TIMEOUT]),
                    parallel_group="context",
                    staleness_budget_seconds=900,
                ),
                input_model=HazardTilesToolInput,
                executor=get_hazard_tiles,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "get_entity_profile",
                    "Load the runtime entity profile used for impact assessment.",
                    EntityProfileToolInput,
                    timeout_ms=700,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.TIMEOUT,
                    ],
                    parallel_group="context",
                    produces_memory_updates=True,
                ),
                input_model=EntityProfileToolInput,
                executor=get_entity_profile,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "get_exposure_summary",
                    "Load area-wide or group-wide exposure to resolve a target entity or summarize affected objects.",
                    ExposureSummaryToolInput,
                    timeout_ms=1100,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.STALE_DATA,
                        ToolFailureMode.TIMEOUT,
                    ],
                    retry_policy=ToolRetryPolicy(max_attempts=2, retryable_failure_modes=[ToolFailureMode.TIMEOUT]),
                    staleness_budget_seconds=900,
                ),
                input_model=ExposureSummaryToolInput,
                executor=get_exposure_summary,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "get_knowledge_evidence",
                    "Load policy, case, and profile evidence for the active target or area.",
                    KnowledgeEvidenceToolInput,
                    timeout_ms=950,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.TIMEOUT,
                    ],
                    parallel_group="context",
                ),
                input_model=KnowledgeEvidenceToolInput,
                executor=get_knowledge_evidence,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "synthesize_entity_impact",
                    "Build the deterministic entity impact view used by downstream answer synthesis.",
                    ImpactSynthesisToolInput,
                    timeout_ms=1200,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.STALE_DATA,
                        ToolFailureMode.TIMEOUT,
                    ],
                    retry_policy=ToolRetryPolicy(max_attempts=2, retryable_failure_modes=[ToolFailureMode.TIMEOUT]),
                    staleness_budget_seconds=900,
                ),
                input_model=ImpactSynthesisToolInput,
                executor=synthesize_entity_impact,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "get_route_options",
                    "Load safe and blocked routes for the target entity.",
                    RouteOptionsToolInput,
                    timeout_ms=1200,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.STALE_DATA,
                        ToolFailureMode.TIMEOUT,
                    ],
                    retry_policy=ToolRetryPolicy(max_attempts=2, retryable_failure_modes=[ToolFailureMode.TIMEOUT]),
                    parallel_group="evacuation",
                    staleness_budget_seconds=900,
                ),
                input_model=RouteOptionsToolInput,
                executor=get_route_options,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "get_shelter_capacity",
                    "Load available shelter capacity for the target area.",
                    ShelterCapacityToolInput,
                    timeout_ms=800,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.TIMEOUT,
                    ],
                    parallel_group="evacuation",
                ),
                input_model=ShelterCapacityToolInput,
                executor=get_shelter_capacity,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "get_live_traffic",
                    "Load live traffic and congestion hints for evacuation routing.",
                    LiveTrafficToolInput,
                    timeout_ms=900,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.STALE_DATA,
                        ToolFailureMode.TIMEOUT,
                        ToolFailureMode.UPSTREAM_UNAVAILABLE,
                    ],
                    retry_policy=ToolRetryPolicy(
                        max_attempts=2,
                        retryable_failure_modes=[ToolFailureMode.TIMEOUT, ToolFailureMode.UPSTREAM_UNAVAILABLE],
                    ),
                    parallel_group="evacuation",
                    staleness_budget_seconds=900,
                ),
                input_model=LiveTrafficToolInput,
                executor=get_live_traffic,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "get_policy_constraints",
                    "Load policy guardrails and approval requirements for the current entity type and risk level.",
                    PolicyConstraintsToolInput,
                    timeout_ms=600,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.POLICY_BLOCKED,
                        ToolFailureMode.TIMEOUT,
                    ],
                ),
                input_model=PolicyConstraintsToolInput,
                executor=get_policy_constraints,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "draft_action_proposal",
                    "Prepare a deterministic action proposal without sending notifications or approving actions.",
                    ActionProposalToolInput,
                    timeout_ms=1300,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.STALE_DATA,
                        ToolFailureMode.TIMEOUT,
                        ToolFailureMode.POLICY_BLOCKED,
                    ],
                    retry_policy=ToolRetryPolicy(max_attempts=2, retryable_failure_modes=[ToolFailureMode.TIMEOUT]),
                    staleness_budget_seconds=900,
                ),
                input_model=ActionProposalToolInput,
                executor=draft_action_proposal,
            ),
            RegisteredTool(
                spec=_build_spec(
                    "create_action_proposal",
                    "Compatibility alias for draft_action_proposal.",
                    ActionProposalToolInput,
                    timeout_ms=1300,
                    failure_modes=[
                        ToolFailureMode.NOT_FOUND,
                        ToolFailureMode.INVALID_INPUT,
                        ToolFailureMode.STALE_DATA,
                        ToolFailureMode.TIMEOUT,
                        ToolFailureMode.POLICY_BLOCKED,
                    ],
                    retry_policy=ToolRetryPolicy(max_attempts=1),
                    fallback_tools=["draft_action_proposal"],
                    staleness_budget_seconds=900,
                ),
                input_model=ActionProposalToolInput,
                executor=create_action_proposal,
            ),
        ]
    )


def _build_spec(
    tool_name: str,
    description: str,
    input_model: type[BaseModel],
    *,
    timeout_ms: int,
    failure_modes: list[ToolFailureMode],
    retry_policy: ToolRetryPolicy | None = None,
    fallback_tools: list[str] | None = None,
    parallel_group: str | None = None,
    staleness_budget_seconds: int | None = None,
    produces_memory_updates: bool = False,
) -> ToolSpec:
    return ToolSpec(
        tool_name=tool_name,
        description=description,
        input_schema=input_model.model_json_schema(),
        timeout_ms=timeout_ms,
        failure_modes=failure_modes,
        retry_policy=retry_policy or ToolRetryPolicy(),
        fallback_tools=fallback_tools or [],
        parallel_group=parallel_group,
        staleness_budget_seconds=staleness_budget_seconds,
        produces_memory_updates=produces_memory_updates,
    )


def _normalize_freshness(value: Any) -> int | None:
    if value in (None, "", "static"):
        return None
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return None


def _classify_failure_reason(exc: Exception) -> ToolFailureMode:
    message = str(exc).lower()
    if "not found" in message or "unknown" in message or "does not exist" in message:
        return ToolFailureMode.NOT_FOUND
    if "invalid" in message or "cannot" in message or "must match" in message:
        return ToolFailureMode.INVALID_INPUT
    if "policy" in message or "not allowed" in message:
        return ToolFailureMode.POLICY_BLOCKED
    if "timeout" in message:
        return ToolFailureMode.TIMEOUT
    if "unavailable" in message or "upstream" in message:
        return ToolFailureMode.UPSTREAM_UNAVAILABLE
    return ToolFailureMode.UNKNOWN
