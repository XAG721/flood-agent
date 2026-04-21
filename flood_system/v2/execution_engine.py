from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from .models import (
    CompletionAssessment,
    CopilotExecutionPlan,
    PlannerRequestContext,
    PlanRunRecord,
    ToolExecutionAuditRecord,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolFailureMode,
)


@dataclass
class ExecutionEngineResult:
    final_plan: CopilotExecutionPlan
    plan_runs: list[PlanRunRecord]
    tool_executions: list[ToolExecutionAuditRecord]
    outputs_by_tool: dict[str, object]
    completion_assessment: CompletionAssessment
    replan_count: int
    used_fallbacks: list[str]
    carried_context_notes: list[str]
    planning_layers_summary: list[str]


class ExecutionEngine:
    MAX_REPLAN_ROUNDS = 2

    def __init__(self, platform, tools, repository) -> None:
        self.platform = platform
        self.tools = tools
        self.repository = repository

    def run_agent_loop(
        self,
        *,
        session_id: str,
        message_id: str,
        event_id: str,
        question: str,
        rule_plan: CopilotExecutionPlan,
        memory_snapshot,
        llm_planner,
        rule_planner,
        evaluate_round,
    ) -> ExecutionEngineResult:
        available_tool_names = self.tools.tool_names()
        llm_context = PlannerRequestContext(
            question=question,
            event_id=event_id,
            memory_snapshot=memory_snapshot,
            available_tools=available_tool_names,
            tool_whitelist=available_tool_names,
        )
        llm_suggestion = llm_planner.plan(
            llm_context,
            rule_plan,
            [spec.model_dump(mode="json") for spec in self.tools.list_specs()],
        )
        current_plan = rule_planner.merge(rule_plan, llm_suggestion, available_tool_names)
        outputs_by_tool: dict[str, object] = {}
        plan_runs: list[PlanRunRecord] = []
        all_tool_executions: list[ToolExecutionAuditRecord] = []
        used_fallbacks: list[str] = []
        planning_layers_summary = [
            f"Rule baseline selected {len(rule_plan.selected_tools)} tools for intent {rule_plan.intent}.",
            "LLM planner fallback to the rule baseline."
            if llm_suggestion.invalid_reason
            else f"LLM planner suggested {len(llm_suggestion.selected_tools)} tool placements within the whitelist.",
            f"Merged plan executes {len(current_plan.selected_tools)} tools after safety reconciliation.",
        ]
        carried_context_notes: list[str] = []
        if current_plan.target_resolution_mode == "memory_focus":
            carried_context_notes.append("Reused the focus entity from the previous turn because the new question did not name a target.")

        round_index = 0
        assessment = CompletionAssessment()
        while True:
            round_plan = current_plan.model_copy(update={"replan_round": round_index})
            plan_record = PlanRunRecord(
                plan_run_id=f"planrun_{uuid4().hex[:10]}",
                session_id=session_id,
                event_id=event_id,
                message_id=message_id,
                plan_id=round_plan.plan_id,
                planning_layer=round_plan.planning_layer,
                replan_round=round_plan.replan_round,
                parent_plan_id=round_plan.parent_plan_id,
                intent=round_plan.intent,
                target_entity_id=round_plan.target_entity_id,
                target_entity_name=round_plan.target_entity_name,
                selected_tools=round_plan.selected_tools,
                tool_selection_reasoning=round_plan.tool_selection_reasoning,
                skipped_tools=round_plan.skipped_tools,
                plan_notes=round_plan.plan_notes,
                created_at=datetime.now(timezone.utc),
            )
            self.repository.save_v2_copilot_plan_run(plan_record)
            plan_runs.append(plan_record)

            round_executions, round_outputs, round_fallbacks = self._execute_round(
                session_id=session_id,
                message_id=message_id,
                event_id=event_id,
                question=question,
                plan=round_plan,
                plan_id=round_plan.plan_id,
                replan_round=round_plan.replan_round,
                cached_outputs=outputs_by_tool,
            )
            outputs_by_tool.update(round_outputs)
            all_tool_executions.extend(round_executions)
            used_fallbacks.extend(round_fallbacks)

            assessment = evaluate_round(
                round_plan,
                round_executions,
                outputs_by_tool,
            )
            if not assessment.should_replan or round_index >= self.MAX_REPLAN_ROUNDS:
                if assessment.should_replan and round_index >= self.MAX_REPLAN_ROUNDS:
                    assessment = assessment.model_copy(
                        update={
                            "should_replan": False,
                            "termination_reason": "Reached the maximum replan budget; downgraded to a conservative stop.",
                        }
                    )
                return ExecutionEngineResult(
                    final_plan=round_plan,
                    plan_runs=plan_runs,
                    tool_executions=all_tool_executions,
                    outputs_by_tool=outputs_by_tool,
                    completion_assessment=assessment,
                    replan_count=round_index,
                    used_fallbacks=list(dict.fromkeys(used_fallbacks)),
                    carried_context_notes=carried_context_notes,
                    planning_layers_summary=planning_layers_summary,
                )

            round_index += 1
            recent_failures = assessment.evidence_gaps or ["reviewer_requested_replan"]
            current_plan = rule_planner.replan(
                round_plan,
                recent_failures=recent_failures,
                replan_round=round_index,
                available_tool_names=available_tool_names,
            )
            planning_layers_summary.append(
                f"Replan round {round_index} triggered by: {', '.join(recent_failures)}."
            )

    def _execute_round(
        self,
        *,
        session_id: str,
        message_id: str,
        event_id: str,
        question: str,
        plan: CopilotExecutionPlan,
        plan_id: str,
        replan_round: int,
        cached_outputs: dict[str, object],
    ) -> tuple[list[ToolExecutionAuditRecord], dict[str, object], list[str]]:
        executions: list[ToolExecutionAuditRecord] = []
        outputs: dict[str, object] = {}
        used_fallbacks: list[str] = []

        first = self._run_tool(
            "resolve_target_entity",
            session_id=session_id,
            message_id=message_id,
            event_id=event_id,
            kwargs={
                "event_id": event_id,
                "question": question,
                "preferred_entity_id": plan.target_entity_id,
                "entity_type": plan.target_entity_type,
            },
            dependency_tools=[],
            plan_id=plan_id,
            replan_round=replan_round,
        )
        executions.extend(first["records"])
        used_fallbacks.extend(first["used_fallbacks"])
        if first["final"].status == ToolExecutionStatus.SUCCESS and first["final"].raw_output is not None:
            outputs["resolve_target_entity"] = first["final"].raw_output

        if "get_exposure_summary" in plan.selected_tools:
            exposure_run = self._maybe_cached_tool(
                "get_exposure_summary",
                session_id=session_id,
                message_id=message_id,
                event_id=event_id,
                kwargs={
                    "event_id": event_id,
                    "entity_type": plan.target_entity_type,
                    "top_k": 3,
                },
                dependency_tools=["resolve_target_entity"],
                plan_id=plan_id,
                replan_round=replan_round,
                cached_outputs=cached_outputs,
            )
            executions.extend(exposure_run["records"])
            used_fallbacks.extend(exposure_run["used_fallbacks"])
            if exposure_run["final"].raw_output is not None:
                outputs["get_exposure_summary"] = exposure_run["final"].raw_output

        target_entity_id = self._resolved_entity_id(outputs, cached_outputs, plan)
        event = self.platform.get_event(event_id)
        area_id = event.area_id
        if target_entity_id:
            try:
                area_id = self.platform.get_entity_profile(target_entity_id).area_id
            except ValueError:
                area_id = event.area_id

        parallel_tasks: list[tuple[str, dict, list[str]]] = []
        if "get_hazard_tiles" in plan.selected_tools:
            parallel_tasks.append(("get_hazard_tiles", {"event_id": event_id}, []))
        if "get_entity_profile" in plan.selected_tools and target_entity_id:
            parallel_tasks.append(("get_entity_profile", {"entity_id": target_entity_id}, ["resolve_target_entity"]))
        if "get_knowledge_evidence" in plan.selected_tools:
            parallel_tasks.append(
                (
                    "get_knowledge_evidence",
                    {
                        "event_id": event_id,
                        "entity_id": target_entity_id,
                        "area_id": area_id,
                    },
                    ["resolve_target_entity"],
                )
            )
        parallel_results = self._run_parallel_group(
            session_id=session_id,
            message_id=message_id,
            event_id=event_id,
            tasks=parallel_tasks,
            cached_outputs=cached_outputs,
            plan_id=plan_id,
            replan_round=replan_round,
        )
        for result in parallel_results:
            executions.extend(result["records"])
            used_fallbacks.extend(result["used_fallbacks"])
            if result["final"].raw_output is not None:
                outputs[result["final"].tool_name] = result["final"].raw_output

        target_entity_id = self._resolved_entity_id(outputs, cached_outputs, plan)
        if target_entity_id and "synthesize_entity_impact" in plan.selected_tools:
            impact_run = self._maybe_cached_tool(
                "synthesize_entity_impact",
                session_id=session_id,
                message_id=message_id,
                event_id=event_id,
                kwargs={"event_id": event_id, "entity_id": target_entity_id},
                dependency_tools=["get_hazard_tiles", "get_entity_profile", "get_knowledge_evidence"],
                plan_id=plan_id,
                replan_round=replan_round,
                cached_outputs=cached_outputs,
            )
            executions.extend(impact_run["records"])
            used_fallbacks.extend(impact_run["used_fallbacks"])
            if impact_run["final"].raw_output is not None:
                outputs["synthesize_entity_impact"] = impact_run["final"].raw_output

        impact_payload = outputs.get("synthesize_entity_impact") or cached_outputs.get("synthesize_entity_impact")
        risk_level = impact_payload.get("risk_level") if isinstance(impact_payload, dict) else None
        entity_type = None
        if isinstance(impact_payload, dict):
            entity_payload = impact_payload.get("entity")
            if isinstance(entity_payload, dict):
                entity_type = entity_payload.get("entity_type")
                area_id = entity_payload.get("area_id", area_id)

        secondary_tasks: list[tuple[str, dict, list[str]]] = []
        if target_entity_id and "get_route_options" in plan.selected_tools:
            secondary_tasks.append(("get_route_options", {"event_id": event_id, "entity_id": target_entity_id}, ["synthesize_entity_impact"]))
        if "get_live_traffic" in plan.selected_tools:
            secondary_tasks.append(("get_live_traffic", {"event_id": event_id}, ["get_hazard_tiles"]))
        if "get_shelter_capacity" in plan.selected_tools:
            secondary_tasks.append(("get_shelter_capacity", {"area_id": area_id}, []))
        if secondary_tasks:
            secondary_results = self._run_parallel_group(
                session_id=session_id,
                message_id=message_id,
                event_id=event_id,
                tasks=secondary_tasks,
                cached_outputs=cached_outputs,
                plan_id=plan_id,
                replan_round=replan_round,
            )
            for result in secondary_results:
                executions.extend(result["records"])
                used_fallbacks.extend(result["used_fallbacks"])
                if result["final"].raw_output is not None:
                    outputs[result["final"].tool_name] = result["final"].raw_output

        if entity_type and risk_level and "get_policy_constraints" in plan.selected_tools:
            policy_run = self._maybe_cached_tool(
                "get_policy_constraints",
                session_id=session_id,
                message_id=message_id,
                event_id=event_id,
                kwargs={"entity_type": entity_type, "risk_level": risk_level},
                dependency_tools=["synthesize_entity_impact"],
                plan_id=plan_id,
                replan_round=replan_round,
                cached_outputs=cached_outputs,
            )
            executions.extend(policy_run["records"])
            used_fallbacks.extend(policy_run["used_fallbacks"])
            if policy_run["final"].raw_output is not None:
                outputs["get_policy_constraints"] = policy_run["final"].raw_output

        if target_entity_id and any(item in plan.selected_tools for item in ("draft_action_proposal", "create_action_proposal")):
            draft_tool = "draft_action_proposal" if "draft_action_proposal" in plan.selected_tools else "create_action_proposal"
            proposal_run = self._maybe_cached_tool(
                draft_tool,
                session_id=session_id,
                message_id=message_id,
                event_id=event_id,
                kwargs={"event_id": event_id, "entity_id": target_entity_id},
                dependency_tools=["synthesize_entity_impact", "get_policy_constraints", "get_knowledge_evidence"],
                plan_id=plan_id,
                replan_round=replan_round,
                cached_outputs=cached_outputs,
            )
            executions.extend(proposal_run["records"])
            used_fallbacks.extend(proposal_run["used_fallbacks"])
            if proposal_run["final"].raw_output is not None:
                outputs[draft_tool] = proposal_run["final"].raw_output

        return executions, outputs, used_fallbacks

    def _run_parallel_group(
        self,
        *,
        session_id: str,
        message_id: str,
        event_id: str,
        tasks: list[tuple[str, dict, list[str]]],
        cached_outputs: dict[str, object],
        plan_id: str,
        replan_round: int,
    ) -> list[dict]:
        if not tasks:
            return []
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [
                pool.submit(
                    self._maybe_cached_tool,
                    tool_name,
                    session_id=session_id,
                    message_id=message_id,
                    event_id=event_id,
                    kwargs=kwargs,
                    dependency_tools=dependencies,
                    cached_outputs=cached_outputs,
                    plan_id=plan_id,
                    replan_round=replan_round,
                )
                for tool_name, kwargs, dependencies in tasks
            ]
        return [future.result() for future in futures]

    def _maybe_cached_tool(
        self,
        tool_name: str,
        *,
        session_id: str,
        message_id: str,
        event_id: str,
        kwargs: dict,
        dependency_tools: list[str],
        cached_outputs: dict[str, object],
        plan_id: str,
        replan_round: int,
    ) -> dict:
        if tool_name in cached_outputs:
            record = ToolExecutionAuditRecord(
                execution_id=f"exec_{uuid4().hex[:10]}",
                session_id=session_id,
                event_id=event_id,
                message_id=message_id,
                tool_name=tool_name,
                status=ToolExecutionStatus.SKIPPED,
                input=kwargs,
                output_summary="Reused successful result from a previous round.",
                raw_output=cached_outputs[tool_name],
                duration_ms=0,
                timed_out=False,
                dependency_tools=dependency_tools,
                cache_hit=True,
                created_at=datetime.now(timezone.utc),
                parallel_group=self.tools.get_spec(tool_name).parallel_group,
                plan_id=plan_id,
                replan_round=replan_round,
            )
            self.repository.save_v2_copilot_tool_execution(record)
            return {"records": [record], "final": record, "used_fallbacks": []}
        return self._run_tool(
            tool_name,
            session_id=session_id,
            message_id=message_id,
            event_id=event_id,
            kwargs=kwargs,
            dependency_tools=dependency_tools,
            plan_id=plan_id,
            replan_round=replan_round,
        )

    def _run_tool(
        self,
        tool_name: str,
        *,
        session_id: str,
        message_id: str,
        event_id: str,
        kwargs: dict,
        dependency_tools: list[str],
        plan_id: str,
        replan_round: int,
    ) -> dict:
        spec = self.tools.get_spec(tool_name)
        records: list[ToolExecutionAuditRecord] = []
        used_fallbacks: list[str] = []
        retry_of_execution_id: str | None = None

        for attempt in range(1, spec.retry_policy.max_attempts + 1):
            execution = self.tools.execute(tool_name, **kwargs).model_copy(
                update={
                    "execution_id": f"exec_{uuid4().hex[:10]}",
                    "attempt": attempt,
                    "retry_of_execution_id": retry_of_execution_id,
                    "dependency_tools": dependency_tools,
                    "parallel_group": spec.parallel_group,
                }
            )
            audit = ToolExecutionAuditRecord(
                session_id=session_id,
                    event_id=event_id,
                    message_id=message_id,
                    plan_id=plan_id,
                    replan_round=replan_round,
                    created_at=datetime.now(timezone.utc),
                    **execution.model_dump(mode="json"),
                )
            self.repository.save_v2_copilot_tool_execution(audit)
            records.append(audit)
            if audit.status == ToolExecutionStatus.SUCCESS:
                return {"records": records, "final": audit, "used_fallbacks": used_fallbacks}
            if audit.failure_reason not in {mode.value for mode in spec.retry_policy.retryable_failure_modes}:
                break
            retry_of_execution_id = audit.execution_id

        final = records[-1]
        if final.status != ToolExecutionStatus.SUCCESS and spec.fallback_tools:
            for fallback_tool in spec.fallback_tools:
                fallback_kwargs = dict(kwargs)
                fallback_result = self.tools.execute(fallback_tool, **fallback_kwargs).model_copy(
                    update={
                        "execution_id": f"exec_{uuid4().hex[:10]}",
                        "attempt": 1,
                        "fallback_from_tool": tool_name,
                        "dependency_tools": dependency_tools,
                        "parallel_group": self.tools.get_spec(fallback_tool).parallel_group,
                    }
                )
                audit = ToolExecutionAuditRecord(
                    session_id=session_id,
                    event_id=event_id,
                    message_id=message_id,
                    plan_id=plan_id,
                    replan_round=replan_round,
                    created_at=datetime.now(timezone.utc),
                    **fallback_result.model_dump(mode="json"),
                )
                self.repository.save_v2_copilot_tool_execution(audit)
                records.append(audit)
                used_fallbacks.append(f"{tool_name}->{fallback_tool}")
                if audit.status == ToolExecutionStatus.SUCCESS:
                    return {"records": records, "final": audit, "used_fallbacks": used_fallbacks}
        return {"records": records, "final": final, "used_fallbacks": used_fallbacks}

    @staticmethod
    def _resolved_entity_id(outputs: dict[str, object], cached_outputs: dict[str, object], plan: CopilotExecutionPlan) -> str | None:
        for payload in (
            outputs.get("resolve_target_entity"),
            cached_outputs.get("resolve_target_entity"),
        ):
            if isinstance(payload, dict) and payload.get("entity_id"):
                return str(payload["entity_id"])
        exposure_payload = outputs.get("get_exposure_summary") or cached_outputs.get("get_exposure_summary")
        if isinstance(exposure_payload, dict):
            affected = exposure_payload.get("affected_entities")
            if isinstance(affected, list) and affected:
                first = affected[0]
                if isinstance(first, dict):
                    entity = first.get("entity")
                    if isinstance(entity, dict) and entity.get("entity_id"):
                        return str(entity["entity_id"])
        return plan.target_entity_id
