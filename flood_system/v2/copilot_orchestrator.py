from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from .copilot_rules import HIGH_RISK_TYPES, ROUTE_KEYWORDS
from .execution_engine import ExecutionEngine
from .llm_planner import LLMPlannerAdapter
from .memory_store import SessionMemoryStore
from .models import (
    ActionProposal,
    AlertSeverity,
    AdvisoryRequest,
    BatchProposalResolutionRequest,
    CompletionAssessment,
    CompletionStatus,
    CopilotExecutionPlan,
    CopilotStructuredAnswer,
    DataFreshnessSummary,
    EntityImpactView,
    EntityProfile,
    EvidenceItem,
    MemorySnapshot,
    PlanRunRecord,
    ProposalResolutionRequest,
    ProposalStatus,
    TriggerEventType,
    ToolExecutionAuditRecord,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolTraceStep,
    V2CopilotMessage,
    V2CopilotMessageRequest,
    V2CopilotSessionRequest,
    V2CopilotSessionView,
)
from .planner import RulePlanner
from .reporting import format_daily_report_message, format_episode_summary_message


class CopilotOrchestrator:
    def __init__(self, platform) -> None:
        self.platform = platform
        self.rule_planner = RulePlanner(platform)
        self.llm_planner = LLMPlannerAdapter()
        self.llm_gateway = platform.llm_gateway
        self.memory_store = SessionMemoryStore(platform.repository)
        self.execution_engine = ExecutionEngine(platform, platform.tools, platform.repository)

    def bootstrap_session(self, request: V2CopilotSessionRequest) -> V2CopilotSessionView:
        event = self.platform.get_event(request.event_id)
        session_id = f"v2_copilot_{uuid4().hex[:10]}"
        self.platform.repository.save_v2_copilot_session(
            session_id,
            event.event_id,
            {"operator_role": request.operator_role},
        )
        self.memory_store.save_snapshot(
            MemorySnapshot(
                session_id=session_id,
                focus_area_id=event.area_id,
                updated_at=datetime.now(timezone.utc),
            )
        )
        welcome = V2CopilotMessage(
            message_id=f"v2_msg_{uuid4().hex[:10]}",
            role="assistant",
            content=(
                f"Connected to {event.title}. Ask what the flood means for a resident, school, factory, route, "
                "shelter, or community target, and the copilot will explain rule planning, LLM planning, replans, "
                "memory state, and tool execution decisions."
            ),
            created_at=datetime.now(timezone.utc),
        )
        self.platform.repository.save_v2_copilot_message(session_id, welcome)
        self._backfill_event_reports_to_session(session_id, event.event_id)
        return self.get_session_view(session_id)

    def get_session_view(self, session_id: str) -> V2CopilotSessionView:
        session = self.platform.repository.get_v2_copilot_session(session_id)
        if session is None:
            raise ValueError("v2 copilot session not found.")
        event = self.platform.get_event(session["event_id"])
        messages = self.platform.repository.list_v2_copilot_messages(session_id)
        latest_answer = next((item.structured_answer for item in reversed(messages) if item.structured_answer), None)
        proposals = self.platform.repository.list_v2_action_proposals(event.event_id)
        notification_drafts = self.platform.repository.list_v2_notification_drafts(event.event_id)
        execution_logs = self.platform.repository.list_v2_execution_logs(event.event_id)
        daily_reports = self.platform.list_daily_reports(event.event_id)
        episode_summaries = self.platform.list_episode_summaries(event.event_id)
        memory_snapshot = self.memory_store.load_snapshot(session_id, area_id=event.area_id)
        session_memory = self.platform.get_session_memory(session_id)
        plan_runs = self.platform.repository.list_v2_copilot_plan_runs(session_id, limit=20)
        recent_tool_executions = self.platform.repository.list_v2_copilot_tool_executions(session_id, limit=40)
        shared_memory_snapshot = self.platform.get_shared_memory_snapshot(event.event_id)
        recent_agent_results = self.platform.list_agent_results(event.event_id)[:10]
        pending_regional_analysis_package = self.platform.get_pending_regional_analysis_package(event.event_id)
        regional_analysis_package_history = self.platform.list_regional_analysis_packages(
            event.event_id,
            include_pending=False,
        )[:5]
        return V2CopilotSessionView(
            session_id=session_id,
            event=event,
            messages=messages,
            latest_answer=latest_answer,
            proposals=proposals,
            notification_drafts=notification_drafts,
            execution_logs=execution_logs,
            daily_reports=daily_reports,
            episode_summaries=episode_summaries,
            memory_snapshot=memory_snapshot,
            session_memory_snapshot=session_memory.memory_snapshot,
            plan_runs=plan_runs,
            recent_tool_executions=recent_tool_executions,
            shared_memory_snapshot=shared_memory_snapshot,
            active_agents=shared_memory_snapshot.active_agents,
            recent_agent_results=recent_agent_results,
            autonomy_level=shared_memory_snapshot.autonomy_level,
            pending_regional_analysis_package=pending_regional_analysis_package,
            regional_analysis_package_history=regional_analysis_package_history,
        )

    def answer(self, session_id: str, content: str) -> V2CopilotSessionView:
        session = self.platform.repository.get_v2_copilot_session(session_id)
        if session is None:
            raise ValueError("v2 copilot session not found.")

        user_message = V2CopilotMessage(
            message_id=f"v2_msg_{uuid4().hex[:10]}",
            role="user",
            content=content,
            created_at=datetime.now(timezone.utc),
        )
        self.platform.repository.save_v2_copilot_message(session_id, user_message)
        self.memory_store.note_user_question(session_id, content)
        self.platform.publish_trigger(
            session["event_id"],
            trigger_type=TriggerEventType.COPILOT_ESCALATION_REQUESTED,
            payload={"session_id": session_id},
            dedupe=False,
        )

        assistant_message_id = f"v2_msg_{uuid4().hex[:10]}"
        structured = self._planner_executor_reviewer(
            session_id=session_id,
            message_id=assistant_message_id,
            event_id=session["event_id"],
            content=content,
        )
        if structured.proposal is not None:
            proposal = structured.proposal.model_copy(update={"source_session_id": session_id})
            structured.proposal = proposal
            self.platform.repository.save_v2_action_proposal(proposal)

        assistant_message = V2CopilotMessage(
            message_id=assistant_message_id,
            role="assistant",
            content=structured.answer,
            created_at=datetime.now(timezone.utc),
            structured_answer=structured,
        )
        self.platform.repository.save_v2_copilot_message(session_id, assistant_message)
        return self.get_session_view(session_id)

    def approve_proposal(
        self,
        session_id: str,
        proposal_id: str,
        request: ProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        self._resolve_single_proposal(session_id, proposal_id, request, approved=True, emit_summary_message=True)
        return self.get_session_view(session_id)

    def reject_proposal(
        self,
        session_id: str,
        proposal_id: str,
        request: ProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        self._resolve_single_proposal(session_id, proposal_id, request, approved=False, emit_summary_message=True)
        return self.get_session_view(session_id)

    def batch_approve_proposals(
        self,
        session_id: str,
        request: BatchProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        resolved = self._resolve_many(session_id, request, approved=True)
        self._save_system_message(session_id, f"当前队列中的 {len(resolved)} 条请示已批准。")
        return self.get_session_view(session_id)

    def batch_reject_proposals(
        self,
        session_id: str,
        request: BatchProposalResolutionRequest,
    ) -> V2CopilotSessionView:
        resolved = self._resolve_many(session_id, request, approved=False)
        self._save_system_message(session_id, f"当前队列中的 {len(resolved)} 条请示已驳回。")
        return self.get_session_view(session_id)

    def _planner_executor_reviewer(
        self,
        *,
        session_id: str,
        message_id: str,
        event_id: str,
        content: str,
    ) -> CopilotStructuredAnswer:
        event = self.platform.get_event(event_id)
        memory_snapshot = self.memory_store.load_snapshot(session_id, area_id=event.area_id)
        rule_plan = self.rule_planner.plan(content, event_id, memory_snapshot)
        loop_result = self.execution_engine.run_agent_loop(
            session_id=session_id,
            message_id=message_id,
            event_id=event_id,
            question=content,
            rule_plan=rule_plan,
            memory_snapshot=memory_snapshot,
            llm_planner=self.llm_planner,
            rule_planner=self.rule_planner,
            evaluate_round=lambda plan, executions, outputs: self._assess_round(content, plan, executions, outputs),
        )
        answer = self._build_structured_answer(content, event_id, loop_result)
        updated_memory = self.memory_store.apply_answer(
            session_id,
            area_id=event.area_id,
            target_entity_id=answer.memory_snapshot.focus_entity_id if answer.memory_snapshot else None,
            target_entity_name=answer.memory_snapshot.focus_entity_name if answer.memory_snapshot else None,
            intent=loop_result.final_plan.intent,
            unresolved_slots=[*answer.missing_data, *answer.evidence_gaps],
            completion_status=answer.completion_status,
            carried_context_notes=answer.carried_context_notes,
            pending_proposal_id=answer.proposal.proposal_id if answer.proposal else None,
        )
        answer.memory_snapshot = updated_memory
        return answer

    def _build_structured_answer(
        self,
        question: str,
        event_id: str,
        loop_result,
    ) -> CopilotStructuredAnswer:
        outputs = loop_result.outputs_by_tool
        impact_payload = outputs.get("synthesize_entity_impact")
        impact = EntityImpactView.model_validate(impact_payload) if isinstance(impact_payload, dict) else None
        knowledge_payload = outputs.get("get_knowledge_evidence")
        knowledge_evidence = self._parse_evidence_items(knowledge_payload)
        proposal_payload = outputs.get("draft_action_proposal") or outputs.get("create_action_proposal")
        proposal = ActionProposal.model_validate(proposal_payload) if isinstance(proposal_payload, dict) else None
        shared_memory = self.platform.get_shared_memory_snapshot(event_id)
        pending_proposals = self.platform.repository.list_v2_action_proposals(
            event_id,
            statuses=[ProposalStatus.PENDING.value],
        )[:4]

        top_risks: list[str] = []
        if impact is not None:
            target_entity = impact.entity
        else:
            exposure_payload = outputs.get("get_exposure_summary")
            target_entity = None
            if isinstance(exposure_payload, dict):
                raw_top_risks = exposure_payload.get("top_risks")
                if isinstance(raw_top_risks, list):
                    top_risks = [item for item in raw_top_risks if isinstance(item, str)]

        llm_answer = self.llm_gateway.generate_copilot_chat(
            {
                "question": question,
                "event_id": event_id,
                "plan": loop_result.final_plan.model_dump(mode="json"),
                "tool_trace": [item.model_dump(mode="json") for item in self._to_tool_execution_results(loop_result.tool_executions)],
                "impact": impact.model_dump(mode="json") if impact is not None else None,
                "top_risks": top_risks,
                "knowledge_evidence": [item.model_dump(mode="json") for item in knowledge_evidence],
                "completion_assessment": loop_result.completion_assessment.model_dump(mode="json"),
                "carried_context_notes": loop_result.carried_context_notes,
                "shared_memory_open_questions": shared_memory.open_questions[:4],
                "pending_proposals": [
                    {
                        "proposal_id": item.proposal_id,
                        "title": item.title,
                        "summary": item.summary,
                        "action_type": item.action_type,
                    }
                    for item in pending_proposals
                ],
            }
        )
        summary = llm_answer.answer
        evidence = knowledge_evidence if knowledge_evidence else (impact.evidence if impact is not None else [])
        impact_summary = llm_answer.impact_summary or top_risks
        recommended_actions = llm_answer.recommended_actions
        follow_up_prompts = list(llm_answer.follow_up_prompts)
        confidence = llm_answer.confidence
        missing_data = list(llm_answer.missing_data)
        requires_human_confirmation = False

        assessment: CompletionAssessment = loop_result.completion_assessment
        all_missing_data = list(dict.fromkeys([*missing_data, *assessment.missing_data]))
        evidence_gaps = list(dict.fromkeys(assessment.evidence_gaps))
        if target_entity and target_entity.entity_type in HIGH_RISK_TYPES:
            requires_human_confirmation = True
        if assessment.status == CompletionStatus.HUMAN_ESCALATION:
            requires_human_confirmation = True

        confidence = self._review_confidence(
            base_confidence=confidence,
            evidence_gaps=evidence_gaps,
            executions=loop_result.tool_executions,
            completion_status=assessment.status,
        )
        target_label = target_entity.name if target_entity else "unresolved target"
        planner_summary = (
            f"Planner intent: {loop_result.final_plan.intent}. Target: {target_label}. "
            f"Executed {len(loop_result.tool_executions)} tool step(s) across {len(loop_result.plan_runs)} plan run(s)."
        )

        provisional_memory = MemorySnapshot(
            session_id="pending",
            focus_entity_id=target_entity.entity_id if target_entity else None,
            focus_entity_name=target_entity.name if target_entity else None,
            focus_area_id=target_entity.area_id if target_entity else self.platform.get_event(event_id).area_id,
            current_goal=loop_result.final_plan.intent,
        )
        experience_records = self.platform.operational_experience_store.query_similar_cases(
            event_id=event_id,
            entity_id=target_entity.entity_id if target_entity else None,
            entity_type=target_entity.entity_type.value if target_entity else None,
            risk_level=impact.risk_level if impact is not None else None,
            limit=4,
        )
        strategy_patterns = self.platform.operational_experience_store.rank_strategy_patterns(
            entity_type=target_entity.entity_type.value if target_entity else None,
            risk_level=impact.risk_level if impact is not None else None,
            limit=3,
        )
        experience_summary = ""
        if strategy_patterns:
            top_pattern = strategy_patterns[0]
            experience_summary = (
                f"Historical pattern: {top_pattern.action_type} appears in {top_pattern.sample_size} similar case(s) "
                f"with approval rate {top_pattern.approval_rate:.0%}."
            )
        elif experience_records:
            experience_summary = f"Loaded {len(experience_records)} similar historical record(s) for comparison."
        outcome_risk_notes = [record.action_summary for record in experience_records[:3] if record.outcome in {"rejected", "failed", "execution_failed"}]

        return CopilotStructuredAnswer(
            answer=summary,
            evidence=evidence[:8],
            impact_summary=impact_summary,
            recommended_actions=recommended_actions,
            follow_up_prompts=follow_up_prompts,
            confidence=confidence,
            confidence_explanation=llm_answer.confidence_explanation,
            requires_human_confirmation=requires_human_confirmation,
            missing_data=all_missing_data,
            proposal=proposal,
            generation_source="llm",
            model_name=self.llm_gateway.model_name,
            grounding_summary=llm_answer.grounding_summary,
            planner_summary=planner_summary,
            tool_selection_reasoning=loop_result.final_plan.tool_selection_reasoning,
            skipped_tools=loop_result.final_plan.skipped_tools,
            tool_executions=self._to_tool_execution_results(loop_result.tool_executions),
            data_freshness=self._data_freshness_summary(loop_result.tool_executions, evidence),
            evidence_gaps=evidence_gaps,
            tool_trace=self._build_tool_trace(loop_result.tool_executions),
            planning_layers_summary=loop_result.planning_layers_summary,
            plan_runs=loop_result.plan_runs,
            completion_status=assessment.status,
            termination_reason=assessment.termination_reason,
            memory_snapshot=provisional_memory,
            replan_count=loop_result.replan_count,
            used_fallbacks=loop_result.used_fallbacks,
            carried_context_notes=loop_result.carried_context_notes,
            experience_summary=experience_summary,
            historical_pattern_refs=[pattern.pattern_id for pattern in strategy_patterns],
            outcome_risk_notes=outcome_risk_notes,
        )

    def _assess_round(
        self,
        question: str,
        plan: CopilotExecutionPlan,
        executions: list[ToolExecutionAuditRecord],
        outputs: dict[str, object],
    ) -> CompletionAssessment:
        evidence_gaps: list[str] = []
        missing_data: list[str] = []
        status = CompletionStatus.DIRECT_ANSWER
        should_replan = False

        required_failures = [
            item for item in executions if item.tool_name in plan.required_tools and item.status in {ToolExecutionStatus.FAILED, ToolExecutionStatus.TIMEOUT}
        ]
        if required_failures:
            should_replan = True
            status = CompletionStatus.CONSERVATIVE_ANSWER
            evidence_gaps.extend(
                [f"{item.tool_name} degraded with {item.failure_reason or item.status.value}" for item in required_failures]
            )

        if "resolve_target_entity" not in outputs or not isinstance(outputs.get("resolve_target_entity"), dict):
            should_replan = True
            status = CompletionStatus.CONSERVATIVE_ANSWER
            evidence_gaps.append("Target resolution did not produce a usable entity.")

        hazard_execution = next((item for item in executions if item.tool_name == "get_hazard_tiles"), None)
        if hazard_execution and hazard_execution.data_freshness_seconds and hazard_execution.data_freshness_seconds > 900:
            should_replan = True
            status = CompletionStatus.CONSERVATIVE_ANSWER
            evidence_gaps.append("Realtime hazard data is older than the freshness threshold.")

        if self._is_route_related(question):
            route_ok = any(item.tool_name == "get_route_options" and item.status == ToolExecutionStatus.SUCCESS for item in executions)
            traffic_ok = any(item.tool_name == "get_live_traffic" and item.status == ToolExecutionStatus.SUCCESS for item in executions)
            if not route_ok:
                should_replan = True
                status = CompletionStatus.CONSERVATIVE_ANSWER
                evidence_gaps.append("Route guidance was requested but route options were unavailable.")
            if not traffic_ok:
                should_replan = True
                status = CompletionStatus.CONSERVATIVE_ANSWER
                evidence_gaps.append("Route guidance is missing live traffic confirmation.")

        impact_payload = outputs.get("synthesize_entity_impact")
        impact = EntityImpactView.model_validate(impact_payload) if isinstance(impact_payload, dict) else None
        if impact is None:
            should_replan = True
            status = CompletionStatus.CONSERVATIVE_ANSWER
            evidence_gaps.append("Entity impact synthesis is not available.")
        else:
            if impact.entity.current_occupancy <= 0:
                missing_data.append("Target occupancy is unavailable and should be confirmed manually.")
            if impact.entity.entity_type in HIGH_RISK_TYPES:
                status = CompletionStatus.HUMAN_ESCALATION
                should_replan = False
            elif evidence_gaps:
                status = CompletionStatus.CONSERVATIVE_ANSWER

        termination_reason = {
            CompletionStatus.DIRECT_ANSWER: "Evidence is sufficient for a direct answer.",
            CompletionStatus.CONSERVATIVE_ANSWER: "The copilot can answer conservatively but key dependencies remain degraded.",
            CompletionStatus.HUMAN_ESCALATION: "The target falls under high-risk governance and must be reviewed by a human operator.",
        }[status]
        return CompletionAssessment(
            status=status,
            termination_reason=termination_reason,
            should_replan=should_replan and status != CompletionStatus.HUMAN_ESCALATION,
            evidence_gaps=list(dict.fromkeys(evidence_gaps)),
            missing_data=list(dict.fromkeys(missing_data)),
        )

    def _resolve_many(
        self,
        session_id: str,
        request: BatchProposalResolutionRequest,
        *,
        approved: bool,
    ) -> list[ActionProposal]:
        proposal_ids = list(dict.fromkeys(request.proposal_ids))
        if not proposal_ids:
            raise ValueError("proposal_ids cannot be empty.")
        resolved: list[ActionProposal] = []
        for proposal_id in proposal_ids:
            resolved.append(
                self._resolve_single_proposal(
                    session_id,
                    proposal_id,
                    ProposalResolutionRequest(
                        operator_id=request.operator_id,
                        operator_role=request.operator_role,
                        note=request.note,
                    ),
                    approved=approved,
                    emit_summary_message=False,
                )
            )
        return resolved

    def _resolve_single_proposal(
        self,
        session_id: str,
        proposal_id: str,
        request: ProposalResolutionRequest,
        *,
        approved: bool,
        emit_summary_message: bool,
    ) -> ActionProposal:
        session = self.platform.repository.get_v2_copilot_session(session_id)
        if session is None:
            raise ValueError("v2 copilot session not found.")

        proposal = self.platform.repository.get_v2_action_proposal(proposal_id)
        if proposal is None:
            raise ValueError("proposal not found.")
        if proposal.event_id != session["event_id"]:
            raise ValueError("proposal does not belong to the current event.")
        if proposal.source_session_id and proposal.source_session_id != session_id:
            raise ValueError("proposal does not belong to the current session.")
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError("proposal has already been resolved.")
        if request.operator_role not in proposal.required_operator_roles:
            raise ValueError("当前角色无权处理这条请示。")

        resolved_status = ProposalStatus.APPROVED if approved else ProposalStatus.REJECTED
        resolved_at = datetime.now(timezone.utc)
        payload = dict(proposal.payload)
        payload.setdefault("entity_type", self._infer_entity_type(proposal))
        payload.setdefault("entity_name", self._infer_entity_name(proposal))

        if approved:
            entity = self.platform.get_entity_profile(proposal.entity_id) if proposal.entity_id else None
            drafts = []
            logs = []
            if entity is not None:
                drafts, logs = self.platform.notification_gateway.build_execution_bundle(
                    event_id=proposal.event_id,
                    proposal=proposal,
                    entity=entity,
                    operator_id=request.operator_id,
                )
                for draft in drafts:
                    self.platform.repository.save_v2_notification_draft(draft)
                for log in logs:
                    self.platform.repository.save_v2_execution_log(log)
                self.platform.add_audit_record(
                    source_type="notification_gateway",
                    action="execution_bundle_generated",
                    summary=f"已为 {proposal.proposal_id} 生成 {len(drafts)} 份通知草稿。",
                    details={"proposal_id": proposal.proposal_id, "draft_count": len(drafts), "log_count": len(logs)},
                    event_id=proposal.event_id,
                    session_id=session_id,
                )
                payload["notification_drafts"] = [draft.model_dump(mode="json") for draft in drafts]
                payload["execution_logs"] = [log.model_dump(mode="json") for log in logs]
                payload["execution_summary"] = {
                    "draft_count": len(drafts),
                    "log_count": len(logs),
                    "last_operator": request.operator_id,
                }
                self.platform.repository.add_v2_stream_record_for_payload(
                    proposal.event_id,
                    "notification_sent",
                    {
                        "proposal_id": proposal.proposal_id,
                        "entity_id": proposal.entity_id,
                        "draft_count": len(drafts),
                    },
                )

        updated = proposal.model_copy(
            update={
                "payload": payload,
                "status": resolved_status,
                "resolved_at": resolved_at,
                "resolved_by": request.operator_id,
                "resolution_note": request.note.strip(),
            }
        )
        self.platform.repository.save_v2_action_proposal(updated)
        self.platform.repository.add_v2_stream_record_for_payload(
            updated.event_id,
            "approval_resolved",
            {
                "proposal_id": updated.proposal_id,
                "status": updated.status.value,
                "resolved_by": request.operator_id,
                "entity_id": updated.entity_id,
            },
        )
        self.platform.add_audit_record(
            source_type="proposal_resolution",
            action="proposal_approved" if approved else "proposal_rejected",
            summary=f"请示 {updated.proposal_id} 已{'批准' if approved else '驳回'}。",
            details={
                "proposal_id": updated.proposal_id,
                "entity_id": updated.entity_id,
                "resolved_by": request.operator_id,
                "note": request.note.strip(),
            },
            severity=AlertSeverity.INFO,
            event_id=updated.event_id,
            session_id=session_id,
        )
        entity_type_value = payload.get("entity_type")
        resolved_entity_type = entity_type_value if isinstance(entity_type_value, str) else None
        risk_level = None
        try:
            if updated.entity_id:
                impact = self.platform.get_entity_impact(updated.entity_id, event_id=updated.event_id)
                risk_level = impact.risk_level
        except Exception:
            risk_level = None
        self.platform.operational_experience_store.record_outcome(
            event_id=updated.event_id,
            entity_id=updated.entity_id,
            entity_type=resolved_entity_type,
            risk_level=risk_level,
            action_type="proposal_resolution",
            action_summary=updated.summary,
            outcome="approved" if approved else "rejected",
            confidence=0.9 if approved else 0.8,
            tags=[updated.severity, request.operator_role],
            payload={
                "proposal_id": updated.proposal_id,
                "resolved_by": request.operator_id,
                "resolution_note": request.note.strip(),
            },
        )
        self.memory_store.apply_proposal_resolution(session_id, updated.proposal_id, approved=approved)
        self.platform.publish_trigger(
            updated.event_id,
            trigger_type=TriggerEventType.PROPOSAL_RESOLVED,
            payload={"session_id": session_id, "proposal_id": updated.proposal_id},
        )

        if approved:
            message = f"已批准《{updated.title}》，并生成 {len(payload.get('notification_drafts', []))} 份通知草稿。"
        else:
            message = f"已驳回《{updated.title}》。"
        if request.note.strip():
            message = f"{message} 备注：{request.note.strip()}"
        if emit_summary_message:
            self._save_system_message(session_id, message, created_at=resolved_at)
        return updated

    @staticmethod
    def _parse_evidence_items(payload) -> list[EvidenceItem]:
        if not isinstance(payload, list):
            return []
        return [EvidenceItem.model_validate(item) for item in payload if isinstance(item, dict)]

    @staticmethod
    def _to_tool_execution_results(records: list[ToolExecutionAuditRecord]) -> list[ToolExecutionResult]:
        return [
            ToolExecutionResult.model_validate(record.model_dump(mode="json"))
            for record in records
        ]

    @staticmethod
    def _review_confidence(
        *,
        base_confidence: float,
        evidence_gaps: list[str],
        executions: list[ToolExecutionAuditRecord],
        completion_status: CompletionStatus,
    ) -> float:
        confidence = base_confidence
        confidence -= min(0.18, len(evidence_gaps) * 0.03)
        if any(item.status == ToolExecutionStatus.TIMEOUT for item in executions):
            confidence -= 0.06
        if any(item.status == ToolExecutionStatus.FAILED for item in executions):
            confidence -= 0.04
        if completion_status == CompletionStatus.CONSERVATIVE_ANSWER:
            confidence = min(confidence, 0.62)
        if completion_status == CompletionStatus.HUMAN_ESCALATION:
            confidence = min(confidence, 0.68)
        return round(max(0.35, confidence), 2)

    @staticmethod
    def _data_freshness_summary(
        executions: list[ToolExecutionAuditRecord],
        evidence: list[EvidenceItem],
    ) -> DataFreshnessSummary:
        hazard_freshness = next(
            (item.data_freshness_seconds for item in executions if item.tool_name == "get_hazard_tiles"),
            None,
        )
        traffic_freshness = next(
            (item.data_freshness_seconds for item in executions if item.tool_name == "get_live_traffic"),
            None,
        )
        knowledge_timestamps = [item.timestamp for item in evidence if item.timestamp is not None and item.evidence_type != "realtime"]
        rag_summary = (
            f"最新知识证据时间：{max(knowledge_timestamps).isoformat()}"
            if knowledge_timestamps
            else "当前回答中没有带时间戳的知识证据。"
        )
        return DataFreshnessSummary(
            hazard_state_freshness_seconds=hazard_freshness,
            traffic_freshness_seconds=traffic_freshness,
            profile_freshness_label="runtime_profile_store",
            rag_document_recency_summary=rag_summary,
        )

    @staticmethod
    def _build_tool_trace(executions: list[ToolExecutionAuditRecord]) -> list[ToolTraceStep]:
        trace: list[ToolTraceStep] = []
        for execution in executions:
            if execution.cache_hit:
                summary = f"命中缓存：{execution.output_summary}"
            elif execution.status == ToolExecutionStatus.SKIPPED:
                summary = f"已跳过：{execution.output_summary}"
            elif execution.status == ToolExecutionStatus.FAILED:
                summary = f"执行失败：{execution.failure_reason or execution.output_summary}"
            elif execution.status == ToolExecutionStatus.TIMEOUT:
                summary = f"执行超时（{execution.duration_ms} 毫秒）：{execution.output_summary}"
            else:
                summary = execution.output_summary
            trace.append(ToolTraceStep(tool_name=execution.tool_name, summary=summary))
        return trace

    @staticmethod
    def _is_route_related(question: str) -> bool:
        normalized_question = question.lower()
        return any(keyword in normalized_question for keyword in ROUTE_KEYWORDS)

    def _save_system_message(self, session_id: str, content: str, *, created_at: datetime | None = None) -> None:
        self.platform.repository.save_v2_copilot_message(
            session_id,
            V2CopilotMessage(
                message_id=f"v2_msg_{uuid4().hex[:10]}",
                role="assistant",
                content=content,
                created_at=created_at or datetime.now(timezone.utc),
            ),
        )

    def _backfill_event_reports_to_session(self, session_id: str, event_id: str) -> None:
        for report in reversed(self.platform.list_daily_reports(event_id)):
            self._save_system_message(
                session_id,
                format_daily_report_message(report),
                created_at=report.created_at,
            )
        for summary in reversed(self.platform.list_episode_summaries(event_id)):
            self._save_system_message(
                session_id,
                format_episode_summary_message(summary),
                created_at=summary.created_at,
            )

    def _infer_entity_type(self, proposal) -> str | None:
        if proposal.entity_id:
            try:
                return self.platform.get_entity_profile(proposal.entity_id).entity_type.value
            except ValueError:
                return None
        return None

    def _infer_entity_name(self, proposal) -> str | None:
        if proposal.entity_id:
            try:
                return self.platform.get_entity_profile(proposal.entity_id).name
            except ValueError:
                return None
        return None
