from __future__ import annotations

from uuid import uuid4

from .copilot_rules import (
    ACTION_KEYWORDS,
    ENTITY_TYPE_KEYWORDS,
    EVIDENCE_KEYWORDS,
    HIGH_RISK_TYPES,
    PROPOSAL_KEYWORDS,
    ROUTE_KEYWORDS,
)
from .models import (
    CopilotExecutionPlan,
    EntityType,
    MemorySnapshot,
    MergedPlanDecision,
    PlannerSuggestion,
    PlanningLayer,
    SkippedToolReason,
)


class RulePlanner:
    def __init__(self, platform) -> None:
        self.platform = platform

    def plan(self, question: str, event_id: str, memory_snapshot: MemorySnapshot | None = None) -> CopilotExecutionPlan:
        target_hint = self._resolve_target_hint(event_id, question, memory_snapshot)
        route_related = self._is_route_related(question)
        needs_action_guidance = self._needs_action_guidance(question)
        group_question = target_hint["mode"] != "explicit"

        required_tools = ["resolve_target_entity", "get_hazard_tiles"]
        optional_tools: list[str] = ["get_knowledge_evidence"]
        selection_reasoning = [
            "Rule planner always resolves the target explicitly before any action synthesis.",
            "Rule planner always checks the latest hazard tiles to preserve a realtime safety baseline.",
        ]

        if group_question:
            required_tools.append("get_exposure_summary")
            selection_reasoning.append(target_hint["reason"])

        optional_tools.extend(["get_entity_profile", "synthesize_entity_impact"])
        selection_reasoning.extend(
            [
                "Load the entity profile to ground vulnerability, occupancy, transport, and notification context.",
                "Synthesize entity impact so downstream advice uses deterministic flood-impact assessment.",
                "Load knowledge evidence so the final answer can cite policy, case, and profile sources.",
            ]
        )

        target_entity_type = target_hint.get("entity_type")
        if route_related:
            optional_tools.extend(["get_route_options", "get_live_traffic", "get_shelter_capacity"])
            selection_reasoning.extend(
                [
                    "The question asks for routing or evacuation guidance, so route options are required.",
                    "Live traffic is required before recommending a route.",
                    "Shelter capacity is required before naming an evacuation destination.",
                ]
            )

        if needs_action_guidance or target_entity_type in {item.value for item in HIGH_RISK_TYPES}:
            optional_tools.append("get_policy_constraints")
            selection_reasoning.append(
                "Policy constraints are required because the target is high-risk or the question asks for action guidance."
            )

        if self._needs_proposal(question, target_entity_type):
            optional_tools.append("draft_action_proposal")
            selection_reasoning.append(
                "Draft an action proposal because the target is high-risk or the operator is implicitly asking for a formal package."
            )

        selected_tools = list(dict.fromkeys([*required_tools, *optional_tools]))
        return CopilotExecutionPlan(
            plan_id=f"plan_{uuid4().hex[:10]}",
            planning_layer=PlanningLayer.RULE,
            intent=self._classify_intent(question),
            target_entity_resolution=target_hint["reason"],
            target_resolution_mode=target_hint["mode"],
            target_entity_id=target_hint.get("entity_id"),
            target_entity_name=target_hint.get("entity_name"),
            target_entity_type=target_hint.get("entity_type"),
            required_tools=list(dict.fromkeys(required_tools)),
            optional_tools=list(dict.fromkeys(optional_tools)),
            selected_tools=selected_tools,
            tool_selection_reasoning=selection_reasoning,
            skipped_tools=[],
            plan_notes=[
                "Rule planner enforces the minimum safe tool set and keeps approval boundaries deterministic."
            ],
        )

    def merge(
        self,
        rule_plan: CopilotExecutionPlan,
        llm_suggestion: PlannerSuggestion,
        available_tool_names: list[str],
    ) -> CopilotExecutionPlan:
        available = set(available_tool_names)
        required = list(rule_plan.required_tools)
        optional = [item for item in rule_plan.optional_tools if item in available]
        ordered = []
        seen = set()

        llm_candidates = [item for item in llm_suggestion.selected_tools if item in available]
        for tool_name in [*required, *llm_candidates, *optional]:
            if tool_name not in seen:
                ordered.append(tool_name)
                seen.add(tool_name)

        skipped_tools = self._build_skipped_tools(ordered, rule_plan.target_resolution_mode)
        merged = rule_plan.model_copy(
            update={
                "plan_id": f"plan_{uuid4().hex[:10]}",
                "planning_layer": PlanningLayer.MERGED,
                "selected_tools": ordered,
                "tool_selection_reasoning": list(dict.fromkeys([*rule_plan.tool_selection_reasoning, *llm_suggestion.tool_selection_reasoning])),
                "skipped_tools": skipped_tools,
                "plan_notes": list(dict.fromkeys([*rule_plan.plan_notes, *llm_suggestion.plan_notes])),
            }
        )
        return merged

    def replan(
        self,
        base_plan: CopilotExecutionPlan,
        *,
        recent_failures: list[str],
        replan_round: int,
        available_tool_names: list[str],
    ) -> CopilotExecutionPlan:
        selected = list(base_plan.selected_tools)
        notes = list(base_plan.plan_notes)
        reasoning = list(base_plan.tool_selection_reasoning)

        if any("resolve_target_entity" in item or "target" in item for item in recent_failures) and "get_exposure_summary" not in selected:
            selected.insert(1, "get_exposure_summary")
            reasoning.append("Replan adds exposure summary because target resolution was degraded in the previous round.")
        if any("traffic" in item for item in recent_failures) and "get_live_traffic" not in selected:
            selected.append("get_live_traffic")
            reasoning.append("Replan adds live traffic because route guidance lacked current road confirmation.")
        if any("route" in item for item in recent_failures) and "get_route_options" not in selected:
            selected.append("get_route_options")
            reasoning.append("Replan adds route options because route coverage was incomplete in the previous round.")

        selected = [item for item in selected if item in available_tool_names]
        notes.append(f"Replan round {replan_round} triggered by: {', '.join(recent_failures) or 'review downgrade'}.")
        return base_plan.model_copy(
            update={
                "plan_id": f"plan_{uuid4().hex[:10]}",
                "planning_layer": PlanningLayer.REPLAN,
                "selected_tools": list(dict.fromkeys(selected)),
                "tool_selection_reasoning": list(dict.fromkeys(reasoning)),
                "skipped_tools": self._build_skipped_tools(selected, base_plan.target_resolution_mode),
                "plan_notes": notes,
                "replan_round": replan_round,
                "parent_plan_id": base_plan.plan_id,
            }
        )

    def _resolve_target_hint(
        self,
        event_id: str,
        question: str,
        memory_snapshot: MemorySnapshot | None,
    ) -> dict[str, str | None]:
        normalized_question = question.lower()
        event = self.platform.get_event(event_id)
        entities = self.platform.list_entity_profiles(area_id=event.area_id)

        for entity in entities:
            name_variants = {
                entity.name.lower(),
                entity.entity_id.lower(),
                entity.location_hint.lower(),
                entity.village.lower(),
            }
            if any(variant and variant in normalized_question for variant in name_variants):
                return {
                    "mode": "explicit",
                    "entity_id": entity.entity_id,
                    "entity_name": entity.name,
                    "entity_type": entity.entity_type.value,
                    "reason": f"Question explicitly names {entity.name}, so the planner targets that entity directly.",
                }

        for entity_type, keywords in ENTITY_TYPE_KEYWORDS.items():
            if any(keyword in normalized_question for keyword in keywords):
                return {
                    "mode": "entity_type_search",
                    "entity_id": None,
                    "entity_name": None,
                    "entity_type": entity_type.value,
                    "reason": (
                        f"Question references {entity_type.value}, so the planner will use exposure ranking to pick "
                        "the highest-risk target in that group."
                    ),
                }

        if memory_snapshot and memory_snapshot.focus_entity_id:
            return {
                "mode": "memory_focus",
                "entity_id": memory_snapshot.focus_entity_id,
                "entity_name": memory_snapshot.focus_entity_name,
                "entity_type": None,
                "reason": "Question does not name a target, so the planner will reuse the focus entity from the previous turn.",
            }

        return {
            "mode": "top_exposure",
            "entity_id": None,
            "entity_name": None,
            "entity_type": None,
            "reason": "Question does not name a target, so the planner will fall back to the highest exposed entity.",
        }

    def _build_skipped_tools(self, selected_tools: list[str], target_resolution_mode: str) -> list[SkippedToolReason]:
        selected = set(selected_tools)
        skipped: list[SkippedToolReason] = []
        for tool_name in self.platform.tools.tool_names():
            if tool_name in selected:
                continue
            reason = self._skip_reason(tool_name, target_resolution_mode)
            skipped.append(SkippedToolReason(tool_name=tool_name, reason=reason))
        return skipped

    @staticmethod
    def _skip_reason(tool_name: str, target_resolution_mode: str) -> str:
        if tool_name == "get_exposure_summary":
            if target_resolution_mode == "explicit":
                return "Question already names a specific entity, so exposure ranking was not needed."
            return "Exposure summary is not required for the current target resolution path."
        if tool_name == "get_route_options":
            return "Question does not ask for routing, evacuation, or destination choice."
        if tool_name == "get_live_traffic":
            return "Question does not require live traffic confirmation."
        if tool_name == "get_shelter_capacity":
            return "Question does not require shelter selection."
        if tool_name == "get_policy_constraints":
            return "Question is informational and does not require policy guardrails."
        if tool_name in {"draft_action_proposal", "create_action_proposal"}:
            return "Current target does not clearly require a proposal package."
        return f"{tool_name} was not required for this question."

    @staticmethod
    def _classify_intent(question: str) -> str:
        normalized_question = question.lower()
        if any(keyword in normalized_question for keyword in ROUTE_KEYWORDS):
            return "route_guidance"
        if any(keyword in normalized_question for keyword in EVIDENCE_KEYWORDS):
            return "evidence_review"
        if any(keyword in normalized_question for keyword in ACTION_KEYWORDS):
            return "action_guidance"
        if any(
            keyword in normalized_question
            for keyword in (
                *ENTITY_TYPE_KEYWORDS[EntityType.SCHOOL],
                *ENTITY_TYPE_KEYWORDS[EntityType.FACTORY],
                *ENTITY_TYPE_KEYWORDS[EntityType.HOSPITAL],
                *ENTITY_TYPE_KEYWORDS[EntityType.NURSING_HOME],
            )
        ):
            return "institution_impact"
        return "entity_impact"

    @staticmethod
    def _is_route_related(question: str) -> bool:
        normalized_question = question.lower()
        return any(keyword in normalized_question for keyword in ROUTE_KEYWORDS)

    @staticmethod
    def _needs_action_guidance(question: str) -> bool:
        normalized_question = question.lower()
        return any(keyword in normalized_question for keyword in ACTION_KEYWORDS)

    @staticmethod
    def _needs_proposal(question: str, entity_type: str | None) -> bool:
        normalized_question = question.lower()
        if entity_type in {item.value for item in HIGH_RISK_TYPES}:
            return True
        return any(keyword in normalized_question for keyword in PROPOSAL_KEYWORDS)
