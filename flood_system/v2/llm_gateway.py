from __future__ import annotations

import json
import os
import socket
import ssl
from http.client import RemoteDisconnected
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import ExecutionMode, LLMErrorCode


class LLMGenerationError(RuntimeError):
    def __init__(self, code: LLMErrorCode | str, message: str) -> None:
        self.code = code.value if isinstance(code, LLMErrorCode) else str(code)
        super().__init__(message)


class GatewayModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class ObjectAdvisoryOutput(GatewayModel):
    answer: str
    impact_summary: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_explanation: str = ""
    missing_data: list[str] = Field(default_factory=list)
    grounding_summary: str = ""


class CopilotChatOutput(GatewayModel):
    answer: str
    impact_summary: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    follow_up_prompts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_explanation: str = ""
    missing_data: list[str] = Field(default_factory=list)
    grounding_summary: str = ""


class RegionalDecisionAction(GatewayModel):
    action_type: str
    execution_mode: ExecutionMode
    decision_reason: str
    action_goal: str


class RegionalDecisionOutput(GatewayModel):
    summary: str
    actions: list[RegionalDecisionAction] = Field(default_factory=list)
    grounding_summary: str = ""


class ProposalDraftOutput(GatewayModel):
    action_display_name: str = ""
    action_display_tagline: str = ""
    action_display_category: str = ""
    title: str
    summary: str
    trigger_reason: str
    recommendation: str
    evidence_summary: str
    high_risk_object_ids: list[str] = Field(default_factory=list)
    action_scope: dict[str, Any] = Field(default_factory=dict)
    grounding_summary: str = ""
    chat_follow_up_prompt: str = ""


class RegionalAnalysisPackageOutput(GatewayModel):
    analysis_message: str = ""
    risk_assessment: str = ""
    rescue_plan: str = ""
    resource_dispatch_plan: str = ""


class ExecutionDraftContent(GatewayModel):
    audience: str
    channel: str
    content: str


class ExecutionBundleOutput(GatewayModel):
    drafts: list[ExecutionDraftContent] = Field(default_factory=list)
    task_summary: str
    task_instructions: list[str] = Field(default_factory=list)
    grounding_summary: str = ""


class ExecutionSummaryOutput(GatewayModel):
    approval_summary: str
    audit_summary: str


class DailyOperationsSummaryOutput(GatewayModel):
    headline: str
    situation_summary: str
    decisions_summary: str
    action_summary: str
    unresolved_risks: list[str] = Field(default_factory=list)
    next_day_recommendations: list[str] = Field(default_factory=list)
    grounding_summary: str = ""


class HighRiskPostmortemOutput(GatewayModel):
    headline: str
    escalation_path: list[str] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    successful_actions: list[str] = Field(default_factory=list)
    failed_actions: list[str] = Field(default_factory=list)
    coordination_gaps: list[str] = Field(default_factory=list)
    reusable_rules: list[str] = Field(default_factory=list)
    memory_tags: list[str] = Field(default_factory=list)


class PromptProfileConfig(GatewayModel):
    system_prompt: str
    model_name: str | None = None
    description: str = ""


class PromptProfileDocument(GatewayModel):
    profiles: dict[str, PromptProfileConfig]


@dataclass
class PromptProfileRegistry:
    profile_path: str | Path | None = None
    _resolved_path: Path = field(init=False)
    _cache_mtime_ns: int | None = field(default=None, init=False)
    _document: PromptProfileDocument | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.profile_path:
            self._resolved_path = Path(self.profile_path)
        else:
            env_path = os.getenv("FLOOD_LLM_PROMPT_PROFILES_PATH", "").strip()
            if env_path:
                self._resolved_path = Path(env_path)
            else:
                self._resolved_path = Path(__file__).resolve().with_name("prompt_profiles.json")

    @property
    def path(self) -> Path:
        return self._resolved_path

    def get(self, prompt_profile: str) -> PromptProfileConfig:
        document = self._load()
        profile = document.profiles.get(prompt_profile)
        if profile is None:
            raise LLMGenerationError(
                LLMErrorCode.INVALID_OUTPUT,
                f"未找到提示词配置：{prompt_profile}。",
            )
        return profile

    def list_profiles(self) -> dict[str, PromptProfileConfig]:
        return dict(self._load().profiles)

    def _load(self) -> PromptProfileDocument:
        if not self.path.exists():
            raise LLMGenerationError(
                LLMErrorCode.UNAVAILABLE,
                f"提示词配置文件不存在：{self.path}",
            )
        stat = self.path.stat()
        if self._document is None or self._cache_mtime_ns != stat.st_mtime_ns:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._document = PromptProfileDocument.model_validate(raw)
            self._cache_mtime_ns = stat.st_mtime_ns
        return self._document


@dataclass
class ResponsesLLMGateway:
    model_name: str = field(default_factory=lambda: os.getenv("FLOOD_LLM_MODEL", "qwen3.5-flash"))
    api_key_env: str = "DASHSCOPE_API_KEY"
    api_key_file: str = "api_key.txt"
    fallback_api_key_env: str = "OPENAI_API_KEY"
    api_protocol: str = field(default_factory=lambda: os.getenv("FLOOD_LLM_PROTOCOL", "chat_completions"))
    api_url: str = field(
        default_factory=lambda: os.getenv(
            "FLOOD_LLM_API_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
    )
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("FLOOD_LLM_TIMEOUT_SECONDS", "60")))
    retry_attempts: int = 2
    prompt_profiles_path: str | None = None
    prompt_registry: PromptProfileRegistry = field(init=False)

    def __post_init__(self) -> None:
        self.prompt_registry = PromptProfileRegistry(self.prompt_profiles_path)

    def generate_object_advisory(self, payload: dict[str, Any]) -> ObjectAdvisoryOutput:
        return self._generate_output(
            prompt_profile="object_advisory",
            payload=payload,
            response_model=ObjectAdvisoryOutput,
        )

    def generate_copilot_chat(self, payload: dict[str, Any]) -> CopilotChatOutput:
        return self._generate_output(
            prompt_profile="copilot_chat",
            payload=payload,
            response_model=CopilotChatOutput,
        )

    def generate_regional_decision(self, payload: dict[str, Any]) -> RegionalDecisionOutput:
        return self._generate_output(
            prompt_profile="regional_decision",
            payload=payload,
            response_model=RegionalDecisionOutput,
        )

    def generate_proposal_draft(self, payload: dict[str, Any]) -> ProposalDraftOutput:
        return self._generate_output(
            prompt_profile="proposal_draft",
            payload=payload,
            response_model=ProposalDraftOutput,
        )

    def generate_regional_analysis_package(self, payload: dict[str, Any]) -> RegionalAnalysisPackageOutput:
        return self._generate_output(
            prompt_profile="regional_analysis_package",
            payload=payload,
            response_model=RegionalAnalysisPackageOutput,
        )

    def generate_regional_analysis_package(self, payload: dict[str, Any]) -> RegionalAnalysisPackageOutput:
        hazard_state = payload.get("hazard_state") or {}
        exposure = payload.get("exposure_summary") or {}
        evidence = payload.get("knowledge_evidence") or []
        proposals = payload.get("pending_proposals") or []
        risk_level = str(hazard_state.get("overall_risk_level") or "Orange")
        top_risks = exposure.get("top_risks") or []
        affected_entities = exposure.get("affected_entities") or []
        focus_names = [
            item.get("entity", {}).get("name")
            for item in affected_entities[:3]
            if isinstance(item, dict) and isinstance(item.get("entity", {}).get("name"), str)
        ]
        evidence_titles = [
            item.get("title")
            for item in evidence[:2]
            if isinstance(item, dict) and isinstance(item.get("title"), str)
        ]
        proposal_titles = [
            item.get("title")
            for item in proposals[:3]
            if isinstance(item, dict) and isinstance(item.get("title"), str)
        ]
        focus_label = ", ".join(focus_names) if focus_names else "priority targets"
        leading_risk = top_risks[0] if top_risks else f"The area remains at {risk_level} risk."
        evidence_label = ", ".join(evidence_titles) if evidence_titles else "simulation and exposure evidence"
        proposal_label = ", ".join(proposal_titles[:2]) if proposal_titles else "notification and rescue coordination"
        return RegionalAnalysisPackageOutput(
            analysis_message=f"A new regional analysis package is ready for {focus_label}.",
            risk_assessment=(
                f"The district is currently at {risk_level} flood risk. "
                f"Primary concern: {leading_risk}. Supporting evidence comes from {evidence_label}."
            ),
            rescue_plan=(
                f"Advance rescue preparation around {focus_label}, "
                f"and prioritize the following actions: {proposal_label}."
            ),
            resource_dispatch_plan=(
                f"Pre-position pumps, traffic coordination teams, and public-warning capacity near {focus_label} "
                "to stay ahead of the next flood peak."
            ),
        )

    def generate_execution_bundle(self, payload: dict[str, Any]) -> ExecutionBundleOutput:
        return self._generate_output(
            prompt_profile="execution_bundle",
            payload=payload,
            response_model=ExecutionBundleOutput,
        )

    def generate_execution_summary(self, payload: dict[str, Any]) -> ExecutionSummaryOutput:
        return self._generate_output(
            prompt_profile="execution_summary",
            payload=payload,
            response_model=ExecutionSummaryOutput,
        )

    def generate_daily_operations_summary(self, payload: dict[str, Any]) -> DailyOperationsSummaryOutput:
        return self._generate_output(
            prompt_profile="daily_operations_summary",
            payload=payload,
            response_model=DailyOperationsSummaryOutput,
        )

    def generate_high_risk_postmortem_summary(self, payload: dict[str, Any]) -> HighRiskPostmortemOutput:
        return self._generate_output(
            prompt_profile="high_risk_postmortem",
            payload=payload,
            response_model=HighRiskPostmortemOutput,
        )

    def _generate_output(
        self,
        *,
        prompt_profile: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
    ) -> Any:
        api_key = self._load_api_key()
        if not api_key:
            raise LLMGenerationError(
                LLMErrorCode.UNAVAILABLE,
                f"{prompt_profile} 阶段无法调用大模型：未配置 OPENAI_API_KEY 或 api_key.txt。",
            )

        profile = self.prompt_registry.get(prompt_profile)
        model_name = profile.model_name or self.model_name
        body = self._build_request_body(
            prompt_profile=prompt_profile,
            profile=profile,
            payload=payload,
            response_model=response_model,
            model_name=model_name,
        )

        last_error: Exception | None = None
        for _ in range(max(1, self.retry_attempts)):
            try:
                response = self._post(api_key, body)
                raw = self._extract_json_output(response, self.api_protocol)
                return response_model.model_validate(raw)
            except LLMGenerationError as exc:
                last_error = exc
                if exc.code == LLMErrorCode.UNAVAILABLE.value:
                    continue
                raise
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
        if isinstance(last_error, LLMGenerationError):
            raise last_error
        raise LLMGenerationError(
            LLMErrorCode.INVALID_OUTPUT,
            f"{prompt_profile} 阶段返回了无法解析的结构化输出：{last_error}",
        )

    def _build_request_body(
        self,
        *,
        prompt_profile: str,
        profile: PromptProfileConfig,
        payload: dict[str, Any],
        response_model: type[BaseModel],
        model_name: str,
    ) -> dict[str, Any]:
        schema = response_model.model_json_schema()
        user_payload = (
            f"阶段：{prompt_profile}\n"
            "请以 JSON 对象输出，并严格满足以下 JSON Schema。\n"
            f"JSON Schema：{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
            f"输入：{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        if self.api_protocol == "responses":
            return {
                "model": model_name,
                "input": (
                    f"{profile.system_prompt}\n\n"
                    "请只返回 JSON 对象，不要返回 Markdown 或额外解释。\n"
                    f"{user_payload}"
                ),
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": prompt_profile,
                        "schema": schema,
                    }
                },
            }
        return {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{profile.system_prompt}\n"
                        "你必须输出严格的 JSON 对象。不要输出 Markdown，不要输出代码块，不要输出额外解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": user_payload,
                },
            ],
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
        }

    def _post(self, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.api_url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - network dependent
            detail = exc.read().decode("utf-8", errors="ignore")
            raise LLMGenerationError(LLMErrorCode.UNAVAILABLE, detail or f"http_{exc.code}") from exc
        except error.URLError as exc:  # pragma: no cover - network dependent
            raise LLMGenerationError(LLMErrorCode.UNAVAILABLE, str(exc.reason)) from exc
        except (RemoteDisconnected, ConnectionResetError, ssl.SSLError) as exc:  # pragma: no cover - network dependent
            raise LLMGenerationError(LLMErrorCode.UNAVAILABLE, str(exc)) from exc
        except (TimeoutError, socket.timeout) as exc:  # pragma: no cover - network dependent
            raise LLMGenerationError(LLMErrorCode.UNAVAILABLE, f"request_timeout: {exc}") from exc

    @staticmethod
    def _extract_json_output(response: dict[str, Any], protocol: str) -> Any:
        if protocol == "chat_completions":
            choices = response.get("choices", [])
            for item in choices:
                message = item.get("message", {})
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return json.loads(content)
            raise ValueError("missing structured output")
        output = response.get("output", [])
        for item in output:
            content = item.get("content", [])
            for chunk in content:
                if chunk.get("type") in {"output_text", "text"} and chunk.get("text"):
                    return json.loads(chunk["text"])
        if response.get("output_text"):
            return json.loads(response["output_text"])
        raise ValueError("missing structured output")

    def _load_api_key(self) -> str:
        env_value = os.getenv(self.api_key_env, "").strip()
        if env_value:
            return env_value

        root = Path(__file__).resolve().parents[2]
        key_file = root / self.api_key_file
        if key_file.exists():
            file_value = key_file.read_text(encoding="utf-8").strip()
            if file_value:
                return file_value

        fallback_value = os.getenv(self.fallback_api_key_env, "").strip()
        if fallback_value:
            return fallback_value
        return ""


@dataclass
class MockLLMGateway:
    model_name: str = "mock-llm"

    def generate_object_advisory(self, payload: dict[str, Any]) -> ObjectAdvisoryOutput:
        impact = payload.get("impact") or {}
        entity = impact.get("entity") or {}
        name = entity.get("name") or "目标对象"
        reasons = [item for item in impact.get("risk_reason", []) if isinstance(item, str)]
        summary = reasons[:2] or [f"{name} 当前处于需要持续关注的风险态势。"]
        actions = self._default_actions(entity.get("entity_type"), name)
        return ObjectAdvisoryOutput(
            answer=f"{name} 当前已进入需要提前处置的风险窗口，建议围绕人员安全、路线可达性和现场联动同步推进。",
            impact_summary=summary,
            recommended_actions=actions,
            confidence=0.82,
            confidence_explanation="已综合对象画像、风险原因、路线状态与补充证据生成建议。",
            missing_data=[],
            grounding_summary=f"基于 {name} 的对象画像、影响评估和证据摘要生成。",
        )

    def generate_copilot_chat(self, payload: dict[str, Any]) -> CopilotChatOutput:
        question = str(payload.get("question") or "当前问题")
        impact = payload.get("impact") or {}
        entity = impact.get("entity") or {}
        name = entity.get("name") or "目标对象"
        reasons = [item for item in impact.get("risk_reason", []) if isinstance(item, str)]
        actions = self._default_actions(entity.get("entity_type"), name)
        top_risks = payload.get("top_risks") or []
        impact_summary = reasons[:2] or [item for item in top_risks if isinstance(item, str)][:2]
        follow_up_prompts = self._default_follow_up_prompts(
            entity_name=name,
            recommended_actions=actions,
            open_questions=self._coerce_strings(payload.get("shared_memory_open_questions")),
            pending_proposals=[
                proposal.get("title")
                for proposal in payload.get("pending_proposals", [])
                if isinstance(proposal, dict) and isinstance(proposal.get("title"), str)
            ],
            question=question,
        )
        return CopilotChatOutput(
            answer=f"针对“{question}”，系统判断 {name} 当前需要优先关注，建议先执行最关键的现场保护与联动动作。",
            impact_summary=impact_summary or [f"{name} 仍处于需要持续关注的风险阶段。"],
            recommended_actions=actions,
            follow_up_prompts=follow_up_prompts,
            confidence=0.78,
            confidence_explanation="答案基于已执行工具返回的影响评估、证据与共享记忆生成。",
            missing_data=[],
            grounding_summary="结合工具执行结果、知识证据和共享记忆生成。",
        )

    def generate_regional_decision(self, payload: dict[str, Any]) -> RegionalDecisionOutput:
        hazard_state = payload.get("hazard_state") or {}
        risk_level = str(hazard_state.get("overall_risk_level") or "Orange")
        actions = [
            RegionalDecisionAction(
                action_type="regional_notification",
                execution_mode=ExecutionMode.NOTIFICATION,
                decision_reason="需要尽快同步全区预警和行动提醒。",
                action_goal="向全区推送风险提醒和关键行动提示。",
            ),
            RegionalDecisionAction(
                action_type="regional_resource_dispatch",
                execution_mode=ExecutionMode.RESOURCE_DISPATCH,
                decision_reason="需要前置部署排涝和交通疏导资源。",
                action_goal="向重点片区预置排涝、交通和现场协同资源。",
            ),
            RegionalDecisionAction(
                action_type="regional_evacuation",
                execution_mode=ExecutionMode.EVACUATION_TASK,
                decision_reason="需要为高风险片区准备分区转移方案。",
                action_goal="形成分区转移建议并锁定优先人群。",
            ),
        ]
        if risk_level == "Red":
            actions.append(
                RegionalDecisionAction(
                    action_type="traffic_control",
                    execution_mode=ExecutionMode.GENERIC_TASK,
                    decision_reason="需要额外控制高风险通道的人车流量。",
                    action_goal="在重点积水走廊执行临时交通管制与分流。",
                )
            )
        return RegionalDecisionOutput(
            summary=f"区域当前为 {risk_level} 风险，建议立即推进区域级动作闭环。",
            actions=actions,
            grounding_summary="基于模拟结果、暴露对象和资源缺口综合生成区域动作集合。",
        )

    def generate_proposal_draft(self, payload: dict[str, Any]) -> ProposalDraftOutput:
        action = payload.get("action") or {}
        exposure = payload.get("exposure_summary") or {}
        entities = exposure.get("affected_entities") or []
        high_risk_ids = [
            item.get("entity", {}).get("entity_id")
            for item in entities[:4]
            if isinstance(item, dict) and isinstance(item.get("entity", {}).get("entity_id"), str)
        ]
        focus_names = [
            item.get("entity", {}).get("name")
            for item in entities[:3]
            if isinstance(item, dict) and isinstance(item.get("entity", {}).get("name"), str)
        ]
        event = payload.get("event") or {}
        action_type = str(action.get("action_type") or "regional_action")
        execution_mode = str(action.get("execution_mode") or ExecutionMode.GENERIC_TASK.value)
        title_map = {
            "regional_notification": "发布区域风险提示",
            "regional_resource_dispatch": "下达区域资源调度",
            "regional_evacuation": "生成区域转移建议",
            "traffic_control": "启动重点通道交通管制",
        }
        scope = self._default_scope(action_type, execution_mode, event.get("title") or "regional event")
        focus_label = ", ".join(focus_names) if focus_names else "priority targets"
        chat_follow_up_prompt = self._proposal_follow_up_prompt(
            title=title_map.get(action_type, f"Regional {action_type}"),
            recommendation=str(action.get("action_goal") or "Clarify the execution goal."),
            focus_label=focus_label,
        )
        return ProposalDraftOutput(
            action_display_name=title_map.get(action_type, f"执行{action_type}动作"),
            action_display_tagline=f"围绕{focus_label}立即组织相关处置与联动。",
            action_display_category=self._display_category_for_action(action_type),
            title=title_map.get(action_type, f"执行 {action_type} 动作"),
            summary=f"建议围绕 {focus_label} 立即执行 {title_map.get(action_type, action_type)}。",
            trigger_reason=str(action.get("decision_reason") or "区域风险进入高位阶段。"),
            recommendation=str(action.get("action_goal") or "请尽快组织执行。"),
            evidence_summary="模拟结果、暴露对象和知识证据显示当前区域已具备主动动作条件。",
            high_risk_object_ids=high_risk_ids,
            action_scope=scope,
            chat_follow_up_prompt=chat_follow_up_prompt,
            grounding_summary="基于区域决策动作、风险摘要和高风险对象清单生成请示草稿。",
        )

    def generate_execution_bundle(self, payload: dict[str, Any]) -> ExecutionBundleOutput:
        proposal = payload.get("proposal") or {}
        execution_mode = str(proposal.get("execution_mode") or ExecutionMode.GENERIC_TASK.value)
        message = (
            (proposal.get("action_scope") or {}).get("message")
            or proposal.get("recommendation")
            or proposal.get("summary")
            or "请立即执行当前区域动作。"
        )
        drafts: list[ExecutionDraftContent] = []
        if execution_mode in {ExecutionMode.NOTIFICATION.value, ExecutionMode.GENERIC_TASK.value}:
            drafts = [
                ExecutionDraftContent(audience="district_public", channel="console", content=str(message)),
                ExecutionDraftContent(audience="operations_desk", channel="briefing", content=f"执行摘要：{message}"),
            ]
        return ExecutionBundleOutput(
            drafts=drafts,
            task_summary=f"围绕动作《{proposal.get('title') or '区域动作'}》生成执行任务。",
            task_instructions=[
                "核对重点区域和高风险对象。",
                "同步值班席与现场协同单元。",
                "记录执行结果并回写审计轨迹。",
            ],
            grounding_summary="基于已确认 proposal、动作参数和高风险对象清单生成执行材料。",
        )

    def generate_execution_summary(self, payload: dict[str, Any]) -> ExecutionSummaryOutput:
        proposal = payload.get("proposal") or {}
        title = proposal.get("title") or "区域动作"
        return ExecutionSummaryOutput(
            approval_summary=f"区域动作《{title}》已进入系统内执行闭环。",
            audit_summary=f"已为《{title}》生成执行任务与审计摘要。",
        )

    def generate_daily_operations_summary(self, payload: dict[str, Any]) -> DailyOperationsSummaryOutput:
        event = payload.get("event") or {}
        counts = payload.get("activity_counts") or {}
        report_date = str(payload.get("report_date") or "昨日")
        title = event.get("title") or "当前事件"
        risk = str(payload.get("latest_risk_level") or "Orange")
        unresolved = payload.get("open_questions") or []
        return DailyOperationsSummaryOutput(
            headline=f"{report_date} 运行日报：{title}",
            situation_summary=(
                f"{title} 在 {report_date} 维持 {risk} 风险态势，"
                f"当日共处理 {counts.get('observations', 0)} 条观测、{counts.get('simulation_updates', 0)} 次模拟更新。"
            ),
            decisions_summary=(
                f"系统累计触发 {counts.get('supervisor_runs', 0)} 次后台巡检，"
                f"处理 {counts.get('proposals', 0)} 条动作请示。"
            ),
            action_summary=(
                f"已生成 {counts.get('notification_drafts', 0)} 份通知草稿，"
                f"记录 {counts.get('execution_logs', 0)} 条执行日志。"
            ),
            unresolved_risks=[item for item in unresolved if isinstance(item, str)][:4],
            next_day_recommendations=[
                "继续核对高风险对象周边的实时水情与可达路线。",
                "优先复核尚未关闭的阻塞项和未决问题。",
            ],
            grounding_summary="基于上一日的事件活动计数、共享记忆、审批记录和执行日志生成。",
        )

    def generate_high_risk_postmortem_summary(self, payload: dict[str, Any]) -> HighRiskPostmortemOutput:
        event = payload.get("event") or {}
        episode = payload.get("episode") or {}
        title = event.get("title") or "当前事件"
        risk = str(episode.get("peak_risk_level") or payload.get("peak_risk_level") or "Red")
        approvals = payload.get("approved_actions") or []
        rejections = payload.get("rejected_actions") or []
        return HighRiskPostmortemOutput(
            headline=f"{title} 高风险阶段复盘",
            escalation_path=[
                f"事件在高风险阶段达到 {risk} 级别。",
                "系统围绕共享记忆、请示审批与执行日志完成复盘。",
            ],
            key_decisions=[*(approvals[:3] if isinstance(approvals, list) else []), *(rejections[:1] if isinstance(rejections, list) else [])],
            successful_actions=[item for item in approvals[:3] if isinstance(item, str)],
            failed_actions=[item for item in rejections[:3] if isinstance(item, str)],
            coordination_gaps=[item for item in (payload.get("open_questions") or [])[:3] if isinstance(item, str)],
            reusable_rules=[
                "高风险对象的请示应优先形成结构化可审批动作。",
                "先确认阻塞项，再推进通知和资源调度闭环。",
            ],
            memory_tags=[item for item in [risk.lower(), "postmortem", "flood_response"] if item],
        )

    @staticmethod
    def _default_actions(entity_type: str | None, name: str) -> list[str]:
        actions = [f"先核对 {name} 周边实时水情和可达路线。", f"同步 {name} 的责任联系人和值班席。"]
        if entity_type in {"school", "nursing_home", "hospital"}:
            actions.append(f"优先准备 {name} 的人员疏导或转移预案。")
        elif entity_type == "factory":
            actions.append(f"优先保护 {name} 的关键库存与生产安全。")
        else:
            actions.append(f"优先确保 {name} 的人员安全与信息触达。")
        return actions

    @staticmethod
    def _default_scope(action_type: str, execution_mode: str, event_title: str) -> dict[str, Any]:
        if action_type == "regional_notification":
            return {
                "target_scope": "全区重点片区",
                "channels": ["broadcast", "sms", "console"],
                "message": f"{event_title} 已进入高风险阶段，请立即关注官方提示并做好防汛准备。",
            }
        if action_type == "regional_resource_dispatch":
            return {
                "resource_types": ["排涝力量", "交通疏导", "现场协同"],
                "resource_count": 6,
                "priority_areas": ["北部片区", "学校周边", "低洼社区"],
                "notes": "优先前置排涝和交通引导力量。",
            }
        if action_type == "regional_evacuation":
            return {
                "evacuation_scope": ["低洼社区", "学校周边"],
                "priority_groups": ["老人", "儿童", "行动不便者"],
                "shelter_direction": "优先引导至北侧安置点",
                "notes": "先组织高风险人群转移准备。",
            }
        return {
            "task_scope": "全区重点风险片区",
            "message": f"{event_title} 需立即执行补充区域动作。",
            "notes": f"执行模式：{execution_mode}",
        }

    @staticmethod
    def _coerce_strings(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        return [item.strip() for item in values if isinstance(item, str) and item.strip()]

    @classmethod
    def _default_follow_up_prompts(
        cls,
        *,
        entity_name: str,
        recommended_actions: list[str],
        open_questions: list[str],
        pending_proposals: list[str],
        question: str,
    ) -> list[str]:
        prompts: list[str] = []
        if entity_name:
            prompts.append(f"继续围绕“{entity_name}”说明接下来 30 分钟的风险变化和最先受影响的环节。")
        if recommended_actions:
            prompts.append(f"把“{recommended_actions[0]}”拆成执行步骤、责任人和确认节点。")
        if open_questions:
            prompts.append(f"优先补充这个未决问题：“{open_questions[0]}”，并说明缺它会影响什么判断。")
        if pending_proposals:
            prompts.append(f"结合待处理事务“{pending_proposals[0]}”，比较现在批准、暂缓和驳回的差异。")
        if not prompts:
            prompts.append(f"结合这轮问题“{question}”，先复述你的判断依据，再说还缺哪一步确认。")
        return list(dict.fromkeys(prompts))[:4]

    @staticmethod
    def _proposal_follow_up_prompt(*, title: str, recommendation: str, focus_label: str) -> str:
        return (
            f"请围绕“{title}”继续追问：先说明为什么现在要处理、会影响哪些对象（{focus_label}），"
            f"再展开执行范围、触发条件，以及“{recommendation}”落地时最需要人工确认的点。"
        )

    @staticmethod
    def _display_category_for_action(action_type: str) -> str:
        category_map = {
            "regional_notification": "预警通知",
            "regional_resource_dispatch": "资源调度",
            "regional_evacuation": "人员转运",
            "traffic_control": "空间管控",
        }
        return category_map.get(action_type, "指挥动作")

