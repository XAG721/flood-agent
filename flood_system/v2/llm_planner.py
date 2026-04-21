from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from .models import CopilotExecutionPlan, PlannerRequestContext, PlannerSuggestion, PlanningLayer


@dataclass
class LLMPlannerAdapter:
    model: str = "gpt-5.4"
    api_key_env: str = "OPENAI_API_KEY"
    api_key_file: str = "api_key.txt"
    api_url: str = "https://api.openai.com/v1/responses"
    timeout_seconds: int = 4

    def plan(
        self,
        context: PlannerRequestContext,
        rule_plan: CopilotExecutionPlan,
        tool_specs: list[dict[str, Any]],
    ) -> PlannerSuggestion:
        api_key = self._load_api_key()
        if not api_key:
            return PlannerSuggestion(
                planning_layer=PlanningLayer.LLM,
                selected_tools=list(rule_plan.selected_tools),
                tool_selection_reasoning=[
                    "LLM planner is unavailable because no default API key source is configured; keep the rule baseline."
                ],
                plan_notes=["LLM planner fallback: missing API key from env or api_key.txt."],
                invalid_reason="missing_api_key",
            )

        prompt = self._build_prompt(context, rule_plan, tool_specs)
        payload = {
            "model": self.model,
            "input": prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "planner_suggestion",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "selected_tools": {"type": "array", "items": {"type": "string"}},
                            "tool_selection_reasoning": {"type": "array", "items": {"type": "string"}},
                            "plan_notes": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["selected_tools", "tool_selection_reasoning", "plan_notes"],
                        "additionalProperties": False,
                    },
                }
            },
        }
        try:
            response = self._post(api_key, payload)
            output = self._extract_json_output(response)
            if not isinstance(output, dict):
                raise ValueError("planner response is not a JSON object")
            selected_tools = [item for item in output.get("selected_tools", []) if isinstance(item, str)]
            return PlannerSuggestion(
                planning_layer=PlanningLayer.LLM,
                selected_tools=selected_tools or list(rule_plan.selected_tools),
                tool_selection_reasoning=[
                    item for item in output.get("tool_selection_reasoning", []) if isinstance(item, str)
                ]
                or ["LLM planner kept the rule baseline ordering."],
                plan_notes=[item for item in output.get("plan_notes", []) if isinstance(item, str)],
            )
        except Exception as exc:
            return PlannerSuggestion(
                planning_layer=PlanningLayer.LLM,
                selected_tools=list(rule_plan.selected_tools),
                tool_selection_reasoning=[
                    f"LLM planner fallback to rule baseline because the planner call failed: {exc}."
                ],
                plan_notes=["LLM planner fallback after API error."],
                invalid_reason="planner_call_failed",
            )

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
        except error.HTTPError as exc:  # pragma: no cover - network-dependent
            detail = exc.read().decode("utf-8", errors="ignore")
            raise ValueError(detail or f"http_{exc.code}") from exc
        except error.URLError as exc:  # pragma: no cover - network-dependent
            raise ValueError(str(exc.reason)) from exc

    @staticmethod
    def _extract_json_output(response: dict[str, Any]) -> Any:
        output = response.get("output", [])
        for item in output:
            content = item.get("content", [])
            for chunk in content:
                if chunk.get("type") in {"output_text", "text"} and chunk.get("text"):
                    return json.loads(chunk["text"])
        if response.get("output_text"):
            return json.loads(response["output_text"])
        raise ValueError("missing planner output")

    @staticmethod
    def _build_prompt(
        context: PlannerRequestContext,
        rule_plan: CopilotExecutionPlan,
        tool_specs: list[dict[str, Any]],
    ) -> str:
        tool_names = [spec.get("tool_name", "") for spec in tool_specs]
        return (
            "You are a flood-response tool planner. You may only reorder or add tools from the whitelist.\n"
            f"Question: {context.question}\n"
            f"Rule planner intent: {rule_plan.intent}\n"
            f"Rule planner selected tools: {', '.join(rule_plan.selected_tools)}\n"
            f"Whitelist: {', '.join(tool_names)}\n"
            f"Recent failures: {', '.join(context.recent_failures) or 'none'}\n"
            "Return JSON with selected_tools, tool_selection_reasoning, and plan_notes."
        )

    def _load_api_key(self) -> str:
        root = Path(__file__).resolve().parents[2]
        key_file = root / self.api_key_file
        if key_file.exists():
            file_value = key_file.read_text(encoding="utf-8").strip()
            if file_value:
                return file_value

        env_value = os.getenv(self.api_key_env, "").strip()
        if env_value:
            return env_value
        return ""
