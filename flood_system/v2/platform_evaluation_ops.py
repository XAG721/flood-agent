from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import (
    BenchmarkScenarioResult,
    CompletionStatus,
    EventCreateRequest,
    EvaluationBenchmark,
    EvaluationReport,
    ObservationBatchRequest,
    ObservationIngestItem,
    V2CopilotMessageRequest,
    V2CopilotSessionRequest,
)


class PlatformEvaluationOpsMixin:
    def get_evaluation_report(self, report_id: str) -> EvaluationReport:
        report = self.repository.get_v2_evaluation_report(report_id)
        if report is None:
            raise ValueError(f"Unknown evaluation report: {report_id}")
        return report

    def list_evaluation_benchmarks(self) -> list[EvaluationBenchmark]:
        return [
            EvaluationBenchmark(
                benchmark_id="elderly-lowland",
                title="低洼老人家庭影响判断",
                question="当前水位对低洼老人家庭意味着什么？",
                scenario_type="elderly",
                expected_tools=["resolve_target_entity", "get_hazard_tiles", "synthesize_entity_impact", "get_knowledge_evidence"],
                expected_completion_status=CompletionStatus.CONSERVATIVE_ANSWER,
                expected_human_confirmation=False,
            ),
            EvaluationBenchmark(
                benchmark_id="school-escalation",
                title="小学风险升级判断",
                question="当前洪水对附近小学意味着什么？",
                scenario_type="school",
                expected_tools=["resolve_target_entity", "get_hazard_tiles", "synthesize_entity_impact", "get_policy_constraints"],
                expected_completion_status=CompletionStatus.HUMAN_ESCALATION,
                expected_human_confirmation=True,
            ),
            EvaluationBenchmark(
                benchmark_id="factory-proposal",
                title="工厂库存与停工建议",
                question="当前洪水对附近工厂的库存和停工安排意味着什么？",
                scenario_type="factory",
                expected_tools=["resolve_target_entity", "synthesize_entity_impact", "draft_action_proposal"],
                expected_completion_status=CompletionStatus.HUMAN_ESCALATION,
                expected_human_confirmation=True,
            ),
            EvaluationBenchmark(
                benchmark_id="route-guidance",
                title="养老机构转移路线建议",
                question="养老机构应通过哪条路线前往最安全的避难点？",
                scenario_type="route",
                expected_tools=["resolve_target_entity", "get_route_options", "get_live_traffic", "get_shelter_capacity"],
                expected_completion_status=CompletionStatus.HUMAN_ESCALATION,
                expected_human_confirmation=True,
            ),
        ]

    def run_evaluation(self) -> EvaluationReport:
        report = self._run_evaluation_for_benchmarks(self.list_evaluation_benchmarks())
        self.repository.save_v2_evaluation_report(report)
        return report

    def replay_evaluation_report(self, report_id: str) -> EvaluationReport:
        source_report = self.get_evaluation_report(report_id)
        benchmark_ids = [item.benchmark_id for item in source_report.scenario_results]
        benchmarks = [item for item in self.list_evaluation_benchmarks() if item.benchmark_id in benchmark_ids]
        if not benchmarks:
            raise ValueError(f"Evaluation report {report_id} does not contain replayable benchmark scenarios.")
        report = self._run_evaluation_for_benchmarks(benchmarks)
        report.notes = [f"已重放评测报告 {report_id}。", *report.notes]
        self.repository.save_v2_evaluation_report(report)
        return report

    def _run_evaluation_for_benchmarks(
        self,
        benchmarks: list[EvaluationBenchmark],
    ) -> EvaluationReport:
        scenario_results: list[BenchmarkScenarioResult] = []
        tool_scores: list[float] = []
        dispatch_scores: list[float] = []
        reuse_scores: list[float] = []
        evidence_scores: list[float] = []
        escalation_scores: list[float] = []
        hallucination_scores: list[float] = []

        for benchmark in benchmarks:
            event_id = self._create_evaluation_event(benchmark)
            session = self.bootstrap_copilot_session(
                V2CopilotSessionRequest(event_id=event_id, operator_role="commander")
            )
            first_view = self.send_copilot_message(
                session.session_id,
                V2CopilotMessageRequest(content=benchmark.question),
            )
            follow_up = self.send_copilot_message(
                session.session_id,
                V2CopilotMessageRequest(content="Continue with that target and restate the highest-priority action."),
            )
            answer = first_view.latest_answer
            if answer is None:
                raise ValueError("evaluation benchmark did not produce a structured answer.")

            used_tools = [item.tool_name for item in answer.tool_executions]
            missing_expected = [tool for tool in benchmark.expected_tools if tool not in used_tools]
            tool_score = 1.0 if not benchmark.expected_tools else round(
                (len(benchmark.expected_tools) - len(missing_expected)) / len(benchmark.expected_tools),
                2,
            )
            dispatch_ok = bool(answer.plan_runs and answer.plan_runs[0].selected_tools)
            memory_reused = bool(follow_up.latest_answer and follow_up.latest_answer.carried_context_notes)
            evidence_ok = bool(answer.evidence)
            escalation_ok = answer.requires_human_confirmation == benchmark.expected_human_confirmation
            hallucination = 1.0 if (not answer.evidence and answer.confidence >= 0.6) else 0.0
            status_ok = benchmark.expected_completion_status is None or answer.completion_status == benchmark.expected_completion_status
            passed = tool_score >= 0.75 and evidence_ok and escalation_ok and status_ok

            notes: list[str] = []
            if missing_expected:
                notes.append(f"Missing expected tools: {', '.join(missing_expected)}.")
            if not evidence_ok:
                notes.append("当前回答没有附带证据条目。")
            if not status_ok and benchmark.expected_completion_status is not None:
                notes.append(
                    f"Expected completion status {benchmark.expected_completion_status.value}, got {answer.completion_status.value}."
                )
            if not escalation_ok:
                notes.append("Human escalation behavior did not match the benchmark expectation.")
            if not memory_reused:
                notes.append("Follow-up turn did not reuse carried context notes.")

            scenario_results.append(
                BenchmarkScenarioResult(
                    benchmark_id=benchmark.benchmark_id,
                    title=benchmark.title,
                    passed=passed,
                    event_id=event_id,
                    session_id=session.session_id,
                    used_tools=used_tools,
                    expected_tools=benchmark.expected_tools,
                    completion_status=answer.completion_status,
                    expected_completion_status=benchmark.expected_completion_status,
                    human_confirmation=answer.requires_human_confirmation,
                    expected_human_confirmation=benchmark.expected_human_confirmation,
                    evidence_count=len(answer.evidence),
                    shared_memory_reused=memory_reused,
                    notes=notes,
                )
            )
            tool_scores.append(tool_score)
            dispatch_scores.append(1.0 if dispatch_ok else 0.0)
            reuse_scores.append(1.0 if memory_reused else 0.0)
            evidence_scores.append(1.0 if evidence_ok else 0.0)
            escalation_scores.append(1.0 if escalation_ok else 0.0)
            hallucination_scores.append(hallucination)

        return EvaluationReport(
            report_id=f"eval_{uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc),
            benchmark_count=len(benchmarks),
            tool_selection_correctness=round(sum(tool_scores) / len(tool_scores), 2) if tool_scores else 0.0,
            dynamic_dispatch_correctness=round(sum(dispatch_scores) / len(dispatch_scores), 2) if dispatch_scores else 0.0,
            shared_memory_reuse_rate=round(sum(reuse_scores) / len(reuse_scores), 2) if reuse_scores else 0.0,
            evidence_coverage_rate=round(sum(evidence_scores) / len(evidence_scores), 2) if evidence_scores else 0.0,
            human_escalation_correctness=round(sum(escalation_scores) / len(escalation_scores), 2) if escalation_scores else 0.0,
            hallucination_rate=round(sum(hallucination_scores) / len(hallucination_scores), 2) if hallucination_scores else 0.0,
            scenario_results=scenario_results,
            notes=[
                "This report was generated from isolated benchmark runs rather than recent live-message sampling.",
                "A second follow-up turn is executed for every benchmark to verify session-memory reuse.",
            ],
        )

    def _create_evaluation_event(self, benchmark: EvaluationBenchmark) -> str:
        event = self.create_event(
            EventCreateRequest(
                area_id="beilin_10km2",
                title=f"Evaluation benchmark: {benchmark.title}",
                trigger_reason=f"evaluation_{benchmark.benchmark_id}",
                operator="evaluation_runner",
            )
        )
        observations = self._load_evaluation_observations(benchmark.scenario_type)
        self.ingest_observations(
            event.event_id,
            ObservationBatchRequest(operator="evaluation_runner", observations=observations),
        )
        self.agent_supervisor.tick(event.event_id)
        return event.event_id

    def _load_evaluation_observations(self, scenario_type: str) -> list[ObservationIngestItem]:
        filename = {
            "elderly": "observations_beilin_warning.csv",
            "school": "observations_beilin_extreme.csv",
            "factory": "observations_beilin_extreme.csv",
            "route": "observations_beilin_warning.csv",
        }.get(scenario_type, "observations_beilin_warning.csv")
        csv_path = Path(__file__).resolve().parent.parent / "bootstrap_data" / filename
        observations: list[ObservationIngestItem] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                observations.append(
                    ObservationIngestItem(
                        observed_at=datetime.fromisoformat(str(row["observed_at"]).replace("Z", "+00:00")),
                        source_type=str(row.get("source_type", "monitoring_point")),
                        source_name=str(row.get("source_name", "")),
                        village=str(row.get("village") or "") or None,
                        rainfall_mm=float(row.get("rainfall_mm") or 0.0),
                        water_level_m=float(row.get("water_level_m") or 0.0),
                        road_blocked=str(row.get("road_blocked", "")).strip().lower() in {"1", "true", "yes"},
                        citizen_reports=int(float(row.get("citizen_reports") or 0)),
                        notes=str(row.get("notes", "")),
                    )
                )
        if not observations:
            raise ValueError(f"Evaluation observations are missing from {csv_path.name}.")
        return observations
