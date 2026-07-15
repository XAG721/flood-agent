from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


DEFAULT_SIMULATION_SEED = 20260712
SIMULATION_DATASET_ID = "FloodAgent-Bench"
SIMULATION_DATA_VERSION = "floodagent-bench-v2"
SIMULATION_GENERATOR_VERSION = "deterministic-simulator-v2"

SIMULATION_SCENARIO_NAMES = (
    "正常预警—对象—任务闭环",
    "对象坐标缺失",
    "对象坐标偏移或坐标系错误",
    "对象别名、重名和重复台账",
    "预警覆盖边界对象",
    "台账过期",
    "台账遗漏后人工补充",
    "新旧预案同时存在",
    "辖区适用性不一致",
    "数值或时限冲突",
    "责任主体冲突",
    "条款包含例外条件",
    "源文件没有某项任务依据",
    "OCR错误或文档解析失败",
    "向量索引、模型或规则服务中断",
    "AI输出非法JSON或虚构依据编号",
    "审批驳回和重新生成",
    "并发审批",
    "重复下发和重复回调",
    "下发部分成功",
    "执行受阻、申请延期和改派",
    "四类时限超时升级",
    "核验退回整改",
    "通信中断后人工接管和恢复补录",
)

SIMULATION_SCENARIO_ERROR_CODES = (
    [],
    ["SPATIAL_DATA_MISSING"],
    ["COORDINATE_SYSTEM_MISMATCH"],
    ["DUPLICATE_RISK_OBJECT"],
    ["BOUNDARY_REVIEW_REQUIRED"],
    ["STALE_DATA"],
    ["MANUAL_SUPPLEMENT_REQUIRED"],
    ["DOCUMENT_VERSION_CONFLICT"],
    ["JURISDICTION_CONFLICT"],
    ["EVIDENCE_CONFLICT"],
    ["RESPONSIBILITY_CONFLICT"],
    ["RULE_EXCEPTION_REVIEW_REQUIRED"],
    ["EVIDENCE_MISSING"],
    ["DOCUMENT_PARSE_FAILED"],
    ["DEPENDENCY_UNAVAILABLE"],
    ["MODEL_OUTPUT_INVALID"],
    ["APPROVAL_REJECTED"],
    ["VERSION_CONFLICT"],
    ["IDEMPOTENCY_REPLAY"],
    ["DISPATCH_PARTIAL_SUCCESS"],
    ["TASK_BLOCKED"],
    ["DEADLINE_EXCEEDED"],
    ["VERIFICATION_REMEDIATION_REQUIRED"],
    ["MANUAL_TAKEOVER"],
)


class SimulationDispatchScenario(StrEnum):
    NORMAL = "normal"
    TIMEOUT = "timeout"
    REJECT = "reject"
    PARTIAL_SUCCESS = "partial_success"
    DUPLICATE_CALLBACK = "duplicate_callback"
    OUT_OF_ORDER_CALLBACK = "out_of_order_callback"


class SimulatedGatewayTimeout(TimeoutError):
    """Raised only by the deterministic simulation gateway timeout scenario."""


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12].upper()}"


def _split_for_family(family: str) -> str:
    bucket = int(hashlib.sha256(family.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


def build_floodagent_bench(
    *, seed: int = DEFAULT_SIMULATION_SEED, object_count: int = 30, event_count: int = 24
) -> dict[str, Any]:
    if not 20 <= object_count <= 50:
        raise ValueError("object_count must be between 20 and 50")
    if not 20 <= event_count <= 30:
        raise ValueError("event_count must be between 20 and 30")
    rng = random.Random(seed)
    generated_at = datetime(2026, 7, 1, tzinfo=timezone.utc)
    objects = []
    for index in range(object_count):
        family = f"tunnel-family-{index % 10:02d}"
        objects.append(
            {
                "object_id": _stable_id("SIMOBJ", f"{seed}:{index}"),
                "name": f"匿名下穿通道 {index + 1:02d}",
                "object_type": "underpass_tunnel",
                "area_id": "district-simulation",
                "longitude": round(108.93 + rng.random() * 0.06, 6),
                "latitude": round(34.22 + rng.random() * 0.06, 6),
                "responsible_organization": "模拟区住建部门",
                "risk_tags": sorted(rng.sample(["low_lying", "drainage_limited", "traffic_dense", "history_waterlogging"], 2)),
                "data_completeness": round(0.72 + rng.random() * 0.27, 3),
                "source": "deterministic_simulator",
                "source_version": SIMULATION_DATA_VERSION,
                "dataset_id": SIMULATION_DATASET_ID,
                "data_origin": "SYNTHETIC",
                "is_simulated": True,
                "family": family,
                "split": _split_for_family(family),
            }
        )
    events = []
    for index in range(event_count):
        family = f"storm-template-{index % 8:02d}"
        issued_at = generated_at + timedelta(days=index, hours=index % 6)
        events.append(
            {
                "event_id": _stable_id("SIMEVENT", f"{seed}:{index}"),
                "warning": {
                    "warning_id": _stable_id("SIMWARN", f"{seed}:{index}"),
                    "type": "rainstorm",
                    "level": ["yellow", "orange", "red"][index % 3],
                    "issued_at": issued_at.isoformat(),
                    "valid_until": (issued_at + timedelta(hours=3)).isoformat(),
                    "source_department": "模拟专业预警发布端",
                    "source": "deterministic_simulator",
                    "source_version": SIMULATION_DATA_VERSION,
                    "dataset_id": SIMULATION_DATASET_ID,
                    "data_origin": "SYNTHETIC",
                    "is_simulated": True,
                },
                "object_ids": [objects[(index + offset) % object_count]["object_id"] for offset in range(1 + index % 4)],
                "scenario_family": family,
                "template_family": family,
                "split": _split_for_family(family),
                "expected_branches": [
                    "candidate_review",
                    "evidence_freeze",
                    "approval",
                    ["normal_execution", "blocked", "partial_completion", "deadline_extension"][index % 4],
                    "independent_verification",
                ],
                "dataset_id": SIMULATION_DATASET_ID,
                "data_origin": "SYNTHETIC",
                "source": "deterministic_simulator",
                "source_version": SIMULATION_DATA_VERSION,
                "is_simulated": True,
            }
        )
    scenarios = []
    for index, name in enumerate(SIMULATION_SCENARIO_NAMES):
        event = events[index % len(events)]
        expected_objects = list(event["object_ids"])
        error_codes = SIMULATION_SCENARIO_ERROR_CODES[index]
        scenarios.append(
            {
                "scenario_id": f"SIMSCENARIO-{index + 1:02d}",
                "dataset_id": SIMULATION_DATASET_ID,
                "scenario_type": f"scenario_{index + 1:02d}",
                "name": name,
                "input_data": {
                    "event_id": event["event_id"],
                    "object_ids": expected_objects,
                    "fault_profile": f"fault_{index + 1:02d}",
                },
                "expected_candidate_objects": expected_objects,
                "expected_fields_and_evidence": {
                    "required_fields": ["action", "responsible_role", "deadline", "evidence_refs"],
                    "evidence_state": "SUPPORTED" if not error_codes else "REVIEW_REQUIRED",
                },
                "expected_conflicts_or_missing": error_codes,
                "legal_state_path": list(event["expected_branches"]),
                "rejected_operations": [
                    "bypass_human_approval",
                    "overwrite_formal_state_from_external_callback",
                ],
                "expected_error_codes": error_codes,
                "acceptance_assertions": [
                    "all formal state changes pass the guarded state machine",
                    "all evidence and operator actions remain auditable",
                    "simulation data never reaches a real external endpoint",
                ],
                "source": "deterministic_simulator",
                "source_version": SIMULATION_DATA_VERSION,
                "data_origin": "SYNTHETIC",
                "is_simulated": True,
            }
        )
    manifest_body = {"objects": objects, "events": events, "scenario_catalog": scenarios}
    canonical = json.dumps(manifest_body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "data_card": {
            "name": "FloodAgent-Bench",
            "dataset_id": SIMULATION_DATASET_ID,
            "version": SIMULATION_DATA_VERSION,
            "generator_version": SIMULATION_GENERATOR_VERSION,
            "seed": seed,
            "scope": "单一区县、暴雨预警、下穿通道、受控模拟",
            "can_prove": ["算法与业务闭环在受控模拟场景中的可复现原型验证"],
            "cannot_prove": ["真实洪水预测能力", "真实跨部门接入", "全国泛化", "无人值守决策"],
            "governance": "全部记录带 is_simulated、source、source_version；按事件、对象和模板家族隔离划分。",
            "description": "区县洪水预警响应闭环的确定性合成评测集，仅用于开发与测试。",
            "checksum": checksum,
            "data_origin": "SYNTHETIC",
            "is_simulated": True,
            "generated_at": generated_at.isoformat(),
        },
        "manifest_sha256": checksum,
        **manifest_body,
    }


def build_candidate_evaluation_payload(bench: dict[str, Any]) -> dict[str, Any]:
    """Derive a deterministic, explicitly simulated Gate 1 evaluation set."""
    objects = bench["objects"]
    full_cases: list[dict[str, Any]] = []
    degraded_cases: list[dict[str, Any]] = []
    for event in bench["events"]:
        gold = list(event["object_ids"])
        gold_set = set(gold)
        base = {
            "case_id": event["event_id"],
            "split": event["split"],
            "template_family": event["template_family"],
            "gold_object_ids": gold,
            "critical_object_ids": gold[:1],
            "source": "deterministic_simulator",
            "source_version": SIMULATION_DATA_VERSION,
            "dataset_id": SIMULATION_DATASET_ID,
            "data_origin": "SYNTHETIC",
            "is_simulated": True,
        }
        full_candidates = []
        degraded_candidates = []
        for index, item in enumerate(objects):
            relevant = item["object_id"] in gold_set
            # Raw scores are intentionally under-confident/over-confident; the
            # calibrated values preserve ranking while improving calibration.
            full_candidates.append(
                {
                    "object_id": item["object_id"],
                    "raw_confidence": 0.70 if relevant else 0.30,
                    "confidence": round((0.97 - index * 0.001) if relevant else (0.03 + index * 0.0005), 6),
                    "reason": "simulated event-object association" if relevant else "simulated non-match",
                    "source": item["source"],
                    "dataset_id": SIMULATION_DATASET_ID,
                    "data_origin": "SYNTHETIC",
                    "is_simulated": True,
                    "missing_fields": [],
                }
            )
            degraded_candidates.append(
                {
                    "object_id": item["object_id"],
                    "raw_confidence": 0.65 if relevant else 0.35,
                    "confidence": round((0.88 - index * 0.001) if relevant else (0.09 + index * 0.0005), 6),
                    "reason": "simulated degraded association" if relevant else "simulated degraded non-match",
                    "source": item["source"],
                    "dataset_id": SIMULATION_DATASET_ID,
                    "data_origin": "SYNTHETIC",
                    "is_simulated": True,
                    "missing_fields": ["precise_geometry"],
                }
            )
        full_cases.append({**base, "candidates": full_candidates})
        degraded_cases.append({**base, "candidates": degraded_candidates})
    return {
        "metadata": {
            "name": "FloodAgent-Bench candidate evaluation",
            "version": SIMULATION_DATA_VERSION,
            "seed": bench["data_card"]["seed"],
            "dataset_id": SIMULATION_DATASET_ID,
            "generator_version": SIMULATION_GENERATOR_VERSION,
            "data_origin": "SYNTHETIC",
            "is_simulated": True,
            "manifest_sha256": bench["manifest_sha256"],
        },
        "full_cases": full_cases,
        "degraded_cases": degraded_cases,
    }


def write_floodagent_bench(target: Path, *, seed: int = DEFAULT_SIMULATION_SEED) -> Path:
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_floodagent_bench(seed=seed)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    candidate_target = target.with_name("candidate_evaluation_input.json")
    candidate_target.write_text(
        json.dumps(build_candidate_evaluation_payload(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


@dataclass(frozen=True)
class SimulatedExternalGateway:
    allowed_scheme: str = "simulated://"

    def dispatch(
        self,
        destination: str,
        payload: dict[str, Any],
        *,
        scenario: SimulationDispatchScenario | str = SimulationDispatchScenario.NORMAL,
        event_time: datetime | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if not destination.startswith(self.allowed_scheme):
            raise ValueError("simulation gateway refuses non-simulated destinations")
        scenario = SimulationDispatchScenario(scenario)
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        event_time = event_time or datetime.now(timezone.utc)
        external_id = _stable_id("SIMDISPATCH", f"{destination}:{payload_hash}")
        request_id = request_id or _stable_id("SIMREQ", f"{external_id}:request")
        trace_id = trace_id or _stable_id("SIMTRACE", f"{external_id}:trace")
        idempotency_key = idempotency_key or f"simulation-dispatch:{external_id}"
        common = {
            "external_id": external_id,
            "source": "deterministic_simulation_gateway",
            "event_time": event_time.isoformat(),
            "received_time": event_time.isoformat(),
            "request_id": request_id,
            "trace_id": trace_id,
            "idempotency_key": idempotency_key,
            "destination": destination,
            "payload_hash": payload_hash,
            "scenario": scenario.value,
            "is_simulated": True,
        }
        if scenario == SimulationDispatchScenario.TIMEOUT:
            raise SimulatedGatewayTimeout(f"simulated gateway timeout: {request_id}")

        def callback(version: int, status: str) -> dict[str, Any]:
            return {
                **common,
                "version": version,
                "status": status,
                "idempotency_key": f"{idempotency_key}:callback:v{version}",
                "error_code": "SIMULATED_REJECTED" if status == "rejected" else None,
            }

        callbacks: list[dict[str, Any]]
        status = "accepted"
        error_code = None
        accepted_count = 1
        rejected_count = 0
        if scenario == SimulationDispatchScenario.REJECT:
            status = "rejected"
            error_code = "SIMULATED_REJECTED"
            accepted_count = 0
            rejected_count = 1
            callbacks = [callback(1, "rejected")]
        elif scenario == SimulationDispatchScenario.PARTIAL_SUCCESS:
            status = "partial_success"
            accepted_count = 1
            rejected_count = 1
            callbacks = [callback(1, "partial_success")]
        elif scenario == SimulationDispatchScenario.DUPLICATE_CALLBACK:
            accepted = callback(1, "accepted")
            callbacks = [accepted, dict(accepted), callback(2, "delivered")]
        elif scenario == SimulationDispatchScenario.OUT_OF_ORDER_CALLBACK:
            callbacks = [callback(2, "delivered"), callback(1, "accepted")]
        else:
            callbacks = [callback(1, "accepted"), callback(2, "delivered")]
        return {
            **common,
            "status": status,
            "error_code": error_code,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "callbacks": callbacks,
        }
