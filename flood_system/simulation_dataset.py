from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_SIMULATION_SEED = 20260712
SIMULATION_DATA_VERSION = "floodagent-bench-v1"


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
                "is_simulated": True,
            }
        )
    manifest_body = {"objects": objects, "events": events}
    canonical = json.dumps(manifest_body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "data_card": {
            "name": "FloodAgent-Bench",
            "version": SIMULATION_DATA_VERSION,
            "seed": seed,
            "scope": "单一区县、暴雨预警、下穿通道、受控模拟",
            "can_prove": ["算法与业务闭环在受控模拟场景中的可复现原型验证"],
            "cannot_prove": ["真实洪水预测能力", "真实跨部门接入", "全国泛化", "无人值守决策"],
            "governance": "全部记录带 is_simulated、source、source_version；按事件、对象和模板家族隔离划分。",
            "generated_at": generated_at.isoformat(),
        },
        "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
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

    def dispatch(self, destination: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not destination.startswith(self.allowed_scheme):
            raise ValueError("simulation gateway refuses non-simulated destinations")
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {
            "status": "accepted",
            "destination": destination,
            "payload_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "is_simulated": True,
        }
