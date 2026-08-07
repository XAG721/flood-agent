"""Prospective zero-tuning MuSiQue transfer of the frozen v76 graph router."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.hotpot_graph_router import (
    CANDIDATE_METHOD,
    COMMON_RESOURCES,
    CONTROL_METHODS,
    FEATURE_NAMES,
    METHODS,
    SOFT_METHOD,
    load_router_artifact,
    write_selection_outputs,
)
from research.frc_rag.public_evidence import evidence_metrics
from research.frc_rag.twowiki_question_router import (
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_support_path_closure import (
    canonical_json_sha256,
    read_jsonl,
    sha256,
    write_jsonl,
    write_jsonl_gzip,
)


EXPERIMENT_ID = "FRC-MUSIQUE-FROZEN-HOTPOT-GRAPH-ROUTER-TRANSFER-V77"
SCHEMA_VERSION = "frc-musique-frozen-hotpot-graph-router-transfer-v77"
STAGE_SALTS = {
    "development": "FRC-MUSIQUE-V77-DEVELOPMENT|",
    "confirmation": "FRC-MUSIQUE-V77-CONFIRMATION|",
}
HOP_QUOTAS = {2: 400, 3: 250, 4: 150}
CASES_PER_STAGE = sum(HOP_QUOTAS.values())
PRIOR_COMMITMENT_UNION = 16900
METRIC_SEEDS = {"development_seed": 20261101, "confirmation_seed": 20261102}
DEVELOPMENT_GATES = {
    "exact_cases": 800,
    "exact_hop_quota": {"2": 400, "3": 250, "4": 150},
    "prior_or_stage_overlap": 0,
    "invalid_selector_output_rate": 0.0,
    "candidate_evidence_macro_f1_at_least": 0.55,
    "candidate_complete_evidence_recall_at_least": 0.55,
    "candidate_minus_strongest_registered_control_f1_at_least": 0.005,
    "candidate_minus_strongest_registered_control_ci_low_above": 0.0,
    "candidate_minus_cross_encoder_f1_at_least": 0.005,
    "candidate_minus_cross_encoder_ci_low_above": 0.0,
    "every_hop_delta_vs_best_control_at_least": -0.005,
    "soft_override_rate_between_inclusive": [0.15, 0.55],
    "mean_selected_tokens_not_above_strongest_control_by_more_than_fraction": 0.05,
}
NONINFERIORITY_ENVELOPE = {
    "candidate_minus_strongest_control_point_at_least": -0.005,
    "candidate_minus_strongest_control_ci_low_at_least": -0.01,
    "every_hop_delta_vs_best_control_at_least": -0.02,
    "invalid_selector_output_rate": 0.0,
}

_HOP_PATTERN = re.compile(r"^([234])hop")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def source_id_commitment(source_id: str) -> str:
    return hashlib.sha256(source_id.encode("utf-8")).hexdigest()


def hop_count_from_id(source_id: str) -> int:
    match = _HOP_PATTERN.match(source_id)
    if match is None:
        raise ValueError("v77 MuSiQue source id has no registered hop prefix")
    return int(match.group(1))


def load_source_commitments(paths: Iterable[Path]) -> set[str]:
    commitments: set[str] = set()
    for path in paths:
        rows = read_jsonl(path)
        local = {
            str(row.get("source_id_commitment", "")).strip() for row in rows
        }
        if "" in local or any(len(value) != 64 for value in local):
            raise ValueError(f"v77 prior map has an invalid commitment: {path}")
        commitments.update(local)
    return commitments


def _order_key(stage: str, source_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{STAGE_SALTS[stage]}{source_id}".encode("utf-8")
    ).hexdigest()
    return digest, source_id


def select_stage_ids(
    metadata_rows: Iterable[dict[str, Any]],
    *,
    excluded_source_commitments: set[str],
    stage: str,
    hop_quotas: dict[int, int] | None = None,
) -> list[str]:
    if stage not in STAGE_SALTS:
        raise ValueError(f"unsupported v77 stage: {stage}")
    quotas = HOP_QUOTAS if hop_quotas is None else hop_quotas
    if not quotas or any(hop not in {2, 3, 4} or quota <= 0 for hop, quota in quotas.items()):
        raise ValueError("v77 hop quotas are invalid")
    rows = [
        {
            "id": str(row.get("id", "")).strip(),
            "answerable": row.get("answerable"),
        }
        for row in metadata_rows
        if row.get("answerable") is True
    ]
    if any(not row["id"] for row in rows):
        raise ValueError("v77 MuSiQue metadata contains an empty id")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("v77 MuSiQue answerable metadata contains duplicate ids")

    def choose(current_stage: str, extra_excluded: set[str]) -> list[str]:
        selected: list[str] = []
        for hop, quota in sorted(quotas.items()):
            eligible = [
                source_id
                for source_id in ids
                if hop_count_from_id(source_id) == hop
                and source_id_commitment(source_id) not in excluded_source_commitments
                and source_id not in extra_excluded
            ]
            eligible.sort(key=lambda source_id: _order_key(current_stage, source_id))
            if len(eligible) < quota:
                raise ValueError(
                    f"v77 lacks {quota} eligible untouched {hop}-hop cases"
                )
            selected.extend(eligible[:quota])
        return selected

    development = choose("development", set())
    if stage == "development":
        return development
    return choose("confirmation", set(development))


def _token_count(text: str) -> int:
    return max(1, len(_TOKEN_PATTERN.findall(text)))


def prepare_case(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_id = str(row.get("id", "")).strip()
    hop_count = hop_count_from_id(source_id)
    if not source_id or row.get("answerable") is not True:
        raise ValueError("v77 requires an answerable MuSiQue case")
    question = str(row.get("question", "")).strip()
    paragraphs = row.get("paragraphs")
    if not question or not isinstance(paragraphs, list) or len(paragraphs) < 2:
        raise ValueError("v77 MuSiQue case has invalid question or paragraphs")
    candidates: list[dict[str, Any]] = []
    gold_ids: list[str] = []
    seen_indices: set[int] = set()
    for position, paragraph in enumerate(paragraphs):
        if (
            not isinstance(paragraph, dict)
            or not isinstance(paragraph.get("idx"), int)
            or not isinstance(paragraph.get("title"), str)
            or not isinstance(paragraph.get("paragraph_text"), str)
            or not isinstance(paragraph.get("is_supporting"), bool)
        ):
            raise ValueError("v77 MuSiQue paragraph schema is invalid")
        paragraph_index = int(paragraph["idx"])
        if paragraph_index in seen_indices:
            raise ValueError("v77 MuSiQue paragraph indices are duplicated")
        seen_indices.add(paragraph_index)
        title = str(paragraph["title"]).strip()
        paragraph_text = str(paragraph["paragraph_text"]).strip()
        text = f"{title}. {paragraph_text}" if title else paragraph_text
        if not text:
            raise ValueError("v77 MuSiQue paragraph text is empty")
        candidate_id = f"musique-p{paragraph_index:02d}"
        if paragraph["is_supporting"]:
            gold_ids.append(candidate_id)
        candidates.append(
            {
                "id": candidate_id,
                "text": text,
                "source": title,
                "metadata": {
                    "title": title,
                    "paragraph_index": paragraph_index,
                    "source_position": position,
                },
                "token_count": _token_count(text),
            }
        )
    if not gold_ids:
        raise ValueError("v77 MuSiQue case has no supporting paragraph")
    blind = {
        "dataset": "MuSiQue",
        "source": "official_full_v1.0_train_answerable_untouched_v77",
        "id": source_id,
        "question": question,
        "hop_count": hop_count,
        "not_answerable": False,
        "required_roles": ["procedure", "answer"],
        "candidates": candidates,
    }
    gold = {
        "id": source_id,
        "hop_count": hop_count,
        "gold_evidence_ids": sorted(gold_ids),
    }
    forbidden = {"answer", "question_decomposition", "is_supporting"}
    if forbidden & set(blind) or any(
        forbidden & set(candidate) for candidate in blind["candidates"]
    ):
        raise AssertionError("v77 blind preparation leaked a forbidden gold field")
    return blind, gold


def prepare_stage(
    source_path: Path,
    exclusion_paths: Sequence[Path],
    *,
    stage: str,
    blind_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    excluded = load_source_commitments(exclusion_paths)
    if len(excluded) != PRIOR_COMMITMENT_UNION:
        raise ValueError("v77 prior source commitment union changed")
    metadata: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                metadata.append(
                    {"id": row.get("id"), "answerable": row.get("answerable")}
                )
    selected_ids = select_stage_ids(
        metadata,
        excluded_source_commitments=excluded,
        stage=stage,
    )
    selected_set = set(selected_ids)
    rows_by_id: dict[str, dict[str, Any]] = {}
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row.get("id", ""))
            if row.get("answerable") is True and source_id in selected_set:
                rows_by_id[source_id] = row
    if set(rows_by_id) != selected_set:
        raise ValueError("v77 selected ids are missing from the registered source")
    blind_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for source_id in selected_ids:
        blind, gold = prepare_case(rows_by_id[source_id])
        blind_rows.append(blind)
        gold_rows.append(gold)
    write_jsonl(blind_path, blind_rows)
    write_jsonl(gold_path, gold_rows)
    counts = Counter(int(row["hop_count"]) for row in gold_rows)
    commitments = {source_id_commitment(source_id) for source_id in selected_ids}
    return {
        "stage": stage,
        "selected_cases": len(selected_ids),
        "hop_count_distribution": {str(key): counts[key] for key in sorted(counts)},
        "prior_excluded_source_commitments": len(excluded),
        "prior_source_overlap": len(commitments & excluded),
        "selected_ids_sha256": canonical_json_sha256(selected_ids),
        "selected_source_commitments_sha256": canonical_json_sha256(
            sorted(commitments)
        ),
        "blind_sha256": sha256(blind_path),
        "gold_sha256": sha256(gold_path),
        "candidate_count": sum(len(row["candidates"]) for row in blind_rows),
        "gold_fields_visible_to_model_or_router": False,
    }


def _paired_bootstrap(
    candidate: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    resamples: int = 10000,
) -> dict[str, Any]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        control, dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    batches: list[np.ndarray] = []
    for start in range(0, resamples, 250):
        batch = min(250, resamples - start)
        indices = generator.integers(0, len(difference), size=(batch, len(difference)))
        batches.append(difference[indices].mean(axis=1))
    estimates = np.concatenate(batches)
    return {
        "point": round(float(difference.mean()), 6),
        "ci_low": round(float(np.quantile(estimates, 0.025)), 6),
        "ci_high": round(float(np.quantile(estimates, 0.975)), 6),
        "resamples": resamples,
        "seed": seed,
    }


def evaluate_stage(
    scored_path: Path,
    gold_path: Path,
    selection_path: Path,
    router_artifact_path: Path,
    v75_model_path: Path,
    *,
    stage: str,
    seed: int,
    output_dir: Path,
    prior_overlap: int,
    stage_overlap: int = 0,
) -> dict[str, Any]:
    if stage not in STAGE_SALTS:
        raise ValueError(f"unsupported v77 stage: {stage}")
    router_artifact = load_router_artifact(router_artifact_path)
    v75_artifact, v75_model = load_v75_model_artifact(v75_model_path)
    if (
        router_artifact["v75_router_model_payload_sha256"]
        != v75_artifact["model_sha256"]
    ):
        raise ValueError("v77 frozen v75 feature model changed")
    scored = read_jsonl(scored_path)
    selections = write_selection_outputs(
        scored, router_artifact, v75_model, selection_path
    )
    gold = {str(row["id"]): row for row in read_jsonl(gold_path)}
    selected = {str(row["case_id"]): row for row in selections}
    if set(gold) != set(selected):
        raise ValueError("v77 gold and selections do not align")
    cases: list[dict[str, Any]] = []
    invalid = 0
    for case_id in sorted(gold):
        gold_row = gold[case_id]
        selection = selected[case_id]
        candidate_ids = set(selection["candidate_ids"])
        methods: dict[str, Any] = {}
        for method in METHODS:
            values = selection["methods"][method]
            ids = list(values["selected_ids"])
            row_invalid = (
                len(ids) != len(set(ids))
                or len(ids) > COMMON_RESOURCES["top_k"]
                or not set(ids) <= candidate_ids
                or int(values["selected_tokens"]) > COMMON_RESOURCES["token_budget"]
            )
            invalid += int(row_invalid)
            metrics = evidence_metrics(gold_row["gold_evidence_ids"], ids)
            methods[method] = {
                **metrics,
                "complete_evidence": float(
                    set(gold_row["gold_evidence_ids"]) <= set(ids)
                ),
                "selected_tokens": int(values["selected_tokens"]),
                "route": values["route"],
                "predicted_soft_gain": values["predicted_soft_gain"],
                "invalid": row_invalid,
            }
        cases.append(
            {
                "case_id": case_id,
                "hop_count": int(gold_row["hop_count"]),
                "gold_evidence_count": len(gold_row["gold_evidence_ids"]),
                "methods": methods,
            }
        )
    aggregates: dict[str, Any] = {}
    for method in METHODS:
        rows = [case["methods"][method] for case in cases]
        aggregates[method] = {
            "cases": len(rows),
            "evidence_macro_f1": round(
                sum(float(row["evidence_f1"]) for row in rows) / len(rows), 6
            ),
            "evidence_macro_recall": round(
                sum(float(row["evidence_recall"]) for row in rows) / len(rows), 6
            ),
            "complete_evidence_recall": round(
                sum(float(row["complete_evidence"]) for row in rows) / len(rows), 6
            ),
            "mean_selected_tokens": round(
                sum(float(row["selected_tokens"]) for row in rows) / len(rows), 6
            ),
        }
    strongest = max(
        CONTROL_METHODS,
        key=lambda method: (
            aggregates[method]["evidence_macro_f1"],
            -CONTROL_METHODS.index(method),
        ),
    )
    candidate_values = np.asarray(
        [case["methods"][CANDIDATE_METHOD]["evidence_f1"] for case in cases]
    )
    comparisons = {
        method: _paired_bootstrap(
            candidate_values,
            np.asarray([case["methods"][method]["evidence_f1"] for case in cases]),
            seed=seed + index,
        )
        for index, method in enumerate(CONTROL_METHODS)
    }
    per_hop: dict[str, Any] = {}
    for hop in sorted(HOP_QUOTAS):
        subset = [case for case in cases if case["hop_count"] == hop]
        if not subset:
            per_hop[str(hop)] = {"cases": 0, "delta": float("-inf")}
            continue
        best_control = max(
            CONTROL_METHODS,
            key=lambda method: (
                sum(float(case["methods"][method]["evidence_f1"]) for case in subset),
                -CONTROL_METHODS.index(method),
            ),
        )
        candidate_f1 = sum(
            float(case["methods"][CANDIDATE_METHOD]["evidence_f1"])
            for case in subset
        ) / len(subset)
        control_f1 = sum(
            float(case["methods"][best_control]["evidence_f1"])
            for case in subset
        ) / len(subset)
        per_hop[str(hop)] = {
            "cases": len(subset),
            "best_control": best_control,
            "candidate_evidence_macro_f1": round(candidate_f1, 6),
            "best_control_evidence_macro_f1": round(control_f1, 6),
            "delta": round(candidate_f1 - control_f1, 6),
        }
    candidate = aggregates[CANDIDATE_METHOD]
    strongest_delta = comparisons[strongest]
    cross_delta = comparisons["cross_encoder_topk"]
    override_rate = sum(
        case["methods"][CANDIDATE_METHOD]["route"] == SOFT_METHOD for case in cases
    ) / len(cases)
    token_ratio = float(candidate["mean_selected_tokens"]) / max(
        1e-12, float(aggregates[strongest]["mean_selected_tokens"])
    )
    counts = Counter(case["hop_count"] for case in cases)
    checks = {
        "exact_cases_equals_800": len(cases) == CASES_PER_STAGE,
        "exact_hop_quota": counts == Counter(HOP_QUOTAS),
        "prior_or_stage_overlap_equals_0": prior_overlap == 0 and stage_overlap == 0,
        "invalid_selector_output_rate_equals_0": invalid == 0,
        "candidate_evidence_macro_f1_at_least_0_55": float(
            candidate["evidence_macro_f1"]
        )
        >= 0.55,
        "candidate_complete_evidence_recall_at_least_0_55": float(
            candidate["complete_evidence_recall"]
        )
        >= 0.55,
        "candidate_minus_strongest_control_f1_at_least_0_005": float(
            strongest_delta["point"]
        )
        >= 0.005,
        "candidate_minus_strongest_control_ci_low_above_0": float(
            strongest_delta["ci_low"]
        )
        > 0.0,
        "candidate_minus_cross_encoder_f1_at_least_0_005": float(
            cross_delta["point"]
        )
        >= 0.005,
        "candidate_minus_cross_encoder_ci_low_above_0": float(cross_delta["ci_low"])
        > 0.0,
        "every_hop_delta_vs_best_control_at_least_minus_0_005": all(
            float(row["delta"]) >= -0.005 for row in per_hop.values()
        ),
        "soft_override_rate_between_0_15_and_0_55": 0.15 <= override_rate <= 0.55,
        "mean_selected_tokens_within_1_05_of_strongest_control": token_ratio <= 1.05,
    }
    passed = all(checks.values())
    noninferiority_checks = {
        "candidate_minus_strongest_control_point_at_least_minus_0_005": float(
            strongest_delta["point"]
        )
        >= -0.005,
        "candidate_minus_strongest_control_ci_low_at_least_minus_0_01": float(
            strongest_delta["ci_low"]
        )
        >= -0.01,
        "every_hop_delta_vs_best_control_at_least_minus_0_02": all(
            float(row["delta"]) >= -0.02 for row in per_hop.values()
        ),
        "invalid_selector_output_rate_equals_0": invalid == 0,
    }
    noninferiority_supported = all(noninferiority_checks.values())
    if stage == "development":
        status = (
            "MUSIQUE_V77_FROZEN_V76_ROUTER_TRANSFER_ADVANTAGE_ESTABLISHED_OPEN_CONFIRMATION"
            if passed
            else "MUSIQUE_V77_FROZEN_V76_ROUTER_TRANSFER_ADVANTAGE_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "MUSIQUE_V77_FROZEN_V76_ROUTER_CASE_DISJOINT_TRANSFER_ADVANTAGE_ESTABLISHED"
            if passed
            else "MUSIQUE_V77_FROZEN_V76_ROUTER_CONFIRMATION_ADVANTAGE_NOT_ESTABLISHED"
        )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "stage": stage,
            "cases": len(cases),
            "hop_count_distribution": {str(key): counts[key] for key in sorted(counts)},
            "prior_source_overlap": prior_overlap,
            "stage_overlap": stage_overlap,
            "invalid_selector_output_count": invalid,
            "router_artifact_sha256": sha256(router_artifact_path),
            "selection_output_sha256": sha256(selection_path),
            "selection_written_before_gold_join": True,
            "answer_decomposition_support_flags_or_gold_used_by_runtime_router": False,
            "official_musique_leaderboard_result": False,
            "v76_router_trained_only_on_hotpotqa_history": True,
            "target_cases_disjoint_from_all_prior_project_musique_exposure": True,
            "broader_method_development_independent_of_musique": False,
        },
        "analysis": {
            "methods": aggregates,
            "strongest_registered_control": {
                "name": strongest,
                **aggregates[strongest],
            },
            "paired_f1_delta": comparisons,
            "router": {
                "soft_override_rate": round(override_rate, 6),
                "v76_history_crossfit_override_rate": router_artifact["crossfit"][
                    "override_rate"
                ],
            },
            "per_hop_delta_vs_best_control": per_hop,
            "support_checks": checks,
            "noninferiority_checks": noninferiority_checks,
            "outcome": {
                "status": status,
                "stage_gate_passed": passed,
                "noninferiority_envelope_supported": noninferiority_supported,
                "confirmation_open_authorized": stage == "development" and passed,
                "reuse_target_stage_for_feature_model_threshold_gate_or_selection": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl_gzip(output_dir / "cases.jsonl.gz", cases)
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    analysis = report["analysis"]
    candidate = analysis["methods"][CANDIDATE_METHOD]
    strongest = analysis["strongest_registered_control"]
    delta = analysis["paired_f1_delta"][strongest["name"]]
    lines = [
        f"# MuSiQue 冻结 v76 路由器迁移实验（v77 {report['metadata']['stage']}）",
        "",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- 候选证据 F1：`{candidate['evidence_macro_f1']:.6f}`",
        f"- 最强登记对照：`{strongest['name']}` / `{strongest['evidence_macro_f1']:.6f}`",
        f"- 差值：`{delta['point']:+.6f}`，95% CI [`{delta['ci_low']:+.6f}`, `{delta['ci_high']:+.6f}`]",
        f"- 软闭包覆盖率：`{analysis['router']['soft_override_rate']:.6f}`",
        f"- 非劣包络：`{analysis['outcome']['noninferiority_envelope_supported']}`",
        f"- Gate 2：`{analysis['outcome']['gate_2']}`",
        "",
        "## 严格迁移门槛",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} `{name}`"
        for name, passed in analysis["support_checks"].items()
    )
    lines.extend(
        [
            "",
            "该实验是完整冻结 v76 路由器的零调参迁移，不是 MuSiQue 官方排行榜、答案生成、真实 SetR、洪水领域专家评测或生产放行证据；Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_registered_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v77 protocol experiment id")
    candidate = protocol.get("candidate", {})
    if (
        candidate.get("name") != CANDIDATE_METHOD
        or candidate.get("frozen_source_experiment")
        != "FRC-HOTPOT-GRAPH-SOFT-OVERRIDE-ROUTER-V76"
        or candidate.get("feature_count") != len(FEATURE_NAMES)
        or candidate.get("feature_names_sha256")
        != canonical_json_sha256(list(FEATURE_NAMES))
        or candidate.get("override_threshold") != 0.0
        or candidate.get("target_training_or_tuning_cases") != 0
    ):
        raise ValueError("v77 frozen candidate contract drifted")
    if tuple(protocol.get("controls", ())) != CONTROL_METHODS:
        raise ValueError("v77 protocol controls drifted")
    if protocol.get("common_resources") != COMMON_RESOURCES:
        raise ValueError("v77 protocol resources drifted")
    if protocol.get("development_gates") != DEVELOPMENT_GATES:
        raise ValueError("v77 protocol gates drifted")
    if protocol.get("noninferiority_envelope") != NONINFERIORITY_ENVELOPE:
        raise ValueError("v77 protocol noninferiority envelope drifted")
    metrics = protocol.get("metrics", {})
    if (
        metrics.get("primary") != "evidence_macro_f1"
        or metrics.get("paired_uncertainty") != "10000 case bootstrap resamples"
        or any(metrics.get(name) != value for name, value in METRIC_SEEDS.items())
    ):
        raise ValueError("v77 protocol metrics drifted")
    for stage in STAGE_SALTS:
        contract = protocol.get("stages", {}).get(stage, {})
        if contract.get("cases") != CASES_PER_STAGE or contract.get(
            "hop_quota"
        ) != {str(key): value for key, value in HOP_QUOTAS.items()}:
            raise ValueError("v77 protocol stage sampling drifted")
    if (
        protocol.get("stages", {})
        .get("confirmation", {})
        .get("open_only_if_every_development_gate_passes")
        is not True
    ):
        raise ValueError("v77 confirmation policy drifted")


__all__ = [
    "CASES_PER_STAGE",
    "DEVELOPMENT_GATES",
    "EXPERIMENT_ID",
    "HOP_QUOTAS",
    "METRIC_SEEDS",
    "NONINFERIORITY_ENVELOPE",
    "PRIOR_COMMITMENT_UNION",
    "evaluate_stage",
    "hop_count_from_id",
    "load_source_commitments",
    "prepare_case",
    "prepare_stage",
    "select_stage_ids",
    "source_id_commitment",
    "validate_registered_protocol",
]
