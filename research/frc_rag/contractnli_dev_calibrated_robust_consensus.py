"""Development-only calibration for the ContractNLI v51 robust abstention guard."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import research.frc_rag.contractnli_native_zero_consensus_abstention as v50
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    LOW_CORE_DIVERGENCE_V49,
    _round_for_display,
    select_v49,
)
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import BUDGETS


SCHEMA_VERSION = "frc-contractnli-dev-calibrated-robust-consensus-v51"
EXPERIMENT_ID = "FRC-CONTRACTNLI-DEV-CALIBRATED-ROBUST-CONSENSUS-V51"
DATASET_ID = "contractnli_official_dev_robust_consensus_calibration_v51"
CAPABILITY = "development_calibration_for_contract_missing_evidence_abstention"
PROTOCOL_SHA256 = "0904a59ace997919a06fe32a4e839bf1fce5158cb7f4c663567506660421a18b"

DEV_SAMPLE_SALT = "FRC-CONTRACTNLI-V51-DEV|"
FOLD_SALT = "FRC-CONTRACTNLI-V51-FOLD|"
TARGET_CASES = 240
TARGET_PER_LABEL = 80
MAX_CASES_PER_DOCUMENT = 6
MINIMUM_DOCUMENTS = 30
FOLD_COUNT = 5
ROBUST_EPSILON = 1e-6
MAX_EVIDENCE_F1_DROP = 0.03
MAX_FIT_ABSTENTION_RATE = 0.8

DEVELOPMENT_CANDIDATE = "dev_calibrated_robust_consensus_v49_frc_v51"
DEVELOPMENT_BASELINE = LOW_CORE_DIVERGENCE_V49


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def read_split_source(source_archive: Path, split: str) -> dict[str, Any]:
    """Open exactly one registered split member and no other JSON content."""

    if split not in {"dev", "train"}:
        raise ValueError("ContractNLI v51 permits only dev or train")
    expected = f"{split}.json"
    with zipfile.ZipFile(source_archive) as archive:
        members = [
            name
            for name in archive.namelist()
            if Path(name).name.lower() == expected
        ]
        if len(members) != 1:
            raise ValueError(f"ContractNLI archive must contain exactly one {expected}")
        raw = archive.read(members[0])
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("ContractNLI split JSON must be an object")
    if not isinstance(value.get("documents"), list) or not isinstance(
        value.get("labels"), dict
    ):
        raise ValueError("ContractNLI split lacks documents or labels")
    return value


def _sample_key(row: dict[str, Any], salt: str) -> tuple[str, str, str]:
    digest = hashlib.sha256(
        (
            salt
            + str(row["label"])
            + "|"
            + str(row["document_id"])
            + "|"
            + str(row["hypothesis_key"])
        ).encode()
    ).hexdigest()
    return digest, str(row["document_id"]), str(row["hypothesis_key"])


def select_balanced_sample(
    source: dict[str, Any],
    *,
    target_per_label: int = TARGET_PER_LABEL,
    maximum_cases_per_document: int = MAX_CASES_PER_DOCUMENT,
    salt: str = DEV_SAMPLE_SALT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(v50._iter_cases(source))
    by_label = {
        label: sorted(
            [row for row in rows if row["label"] == label],
            key=lambda row: _sample_key(row, salt),
        )
        for label in v50.LABELS
    }
    if any(len(by_label[label]) < target_per_label for label in v50.LABELS):
        raise ValueError("ContractNLI v51 dev lacks the frozen per-label quota")

    selected: list[dict[str, Any]] = []
    document_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    cursors = {label: 0 for label in v50.LABELS}
    while any(label_counts[label] < target_per_label for label in v50.LABELS):
        progress = False
        for label in v50.LABELS:
            if label_counts[label] >= target_per_label:
                continue
            values = by_label[label]
            while cursors[label] < len(values):
                row = values[cursors[label]]
                cursors[label] += 1
                document_id = str(row["document_id"])
                if document_counts[document_id] >= maximum_cases_per_document:
                    continue
                selected.append(row)
                document_counts[document_id] += 1
                label_counts[label] += 1
                progress = True
                break
        if not progress:
            break

    expected_cases = target_per_label * len(v50.LABELS)
    if len(selected) != expected_cases:
        raise ValueError("ContractNLI v51 could not satisfy its balanced dev sample")
    selected = sorted(selected, key=lambda row: _sample_key(row, salt))
    return selected, {
        "eligible_cases": len(rows),
        "eligible_label_counts": {
            label: len(by_label[label]) for label in v50.LABELS
        },
        "selected_cases": len(selected),
        "selected_label_counts": {
            label: label_counts[label] for label in v50.LABELS
        },
        "selected_documents": len(document_counts),
        "maximum_cases_per_document": max(document_counts.values(), default=0),
        "target_per_label": target_per_label,
        "document_cap": maximum_cases_per_document,
        "sample_salt": salt,
    }


def prepare_blind_cases(
    selected: Sequence[dict[str, Any]], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prepared, maps, structural = v50.prepare_blind_cases(selected, tokenizer)
    for row in prepared:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
        row["capability"] = CAPABILITY
    return prepared, maps, structural


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = v50.build_deterministic_queries(prepared_rows)
    for row in rows:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
    return rows


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    summary = v50.validate_query_cache(prepared_rows, query_rows)
    if any(row.get("schema_version") != SCHEMA_VERSION for row in query_rows):
        raise ValueError("ContractNLI v51 query schema changed")
    return summary


class FrozenDevelopmentScorer(v50.FrozenContractNliScorer):
    """The frozen v50 scorer with v51 development-only cache identity."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def robust_consensus_details(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("ContractNLI v51 requires at least one candidate")
    normalised: dict[str, np.ndarray] = {}
    quantiles: dict[str, dict[str, float]] = {}
    for role in ("anchor", "first_fact", "second_fact_or_bridge"):
        values = np.asarray(
            [float(item["raw_dynamic_role_scores"][role]) for item in candidates],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("ContractNLI v51 raw role logits must be finite")
        q25, median, q75 = np.quantile(
            values, [0.25, 0.5, 0.75], method="linear"
        )
        scale = max(float(q75 - q25), ROBUST_EPSILON)
        normalised[role] = (values - float(median)) / scale
        quantiles[role] = {
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "scale": scale,
        }
    polarity = np.maximum(
        normalised["first_fact"], normalised["second_fact_or_bridge"]
    )
    consensus = np.minimum(normalised["anchor"], polarity)
    winner = sorted(
        (
            (-float(consensus[index]), str(candidate["id"]), index)
            for index, candidate in enumerate(candidates)
        )
    )[0]
    winner_index = winner[2]
    return {
        "case_score": float(consensus[winner_index]),
        "case_score_hex": float(consensus[winner_index]).hex(),
        "winning_candidate_id": str(candidates[winner_index]["id"]),
        "winning_anchor_z": float(normalised["anchor"][winner_index]),
        "winning_polarity_z": float(polarity[winner_index]),
        "quantiles": quantiles,
    }


def fold_for_document(document_id: str) -> int:
    digest = hashlib.sha256((FOLD_SALT + document_id).encode()).hexdigest()
    return int(digest[:16], 16) % FOLD_COUNT


def build_development_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    *,
    pool_quartile_boundaries: Sequence[float],
    document_length_quartile_boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    rows = v50.build_gold_rows(
        source,
        candidate_maps,
        pool_quartile_boundaries=pool_quartile_boundaries,
        document_length_quartile_boundaries=document_length_quartile_boundaries,
    )
    document_by_commitment: dict[str, str] = {}
    for source_row in v50._iter_cases(source):
        document_id = str(source_row["document_id"])
        commitment = hashlib.sha256(document_id.encode()).hexdigest()
        previous = document_by_commitment.setdefault(commitment, document_id)
        if previous != document_id:
            raise ValueError("ContractNLI document commitment collision")
    for row in rows:
        document_id = document_by_commitment.get(str(row["document_cluster"]))
        if document_id is None:
            raise ValueError("ContractNLI development document commitment is absent")
        row["development_fold"] = fold_for_document(document_id)
    return rows


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    evidence_rows = [
        row for row in gold_rows if row["label_group"] in v50.EVIDENCE_LABELS
    ]
    missing_rows = [
        row for row in gold_rows if row["label_group"] == "NotMentioned"
    ]
    ceiling = (
        float(
            np.mean(
                [row["candidate_ceiling_complete"] for row in evidence_rows]
            )
        )
        if evidence_rows
        else 0.0
    )
    empty_rate = (
        float(np.mean([not row["gold_candidate_ids"] for row in missing_rows]))
        if missing_rows
        else 0.0
    )
    label_counts = Counter(str(row["label_group"]) for row in gold_rows)
    documents = len({row["document_cluster"] for row in gold_rows})
    fold_counts = dict(
        sorted(Counter(str(row["development_fold"]) for row in gold_rows).items())
    )
    checks = {
        "exact_target_cases_met": len(gold_rows) == TARGET_CASES,
        "exact_target_per_label_met": all(
            label_counts[label] == TARGET_PER_LABEL for label in v50.LABELS
        ),
        "candidate_ceiling_complete_rate_on_evidence_cases_equals_1": ceiling
        == 1.0,
        "not_mentioned_official_empty_evidence_rate_equals_1": empty_rate == 1.0,
        "minimum_documents_met": documents >= MINIMUM_DOCUMENTS,
        "all_five_folds_nonempty": set(fold_counts)
        == {str(index) for index in range(FOLD_COUNT)},
        "sampling_selected_cases_equals_target": sampling.get("selected_cases")
        == TARGET_CASES,
        "sampling_maximum_cases_per_document_within_cap": int(
            sampling.get("maximum_cases_per_document", MAX_CASES_PER_DOCUMENT + 1)
        )
        <= MAX_CASES_PER_DOCUMENT,
    }
    return {
        "schema_version": "frc-contractnli-v51-dev-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "cases": len(gold_rows),
        "evidence_cases": len(evidence_rows),
        "not_mentioned_cases": len(missing_rows),
        "candidate_ceiling_complete_rate_on_evidence_cases": round(ceiling, 6),
        "not_mentioned_official_empty_evidence_rate": round(empty_rate, 6),
        "label_groups": dict(label_counts),
        "candidate_pool_quartiles": dict(
            Counter(row["candidate_pool_quartile"] for row in gold_rows)
        ),
        "document_length_quartiles": dict(
            Counter(row["document_length_quartile"] for row in gold_rows)
        ),
        "document_types": dict(
            Counter(row["document_type"] for row in gold_rows)
        ),
        "development_documents": documents,
        "development_fold_counts": fold_counts,
        "sampling": sampling,
        "checks": checks,
        "minimum_cases_and_ceiling_checks_passed": all(checks.values()),
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }


def _empty_metrics(label: str, gold_ids: Sequence[str]) -> dict[str, Any]:
    return v50._selection_metrics([], label, gold_ids)


def build_development_records(
    gold_rows: Sequence[dict[str, Any]], scored_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(gold_rows) != len(scored_rows) or not gold_rows:
        raise ValueError("ContractNLI v51 dev gold and score caches differ")
    records: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("ContractNLI v51 dev ids differ")
        candidates = v50.merge_v50_scored_candidates(dict(scored))
        robust = robust_consensus_details(candidates)
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            selected = select_v49(
                candidates, LOW_CORE_DIVERGENCE_V49, token_budget=budget
            )
            configurations[str(budget)] = {
                "pass_metrics": v50._selection_metrics(
                    selected,
                    str(gold["label_group"]),
                    list(gold["gold_candidate_ids"]),
                ),
                "abstain_metrics": _empty_metrics(
                    str(gold["label_group"]), list(gold["gold_candidate_ids"])
                ),
            }
        records.append(
            {
                "case_id": str(gold["case_id"]),
                "document_cluster": str(gold["document_cluster"]),
                "development_fold": int(gold["development_fold"]),
                "label_group": str(gold["label_group"]),
                "candidate_unit_count": int(gold["candidate_unit_count"]),
                "candidate_pool_quartile": str(gold["candidate_pool_quartile"]),
                "document_length_quartile": str(
                    gold["document_length_quartile"]
                ),
                "document_type": str(gold["document_type"]),
                "candidate_ceiling_complete": bool(
                    gold["candidate_ceiling_complete"]
                ),
                "robust_score": float(robust["case_score"]),
                "robust_score_hex": str(robust["case_score_hex"]),
                "configurations": configurations,
            }
        )
    return records


def threshold_candidates(records: Sequence[dict[str, Any]]) -> list[float]:
    values = sorted({float(row["robust_score"]) for row in records})
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("ContractNLI v51 robust scores are absent or non-finite")
    result = [float(np.nextafter(values[0], -math.inf))]
    result.extend(
        float(left + (right - left) / 2.0)
        for left, right in zip(values, values[1:], strict=False)
    )
    result.append(float(np.nextafter(values[-1], math.inf)))
    return result


def _mean(values: Sequence[float | int | bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def aggregate_records(
    records: Sequence[dict[str, Any]],
    *,
    threshold: float | None,
    decisions: dict[str, bool] | None = None,
) -> dict[str, float]:
    values: list[dict[str, Any]] = []
    for row in records:
        if decisions is not None:
            passed = bool(decisions[str(row["case_id"])])
        elif threshold is None:
            passed = True
        else:
            passed = float(row["robust_score"]) >= threshold
        key = "pass_metrics" if passed else "abstain_metrics"
        values.extend(
            configuration[key] for configuration in row["configurations"].values()
        )
    evidence = [value for value in values if value["evidence_f1"] is not None]
    missing = [
        configuration[
            "pass_metrics"
            if (
                decisions[str(row["case_id"])]
                if decisions is not None
                else threshold is None or float(row["robust_score"]) >= threshold
            )
            else "abstain_metrics"
        ]
        for row in records
        if row["label_group"] == "NotMentioned"
        for configuration in row["configurations"].values()
    ]
    return {
        "evidence_or_abstention_macro_f1": round(
            _mean([value["utility_f1"] for value in values]), 6
        ),
        "evidence_bearing_macro_f1": round(
            _mean([value["evidence_f1"] for value in evidence]), 6
        ),
        "not_mentioned_abstention_accuracy": round(
            _mean([value["utility_f1"] for value in missing]), 6
        ),
        "abstention_rate": round(
            _mean([value["abstained"] for value in values]), 6
        ),
        "mean_selected_unit_count": round(
            _mean([value["selected_unit_count"] for value in values]), 6
        ),
        "mean_selected_token_cost": round(
            _mean([value["selected_token_cost"] for value in values]), 6
        ),
    }


def fit_threshold(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    baseline = aggregate_records(records, threshold=None)
    candidates: list[dict[str, Any]] = []
    for threshold in threshold_candidates(records):
        aggregate = aggregate_records(records, threshold=threshold)
        evidence_drop = round(
            baseline["evidence_bearing_macro_f1"]
            - aggregate["evidence_bearing_macro_f1"],
            6,
        )
        feasible = (
            evidence_drop <= MAX_EVIDENCE_F1_DROP
            and aggregate["abstention_rate"] <= MAX_FIT_ABSTENTION_RATE
        )
        candidates.append(
            {
                "threshold": threshold,
                "threshold_hex": threshold.hex(),
                "aggregate": aggregate,
                "evidence_f1_drop": evidence_drop,
                "feasible": feasible,
            }
        )
    feasible = [row for row in candidates if row["feasible"]]
    if not feasible:
        raise ValueError("ContractNLI v51 has no feasible development threshold")
    best = sorted(
        feasible,
        key=lambda row: (
            -float(row["aggregate"]["evidence_or_abstention_macro_f1"]),
            -float(row["aggregate"]["evidence_bearing_macro_f1"]),
            -float(row["aggregate"]["not_mentioned_abstention_accuracy"]),
            float(row["threshold"]),
        ),
    )[0]
    return {
        "threshold": float(best["threshold"]),
        "threshold_hex": str(best["threshold_hex"]),
        "aggregate": dict(best["aggregate"]),
        "baseline": baseline,
        "evidence_f1_drop": float(best["evidence_f1_drop"]),
        "threshold_candidates": len(candidates),
        "feasible_threshold_candidates": len(feasible),
    }


def calibrate_development(
    records: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = sorted({int(row["development_fold"]) for row in records})
    if folds != list(range(FOLD_COUNT)):
        raise ValueError("ContractNLI v51 development folds are incomplete")
    oof_decisions: dict[str, bool] = {}
    fold_results: list[dict[str, Any]] = []
    for fold in folds:
        fit_rows = [row for row in records if row["development_fold"] != fold]
        held_out = [row for row in records if row["development_fold"] == fold]
        fitted = fit_threshold(fit_rows)
        threshold = float(fitted["threshold"])
        for row in held_out:
            oof_decisions[str(row["case_id"])] = (
                float(row["robust_score"]) >= threshold
            )
        fold_results.append(
            {
                "fold": fold,
                "fit_cases": len(fit_rows),
                "held_out_cases": len(held_out),
                "fit_documents": len(
                    {row["document_cluster"] for row in fit_rows}
                ),
                "held_out_documents": len(
                    {row["document_cluster"] for row in held_out}
                ),
                "threshold_hex": str(fitted["threshold_hex"]),
                "fit_aggregate": fitted["aggregate"],
                "held_out_aggregate": aggregate_records(
                    held_out,
                    threshold=None,
                    decisions=oof_decisions,
                ),
            }
        )
    if set(oof_decisions) != {str(row["case_id"]) for row in records}:
        raise AssertionError("ContractNLI v51 OOF decisions are incomplete")

    baseline = aggregate_records(records, threshold=None)
    oof = aggregate_records(records, threshold=None, decisions=oof_decisions)
    utility_gain = round(
        oof["evidence_or_abstention_macro_f1"]
        - baseline["evidence_or_abstention_macro_f1"],
        6,
    )
    evidence_drop = round(
        baseline["evidence_bearing_macro_f1"]
        - oof["evidence_bearing_macro_f1"],
        6,
    )
    final_fit = fit_threshold(records)
    label_counts = Counter(str(row["label_group"]) for row in records)
    checks = {
        "exact_cases_equals_240": len(records) == TARGET_CASES,
        "exact_cases_per_label_equals_80": all(
            label_counts[label] == TARGET_PER_LABEL for label in v50.LABELS
        ),
        "minimum_documents_met": len(
            {row["document_cluster"] for row in records}
        )
        >= MINIMUM_DOCUMENTS,
        "all_five_folds_nonempty": folds == list(range(FOLD_COUNT)),
        "oof_candidate_minus_ungated_v49_utility_at_least_0_02": utility_gain
        >= 0.02,
        "oof_not_mentioned_abstention_accuracy_at_least_0_2": oof[
            "not_mentioned_abstention_accuracy"
        ]
        >= 0.2,
        "oof_evidence_bearing_f1_drop_at_most_0_03": evidence_drop <= 0.03,
        "oof_candidate_abstention_rate_at_least_0_05": oof["abstention_rate"]
        >= 0.05,
        "oof_candidate_abstention_rate_at_most_0_8": oof["abstention_rate"]
        <= 0.8,
        "deterministic_query_fallback_rate_equals_0": query_summary[
            "fallback_rate"
        ]
        == 0.0,
    }
    train_open_authorized = all(checks.values())
    status = (
        "CONTRACTNLI_V51_DEVELOPMENT_GATE_PASSED_TRAIN_OPEN_AUTHORIZED"
        if train_open_authorized
        else "CONTRACTNLI_V51_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_TRAIN"
    )
    evidence = [
        {
            "case_id": str(row["case_id"]),
            "document_cluster": str(row["document_cluster"]),
            "development_fold": int(row["development_fold"]),
            "label_group": str(row["label_group"]),
            "candidate_unit_count": int(row["candidate_unit_count"]),
            "candidate_pool_quartile": str(row["candidate_pool_quartile"]),
            "document_length_quartile": str(row["document_length_quartile"]),
            "document_type": str(row["document_type"]),
            "candidate_ceiling_complete": bool(row["candidate_ceiling_complete"]),
            "robust_score_hex": str(row["robust_score_hex"]),
            "oof_gate_passed": bool(oof_decisions[str(row["case_id"])]),
        }
        for row in records
    ]
    report = {
        "schema_version": "frc-contractnli-v51-development-calibration-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "split": "official dev",
            "cases": len(records),
            "documents": len({row["document_cluster"] for row in records}),
            "label_counts": {label: label_counts[label] for label in v50.LABELS},
            "budgets": list(BUDGETS),
            "development_only": True,
            "official_leaderboard_result": False,
            "test_split_used": False,
            "train_split_opened": False,
            "gold_joined_after_complete_dev_score_cache": True,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "robust_score_formula": (
                "max_i min(z_anchor_i, max(z_support_i, z_contradiction_i))"
            ),
            "ungated_v49_aggregate": baseline,
            "oof_candidate_aggregate": oof,
            "oof_utility_gain_vs_ungated_v49": utility_gain,
            "oof_evidence_f1_drop_vs_ungated_v49": evidence_drop,
            "fold_results": fold_results,
            "final_all_dev_fit": final_fit,
            "development_checks": checks,
            "outcome": {
                "status": status,
                "train_open_authorized": train_open_authorized,
                "final_threshold_hex": str(final_fit["threshold_hex"]),
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "reuse_dev_for_confirmation_metrics": False,
                "reuse_v50_test_for_v51": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    return report, evidence


def write_development_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> None:
    rounded = _round_for_display(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(rounded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = rounded["analysis"]
    outcome = analysis["outcome"]
    baseline = analysis["ungated_v49_aggregate"]
    candidate = analysis["oof_candidate_aggregate"]
    lines = [
        "# ContractNLI robust consensus development calibration (v51)",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases/documents: {rounded['metadata']['cases']}/{rounded['metadata']['documents']}",
        f"- Train open authorized: `{str(outcome['train_open_authorized']).lower()}`",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "## Five-fold OOF result",
        "",
        f"- Ungated v49 utility F1: {baseline['evidence_or_abstention_macro_f1']:.6f}",
        f"- OOF candidate utility F1: {candidate['evidence_or_abstention_macro_f1']:.6f}",
        f"- OOF utility gain: {analysis['oof_utility_gain_vs_ungated_v49']:.6f}",
        f"- OOF evidence F1 drop: {analysis['oof_evidence_f1_drop_vs_ungated_v49']:.6f}",
        f"- OOF missing abstention accuracy: {candidate['not_mentioned_abstention_accuracy']:.6f}",
        f"- OOF abstention rate: {candidate['abstention_rate']:.6f}",
        f"- Final threshold hex: `{outcome['final_threshold_hex']}`",
        "",
        "## Development checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in analysis["development_checks"].items()
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This supervised dev result is not confirmation, does not authorize selector adoption, and may open the previously untouched train split only when every registered development check passes.",
            "",
        ]
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in evidence:
                compressed.write(
                    (
                        json.dumps(
                            _round_for_display(row),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode()
                )


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("ContractNLI v51 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    boundary = value["development_boundary"]
    if boundary["official_dev_member_content_opened_before_v51_registration"]:
        raise ValueError("ContractNLI v51 dev was opened before registration")
    if boundary["official_train_member_content_opened_before_v51_registration"]:
        raise ValueError("ContractNLI v51 train was opened before registration")
    if not boundary["official_test_split_permanently_excluded_from_v51"]:
        raise ValueError("ContractNLI v51 test exclusion changed")
    if value["development_scope"]["target_cases"] != TARGET_CASES:
        raise ValueError("ContractNLI v51 development size changed")
    if value["development_threshold_selection"]["budgets"] != list(BUDGETS):
        raise ValueError("ContractNLI v51 budgets changed")
    return value


def validate_implementation_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    expected = {
        "protocol_sha256": _sha256(protocol_path),
        "module_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "test_sha256": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("ContractNLI v51 implementation hash mismatch")
    if value.get("dev_or_train_content_opened_before_registration") is not False:
        raise ValueError("ContractNLI v51 implementation was registered too late")
    if not all(value.get("synthetic_invariants_verified", {}).values()):
        raise ValueError("ContractNLI v51 synthetic invariants are incomplete")
    return value


def validate_development_execution(
    registration_path: Path,
    *,
    implementation_path: Path,
    source_archive: Path,
    prepared_path: Path,
    candidate_map_path: Path,
    census_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    expected = {
        "implementation_registration_sha256": _sha256(implementation_path),
        "source_archive_sha256": _sha256(source_archive),
        "prepared_blind_sha256": _sha256(prepared_path),
        "candidate_map_sha256": _sha256(candidate_map_path),
        "blind_census_sha256": _sha256(census_path),
        "candidate_coverage_sha256": _sha256(coverage_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("ContractNLI v51 development execution hash mismatch")
    if value.get("dev_query_generation_started") is not False:
        raise ValueError("ContractNLI v51 dev execution was registered too late")
    if value.get("train_member_opened") is not False:
        raise ValueError("ContractNLI v51 train was opened before development")
    return value


__all__ = [
    "CAPABILITY",
    "DATASET_ID",
    "DEVELOPMENT_BASELINE",
    "DEVELOPMENT_CANDIDATE",
    "DEV_SAMPLE_SALT",
    "EXPERIMENT_ID",
    "FOLD_COUNT",
    "FrozenDevelopmentScorer",
    "MAX_CASES_PER_DOCUMENT",
    "PROTOCOL_SHA256",
    "SCHEMA_VERSION",
    "TARGET_CASES",
    "TARGET_PER_LABEL",
    "aggregate_records",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_development_gold_rows",
    "build_development_records",
    "calibrate_development",
    "fit_threshold",
    "fold_for_document",
    "prepare_blind_cases",
    "read_split_source",
    "robust_consensus_details",
    "select_balanced_sample",
    "threshold_candidates",
    "validate_development_execution",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "write_development_report",
]
