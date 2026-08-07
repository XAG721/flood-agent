"""Calibrated RoBERTa QA support gate on SQuAD 2.0 (v64)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import research.frc_rag.quac_roberta_qa_support_transfer as v62
import research.frc_rag.squad2_dual_support_union as v61
import research.frc_rag.squad2_extractive_support_gate as v59
import research.frc_rag.squad2_generative_answerability_gate as v58
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    _round_for_display,
)


SCHEMA_VERSION = "frc-squad2-calibrated-roberta-support-v64"
EXPERIMENT_ID = "FRC-SQUAD2-CALIBRATED-ROBERTA-SUPPORT-V64"
DATASET_ID = "squad2_calibrated_roberta_support_v64"
CAPABILITY = "self_domain_calibrated_qa_support_and_evidence_selection"
PROTOCOL_SHA256 = "2585d468c257de8912c380718d3ea27b7dbd96b6a61d75ff09d5a0901f8f0835"
SOURCE_REGISTRATION_SHA256 = (
    "85f1e206ab0ef1d566f398974ad665045d98789bc2130b81adb9cf1eac6a331c"
)
MODEL_REGISTRATION_SHA256 = (
    "b80ee8b127713e1440c25eee58974fe5699a808fdec37980f9729d239d048325"
)
MODEL_REPOSITORY = v62.MODEL_REPOSITORY
MODEL_REVISION = v62.MODEL_REVISION

STAGES = ("calibration", "confirmation")
SOURCE_FILES = {
    "calibration": "train-v2.0.json",
    "confirmation": "dev-v2.0.json",
}
TARGET_CASES = {"calibration": 2000, "confirmation": 600}
TARGET_PER_GROUP = {"calibration": 1000, "confirmation": 300}
MAX_CASES_PER_PARAGRAPH = 2
MAX_CASES_PER_ARTICLE_PER_STATE = {"calibration": 20, "confirmation": 12}
MINIMUM_PARAGRAPHS = {"calibration": 700, "confirmation": 250}
MINIMUM_ARTICLES = {"calibration": 200, "confirmation": 25}
SAMPLE_SALTS = {
    "calibration": "FRC-SQUAD2-V64-CALIBRATION|",
    "confirmation": "FRC-SQUAD2-V64-CONFIRMATION|",
}
FOLD_SALT = "FRC-SQUAD2-V64-FOLD|"
FOLD_COUNT = 5
BUDGETS = v59.BUDGETS

GATED_NON_FRC_BY_BASE = {
    base: f"calibrated_roberta_supported_{base}_v64"
    for base in v59.GATED_NON_FRC_BY_BASE
}
GATED_FRC_BY_BASE = {
    base: f"calibrated_roberta_supported_{base}_v64" for base in v59.GATED_FRC_BY_BASE
}
GATED_NON_FRC_METHODS = tuple(GATED_NON_FRC_BY_BASE.values())
GATED_FRC_CONTROLS = tuple(GATED_FRC_BY_BASE.values())
CANDIDATE = "calibrated_roberta_supported_adaptive_argmax_cardinality_frc_v43_v64"
EXACT_ANCHOR_GATED = GATED_NON_FRC_BY_BASE["cross_encoder_topk"]
EXACT_ANCHOR_UNGATED = v59.EXACT_ANCHOR_UNGATED
METHOD_RENAMES = {
    **{
        old: GATED_NON_FRC_BY_BASE[base]
        for base, old in v59.GATED_NON_FRC_BY_BASE.items()
    },
    **{old: GATED_FRC_BY_BASE[base] for base, old in v59.GATED_FRC_BY_BASE.items()},
    v59.CANDIDATE: CANDIDATE,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _opaque(prefix: str, *parts: str) -> str:
    return f"{prefix}{_hash(*parts)[:22]}"


def read_stage_source(source_root: Path, stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported SQuAD2 v64 stage: {stage}")
    source_stage = "development" if stage == "calibration" else "confirmation"
    return v58.read_stage_source(source_root, source_stage)


def load_prior_exclusion_union(paths: Sequence[Path]) -> set[str]:
    if len(paths) != 4:
        raise ValueError("SQuAD2 v64 requires exactly four prior commitment maps")
    groups = [v59.load_v58_excluded_commitments(path) for path in paths]
    union = set().union(*groups)
    if len(union) != 2400:
        raise ValueError("SQuAD2 v64 prior exclusion union must contain 2,400 ids")
    return union


def _sample_key(row: dict[str, Any], stage: str) -> tuple[str, str]:
    return (
        _hash(
            SAMPLE_SALTS[stage],
            str(row["answer_state"]),
            str(row["article_id"]),
            str(row["paragraph_id"]),
            str(row["raw_id"]),
        ),
        str(row["raw_id"]),
    )


def select_balanced_sample(
    source: dict[str, Any],
    *,
    stage: str,
    excluded_commitments: set[str],
    target_per_group: int | None = None,
    maximum_cases_per_paragraph: int = MAX_CASES_PER_PARAGRAPH,
    maximum_cases_per_article_per_state: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported SQuAD2 v64 stage: {stage}")
    target = target_per_group or TARGET_PER_GROUP[stage]
    article_cap = (
        maximum_cases_per_article_per_state or MAX_CASES_PER_ARTICLE_PER_STATE[stage]
    )
    rows, schema = v58.extract_cases(source)
    available = [
        row for row in rows if _hash(str(row["raw_id"])) not in excluded_commitments
    ]
    groups = ("answer_bearing", "no_answer")
    by_group = {
        group: sorted(
            [row for row in available if row["answer_state"] == group],
            key=lambda row: _sample_key(row, stage),
        )
        for group in groups
    }
    if any(len(by_group[group]) < target for group in groups):
        raise ValueError("SQuAD2 v64 source lacks the registered balanced quota")
    selected: list[dict[str, Any]] = []
    selected_by_group: Counter[str] = Counter()
    paragraph_counts: Counter[str] = Counter()
    article_state_counts: Counter[tuple[str, str]] = Counter()
    cursors = {group: 0 for group in groups}
    while any(selected_by_group[group] < target for group in groups):
        progressed = False
        for group in groups:
            if selected_by_group[group] >= target:
                continue
            values = by_group[group]
            while cursors[group] < len(values):
                row = values[cursors[group]]
                cursors[group] += 1
                paragraph_id = str(row["paragraph_id"])
                article_state = (str(row["article_id"]), group)
                if paragraph_counts[paragraph_id] >= maximum_cases_per_paragraph:
                    continue
                if article_state_counts[article_state] >= article_cap:
                    continue
                selected.append(row)
                selected_by_group[group] += 1
                paragraph_counts[paragraph_id] += 1
                article_state_counts[article_state] += 1
                progressed = True
                break
        if not progressed:
            raise ValueError("SQuAD2 v64 caps prevent the registered balanced sample")
    selected.sort(key=lambda row: (str(row["answer_state"]), _sample_key(row, stage)))
    overlap = sum(_hash(str(row["raw_id"])) in excluded_commitments for row in selected)
    return selected, {
        "stage": stage,
        "target_cases": target * 2,
        "selected_cases": len(selected),
        "eligible_answer_state_counts_after_exclusion": dict(
            Counter(row["answer_state"] for row in available)
        ),
        "selected_answer_state_counts": dict(selected_by_group),
        "selected_paragraphs": len(paragraph_counts),
        "selected_articles": len({str(row["article_id"]) for row in selected}),
        "maximum_cases_per_paragraph": max(paragraph_counts.values(), default=0),
        "maximum_cases_per_article_per_answer_state": max(
            article_state_counts.values(), default=0
        ),
        "prior_excluded_commitment_count": len(excluded_commitments),
        "selected_prior_commitment_overlap": overlap,
        **schema,
    }


def prepare_blind_cases(
    selected: Sequence[dict[str, Any]], tokenizer: Any, *, stage: str
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    source_stage = "development" if stage == "calibration" else "confirmation"
    prepared, maps, qa_inputs, structural = v58.prepare_blind_cases(
        selected, tokenizer, stage=source_stage
    )
    for source_row, blind, mapping, qa_input in zip(
        selected, prepared, maps, qa_inputs, strict=True
    ):
        case_id = _opaque("s64c", stage, str(source_row["raw_id"]))
        blind.update(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "capability": CAPABILITY,
                "stage": stage,
                "id": case_id,
            }
        )
        mapping.update(
            {
                "schema_version": "frc-squad2-v64-candidate-map-v1",
                "id": case_id,
            }
        )
        qa_input.update(
            {
                "schema_version": "frc-squad2-v64-qa-input-v1",
                "dataset_id": DATASET_ID,
                "stage": stage,
                "id": case_id,
            }
        )
    structural.update(
        {
            "schema_version": "frc-squad2-v64-structural-census-v1",
            "stage": stage,
            "inherited_candidate_construction": "frozen v58 exact implementation",
        }
    )
    return prepared, maps, qa_inputs, structural


class LocalRobertaQASupportVerifier(v62.LocalRobertaQASupportVerifier):
    def verify(
        self, rows: Sequence[dict[str, Any]], *, cache_key: str
    ) -> list[dict[str, Any]]:
        result = super().verify(rows, cache_key=cache_key)
        for row in result:
            row["schema_version"] = "frc-squad2-v64-raw-qa-decision-v1"
        return result


def validate_raw_qa_cache(
    qa_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> dict[str, Any]:
    expected = [str(row["id"]) for row in qa_inputs]
    if [str(row["id"]) for row in decisions] != expected:
        raise ValueError("SQuAD2 v64 QA cache is incomplete or out of order")
    if any(row.get("cache_key") != cache_key for row in decisions):
        raise ValueError("SQuAD2 v64 QA cache key changed")
    invalid = sum(bool(row.get("invalid_fail_closed_used")) for row in decisions)
    return {
        "rows": len(decisions),
        "invalid_output_count": invalid,
        "invalid_output_rate": invalid / len(decisions) if decisions else 1.0,
        "gold_fields_visible_to_verifier": False,
    }


def build_support_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    raw_decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_rows, _ = v58.extract_cases(source)
    source_by_commitment = {_hash(str(row["raw_id"])): row for row in source_rows}
    decisions = {str(row["id"]): row for row in raw_decisions}
    result: list[dict[str, Any]] = []
    for mapping in candidate_maps:
        source_row = source_by_commitment.get(str(mapping["source_case_commitment"]))
        decision = decisions.get(str(mapping["id"]))
        if source_row is None or decision is None:
            raise ValueError("SQuAD2 v64 support gold join is incomplete")
        result.append(
            {
                "case_id": str(mapping["id"]),
                "article_cluster": str(mapping["article_commitment"]),
                "paragraph_cluster": str(mapping["paragraph_commitment"]),
                "answer_state": str(source_row["answer_state"]),
                "score_margin": decision.get("score_margin"),
                "invalid_fail_closed_used": bool(
                    decision.get("invalid_fail_closed_used")
                ),
                "raw_prediction_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "best_span_score": decision.get("best_span_score"),
                            "null_score": decision.get("null_score"),
                            "score_margin": decision.get("score_margin"),
                            "span_sha256": decision.get("span_sha256"),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
        )
    return result


def threshold_candidates(rows: Sequence[dict[str, Any]]) -> list[float]:
    margins = sorted(
        {
            float(row["score_margin"])
            for row in rows
            if not row["invalid_fail_closed_used"]
            and row.get("score_margin") is not None
            and math.isfinite(float(row["score_margin"]))
        }
    )
    if not margins:
        raise ValueError("SQuAD2 v64 has no finite calibration margin")
    values = [math.nextafter(margins[0], -math.inf)]
    values.extend(
        left + (right - left) / 2.0 for left, right in zip(margins, margins[1:])
    )
    values.append(margins[-1])
    return values


def apply_threshold_decision(row: dict[str, Any], threshold: float) -> bool:
    margin = row.get("score_margin")
    return bool(
        not row.get("invalid_fail_closed_used")
        and margin is not None
        and math.isfinite(float(margin))
        and float(margin) > threshold
    )


def support_metrics(
    rows: Sequence[dict[str, Any]], threshold: float
) -> dict[str, float | int]:
    answer = [row for row in rows if row["answer_state"] == "answer_bearing"]
    no_answer = [row for row in rows if row["answer_state"] == "no_answer"]
    if not answer or not no_answer:
        raise ValueError("SQuAD2 v64 support metrics require both answer states")
    answer_pass = float(
        np.mean([apply_threshold_decision(row, threshold) for row in answer])
    )
    no_answer_reject = float(
        np.mean([not apply_threshold_decision(row, threshold) for row in no_answer])
    )
    return {
        "rows": len(rows),
        "answer_bearing_rows": len(answer),
        "no_answer_rows": len(no_answer),
        "answer_bearing_support_pass_rate": answer_pass,
        "no_answer_rejection_rate": no_answer_reject,
        "balanced_accuracy": (answer_pass + no_answer_reject) / 2.0,
    }


def choose_threshold(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    candidates = threshold_candidates(rows)
    ranked: list[tuple[tuple[float, ...], float, dict[str, float | int]]] = []
    for threshold in candidates:
        metrics = support_metrics(rows, threshold)
        answer_pass = float(metrics["answer_bearing_support_pass_rate"])
        no_answer_reject = float(metrics["no_answer_rejection_rate"])
        feasible = answer_pass >= 0.85 and no_answer_reject >= 0.75
        key = (
            float(feasible),
            float(metrics["balanced_accuracy"]),
            min(answer_pass, no_answer_reject),
            -abs(threshold),
            -threshold,
        )
        ranked.append((key, threshold, metrics))
    _, threshold, metrics = max(ranked, key=lambda item: item[0])
    return {
        "threshold": threshold,
        "candidate_threshold_count": len(candidates),
        "feasibility_constraints_met": (
            float(metrics["answer_bearing_support_pass_rate"]) >= 0.85
            and float(metrics["no_answer_rejection_rate"]) >= 0.75
        ),
        "metrics": metrics,
    }


def fold_for_article(article_cluster: str) -> int:
    return int(_hash(FOLD_SALT, article_cluster)[:16], 16) % FOLD_COUNT


def calibrate_article_disjoint_oof(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("SQuAD2 v64 calibration rows are empty")
    fold_articles: dict[int, set[str]] = {fold: set() for fold in range(FOLD_COUNT)}
    fold_rows: dict[int, list[dict[str, Any]]] = {
        fold: [] for fold in range(FOLD_COUNT)
    }
    for row in rows:
        article = str(row["article_cluster"])
        fold = fold_for_article(article)
        fold_articles[fold].add(article)
        fold_rows[fold].append(row)
    if any(not fold_rows[fold] for fold in range(FOLD_COUNT)):
        raise ValueError("SQuAD2 v64 calibration fold is empty")
    if any(
        {row["answer_state"] for row in fold_rows[fold]}
        != {"answer_bearing", "no_answer"}
        for fold in range(FOLD_COUNT)
    ):
        raise ValueError("SQuAD2 v64 calibration fold lacks one answer state")
    oof_rows: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        train = [
            row
            for other in range(FOLD_COUNT)
            if other != fold
            for row in fold_rows[other]
        ]
        fitted = choose_threshold(train)
        threshold = float(fitted["threshold"])
        validation = fold_rows[fold]
        validation_metrics = support_metrics(validation, threshold)
        oof_rows.extend(
            {
                **row,
                "support_passed": apply_threshold_decision(row, threshold),
                "fold": fold,
            }
            for row in validation
        )
        folds.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_articles": len(
                    set().union(
                        *(
                            fold_articles[item]
                            for item in range(FOLD_COUNT)
                            if item != fold
                        )
                    )
                ),
                "validation_articles": len(fold_articles[fold]),
                "threshold": threshold,
                "train_selection": fitted,
                "validation_metrics": validation_metrics,
            }
        )
    answer = [row for row in oof_rows if row["answer_state"] == "answer_bearing"]
    no_answer = [row for row in oof_rows if row["answer_state"] == "no_answer"]
    answer_pass = float(np.mean([row["support_passed"] for row in answer]))
    no_answer_reject = float(np.mean([not row["support_passed"] for row in no_answer]))
    global_fit = choose_threshold(rows)
    return {
        "fold_count": FOLD_COUNT,
        "article_disjoint": sum(len(value) for value in fold_articles.values())
        == len(set().union(*fold_articles.values())),
        "folds": folds,
        "oof_metrics": {
            "rows": len(oof_rows),
            "answer_bearing_rows": len(answer),
            "no_answer_rows": len(no_answer),
            "answer_bearing_support_pass_rate": answer_pass,
            "no_answer_rejection_rate": no_answer_reject,
            "balanced_accuracy": (answer_pass + no_answer_reject) / 2.0,
        },
        "global_fit": global_fit,
        "oof_decisions": sorted(oof_rows, key=lambda row: str(row["case_id"])),
    }


def threshold_support_rows(
    raw_decisions: Sequence[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in raw_decisions:
        support_passed = apply_threshold_decision(raw, threshold)
        digest = hashlib.sha256(
            json.dumps(
                {
                    "score_margin": raw.get("score_margin"),
                    "threshold": threshold,
                    "support_passed": support_passed,
                    "span_sha256": raw.get("span_sha256"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        result.append(
            {
                **raw,
                "schema_version": "frc-squad2-v64-thresholded-support-v1",
                "decision": "SUPPORTED" if support_passed else "UNSUPPORTED",
                "support_passed": support_passed,
                "locked_threshold": threshold,
                "raw_prediction_sha256": digest,
                "raw_prediction_codepoints": 0,
            }
        )
    return result


def evaluate_calibration_gate(
    support_gold: Sequence[dict[str, Any]],
    verifier_summary: dict[str, Any],
    sampling: dict[str, Any],
    source_artifacts: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calibration = calibrate_article_disjoint_oof(support_gold)
    oof = calibration["oof_metrics"]
    checks = {
        "exact_cases_equals_2000": len(support_gold) == 2000,
        "exact_answer_state_balance": Counter(
            row["answer_state"] for row in support_gold
        )
        == {"answer_bearing": 1000, "no_answer": 1000},
        "selected_prior_commitment_overlap_equals_0": sampling[
            "selected_prior_commitment_overlap"
        ]
        == 0,
        "schema_exclusion_rate_at_most_0_01": sampling["schema_exclusion_rate"] <= 0.01,
        "qa_invalid_rate_equals_0": verifier_summary["invalid_output_rate"] == 0.0,
        "five_article_disjoint_nonempty_folds": calibration["fold_count"] == 5
        and calibration["article_disjoint"]
        and all(item["validation_rows"] > 0 for item in calibration["folds"]),
        "oof_balanced_accuracy_at_least_0_85": oof["balanced_accuracy"] >= 0.85,
        "oof_answer_bearing_pass_rate_at_least_0_85": oof[
            "answer_bearing_support_pass_rate"
        ]
        >= 0.85,
        "oof_no_answer_rejection_rate_at_least_0_75": oof["no_answer_rejection_rate"]
        >= 0.75,
        "minimum_paragraphs_at_least_700": sampling["selected_paragraphs"] >= 700,
        "minimum_articles_at_least_200": sampling["selected_articles"] >= 200,
    }
    opened = all(checks.values())
    status = (
        "SQUAD2_V64_CALIBRATION_ESTABLISHED_OPEN_DEV"
        if opened
        else "SQUAD2_V64_CALIBRATION_NOT_ESTABLISHED_STOP_BEFORE_DEV"
    )
    evidence = [
        {
            "case_id": str(row["case_id"]),
            "article_cluster": str(row["article_cluster"]),
            "paragraph_cluster": str(row["paragraph_cluster"]),
            "answer_state": str(row["answer_state"]),
            "score_margin": row["score_margin"],
            "invalid_fail_closed_used": bool(row["invalid_fail_closed_used"]),
            "fold": int(row["fold"]),
            "support_passed": bool(row["support_passed"]),
            "raw_prediction_sha256": str(row["raw_prediction_sha256"]),
        }
        for row in calibration.pop("oof_decisions")
    ]
    report = {
        "schema_version": "frc-squad2-v64-calibration-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "stage": "calibration",
            "cases": len(evidence),
            "answer_state_counts": dict(
                Counter(row["answer_state"] for row in evidence)
            ),
            "articles": sampling["selected_articles"],
            "paragraphs": sampling["selected_paragraphs"],
            "official_squad2_answer_string_result": False,
            "model_training_independent_confirmation": False,
            "gold_joined_after_complete_qa_cache": True,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "calibration": calibration,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "calibration_gate_passed": opened,
                "confirmation_dev_open_authorized": opened,
                "locked_threshold": calibration["global_fit"]["threshold"]
                if opened
                else None,
                "calibration_or_confirmation_reuse_for_additional_tuning_or_selection": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    return report, evidence


def evaluate_confirmation_support_gate(
    support_gold: Sequence[dict[str, Any]],
    verifier_summary: dict[str, Any],
    sampling: dict[str, Any],
    source_artifacts: dict[str, Any],
    *,
    locked_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = support_metrics(support_gold, locked_threshold)
    checks = {
        "exact_cases_equals_600": len(support_gold) == 600,
        "exact_answer_state_balance": Counter(
            row["answer_state"] for row in support_gold
        )
        == {"answer_bearing": 300, "no_answer": 300},
        "schema_exclusion_rate_at_most_0_01": sampling["schema_exclusion_rate"] <= 0.01,
        "qa_invalid_rate_equals_0": verifier_summary["invalid_output_rate"] == 0.0,
        "balanced_accuracy_at_least_0_8": metrics["balanced_accuracy"] >= 0.8,
        "answer_bearing_pass_rate_at_least_0_8": metrics[
            "answer_bearing_support_pass_rate"
        ]
        >= 0.8,
        "no_answer_rejection_rate_at_least_0_7": metrics["no_answer_rejection_rate"]
        >= 0.7,
        "minimum_paragraphs_at_least_250": sampling["selected_paragraphs"] >= 250,
        "minimum_articles_at_least_25": sampling["selected_articles"] >= 25,
    }
    opened = all(checks.values())
    status = (
        "SQUAD2_V64_CONFIRMATION_SUPPORT_ESTABLISHED_OPEN_RETRIEVAL"
        if opened
        else "SQUAD2_V64_CONFIRMATION_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_RETRIEVAL"
    )
    evidence = [
        {
            **row,
            "support_passed": apply_threshold_decision(row, locked_threshold),
            "locked_threshold": locked_threshold,
        }
        for row in support_gold
    ]
    report = {
        "schema_version": "frc-squad2-v64-confirmation-support-result-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "stage": "confirmation_support_gate",
            "cases": len(evidence),
            "answer_state_counts": dict(
                Counter(row["answer_state"] for row in evidence)
            ),
            "articles": sampling["selected_articles"],
            "paragraphs": sampling["selected_paragraphs"],
            "official_squad2_answer_string_result": False,
            "model_training_independent_confirmation": False,
            "gold_joined_after_complete_qa_cache": True,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "locked_threshold": locked_threshold,
            "support_verifier": {**metrics, **verifier_summary},
            "support_checks": checks,
            "outcome": {
                "status": status,
                "support_gate_passed": opened,
                "retrieval_scoring_open_authorized": opened,
                "independent_model_training_confirmation_claimed": False,
                "calibration_or_confirmation_reuse_for_additional_tuning_or_selection": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    return report, evidence


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = v61.build_deterministic_queries(prepared_rows)
    for row in rows:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
        row["stage"] = "confirmation"
    return rows


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    expected = build_deterministic_queries(prepared_rows)
    if list(query_rows) != expected:
        raise ValueError("SQuAD2 v64 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
    }


class FrozenSquad2V64Scorer(v61.FrozenSquad2Scorer):
    def score(self, cases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["stage"] = "confirmation"
        return rows


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    structural_census: dict[str, Any],
) -> list[dict[str, Any]]:
    return v61.build_gold_rows(
        source,
        candidate_maps,
        scored_rows,
        structural_census=structural_census,
    )


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    # The inherited v59 coverage helper names this field after its original
    # v58-only exclusion boundary.  V64 excludes the union of v58-v61 and
    # records that stricter boundary under ``selected_prior_commitment_overlap``.
    # Adapt the audit-key name without changing the frozen sample or scores.
    compatible_sampling = {
        **sampling,
        "selected_v58_commitment_overlap": int(
            sampling.get("selected_prior_commitment_overlap", 0)
        ),
    }
    value = v61.build_candidate_coverage(gold_rows, compatible_sampling)
    value["schema_version"] = "frc-squad2-v64-candidate-coverage-v1"
    value["experiment_id"] = EXPERIMENT_ID
    return value


def _rename_methods(report: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
    aggregates = report["analysis"]["aggregates"]
    report["analysis"]["aggregates"] = {
        METHOD_RENAMES.get(name, name): value for name, value in aggregates.items()
    }
    for key in ("strongest_shared_gate_non_frc", "strongest_same_gate_frc"):
        value = report["analysis"][key]
        report["analysis"][key] = METHOD_RENAMES.get(value, value)
    report["analysis"]["mechanism"] = (
        "one shared calibrated RoBERTa null-vs-span decision followed by frozen "
        "selectors; candidate is transferred v43 adaptive FRC"
    )
    for stratum in report["analysis"]["supported_stratum_deltas"].values():
        value = stratum["strongest_shared_gate_non_frc"]
        stratum["strongest_shared_gate_non_frc"] = METHOD_RENAMES.get(value, value)
    for row in evidence:
        for configuration in row["configurations"].values():
            configuration["methods"] = {
                METHOD_RENAMES.get(name, name): value
                for name, value in configuration["methods"].items()
            }


def evaluate_final_method(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    support_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    verifier_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    compatible_source = {
        **source_artifacts,
        "sampling": {
            **source_artifacts["sampling"],
            "selected_v58_commitment_overlap": int(
                source_artifacts["sampling"].get(
                    "selected_prior_commitment_overlap", 0
                )
            ),
        },
    }
    report, evidence = v59.evaluate_stage(
        gold_rows,
        scored_rows,
        support_rows,
        query_summary,
        verifier_summary,
        compatible_source,
        stage="confirmation",
    )
    _rename_methods(report, evidence)
    report["schema_version"] = "frc-squad2-v64-final-result-v1"
    report["experiment_id"] = EXPERIMENT_ID
    metadata = report["metadata"]
    metadata.update(
        {
            "dataset_id": DATASET_ID,
            "stage": "confirmation",
            "self_domain_held_out_component_feasibility_only": True,
            "independent_model_training_confirmation": False,
            "official_squad2_answer_string_result": False,
            "prior_case_level_artifact_reused": False,
        }
    )
    analysis = report["analysis"]
    verifier = analysis["support_verifier"]
    verifier["answer_bearing_support_pass_rate"] = verifier.pop(
        "answer_bearing_span_pass_rate"
    )
    verifier["locked_threshold"] = source_artifacts["locked_threshold"]
    candidate = analysis["aggregates"][CANDIDATE]
    comparisons = analysis["family_comparison"]
    checks = analysis["support_checks"]
    for key in (
        "support_verifier_balanced_accuracy_at_least_0_75",
        "answer_bearing_span_pass_rate_at_least_0_75",
        "no_answer_rejection_rate_at_least_0_65",
        "invalid_output_rate_at_most_0_02",
        "answer_recall_drop_vs_ungated_exact_anchor_at_most_0_1",
        "candidate_no_answer_abstention_accuracy_at_least_0_65",
        "candidate_abstention_rate_at_least_0_2",
    ):
        checks.pop(key, None)
    checks.update(
        {
            "confirmation_support_gate_passed": source_artifacts[
                "confirmation_support_gate_passed"
            ],
            "qa_invalid_rate_equals_0": verifier_summary["invalid_output_rate"] == 0.0,
            "candidate_answer_recall_at_least_0_75": candidate["answer_macro_recall"]
            >= 0.75,
            "candidate_no_answer_abstention_accuracy_at_least_0_7": candidate[
                "no_answer_abstention_accuracy"
            ]
            >= 0.7,
            "candidate_abstention_rate_at_least_0_15": candidate["abstention_rate"]
            >= 0.15,
            "candidate_minus_gated_exact_anchor_point_at_least_0_05": comparisons[
                "candidate_minus_gated_exact_anchor"
            ]["point"]
            >= 0.05,
            "candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_05": comparisons[
                "candidate_minus_strongest_shared_gate_non_frc"
            ]["point"]
            >= 0.05,
            "calibration_or_confirmation_reuse_for_additional_tuning_or_selection": False,
        }
    )
    supported = all(
        value
        for key, value in checks.items()
        if key != "calibration_or_confirmation_reuse_for_additional_tuning_or_selection"
    )
    status = (
        "SQUAD2_V64_CALIBRATED_QA_FRC_COMPONENT_FEASIBILITY_ESTABLISHED"
        if supported
        else "SQUAD2_V64_CALIBRATED_QA_FRC_COMPONENT_FEASIBILITY_NOT_ESTABLISHED"
    )
    analysis["outcome"].update(
        {
            "status": status,
            "support_established": supported,
            "confirmation_open_authorized": False,
            "independent_model_training_confirmation_claimed": False,
            "calibration_or_confirmation_reuse_for_additional_tuning_or_selection": False,
            "selector_adoption_authorized": False,
            "canary_or_default_authorized": False,
            "gate_2": "NO-GO/SHADOW",
        }
    )
    return report, evidence


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
    *,
    title: str,
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
    lines = [
        f"# {title}",
        "",
        f"- status: `{outcome['status']}`",
        f"- cases: {rounded['metadata']['cases']}",
        f"- selector adoption authorized: `{outcome['selector_adoption_authorized']}`",
        f"- Gate 2: `{outcome['gate_2']}`",
        "",
    ]
    if "support_verifier" in analysis:
        verifier = analysis["support_verifier"]
        lines.extend(
            [
                f"- balanced accuracy: {verifier['balanced_accuracy']:.6f}",
                f"- answer-bearing pass rate: {verifier['answer_bearing_support_pass_rate']:.6f}",
                f"- no-answer rejection rate: {verifier['no_answer_rejection_rate']:.6f}",
                "",
            ]
        )
    if "calibration" in analysis:
        oof = analysis["calibration"]["oof_metrics"]
        lines.extend(
            [
                f"- OOF balanced accuracy: {oof['balanced_accuracy']:.6f}",
                f"- OOF answer-bearing pass rate: {oof['answer_bearing_support_pass_rate']:.6f}",
                f"- OOF no-answer rejection rate: {oof['no_answer_rejection_rate']:.6f}",
                f"- locked threshold: {analysis['outcome']['locked_threshold']}",
                "",
            ]
        )
    lines.extend(
        [
            "> This is a balanced closed-paragraph component experiment. It is not an official SQuAD answer score, independent model-training confirmation, open-corpus retrieval, SetR reproduction, selector adoption, or flood-domain validation.",
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
                    ).encode("utf-8")
                )


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("SQuAD2 v64 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value["prior_boundary"]["dev_v2_content_opened_or_parsed"] is not False:
        raise ValueError("SQuAD2 v64 dev boundary changed")
    if value["prior_boundary"]["expected_prior_exclusion_commitments"] != 2400:
        raise ValueError("SQuAD2 v64 exclusion count changed")
    if value["threshold_calibration"]["learned_parameter_count"] != 1:
        raise ValueError("SQuAD2 v64 learned-parameter count changed")
    return value


def validate_source_registration(
    source_registration_path: Path, *, source_root: Path
) -> dict[str, Any]:
    if _sha256(source_registration_path) != SOURCE_REGISTRATION_SHA256:
        raise ValueError("SQuAD2 v64 source registration changed")
    value = json.loads(source_registration_path.read_text(encoding="utf-8"))
    expected = {str(row["path"]): row for row in value["files"]}
    for name in SOURCE_FILES.values():
        path = source_root / name
        row = expected.get(name)
        if (
            row is None
            or path.stat().st_size != int(row["bytes"])
            or _sha256(path) != row["sha256"]
        ):
            raise ValueError(f"SQuAD2 v64 source file changed: {name}")
    return value


def validate_model_registration(
    model_registration_path: Path, *, model_root: Path
) -> dict[str, Any]:
    if _sha256(model_registration_path) != MODEL_REGISTRATION_SHA256:
        raise ValueError("SQuAD2 v64 model registration changed")
    value = json.loads(model_registration_path.read_text(encoding="utf-8"))
    if value["repository"] != MODEL_REPOSITORY or value["revision"] != MODEL_REVISION:
        raise ValueError("SQuAD2 v64 model identity changed")
    for name, expected in value["files"].items():
        path = model_root / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"SQuAD2 v64 model file changed: {name}")
    return value


def validate_implementation_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    source_registration_path: Path,
    model_registration_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
    successor_erratum_path: Path | None = None,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    current = {
        "protocol": _sha256(protocol_path),
        "source_registration": _sha256(source_registration_path),
        "model_registration": _sha256(model_registration_path),
        "module": _sha256(module_path),
        "runner": _sha256(runner_path),
        "tests": _sha256(test_path),
    }
    frozen = value.get("hashes", {})
    for name in ("protocol", "source_registration", "model_registration"):
        if frozen.get(name) != current[name]:
            raise ValueError("SQuAD2 v64 implementation registration changed")
    if successor_erratum_path is None:
        if any(frozen.get(name) != current[name] for name in ("module", "runner", "tests")):
            raise ValueError("SQuAD2 v64 implementation registration changed")
    else:
        successor = json.loads(successor_erratum_path.read_text(encoding="utf-8"))
        if successor.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError("SQuAD2 v64 successor erratum experiment mismatch")
        successor_hashes = successor.get("hashes", {})
        expected_successor = {
            "prior_implementation_erratum": _sha256(registration_path),
            "corrected_module": current["module"],
            "corrected_runner": current["runner"],
            "regression_tests": current["tests"],
        }
        if any(
            successor_hashes.get(name) != expected_hash
            for name, expected_hash in expected_successor.items()
        ):
            raise ValueError("SQuAD2 v64 successor erratum hash chain changed")
    if value.get("dev_content_opened_or_parsed") is not False:
        raise ValueError("SQuAD2 v64 implementation was frozen after dev access")
    return value


__all__ = [
    "BUDGETS",
    "CANDIDATE",
    "EXACT_ANCHOR_GATED",
    "EXPERIMENT_ID",
    "FrozenSquad2V64Scorer",
    "LocalRobertaQASupportVerifier",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "SOURCE_FILES",
    "STAGES",
    "TARGET_CASES",
    "apply_threshold_decision",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "build_support_gold_rows",
    "calibrate_article_disjoint_oof",
    "choose_threshold",
    "evaluate_calibration_gate",
    "evaluate_confirmation_support_gate",
    "evaluate_final_method",
    "fold_for_article",
    "load_prior_exclusion_union",
    "prepare_blind_cases",
    "read_stage_source",
    "select_balanced_sample",
    "support_metrics",
    "threshold_support_rows",
    "validate_implementation_registration",
    "validate_model_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_raw_qa_cache",
    "validate_source_registration",
    "write_report",
]
