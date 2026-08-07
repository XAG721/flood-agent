"""Prospective ContractNLI train confirmation of a rank-concurrence guard."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import research.frc_rag.contractnli_native_zero_consensus_abstention as v50
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    LOW_CORE_DIVERGENCE_V49,
    METHODS as V49_METHODS,
    _round_for_display,
    select_v49,
)
from research.frc_rag.feverous_adaptive_atomic_roles import ADAPTIVE_ARGMAX
from research.frc_rag.hover_dynamic_atomic_roles import DYNAMIC_RANK
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import (
    BUDGETS,
    NON_FRC_BASELINES,
)


SCHEMA_VERSION = "frc-contractnli-rank-concurrence-confirmation-v52"
EXPERIMENT_ID = "FRC-CONTRACTNLI-RANK-CONCURRENCE-CONFIRMATION-V52"
DATASET_ID = "contractnli_official_train_rank_concurrence_confirmation_v52"
CAPABILITY = "parameter_free_rank_concurrence_missing_evidence_abstention"
PROTOCOL_SHA256 = "048fe13d87a58e8b6630387c9c0cc8e7c1e72119a54ed8719498d32ef6bc9251"

SAMPLE_SALT = "FRC-CONTRACTNLI-V52-RANK-CONCURRENCE|"
TARGET_CASES = 600
TARGET_PER_LABEL = 200
MAX_CASES_PER_DOCUMENT = 6
MINIMUM_DOCUMENTS = 80
MINIMUM_STRATUM_CASES = 60
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260812

RANK_CONCURRENCE_PREFIX = "rank_concurrence_"
GATED_NON_FRC_METHODS = tuple(
    f"{RANK_CONCURRENCE_PREFIX}{method}_v52" for method in NON_FRC_BASELINES
)
CANDIDATE = "rank_concurrence_low_core_divergence_frc_v52"
NATIVE_ZERO_CONTROL = "native_zero_consensus_low_core_divergence_frc_control_v52"
METHODS = (*V49_METHODS, *GATED_NON_FRC_METHODS, NATIVE_ZERO_CONTROL, CANDIDATE)
UNGATED_FRC_CONTROLS = (DYNAMIC_RANK, ADAPTIVE_ARGMAX, LOW_CORE_DIVERGENCE_V49)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_train_source(source_archive: Path) -> dict[str, Any]:
    """Open exactly the registered train member and no dev/test content."""

    with zipfile.ZipFile(source_archive) as archive:
        members = [
            name
            for name in archive.namelist()
            if Path(name).name.lower() == "train.json"
        ]
        if len(members) != 1:
            raise ValueError("ContractNLI v52 archive must contain one train.json")
        raw = archive.read(members[0])
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("ContractNLI v52 train JSON must be an object")
    if not isinstance(value.get("documents"), list) or not isinstance(
        value.get("labels"), dict
    ):
        raise ValueError("ContractNLI v52 train lacks documents or labels")
    return value


def _sample_key(row: dict[str, Any]) -> tuple[str, str, str]:
    digest = hashlib.sha256(
        (
            SAMPLE_SALT
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(v50._iter_cases(source))
    by_label = {
        label: sorted(
            [row for row in rows if row["label"] == label], key=_sample_key
        )
        for label in v50.LABELS
    }
    if any(len(by_label[label]) < target_per_label for label in v50.LABELS):
        raise ValueError("ContractNLI v52 train lacks the frozen label quota")

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

    expected = target_per_label * len(v50.LABELS)
    if len(selected) != expected:
        raise ValueError("ContractNLI v52 could not form its balanced sample")
    selected = sorted(selected, key=_sample_key)
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
        "sample_salt": SAMPLE_SALT,
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
        raise ValueError("ContractNLI v52 query schema changed")
    return summary


class FrozenRankConcurrenceScorer(v50.FrozenContractNliScorer):
    """Frozen BGE scorer with the v52 cache identity."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    *,
    pool_quartile_boundaries: Sequence[float],
    document_length_quartile_boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    return v50.build_gold_rows(
        source,
        candidate_maps,
        pool_quartile_boundaries=pool_quartile_boundaries,
        document_length_quartile_boundaries=document_length_quartile_boundaries,
    )


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    evidence = [
        row for row in gold_rows if row["label_group"] in v50.EVIDENCE_LABELS
    ]
    missing = [
        row for row in gold_rows if row["label_group"] == "NotMentioned"
    ]
    ceiling = float(
        np.mean([row["candidate_ceiling_complete"] for row in evidence])
    )
    empty_rate = float(np.mean([not row["gold_candidate_ids"] for row in missing]))
    labels = Counter(str(row["label_group"]) for row in gold_rows)
    documents = len({str(row["document_cluster"]) for row in gold_rows})
    checks = {
        "exact_target_cases_met": len(gold_rows) == TARGET_CASES,
        "exact_target_per_label_met": all(
            labels[label] == TARGET_PER_LABEL for label in v50.LABELS
        ),
        "minimum_documents_met": documents >= MINIMUM_DOCUMENTS,
        "candidate_ceiling_complete_rate_on_evidence_cases_equals_1": ceiling
        == 1.0,
        "not_mentioned_official_empty_evidence_rate_equals_1": empty_rate == 1.0,
        "sampling_selected_cases_equals_target": sampling.get("selected_cases")
        == TARGET_CASES,
        "sampling_maximum_cases_per_document_within_cap": int(
            sampling.get("maximum_cases_per_document", MAX_CASES_PER_DOCUMENT + 1)
        )
        <= MAX_CASES_PER_DOCUMENT,
    }
    return {
        "schema_version": "frc-contractnli-v52-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "cases": len(gold_rows),
        "documents": documents,
        "evidence_cases": len(evidence),
        "not_mentioned_cases": len(missing),
        "label_groups": dict(labels),
        "candidate_ceiling_complete_rate_on_evidence_cases": round(ceiling, 6),
        "not_mentioned_official_empty_evidence_rate": round(empty_rate, 6),
        "candidate_pool_quartiles": dict(
            Counter(row["candidate_pool_quartile"] for row in gold_rows)
        ),
        "document_length_quartiles": dict(
            Counter(row["document_length_quartile"] for row in gold_rows)
        ),
        "document_types": dict(Counter(row["document_type"] for row in gold_rows)),
        "sampling": sampling,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def _role_winner(candidates: Sequence[dict[str, Any]], role: str) -> str:
    ranked: list[tuple[float, str]] = []
    for candidate in candidates:
        raw = candidate.get("raw_dynamic_role_scores")
        if not isinstance(raw, dict) or role not in raw:
            raise ValueError("ContractNLI v52 raw role score is missing")
        score = float(raw[role])
        if not math.isfinite(score):
            raise ValueError("ContractNLI v52 raw role score must be finite")
        ranked.append((-score, str(candidate["id"])))
    if not ranked:
        raise ValueError("ContractNLI v52 requires at least one candidate")
    return sorted(ranked)[0][1]


def rank_concurrence_gate_details(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    winners = {
        role: _role_winner(candidates, role)
        for role in ("anchor", "first_fact", "second_fact_or_bridge")
    }
    matched_roles = [
        role
        for role in ("first_fact", "second_fact_or_bridge")
        if winners[role] == winners["anchor"]
    ]
    return {
        "anchor_winner_id": winners["anchor"],
        "support_winner_id": winners["first_fact"],
        "contradiction_winner_id": winners["second_fact_or_bridge"],
        "matched_polarity_roles": matched_roles,
        "rank_concurrence_passed": bool(matched_roles),
    }


def select_v52(
    candidates: Sequence[dict[str, Any]], method: str, *, token_budget: int
) -> list[dict[str, Any]]:
    values = list(candidates)
    if method in V49_METHODS:
        return select_v49(values, method, token_budget=token_budget)
    if method == NATIVE_ZERO_CONTROL:
        if not v50.native_zero_gate_details(values)["consensus_gate_passed"]:
            return []
        return select_v49(
            values, LOW_CORE_DIVERGENCE_V49, token_budget=token_budget
        )
    if not rank_concurrence_gate_details(values)["rank_concurrence_passed"]:
        return []
    if method == CANDIDATE:
        return select_v49(
            values, LOW_CORE_DIVERGENCE_V49, token_budget=token_budget
        )
    if method in GATED_NON_FRC_METHODS:
        base = method[len(RANK_CONCURRENCE_PREFIX) : -len("_v52")]
        if base not in NON_FRC_BASELINES:
            raise ValueError("ContractNLI v52 gated baseline mapping is invalid")
        return select_v49(values, base, token_budget=token_budget)
    raise ValueError(f"Unsupported ContractNLI v52 method: {method}")


def _aggregate(rows: Sequence[dict[str, Any]], method: str) -> dict[str, float]:
    return v50._aggregate(rows, method)


def _case_utility(row: dict[str, Any], method: str) -> float:
    return float(
        np.mean(
            [
                configuration["methods"][method]["metrics"]["utility_f1"]
                for configuration in row["configurations"].values()
            ]
        )
    )


def _cluster_bootstrap(
    rows: Sequence[dict[str, Any]], candidate: str, baseline: str
) -> dict[str, float | int]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row["document_cluster"])].append(
            _case_utility(row, candidate) - _case_utility(row, baseline)
        )
    clusters = sorted(by_cluster)
    arrays = [np.asarray(by_cluster[cluster], dtype=float) for cluster in clusters]
    point = float(np.mean(np.concatenate(arrays)))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        picked = rng.integers(0, len(arrays), size=len(arrays))
        samples[index] = float(
            np.mean(np.concatenate([arrays[item] for item in picked]))
        )
    return {
        "point": round(point, 6),
        "ci_low": round(float(np.quantile(samples, 0.025)), 6),
        "ci_high": round(float(np.quantile(samples, 0.975)), 6),
        "clusters": len(clusters),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _strongest(rows: Sequence[dict[str, Any]], methods: Sequence[str]) -> str:
    return sorted(
        methods,
        key=lambda method: (
            -float(_aggregate(rows, method)["evidence_or_abstention_macro_f1"]),
            method,
        ),
    )[0]


def _stratum_delta(
    rows: Sequence[dict[str, Any]], candidate: str
) -> tuple[str, float]:
    strongest = _strongest(rows, GATED_NON_FRC_METHODS)
    candidate_value = float(np.mean([_case_utility(row, candidate) for row in rows]))
    baseline_value = float(np.mean([_case_utility(row, strongest) for row in rows]))
    return strongest, round(candidate_value - baseline_value, 6)


def evaluate_confirmation(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(gold_rows) != len(scored_rows) or not gold_rows:
        raise ValueError("ContractNLI v52 gold and score caches differ")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("ContractNLI v52 gold and score ids differ")
        candidates = v50.merge_v50_scored_candidates(dict(scored))
        rank_gate = rank_concurrence_gate_details(candidates)
        native_gate = v50.native_zero_gate_details(candidates)
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v52(candidates, method, token_budget=budget)
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": v50._selection_metrics(
                        selected,
                        str(gold["label_group"]),
                        list(gold["gold_candidate_ids"]),
                    ),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "document_cluster": str(gold["document_cluster"]),
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
                "rank_gate": rank_gate,
                "native_zero_consensus_passed": bool(
                    native_gate["consensus_gate_passed"]
                ),
                "configurations": configurations,
            }
        )

    aggregates = {method: _aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = _strongest(evidence, GATED_NON_FRC_METHODS)
    strongest_frc = _strongest(evidence, UNGATED_FRC_CONTROLS)
    comparisons = {
        "candidate_minus_strongest_shared_gate_non_frc": _cluster_bootstrap(
            evidence, CANDIDATE, strongest_non_frc
        ),
        "candidate_minus_ungated_v49": _cluster_bootstrap(
            evidence, CANDIDATE, LOW_CORE_DIVERGENCE_V49
        ),
        "candidate_minus_strongest_ungated_frozen_frc": _cluster_bootstrap(
            evidence, CANDIDATE, strongest_frc
        ),
        "candidate_minus_native_zero_control": _cluster_bootstrap(
            evidence, CANDIDATE, NATIVE_ZERO_CONTROL
        ),
    }
    candidate = aggregates[CANDIDATE]
    ungated = aggregates[LOW_CORE_DIVERGENCE_V49]
    native = aggregates[NATIVE_ZERO_CONTROL]
    evidence_drop = round(
        ungated["evidence_bearing_macro_f1"]
        - candidate["evidence_bearing_macro_f1"],
        6,
    )
    missing_improvement = round(
        candidate["not_mentioned_abstention_accuracy"]
        - native["not_mentioned_abstention_accuracy"],
        6,
    )

    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        subset = []
        for row in evidence:
            clone = dict(row)
            clone["configurations"] = {
                str(budget): row["configurations"][str(budget)]
            }
            subset.append(clone)
        _, delta = _stratum_delta(subset, CANDIDATE)
        budget_deltas[str(budget)] = delta

    dimensions = {
        "label_group": lambda row: row["label_group"],
        "candidate_pool_quartile": lambda row: row["candidate_pool_quartile"],
        "document_length_quartile": lambda row: row["document_length_quartile"],
        "document_type": lambda row: row["document_type"],
        "gate_outcome": lambda row: (
            "passed"
            if row["rank_gate"]["rank_concurrence_passed"]
            else "abstained"
        ),
    }
    strata: dict[str, dict[str, Any]] = {}
    for dimension, getter in dimensions.items():
        for value in sorted({str(getter(row)) for row in evidence}):
            subset = [row for row in evidence if str(getter(row)) == value]
            if len(subset) < MINIMUM_STRATUM_CASES:
                continue
            strongest, delta = _stratum_delta(subset, CANDIDATE)
            strata[f"{dimension}:{value}"] = {
                "cases": len(subset),
                "strongest_shared_gate_non_frc": strongest,
                "delta": delta,
            }
    minimum_delta = min(
        [*budget_deltas.values(), *(row["delta"] for row in strata.values())],
        default=-math.inf,
    )
    checks = {
        "exact_cases_equals_600": len(evidence) == TARGET_CASES,
        "candidate_ceiling_complete_rate_on_evidence_cases_equals_1": all(
            row["candidate_ceiling_complete"]
            for row in evidence
            if row["label_group"] in v50.EVIDENCE_LABELS
        ),
        "not_mentioned_official_empty_evidence_rate_equals_1": all(
            row["label_group"] != "NotMentioned" or not row["gold_candidate_ids"]
            for row in gold_rows
        ),
        "candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_01": comparisons[
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["point"]
        >= 0.01,
        "candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0": comparisons[
            "candidate_minus_strongest_shared_gate_non_frc"
        ]["ci_low"]
        > 0.0,
        "candidate_minus_ungated_v49_point_at_least_0_01": comparisons[
            "candidate_minus_ungated_v49"
        ]["point"]
        >= 0.01,
        "candidate_minus_ungated_v49_ci_low_above_0": comparisons[
            "candidate_minus_ungated_v49"
        ]["ci_low"]
        > 0.0,
        "candidate_minus_native_zero_control_point_at_least_0_02": comparisons[
            "candidate_minus_native_zero_control"
        ]["point"]
        >= 0.02,
        "candidate_minus_native_zero_control_ci_low_above_0": comparisons[
            "candidate_minus_native_zero_control"
        ]["ci_low"]
        > 0.0,
        "evidence_bearing_macro_f1_drop_vs_ungated_v49_at_most_0_03": evidence_drop
        <= 0.03,
        "not_mentioned_abstention_accuracy_at_least_0_2": candidate[
            "not_mentioned_abstention_accuracy"
        ]
        >= 0.2,
        "not_mentioned_improvement_vs_native_zero_at_least_0_2": missing_improvement
        >= 0.2,
        "candidate_abstention_rate_at_least_0_05": candidate["abstention_rate"]
        >= 0.05,
        "candidate_abstention_rate_at_most_0_8": candidate["abstention_rate"]
        <= 0.8,
        "every_budget_and_supported_stratum_delta_at_least_minus_0_03": minimum_delta
        >= -0.03,
        "deterministic_query_fallback_rate_equals_0": query_summary[
            "fallback_rate"
        ]
        == 0.0,
    }
    supported = all(checks.values())
    status = (
        "CONTRACTNLI_V52_RANK_CONCURRENCE_SUPPORT_ESTABLISHED"
        if supported
        else "CONTRACTNLI_V52_RANK_CONCURRENCE_SUPPORT_NOT_ESTABLISHED"
    )
    report = {
        "schema_version": "frc-contractnli-rank-concurrence-report-v52",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "split": "official train role-reversed as untouched confirmation",
            "cases": len(evidence),
            "documents": len({row["document_cluster"] for row in evidence}),
            "label_counts": dict(Counter(row["label_group"] for row in evidence)),
            "budgets": list(BUDGETS),
            "official_leaderboard_result": False,
            "full_contract_bounded_pool": True,
            "balanced_mechanism_sample_not_natural_prevalence": True,
            "gold_joined_after_complete_score_cache": True,
            "v50_test_or_v51_dev_reused": False,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "gate_formula": (
                "anchor_argmax_id == support_argmax_id or "
                "anchor_argmax_id == contradiction_argmax_id"
            ),
            "aggregates": aggregates,
            "strongest_shared_gate_non_frc": strongest_non_frc,
            "strongest_ungated_frozen_frc": strongest_frc,
            "family_comparison": comparisons,
            "evidence_bearing_macro_f1_drop_vs_ungated_v49": evidence_drop,
            "not_mentioned_improvement_vs_native_zero": missing_improvement,
            "budget_deltas": budget_deltas,
            "supported_stratum_deltas": strata,
            "minimum_budget_or_supported_stratum_delta": round(minimum_delta, 6),
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "support_established": supported,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "reuse_train_for_tuning_or_selection": False,
                "reuse_v50_test_or_v51_dev": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    return report, evidence


def write_report(
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
    candidate = analysis["aggregates"][CANDIDATE]
    lines = [
        "# ContractNLI rank-concurrence confirmation (v52)",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases/documents: {rounded['metadata']['cases']}/{rounded['metadata']['documents']}",
        f"- Candidate utility F1: {candidate['evidence_or_abstention_macro_f1']:.6f}",
        f"- Candidate evidence F1: {candidate['evidence_bearing_macro_f1']:.6f}",
        f"- Missing abstention accuracy: {candidate['not_mentioned_abstention_accuracy']:.6f}",
        f"- Abstention rate: {candidate['abstention_rate']:.6f}",
        "- Selector adoption: `false`",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "## Family comparisons",
        "",
    ]
    for name, value in analysis["family_comparison"].items():
        lines.append(
            f"- `{name}`: {value['point']:+.6f} "
            f"(95% CI [{value['ci_low']:+.6f}, {value['ci_high']:+.6f}])"
        )
    lines.extend(["", "## Support checks", ""])
    lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in analysis["support_checks"].items()
    )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a one-time, role-reversed ContractNLI train confirmation of a parameter-free full-contract span-selection guard. It is not an official leaderboard result, open-corpus retrieval, SetR reproduction, selector adoption, or flood-domain expert validation.",
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
        raise ValueError("ContractNLI v52 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    boundary = value["development_history_boundary"]
    if boundary["official_train_member_content_opened_before_v52_registration"]:
        raise ValueError("ContractNLI v52 train was opened before registration")
    if not boundary["v50_test_and_v51_dev_permanently_excluded_from_v52_fitting_selection_and_evaluation"]:
        raise ValueError("ContractNLI v52 historical split exclusion changed")
    if not boundary["v52_has_no_learned_threshold_or_dataset_fitted_parameter"]:
        raise ValueError("ContractNLI v52 parameter-free boundary changed")
    if value["confirmation_scope"]["target_cases"] != TARGET_CASES:
        raise ValueError("ContractNLI v52 confirmation size changed")
    if value["confirmation_evaluation"]["token_budgets"] != list(BUDGETS):
        raise ValueError("ContractNLI v52 budgets changed")
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
        raise ValueError("ContractNLI v52 implementation hash mismatch")
    if value.get("train_content_opened_before_registration") is not False:
        raise ValueError("ContractNLI v52 implementation was registered too late")
    if not all(value.get("synthetic_invariants_verified", {}).values()):
        raise ValueError("ContractNLI v52 synthetic invariants are incomplete")
    return value


def validate_execution_registration(
    registration_path: Path,
    *,
    implementation_path: Path,
    source_archive: Path,
    prepared_path: Path,
    candidate_map_path: Path,
    census_path: Path,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    expected = {
        "implementation_registration_sha256": _sha256(implementation_path),
        "source_archive_sha256": _sha256(source_archive),
        "prepared_blind_sha256": _sha256(prepared_path),
        "candidate_map_sha256": _sha256(candidate_map_path),
        "blind_census_sha256": _sha256(census_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("ContractNLI v52 execution hash mismatch")
    if value.get("query_generation_started") is not False:
        raise ValueError("ContractNLI v52 execution was registered too late")
    if value.get("gold_joined_for_coverage_or_metrics") is not False:
        raise ValueError("ContractNLI v52 gold was joined before scoring")
    return value


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CANDIDATE",
    "CAPABILITY",
    "DATASET_ID",
    "EXPERIMENT_ID",
    "FrozenRankConcurrenceScorer",
    "GATED_NON_FRC_METHODS",
    "MAX_CASES_PER_DOCUMENT",
    "METHODS",
    "NATIVE_ZERO_CONTROL",
    "PROTOCOL_SHA256",
    "SAMPLE_SALT",
    "SCHEMA_VERSION",
    "TARGET_CASES",
    "TARGET_PER_LABEL",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "evaluate_confirmation",
    "prepare_blind_cases",
    "rank_concurrence_gate_details",
    "read_train_source",
    "select_balanced_sample",
    "select_v52",
    "validate_execution_registration",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "write_report",
]
