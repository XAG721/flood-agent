"""Prospective QASPER test of the frozen top-two proposal selector (v48)."""

from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

from research.frc_rag.hover_dynamic_atomic_roles import (
    DYNAMIC_RANK,
    DYNAMIC_ROLES,
    merge_scored_candidates,
    select_candidates,
)
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import (
    BUDGETS,
    CONSENSUS_GUARDED_V46,
    FRC_CONTROLS as V46_FRC_CONTROLS,
    METHODS as V46_METHODS,
    NON_FRC_BASELINES,
    FrozenTatqaScorer,
    select_v46,
)


SCHEMA_VERSION = "frc-qasper-top2-proposal-guarded-atomic-roles-v48"
EXPERIMENT_ID = "FRC-QASPER-TOP2-PROPOSAL-GUARDED-ATOMIC-ROLES-V48"
DATASET_ID = "qasper_dev_full_paper_evidence_selection_v48"
CAPABILITY = "full_paper_paragraph_and_float_evidence_selection"
PROTOCOL_SHA256 = "34c96fda996ab635911a3736b6d0df56598b79f87ed671c1f0050a366f475501"
OFFICIAL_REPOSITORY_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
OFFICIAL_ARCHIVE_SHA256 = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
OFFICIAL_ARCHIVE_BYTES = 10_835_856
OFFICIAL_DEV_MEMBER = "qasper-dev-v0.3.json"

TOP2_PROPOSAL_GUARDED_V48 = "top2_proposal_guarded_frc_v48"
FRC_CONTROLS = (*V46_FRC_CONTROLS, CONSENSUS_GUARDED_V46)
METHODS = (*V46_METHODS, TOP2_PROPOSAL_GUARDED_V48)
SAMPLE_SALT = "FRC-QASPER-V48|"
TARGET_CASES = 600
MINIMUM_CASES = 500
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260808
MINIMUM_STRATUM_CASES = 60

_FORBIDDEN_BLIND_KEYS = {
    "answer",
    "answers",
    "annotation_id",
    "worker_id",
    "unanswerable",
    "extractive_spans",
    "free_form_answer",
    "yes_no",
    "evidence",
    "highlighted_evidence",
    "gold",
    "gold_reference_ids",
    "gold_count",
    "answer_type",
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_nested_keys(item))
    return keys


def _contains_forbidden_blind_key(value: Any) -> bool:
    return bool(_nested_keys(value) & _FORBIDDEN_BLIND_KEYS)


def _token_count(tokenizer: Any, text: str) -> int:
    return max(1, len(tokenizer.encode(text, add_special_tokens=False)))


class _CharacterTokenizer:
    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return list(text)


def _parallel_records(value: Any) -> list[dict[str, Any]]:
    """Accept raw list records and the HF loader's dict-of-lists form."""
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []
    lengths = [len(item) for item in value.values() if isinstance(item, list)]
    if not lengths:
        return []
    size = min(lengths)
    return [
        {
            key: item[index] if isinstance(item, list) and index < len(item) else item
            for key, item in value.items()
        }
        for index in range(size)
    ]


def _sections(paper: dict[str, Any]) -> list[dict[str, Any]]:
    raw = paper.get("full_text")
    if isinstance(raw, dict):
        names = raw.get("section_name", [])
        paragraphs = raw.get("paragraphs", [])
        size = min(len(names), len(paragraphs)) if isinstance(names, list) and isinstance(paragraphs, list) else 0
        return [
            {"section_name": names[index], "paragraphs": paragraphs[index]}
            for index in range(size)
        ]
    return _parallel_records(raw)


def _floats(paper: dict[str, Any]) -> list[dict[str, Any]]:
    return _parallel_records(paper.get("figures_and_tables"))


def _questions(paper: dict[str, Any]) -> list[dict[str, Any]]:
    return _parallel_records(paper.get("qas"))


def _answers(question: dict[str, Any]) -> list[dict[str, Any]]:
    return _parallel_records(question.get("answers"))


def _answer_payload(annotation: dict[str, Any]) -> dict[str, Any]:
    raw = annotation.get("answer", annotation)
    return dict(raw) if isinstance(raw, dict) else {}


def build_candidate_units(
    paper: dict[str, Any], tokenizer: Any
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for section_index, section in enumerate(_sections(paper)):
        section_name = _normalise(section.get("section_name"))
        paragraphs = section.get("paragraphs")
        if not isinstance(paragraphs, list):
            continue
        for paragraph_index, raw_paragraph in enumerate(paragraphs):
            content = _normalise(raw_paragraph)
            if not content:
                continue
            text = f"section {section_name}: {content}" if section_name else content
            units.append(
                {
                    "canonical_id": f"paragraph_{section_index}_{paragraph_index}",
                    "source_kind": "paragraph",
                    "content": content,
                    "text": text,
                    "token_count": _token_count(tokenizer, text),
                }
            )
    for index, item in enumerate(_floats(paper)):
        caption = _normalise(item.get("caption"))
        if not caption:
            continue
        text = f"figure or table caption: {caption}"
        units.append(
            {
                "canonical_id": f"float_{index}",
                "source_kind": "float",
                "content": caption,
                "text": text,
                "token_count": _token_count(tokenizer, text),
            }
        )
    return units


def _evidence_aliases(unit: dict[str, Any]) -> tuple[str, ...]:
    content = _normalise(unit.get("content"))
    if unit.get("source_kind") == "float":
        return (content, _normalise(f"FLOAT SELECTED: {content}"))
    return (content,)


def _evidence_index(units: Sequence[dict[str, Any]]) -> tuple[dict[str, str], bool]:
    index: dict[str, set[str]] = {}
    primary: dict[str, set[str]] = {}
    for unit in units:
        candidate_id = str(unit["canonical_id"])
        content = _normalise(unit.get("content"))
        primary.setdefault(content, set()).add(candidate_id)
        for alias in _evidence_aliases(unit):
            index.setdefault(alias, set()).add(candidate_id)
    ambiguous = any(len(values) != 1 for values in primary.values()) or any(
        len(values) != 1 for values in index.values()
    )
    return (
        {key: next(iter(values)) for key, values in index.items() if len(values) == 1},
        ambiguous,
    )


def _answer_type(answer: dict[str, Any]) -> str:
    yes_no = answer.get("yes_no")
    if isinstance(yes_no, bool):
        return "yes_no"
    spans = answer.get("extractive_spans")
    if isinstance(spans, list) and any(_normalise(value) for value in spans):
        return "extractive"
    if _normalise(answer.get("free_form_answer")):
        return "free_form"
    return "unknown"


def mapped_answer_references(
    question: dict[str, Any], units: Sequence[dict[str, Any]]
) -> tuple[tuple[tuple[str, ...], ...], str, bool]:
    index, ambiguous = _evidence_index(units)
    if ambiguous:
        return (), "unknown", False
    references: set[tuple[str, ...]] = set()
    answer_types: set[str] = set()
    for annotation in _answers(question):
        answer = _answer_payload(annotation)
        if bool(answer.get("unanswerable")):
            continue
        evidence = answer.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            continue
        mapped: list[str] = []
        for raw in evidence:
            key = _normalise(raw)
            if not key or key not in index:
                return (), "unknown", False
            mapped.append(index[key])
        reference = tuple(sorted(set(mapped)))
        if reference:
            references.add(reference)
            answer_types.add(_answer_type(answer))
    answer_type = next(iter(answer_types)) if len(answer_types) == 1 else "mixed_or_unknown"
    return tuple(sorted(references)), answer_type, bool(references)


def _iter_question_rows(
    papers: Iterable[tuple[str, dict[str, Any]]] | dict[str, Any],
) -> Iterator[dict[str, Any]]:
    items = papers.items() if isinstance(papers, dict) else papers
    for raw_paper_id, raw_paper in items:
        if not isinstance(raw_paper, dict):
            continue
        paper_id = _normalise(raw_paper_id)
        paper = dict(raw_paper)
        for question in _questions(paper):
            yield {
                "paper_id": paper_id,
                "paper": paper,
                "question": question,
                "question_id": _normalise(question.get("question_id", question.get("id"))),
            }


def _eligibility(row: dict[str, Any]) -> tuple[bool, str]:
    if not row["paper_id"] or not row["question_id"]:
        return False, "missing_id"
    if not _normalise(row["question"].get("question")):
        return False, "missing_question"
    units = build_candidate_units(row["paper"], _CharacterTokenizer())
    if len(units) < 3:
        return False, "fewer_than_three_candidates"
    references, _answer_type_value, complete = mapped_answer_references(
        row["question"], units
    )
    if not complete or not references:
        return False, "missing_unambiguous_evidence_reference"
    return True, "eligible"


def _sample_key(row: dict[str, Any]) -> tuple[str, str]:
    question_id = str(row["question_id"])
    return hashlib.sha256((SAMPLE_SALT + question_id).encode()).hexdigest(), question_id


def select_sample(
    papers: Iterable[tuple[str, dict[str, Any]]] | dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total = 0
    excluded: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _iter_question_rows(papers):
        total += 1
        question_id = str(row["question_id"])
        if question_id in seen:
            raise ValueError("QASPER question id is not unique")
        if question_id:
            seen.add(question_id)
        is_eligible, reason = _eligibility(row)
        if not is_eligible:
            excluded[reason] += 1
            continue
        eligible.append(row)
    eligible.sort(key=_sample_key)
    selected = eligible[:TARGET_CASES]
    checks = {"minimum_cases_met": len(selected) >= MINIMUM_CASES}
    return selected, {
        "total_questions": total,
        "eligible_questions": len(eligible),
        "selected_questions": len(selected),
        "excluded": dict(sorted(excluded.items())),
        "checks": checks,
        "ready_for_blind_preparation": all(checks.values()),
        "sample_order_commitment": canonical_json_sha256(
            [hashlib.sha256(str(row["question_id"]).encode()).hexdigest() for row in selected]
        ),
    }


def prepare_blind_cases(
    papers: Iterable[tuple[str, dict[str, Any]]] | dict[str, Any], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected, sampling = select_sample(papers)
    if not sampling["ready_for_blind_preparation"]:
        return [], [], {
            "sampling": sampling,
            "prepared_cases": 0,
            "ready_for_query_generation": False,
            "gold_fields_exported_to_blind_cache": False,
        }
    prepared: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    pool_sizes: list[int] = []
    token_costs: list[int] = []
    for row in selected:
        question_id = str(row["question_id"])
        paper_id = str(row["paper_id"])
        units = build_candidate_units(row["paper"], tokenizer)
        identifiers = [str(unit["canonical_id"]) for unit in units]
        if len(set(identifiers)) != len(identifiers):
            raise AssertionError("QASPER candidate ids are not unique")
        references, _type, complete = mapped_answer_references(row["question"], units)
        if not complete or not references:
            raise AssertionError("QASPER evidence mapping changed after sampling")
        case_id = "q" + hashlib.sha256((SAMPLE_SALT + question_id).encode()).hexdigest()[:20]
        paper_commitment = hashlib.sha256(paper_id.encode()).hexdigest()
        candidates = [
            {
                "id": str(unit["canonical_id"]),
                "source_id": str(unit["canonical_id"]),
                "source_kind": str(unit["source_kind"]),
                "text": str(unit["text"]),
                "token_count": int(unit["token_count"]),
            }
            for unit in units
        ]
        blind = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "id": case_id,
            "paper_id_sha256": paper_commitment,
            "query": _normalise(row["question"].get("question")),
            "candidates": candidates,
            "gold_fields_visible_to_generator": False,
            "gold_fields_visible_to_scorer": False,
        }
        if _contains_forbidden_blind_key(blind):
            raise AssertionError("QASPER gold field leaked into blind cache")
        prepared.append(blind)
        maps.append(
            {
                "schema_version": "frc-qasper-v48-candidate-map-v1",
                "id": case_id,
                "question_id_sha256": hashlib.sha256(question_id.encode()).hexdigest(),
                "paper_id_sha256": paper_commitment,
                "units": [
                    {
                        "candidate_id": str(unit["canonical_id"]),
                        "canonical_id": str(unit["canonical_id"]),
                        "source_kind": str(unit["source_kind"]),
                        "content_sha256": hashlib.sha256(str(unit["content"]).encode()).hexdigest(),
                        "text": str(unit["text"]),
                        "text_sha256": hashlib.sha256(str(unit["text"]).encode()).hexdigest(),
                        "token_count": int(unit["token_count"]),
                    }
                    for unit in units
                ],
            }
        )
        pool_sizes.append(len(candidates))
        token_costs.extend(int(item["token_count"]) for item in candidates)

    def distribution(values: Sequence[int]) -> dict[str, float | int]:
        return {
            "minimum": min(values) if values else 0,
            "mean": round(float(np.mean(values)), 6) if values else 0.0,
            "maximum": max(values) if values else 0,
        }

    boundaries = [round(float(value), 6) for value in np.quantile(pool_sizes, [0.25, 0.5, 0.75])]
    return prepared, maps, {
        "sampling": sampling,
        "prepared_cases": len(prepared),
        "candidate_pool_size": distribution(pool_sizes),
        "candidate_pool_quartile_boundaries": boundaries,
        "candidate_token_cost": distribution(token_costs),
        "ready_for_query_generation": len(prepared) == len(selected),
        "gold_fields_exported_to_blind_cache": False,
    }


class FrozenQasperScorer(FrozenTatqaScorer):
    """The unchanged neural scorer with QASPER metadata."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def _pool_quartile(size: int, boundaries: Sequence[float]) -> str:
    if size <= boundaries[0]:
        return "q1"
    if size <= boundaries[1]:
        return "q2"
    if size <= boundaries[2]:
        return "q3"
    return "q4"


def build_gold_rows(
    source_papers: Iterable[tuple[str, dict[str, Any]]] | dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    *,
    pool_quartile_boundaries: Sequence[float],
) -> list[dict[str, Any]]:
    source: dict[str, dict[str, Any]] = {}
    for row in _iter_question_rows(source_papers):
        commitment = hashlib.sha256(str(row["question_id"]).encode()).hexdigest()
        if commitment in source:
            raise ValueError("QASPER question commitment is not unique")
        source[commitment] = row
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        row = source.get(str(candidate_map["question_id_sha256"]))
        if row is None:
            raise ValueError("QASPER candidate map question is missing from source")
        units = build_candidate_units(row["paper"], _CharacterTokenizer())
        references, answer_type, complete = mapped_answer_references(row["question"], units)
        candidate_by_canonical = {
            str(unit["canonical_id"]): str(unit["candidate_id"])
            for unit in candidate_map["units"]
        }
        if not complete or any(not set(reference) <= set(candidate_by_canonical) for reference in references):
            raise ValueError("QASPER alternative evidence mapping is incomplete")
        mapped_references = [
            [candidate_by_canonical[value] for value in reference]
            for reference in references
        ]
        smallest = min(len(reference) for reference in mapped_references)
        all_ids = {value for reference in references for value in reference}
        pool_size = len(candidate_by_canonical)
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "question_id": str(row["question_id"]),
                "gold_references": mapped_references,
                "reference_count": len(mapped_references),
                "reference_count_group": "one" if len(mapped_references) == 1 else "multiple",
                "smallest_reference_count": smallest,
                "smallest_reference_count_group": "one" if smallest == 1 else "multiple",
                "evidence_source_mode": "includes_float" if any(value.startswith("float_") for value in all_ids) else "paragraph_only",
                "answer_type": answer_type,
                "candidate_unit_count": pool_size,
                "candidate_pool_quartile": _pool_quartile(pool_size, pool_quartile_boundaries),
                "candidate_ceiling_complete": True,
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]],
    sampling: dict[str, Any],
    pool_quartile_boundaries: Sequence[float],
) -> dict[str, Any]:
    ceiling = float(np.mean([bool(row["candidate_ceiling_complete"]) for row in gold_rows])) if gold_rows else 0.0
    checks = {
        "minimum_cases_met": len(gold_rows) >= MINIMUM_CASES,
        "candidate_ceiling_complete_rate_equals_1": ceiling == 1.0,
    }
    return {
        "schema_version": "frc-qasper-v48-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "cases": len(gold_rows),
        "candidate_ceiling_complete_rate": round(ceiling, 6),
        "smallest_reference_count_groups": dict(Counter(row["smallest_reference_count_group"] for row in gold_rows)),
        "reference_count_groups": dict(Counter(row["reference_count_group"] for row in gold_rows)),
        "evidence_source_modes": dict(Counter(row["evidence_source_mode"] for row in gold_rows)),
        "answer_types": dict(Counter(row["answer_type"] for row in gold_rows)),
        "candidate_pool_quartiles": dict(Counter(row["candidate_pool_quartile"] for row in gold_rows)),
        "candidate_pool_quartile_boundaries": list(pool_quartile_boundaries),
        "sampling": sampling,
        "checks": checks,
        "minimum_cases_and_ceiling_checks_passed": all(checks.values()),
        "candidate_method_or_threshold_changed": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }


def top2_proposal_details(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rankings: dict[str, list[str]] = {}
    for role in DYNAMIC_ROLES:
        ordered = sorted(
            candidates,
            key=lambda item: (-float(item["dynamic_role_scores"][role]), str(item["id"])),
        )
        rankings[role] = [str(item["id"]) for item in ordered]
    proposal_ids = sorted({value for ranking in rankings.values() for value in ranking[:2]})
    target = min(5, max(1, len(proposal_ids))) if candidates else 0
    return {
        "role_top2_ids": {role: values[:2] for role, values in rankings.items()},
        "proposal_ids": proposal_ids,
        "proposal_count": len(proposal_ids),
        "target_cardinality": target,
    }


def select_v48(
    candidates: Sequence[dict[str, Any]], method: str, *, token_budget: int
) -> list[dict[str, Any]]:
    if method != TOP2_PROPOSAL_GUARDED_V48:
        return select_v46(candidates, method, token_budget=token_budget)
    details = top2_proposal_details(candidates)
    proposal_ids = set(details["proposal_ids"])
    proposals = [item for item in candidates if str(item["id"]) in proposal_ids]
    return select_candidates(proposals, DYNAMIC_RANK, token_budget=token_budget)


def _selection_metrics(
    selected: Sequence[dict[str, Any]], references: Sequence[Sequence[str]]
) -> dict[str, float | int | bool | list[str]]:
    selected_ids = {str(item["id"]) for item in selected}
    scored: list[tuple[float, tuple[str, ...], float, float, int]] = []
    for raw_reference in references:
        reference = tuple(sorted({str(value) for value in raw_reference}))
        gold = set(reference)
        hits = len(selected_ids & gold)
        precision = hits / len(selected_ids) if selected_ids else 1.0
        recall = hits / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scored.append((f1, reference, precision, recall, hits))
    if not scored:
        raise ValueError("QASPER case has no alternative gold reference")
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
    complete = any(set(reference) <= selected_ids for reference in references)
    return {
        "precision": best[2],
        "recall": best[3],
        "f1": best[0],
        "complete_recall": complete,
        "selected_unit_count": len(selected),
        "selected_token_cost": sum(int(item["token_count"]) for item in selected),
        "gold_unit_hits": best[4],
        "matched_reference": list(best[1]),
    }


def _aggregate(
    rows: Sequence[dict[str, Any]], method: str, budgets: Sequence[int] = BUDGETS
) -> dict[str, float]:
    metrics = [
        row["configurations"][str(budget)]["methods"][method]["metrics"]
        for row in rows
        for budget in budgets
    ]
    return {
        "evidence_macro_f1": float(np.mean([item["f1"] for item in metrics])),
        "macro_precision": float(np.mean([item["precision"] for item in metrics])),
        "macro_recall": float(np.mean([item["recall"] for item in metrics])),
        "complete_reference_recall": float(np.mean([item["complete_recall"] for item in metrics])),
        "mean_selected_unit_count": float(np.mean([item["selected_unit_count"] for item in metrics])),
        "mean_selected_token_cost": float(np.mean([item["selected_token_cost"] for item in metrics])),
    }


def _comparison(point: float, values: np.ndarray) -> dict[str, float]:
    return {
        "point": round(float(point), 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
    }


def _paired_bootstrap(rows: Sequence[dict[str, Any]], *, resamples: int) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    versus_frc = np.empty(resamples, dtype=float)
    versus_non_frc = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = [rows[int(value)] for value in rng.integers(0, len(rows), size=len(rows))]
        candidate = _aggregate(sample, TOP2_PROPOSAL_GUARDED_V48)["evidence_macro_f1"]
        versus_frc[index] = candidate - max(_aggregate(sample, method)["evidence_macro_f1"] for method in FRC_CONTROLS)
        versus_non_frc[index] = candidate - max(_aggregate(sample, method)["evidence_macro_f1"] for method in NON_FRC_BASELINES)
    candidate = _aggregate(rows, TOP2_PROPOSAL_GUARDED_V48)["evidence_macro_f1"]
    return {
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "v48_minus_strongest_frozen_frc_control": _comparison(
            candidate - max(_aggregate(rows, method)["evidence_macro_f1"] for method in FRC_CONTROLS), versus_frc
        ),
        "v48_minus_strongest_non_frc": _comparison(
            candidate - max(_aggregate(rows, method)["evidence_macro_f1"] for method in NON_FRC_BASELINES), versus_non_frc
        ),
    }


def _set_difference_rate(rows: Sequence[dict[str, Any]], reference: str) -> float:
    values = [
        set(row["configurations"][str(budget)]["methods"][TOP2_PROPOSAL_GUARDED_V48]["selected_ids"])
        != set(row["configurations"][str(budget)]["methods"][reference]["selected_ids"])
        for row in rows
        for budget in BUDGETS
    ]
    return float(np.mean(values)) if values else 0.0


def _stratum_delta(rows: Sequence[dict[str, Any]]) -> tuple[str, float]:
    strongest = max(
        NON_FRC_BASELINES,
        key=lambda method: (_aggregate(rows, method)["evidence_macro_f1"], method),
    )
    delta = _aggregate(rows, TOP2_PROPOSAL_GUARDED_V48)["evidence_macro_f1"] - _aggregate(rows, strongest)["evidence_macro_f1"]
    return strongest, round(delta, 6)


def evaluate_qasper(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not gold_rows or len(gold_rows) != len(scored_rows):
        raise ValueError("QASPER gold and scored rows are empty or misaligned")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("QASPER gold and scored case ids differ")
        candidates = merge_scored_candidates(dict(scored))
        proposal = top2_proposal_details(candidates)
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v48(candidates, method, token_budget=budget)
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(selected, gold["gold_references"]),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "smallest_reference_count_group": str(gold["smallest_reference_count_group"]),
                "reference_count_group": str(gold["reference_count_group"]),
                "evidence_source_mode": str(gold["evidence_source_mode"]),
                "answer_type": str(gold["answer_type"]),
                "candidate_pool_quartile": str(gold["candidate_pool_quartile"]),
                "candidate_ceiling_complete": bool(gold["candidate_ceiling_complete"]),
                "candidate_unit_count": len(candidates),
                "proposal": proposal,
                "target_cardinality_group": "below_five" if int(proposal["target_cardinality"]) < 5 else "five",
                "configurations": configurations,
            }
        )
    aggregates = {method: _aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = max(NON_FRC_BASELINES, key=lambda method: (aggregates[method]["evidence_macro_f1"], method))
    strongest_frc = max(FRC_CONTROLS, key=lambda method: (aggregates[method]["evidence_macro_f1"], method))
    comparisons = _paired_bootstrap(evidence, resamples=resamples)
    target_below = float(np.mean([int(row["proposal"]["target_cardinality"]) < 5 for row in evidence]))
    difference_dynamic = _set_difference_rate(evidence, DYNAMIC_RANK)
    difference_v46 = _set_difference_rate(evidence, CONSENSUS_GUARDED_V46)
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        strongest = max(NON_FRC_BASELINES, key=lambda method: (_aggregate(evidence, method, (budget,))["evidence_macro_f1"], method))
        budget_deltas[str(budget)] = round(
            _aggregate(evidence, TOP2_PROPOSAL_GUARDED_V48, (budget,))["evidence_macro_f1"]
            - _aggregate(evidence, strongest, (budget,))["evidence_macro_f1"], 6
        )
    strata: dict[str, Any] = {}
    for field, values in (
        ("smallest_reference_count_group", ("one", "multiple")),
        ("reference_count_group", ("one", "multiple")),
        ("evidence_source_mode", ("paragraph_only", "includes_float")),
        ("answer_type", ("extractive", "free_form", "yes_no")),
        ("candidate_pool_quartile", ("q1", "q2", "q3", "q4")),
        ("target_cardinality_group", ("below_five", "five")),
    ):
        for value in values:
            subset = [row for row in evidence if row[field] == value]
            if len(subset) < MINIMUM_STRATUM_CASES:
                continue
            strongest, delta = _stratum_delta(subset)
            strata[f"{field}:{value}"] = {"cases": len(subset), "strongest_non_frc": strongest, "delta": delta}
    ceiling = float(np.mean([bool(row["candidate_ceiling_complete"]) for row in evidence]))
    unit_reduction = aggregates[DYNAMIC_RANK]["mean_selected_unit_count"] - aggregates[TOP2_PROPOSAL_GUARDED_V48]["mean_selected_unit_count"]
    recall_drop = aggregates[DYNAMIC_RANK]["macro_recall"] - aggregates[TOP2_PROPOSAL_GUARDED_V48]["macro_recall"]
    safety = [*budget_deltas.values(), *(float(item["delta"]) for item in strata.values())]
    frc = comparisons["v48_minus_strongest_frozen_frc_control"]
    non_frc = comparisons["v48_minus_strongest_non_frc"]
    checks = {
        "minimum_cases_met": len(evidence) >= MINIMUM_CASES,
        "candidate_ceiling_complete_rate_equals_1": ceiling == 1.0,
        "query_parser_fallback_rate_at_most_0_05": float(query_summary.get("fallback_rate", 1.0)) <= 0.05,
        "adaptive_target_below_five_rate_at_least_0_05": target_below >= 0.05,
        "adaptive_target_below_five_rate_at_most_0_95": target_below <= 0.95,
        "selection_set_difference_rate_vs_dynamic_at_least_0_05": difference_dynamic >= 0.05,
        "selection_set_difference_rate_vs_v46_at_least_0_05": difference_v46 >= 0.05,
        "v48_minus_strongest_frc_point_at_least_0_005": frc["point"] >= 0.005,
        "v48_minus_strongest_frc_ci_low_above_0": frc["ci_low"] > 0,
        "v48_minus_strongest_non_frc_point_at_least_0_01": non_frc["point"] >= 0.01,
        "v48_minus_strongest_non_frc_ci_low_above_0": non_frc["ci_low"] > 0,
        "every_budget_and_supported_stratum_delta_at_least_minus_0_02": min(safety, default=-1.0) >= -0.02,
        "mean_selected_unit_reduction_at_least_0_25": unit_reduction >= 0.25,
        "evidence_recall_drop_at_most_0_01": recall_drop <= 0.01,
    }
    if not (checks["minimum_cases_met"] and checks["candidate_ceiling_complete_rate_equals_1"]):
        status = "QASPER_FULL_PAPER_POOL_INCONCLUSIVE"
    elif all(checks.values()):
        status = "QASPER_TOP2_PROPOSAL_GUARDED_SUPPORT_ESTABLISHED_ON_FULL_PAPER_POOL"
    else:
        status = "QASPER_TOP2_PROPOSAL_GUARDED_SUPPORT_NOT_ESTABLISHED"
    proposal_counts = Counter(int(row["proposal"]["proposal_count"]) for row in evidence)
    targets = Counter(int(row["proposal"]["target_cardinality"]) for row in evidence)
    report = {
        "schema_version": "frc-qasper-top2-proposal-guarded-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "cases": len(evidence),
            "budgets": list(BUDGETS),
            "official_leaderboard_result": False,
            "full_paper_bounded_pool": True,
            "multi_reference_gold_not_unioned": True,
            "gold_joined_after_complete_score_cache": True,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "aggregates": aggregates,
            "strongest_non_frc": strongest_non_frc,
            "strongest_frozen_frc_control": strongest_frc,
            "family_comparison": comparisons,
            "candidate_ceiling_complete_rate": round(ceiling, 6),
            "adaptive_target_below_five_rate": round(target_below, 6),
            "selection_set_difference_rate_vs_dynamic": round(difference_dynamic, 6),
            "selection_set_difference_rate_vs_v46": round(difference_v46, 6),
            "mean_selected_unit_reduction_vs_dynamic": round(unit_reduction, 6),
            "evidence_recall_drop_vs_dynamic": round(recall_drop, 6),
            "proposal_union_size_distribution": {str(key): proposal_counts[key] for key in sorted(proposal_counts)},
            "target_cardinality_distribution": {str(key): targets[key] for key in sorted(targets)},
            "budget_deltas": budget_deltas,
            "supported_stratum_deltas": strata,
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "reuse_v48_cases_for_tuning_or_selection": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
        "limitations": [
            "QASPER supplies an attached full paper rather than open-corpus retrieval.",
            "Evidence is evaluated at paragraph or figure/table-caption granularity.",
            "The endpoint is evidence selection, not answer generation.",
            "This cannot establish SetR reproduction, flood-domain validity, or production readiness.",
        ],
    }
    return report, evidence


def _round_for_display(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_for_display(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_for_display(item) for item in value]
    return value


def write_report(
    report: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    *,
    json_path: Path,
    markdown_path: Path,
    evidence_path: Path,
) -> None:
    rounded = _round_for_display(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rounded, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    analysis = rounded["analysis"]
    lines = [
        "# QASPER top-2 proposal-guarded atomic-role evaluation (v48)", "",
        f"- Status: `{analysis['outcome']['status']}`",
        f"- Cases: {rounded['metadata']['cases']}",
        f"- Strongest frozen FRC control: `{analysis['strongest_frozen_frc_control']}`",
        f"- Strongest non-FRC: `{analysis['strongest_non_frc']}`",
        f"- Target-below-five rate: {analysis['adaptive_target_below_five_rate']:.6f}",
        "- Gate 2: `NO-GO/SHADOW`", "", "## Aggregate methods", "",
        "| Method | F1 | Precision | Recall | Complete | Units | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in analysis["aggregates"].items():
        lines.append(
            f"| `{method}` | {values['evidence_macro_f1']:.6f} | {values['macro_precision']:.6f} | "
            f"{values['macro_recall']:.6f} | {values['complete_reference_recall']:.6f} | "
            f"{values['mean_selected_unit_count']:.6f} | {values['mean_selected_token_cost']:.6f} |"
        )
    lines.extend(["", "## Boundary", "", "This is a full-paper evidence-selection experiment, not an official QASPER answer-generation leaderboard result. Alternative answer-level references are not unioned, and v48 cases may not be reused for tuning or method selection.", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in evidence:
                compressed.write((json.dumps(_round_for_display(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def read_source_papers(source_archive: Path) -> dict[str, dict[str, Any]]:
    with tarfile.open(source_archive, mode="r:gz") as archive:
        matches = [member for member in archive.getmembers() if Path(member.name).name == OFFICIAL_DEV_MEMBER]
        if len(matches) != 1:
            raise ValueError("QASPER archive must contain one registered dev JSON member")
        handle = archive.extractfile(matches[0])
        if handle is None:
            raise ValueError("QASPER dev member is not a regular file")
        value = json.load(handle)
    if not isinstance(value, dict) or not value or not all(isinstance(item, dict) for item in value.values()):
        raise ValueError("QASPER dev source schema mismatch")
    return {str(key): dict(item) for key, item in value.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("QASPER v48 protocol hash mismatch")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("QASPER v48 experiment mismatch")
    if value.get("methods", {}).get("candidate_method") != TOP2_PROPOSAL_GUARDED_V48:
        raise ValueError("QASPER v48 candidate method mismatch")
    if value.get("methods", {}).get("token_budgets") != list(BUDGETS):
        raise ValueError("QASPER v48 budget registration mismatch")
    if value.get("development_boundary", {}).get("qasper_train_split_permanently_excluded") is not True:
        raise ValueError("QASPER v48 train exclusion is missing")
    return value


def validate_implementation_registration(
    registration_path: Path, *, protocol_path: Path, module_path: Path, runner_path: Path, test_path: Path
) -> dict[str, Any]:
    validate_protocol(protocol_path)
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "frc-qasper-top2-proposal-guarded-atomic-roles-implementation-v48":
        raise ValueError("QASPER v48 implementation schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("QASPER v48 implementation experiment mismatch")
    if value.get("validation_member_opened_before_registration") is not False:
        raise ValueError("QASPER v48 crossed the validation-content boundary")
    expected = {
        "protocol_sha256": _sha256(protocol_path),
        "module_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "test_sha256": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("QASPER v48 implementation hash mismatch")
    invariants = value.get("synthetic_invariants_verified")
    if not isinstance(invariants, dict) or not all(invariants.values()):
        raise ValueError("QASPER v48 synthetic invariants are incomplete")
    if value.get("method_or_threshold_change_after_registration_forbidden") is not True:
        raise ValueError("QASPER v48 implementation is not frozen")
    return value


def validate_execution_registration(
    execution_path: Path, *, implementation_path: Path, source_archive_path: Path,
    source_card_path: Path, prepared_path: Path, candidate_map_path: Path,
    census_path: Path, coverage_path: Path,
) -> dict[str, Any]:
    value = json.loads(execution_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "frc-qasper-top2-proposal-guarded-atomic-roles-execution-v48":
        raise ValueError("QASPER v48 execution schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("QASPER v48 execution experiment mismatch")
    expected = {
        "implementation_registration_sha256": _sha256(implementation_path),
        "source_archive_sha256": _sha256(source_archive_path),
        "source_card_sha256": _sha256(source_card_path),
        "prepared_blind_sha256": _sha256(prepared_path),
        "candidate_map_sha256": _sha256(candidate_map_path),
        "blind_census_sha256": _sha256(census_path),
        "candidate_coverage_sha256": _sha256(coverage_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("QASPER v48 execution artifact hash mismatch")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("minimum_cases_and_ceiling_checks_passed") is not True:
        raise ValueError("QASPER v48 coverage gate is not open")
    for field in ("query_generation_started", "neural_scoring_started", "metrics_computed"):
        if value.get(field) is not False:
            raise ValueError(f"QASPER v48 execution field changed: {field}")
    if value.get("negative_null_or_inconclusive_result_must_be_published") is not True:
        raise ValueError("QASPER v48 publication boundary is missing")
    return value


__all__ = [
    "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED", "EXPERIMENT_ID",
    "FrozenQasperScorer", "OFFICIAL_ARCHIVE_BYTES", "OFFICIAL_ARCHIVE_SHA256",
    "OFFICIAL_REPOSITORY_REVISION", "TOP2_PROPOSAL_GUARDED_V48",
    "build_candidate_coverage", "build_candidate_units", "build_gold_rows",
    "evaluate_qasper", "mapped_answer_references", "prepare_blind_cases",
    "read_source_papers", "select_sample", "select_v48", "top2_proposal_details",
    "validate_execution_registration", "validate_implementation_registration",
    "validate_protocol", "write_report",
]
