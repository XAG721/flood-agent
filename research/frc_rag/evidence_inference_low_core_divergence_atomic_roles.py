"""Prospective Evidence Inference 2.0 validation test for the v49 selector."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.feverous_adaptive_atomic_roles import ADAPTIVE_ARGMAX
from research.frc_rag.hover_dynamic_atomic_roles import (
    DYNAMIC_RANK,
    DYNAMIC_ROLES,
    RANK_COVERAGE_WEIGHT,
    merge_scored_candidates,
)
from research.frc_rag.qasper_top2_proposal_guarded_atomic_roles import (
    _pool_quartile,
    _round_for_display,
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


SCHEMA_VERSION = "frc-evidence-inference-low-core-divergence-atomic-roles-v49"
EXPERIMENT_ID = (
    "FRC-EVIDENCE-INFERENCE-LOW-CORE-DIVERGENCE-GUARDED-ATOMIC-ROLES-V49"
)
DATASET_ID = "evidence_inference_v2_validation_full_article_sentence_selection_v49"
CAPABILITY = "full_article_sentence_evidence_selection"
PROTOCOL_SHA256 = "e937f230bbb4a359cc2787edd6f33e90e5e6b084c5b7f14a8ca9f4f58c3d5d0c"
OFFICIAL_REPOSITORY_REVISION = "a661e8c14f973398380c8865cf2f27a535aaaf6d"
BIGBIO_REPOSITORY_REVISION = "35dce6aba1b3eb9eb9af9bdc38ebeda73dad15b9"
OFFICIAL_ARCHIVE_BYTES = 36_528_800

LOW_CORE_DIVERGENCE_V49 = "low_core_divergence_guarded_frc_v49"
FRC_CONTROLS = (*V46_FRC_CONTROLS, CONSENSUS_GUARDED_V46)
METHODS = (*V46_METHODS, LOW_CORE_DIVERGENCE_V49)
SAMPLE_SALT = "FRC-EVIDENCE-INFERENCE-V49|"
TARGET_CASES = 600
MINIMUM_CASES = 500
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260809
MINIMUM_STRATUM_CASES = 60

# Frozen verbatim from the original repository caveat and the BigBio loader.
SKIP_PROMPT_IDS = frozenset(
    {
        346,
        911,
        912,
        1261,
        1262,
        3044,
        3111,
        3248,
        3514,
        3620,
        4308,
        4324,
        4325,
        4490,
        4491,
        4492,
        4715,
        4807,
        4824,
        4948,
        5000,
        5001,
        5002,
        5037,
        5046,
        5047,
        5639,
        5710,
        5752,
        5775,
        5776,
        5777,
        5778,
        5779,
        5780,
        5781,
        5782,
        5841,
        5843,
        5861,
        5862,
        5863,
        5964,
        5965,
        5966,
        5975,
        6034,
        6065,
        6066,
        6666,
        6667,
        6668,
        6669,
        7040,
        7042,
        7811,
        7812,
        7813,
        7814,
        7815,
        7944,
        8118,
        8197,
        8198,
        8199,
        8200,
        8201,
        8536,
        8590,
        8593,
        8605,
        8606,
        8631,
        8635,
        8639,
        8640,
        8745,
        8747,
        8749,
        8767,
        8773,
        8870,
        8875,
        8876,
        8877,
        8878,
        8884,
        8885,
        8886,
        8917,
        8921,
        9295,
        9297,
        9429,
        9430,
        9431,
        9432,
        9862,
        10032,
        10035,
        10885,
        10886,
        10887,
        10888,
        10889,
        10890,
    }
)

_SENTENCE_BOUNDARY = re.compile(r"(?:[.!?][\"')\]]*|\n)\s+")
_FORBIDDEN_BLIND_KEYS = {
    "promptid",
    "pmcid",
    "userid",
    "label",
    "classification",
    "in abstract",
    "in_abstract",
    "annotations",
    "evidence start",
    "evidence end",
    "evidence_start",
    "evidence_end",
    "gold",
    "gold_references",
    "gold_count",
    "annotator_count",
}


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).lower())
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


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _integer(value: Any) -> int | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _truthy(value: Any) -> bool:
    return _normalise(value).lower() in {"1", "1.0", "true", "yes"}


def _field(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def build_sentence_units(article_text: str, tokenizer: Any) -> list[dict[str, Any]]:
    """Split complete plain text with the preregistered boundary and retain offsets."""
    units: list[dict[str, Any]] = []
    cursor = 0
    raw_spans: list[tuple[int, int]] = []
    for match in _SENTENCE_BOUNDARY.finditer(article_text):
        boundary_end = match.start() + len(match.group(0).rstrip())
        raw_spans.append((cursor, boundary_end))
        cursor = match.end()
    raw_spans.append((cursor, len(article_text)))
    for raw_start, raw_end in raw_spans:
        chunk = article_text[raw_start:raw_end]
        leading = len(chunk) - len(chunk.lstrip())
        trailing_end = len(chunk.rstrip())
        start = raw_start + leading
        end = raw_start + trailing_end
        if end <= start:
            continue
        content = _normalise(article_text[start:end])
        if not content:
            continue
        identifier = f"sentence_{len(units):04d}"
        units.append(
            {
                "canonical_id": identifier,
                "source_kind": "sentence",
                "content": content,
                "text": content,
                "start": start,
                "end": end,
                "token_count": _token_count(tokenizer, content),
            }
        )
    return units


def _reference_metadata(
    annotations: Sequence[dict[str, Any]],
    units: Sequence[dict[str, Any]],
    *,
    article_length: int,
) -> tuple[tuple[tuple[str, ...], ...], str, str, bool]:
    grouped: dict[str, set[str]] = defaultdict(set)
    labels: set[str] = set()
    abstract_flags: list[bool] = []
    retained = 0
    for row in annotations:
        if not (_truthy(_field(row, "Valid Label")) and _truthy(_field(row, "Valid Reasoning"))):
            continue
        label = _normalise(_field(row, "Label")).lower()
        if not label or label == "invalid prompt":
            continue
        user_id = _normalise(_field(row, "UserID", "User ID"))
        start = _integer(_field(row, "Evidence Start", "Start Evidence"))
        end_inclusive = _integer(_field(row, "Evidence End", "End Evidence"))
        if (
            not user_id
            or start is None
            or end_inclusive is None
            or start < 0
            or end_inclusive < start
            or end_inclusive >= article_length
        ):
            continue
        end = end_inclusive + 1
        mapped = {
            str(unit["canonical_id"])
            for unit in units
            if int(unit["start"]) < end and int(unit["end"]) > start
        }
        if not mapped:
            return (), "mixed_or_unknown", "unknown", False
        grouped[user_id].update(mapped)
        labels.add(label)
        abstract_flags.append(_truthy(_field(row, "In Abstract")))
        retained += 1
    references = tuple(sorted({tuple(sorted(values)) for values in grouped.values() if values}))
    label_group = next(iter(labels)) if len(labels) == 1 else "mixed_or_unknown"
    location = "all_abstract" if abstract_flags and all(abstract_flags) else "includes_non_abstract"
    return references, label_group, location, bool(references and retained)


def _prompt_id(row: dict[str, Any]) -> int | None:
    return _integer(_field(row, "PromptID", "Prompt ID"))


def _pmcid(row: dict[str, Any]) -> int | None:
    return _integer(_field(row, "PMCID", "PMC ID"))


def _query(prompt: dict[str, Any]) -> str:
    return (
        f"Compared with {_normalise(_field(prompt, 'Comparator'))}, what effect did "
        f"{_normalise(_field(prompt, 'Intervention'))} have on "
        f"{_normalise(_field(prompt, 'Outcome'))}?"
    )


def _member_by_basename(
    archive: tarfile.TarFile, basename: str
) -> tarfile.TarInfo:
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile() and Path(member.name).name == basename
    ]
    if len(matches) != 1:
        raise ValueError(f"Evidence Inference archive member mismatch: {basename}")
    return matches[0]


def _read_member_text(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"Evidence Inference member is unreadable: {member.name}")
    return handle.read().decode("utf-8-sig", errors="strict")


def _stream_filtered_csv(
    archive: tarfile.TarFile, basename: str, validation_ids: set[int]
) -> list[dict[str, str]]:
    member = _member_by_basename(archive, basename)
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"Evidence Inference CSV is unreadable: {basename}")
    rows: list[dict[str, str]] = []
    with io.TextIOWrapper(handle, encoding="utf-8-sig", newline="") as text:
        reader = csv.DictReader(text)
        if not reader.fieldnames or "PMCID" not in reader.fieldnames:
            raise ValueError(f"Evidence Inference CSV schema mismatch: {basename}")
        for raw in reader:
            row = {str(key): str(value or "") for key, value in raw.items()}
            if _pmcid(row) in validation_ids:
                rows.append(row)
    return rows


def read_validation_source(source_archive: Path) -> dict[str, Any]:
    """Read only the registered validation ids/articles while streaming shared CSVs."""
    with tarfile.open(source_archive, mode="r:gz") as archive:
        validation_member = _member_by_basename(archive, "validation_article_ids.txt")
        validation_ids = {
            value
            for raw in _read_member_text(archive, validation_member).splitlines()
            if (value := _integer(raw)) is not None
        }
        if not validation_ids:
            raise ValueError("Evidence Inference validation split is empty")
        prompts = _stream_filtered_csv(archive, "prompts_merged.csv", validation_ids)
        annotations = _stream_filtered_csv(
            archive, "annotations_merged.csv", validation_ids
        )
        articles: dict[int, str] = {}
        for pmcid in sorted(validation_ids):
            member = _member_by_basename(archive, f"PMC{pmcid}.txt")
            articles[pmcid] = _read_member_text(archive, member)
    return {
        "validation_article_ids": sorted(validation_ids),
        "prompts": prompts,
        "annotations": annotations,
        "articles": articles,
    }


def _iter_prompt_rows(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    annotations_by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in source["annotations"]:
        prompt_id = _prompt_id(annotation)
        if prompt_id is not None:
            annotations_by_prompt[prompt_id].append(annotation)
    for prompt in source["prompts"]:
        prompt_id = _prompt_id(prompt)
        pmcid = _pmcid(prompt)
        if prompt_id is None or pmcid is None:
            continue
        article = source["articles"].get(pmcid)
        if article is None:
            continue
        yield {
            "prompt_id": prompt_id,
            "pmcid": pmcid,
            "prompt": prompt,
            "annotations": annotations_by_prompt.get(prompt_id, []),
            "article": article,
        }


def _eligibility(row: dict[str, Any]) -> tuple[bool, str]:
    if int(row["prompt_id"]) in SKIP_PROMPT_IDS:
        return False, "frozen_caveat_prompt"
    if any(
        not _normalise(_field(row["prompt"], field))
        for field in ("Intervention", "Comparator", "Outcome")
    ):
        return False, "missing_ico"
    units = build_sentence_units(str(row["article"]), _CharacterTokenizer())
    if len(units) < 5:
        return False, "fewer_than_five_sentences"
    references, _label, _location, complete = _reference_metadata(
        row["annotations"], units, article_length=len(str(row["article"]))
    )
    if not complete or not references:
        return False, "missing_complete_annotator_reference"
    return True, "eligible"


def _sample_key(row: dict[str, Any]) -> tuple[str, int]:
    prompt_id = int(row["prompt_id"])
    return hashlib.sha256(f"{SAMPLE_SALT}{prompt_id}".encode()).hexdigest(), prompt_id


def select_sample(source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total = 0
    excluded: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in _iter_prompt_rows(source):
        total += 1
        prompt_id = int(row["prompt_id"])
        if prompt_id in seen:
            raise ValueError("Evidence Inference PromptID is not unique")
        seen.add(prompt_id)
        accepted, reason = _eligibility(row)
        if not accepted:
            excluded[reason] += 1
            continue
        eligible.append(row)
    eligible.sort(key=_sample_key)
    selected = eligible[:TARGET_CASES]
    checks = {"minimum_cases_met": len(selected) >= MINIMUM_CASES}
    return selected, {
        "validation_prompts": total,
        "eligible_prompts": len(eligible),
        "selected_prompts": len(selected),
        "excluded": dict(sorted(excluded.items())),
        "checks": checks,
        "ready_for_blind_preparation": all(checks.values()),
        "sample_order_commitment": canonical_json_sha256(
            [hashlib.sha256(str(row["prompt_id"]).encode()).hexdigest() for row in selected]
        ),
    }


def prepare_blind_cases(
    source: dict[str, Any], tokenizer: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selected, sampling = select_sample(source)
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
        prompt_id = int(row["prompt_id"])
        pmcid = int(row["pmcid"])
        units = build_sentence_units(str(row["article"]), tokenizer)
        references, _label, _location, complete = _reference_metadata(
            row["annotations"], units, article_length=len(str(row["article"]))
        )
        if not complete or not references:
            raise AssertionError("Evidence Inference mapping changed after sampling")
        case_id = "ei" + hashlib.sha256(f"{SAMPLE_SALT}{prompt_id}".encode()).hexdigest()[:20]
        candidates = [
            {
                "id": str(unit["canonical_id"]),
                "source_id": str(unit["canonical_id"]),
                "source_kind": "sentence",
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
            "document_id_sha256": hashlib.sha256(str(pmcid).encode()).hexdigest(),
            "query": _query(row["prompt"]),
            "candidates": candidates,
            "gold_fields_visible_to_generator": False,
            "gold_fields_visible_to_scorer": False,
        }
        if _contains_forbidden_blind_key(blind):
            raise AssertionError("Evidence Inference gold field leaked into blind cache")
        prepared.append(blind)
        maps.append(
            {
                "schema_version": "frc-evidence-inference-v49-candidate-map-v1",
                "id": case_id,
                "prompt_id_sha256": hashlib.sha256(str(prompt_id).encode()).hexdigest(),
                "document_id_sha256": hashlib.sha256(str(pmcid).encode()).hexdigest(),
                "units": [
                    {
                        "candidate_id": str(unit["canonical_id"]),
                        "canonical_id": str(unit["canonical_id"]),
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

    boundaries = [
        round(float(value), 6) for value in np.quantile(pool_sizes, [0.25, 0.5, 0.75])
    ]
    return prepared, maps, {
        "sampling": sampling,
        "prepared_cases": len(prepared),
        "validation_articles": len(source["validation_article_ids"]),
        "candidate_pool_size": distribution(pool_sizes),
        "candidate_pool_quartile_boundaries": boundaries,
        "candidate_token_cost": distribution(token_costs),
        "ready_for_query_generation": len(prepared) == len(selected),
        "gold_fields_exported_to_blind_cache": False,
    }


class FrozenEvidenceInferenceScorer(FrozenTatqaScorer):
    """The unchanged neural scorer with v49 dataset metadata."""

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
) -> list[dict[str, Any]]:
    source_by_commitment: dict[str, dict[str, Any]] = {}
    for row in _iter_prompt_rows(source):
        commitment = hashlib.sha256(str(row["prompt_id"]).encode()).hexdigest()
        if commitment in source_by_commitment:
            raise ValueError("Evidence Inference PromptID commitment is not unique")
        source_by_commitment[commitment] = row
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        row = source_by_commitment.get(str(candidate_map["prompt_id_sha256"]))
        if row is None:
            raise ValueError("Evidence Inference candidate map PromptID is absent")
        units = build_sentence_units(str(row["article"]), _CharacterTokenizer())
        references, label, location, complete = _reference_metadata(
            row["annotations"], units, article_length=len(str(row["article"]))
        )
        available = {str(item["canonical_id"]) for item in candidate_map["units"]}
        if not complete or any(not set(reference) <= available for reference in references):
            raise ValueError("Evidence Inference candidate ceiling is incomplete")
        smallest = min(len(reference) for reference in references)
        pool_size = len(available)
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "gold_references": [list(reference) for reference in references],
                "reference_count": len(references),
                "reference_count_group": "one" if len(references) == 1 else "multiple",
                "smallest_reference_count": smallest,
                "smallest_reference_count_group": "one" if smallest == 1 else "multiple",
                "evidence_location": location,
                "label_group": label,
                "candidate_unit_count": pool_size,
                "candidate_pool_quartile": _pool_quartile(
                    pool_size, pool_quartile_boundaries
                ),
                "candidate_ceiling_complete": True,
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]],
    sampling: dict[str, Any],
    pool_quartile_boundaries: Sequence[float],
) -> dict[str, Any]:
    ceiling = (
        float(np.mean([bool(row["candidate_ceiling_complete"]) for row in gold_rows]))
        if gold_rows
        else 0.0
    )
    checks = {
        "minimum_cases_met": len(gold_rows) >= MINIMUM_CASES,
        "candidate_ceiling_complete_rate_equals_1": ceiling == 1.0,
    }
    return {
        "schema_version": "frc-evidence-inference-v49-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "cases": len(gold_rows),
        "candidate_ceiling_complete_rate": round(ceiling, 6),
        "smallest_reference_count_groups": dict(
            Counter(row["smallest_reference_count_group"] for row in gold_rows)
        ),
        "reference_count_groups": dict(
            Counter(row["reference_count_group"] for row in gold_rows)
        ),
        "evidence_locations": dict(Counter(row["evidence_location"] for row in gold_rows)),
        "label_groups": dict(Counter(row["label_group"] for row in gold_rows)),
        "candidate_pool_quartiles": dict(
            Counter(row["candidate_pool_quartile"] for row in gold_rows)
        ),
        "candidate_pool_quartile_boundaries": list(pool_quartile_boundaries),
        "sampling": sampling,
        "checks": checks,
        "minimum_cases_and_ceiling_checks_passed": all(checks.values()),
        "candidate_method_or_threshold_changed": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }


def low_core_divergence_details(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rankings: dict[str, list[str]] = {}
    for role in DYNAMIC_ROLES:
        ordered = sorted(
            candidates,
            key=lambda item: (
                -float(item["dynamic_role_scores"][role]),
                str(item["id"]),
            ),
        )
        rankings[role] = [str(item["id"]) for item in ordered]
    argmax_ids = sorted({values[0] for values in rankings.values() if values})
    top2_ids = sorted(
        {identifier for values in rankings.values() for identifier in values[:2]}
    )
    noncore_ids = sorted(set(top2_ids) - set(argmax_ids))
    triggered = bool(candidates) and len(argmax_ids) <= 2 and len(noncore_ids) >= 3
    target = min(5, max(1, len(argmax_ids) + int(triggered))) if candidates else 0
    return {
        "role_top2_ids": {role: values[:2] for role, values in rankings.items()},
        "argmax_core_ids": argmax_ids,
        "top2_proposal_ids": top2_ids,
        "argmax_core_size": len(argmax_ids),
        "runner_up_surplus": len(noncore_ids),
        "expansion_triggered": triggered,
        "target_cardinality": target,
    }


def _rank_utilities(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    result = {str(candidate["id"]): {} for candidate in candidates}
    for role in DYNAMIC_ROLES:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate["dynamic_role_scores"][role]),
                str(candidate["id"]),
            ),
        )
        for rank, candidate in enumerate(ordered):
            result[str(candidate["id"])][role] = 1.0 / math.log2(2 + rank)
    return result


def _target_rank_coverage_select(
    candidates: Sequence[dict[str, Any]], *, target: int, token_budget: int
) -> list[dict[str, Any]]:
    utilities = _rank_utilities(candidates)
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    best_by_role = {role: 0.0 for role in DYNAMIC_ROLES}
    total = 0
    while remaining and len(selected) < target:
        eligible: list[tuple[float, str, dict[str, Any]]] = []
        for candidate in remaining:
            cost = int(candidate["token_count"])
            if total + cost > token_budget:
                continue
            role_gain = sum(
                max(
                    0.0,
                    utilities[str(candidate["id"])][role] - best_by_role[role],
                )
                for role in DYNAMIC_ROLES
            )
            gain = float(candidate["scores"]["cross_encoder"]) + (
                RANK_COVERAGE_WEIGHT * role_gain
            )
            eligible.append((gain, str(candidate["id"]), candidate))
        if not eligible:
            break
        eligible.sort(key=lambda item: (-item[0], item[1]))
        best = eligible[0][2]
        selected.append(best)
        remaining = [item for item in remaining if item["id"] != best["id"]]
        total += int(best["token_count"])
        for role in DYNAMIC_ROLES:
            best_by_role[role] = max(
                best_by_role[role], utilities[str(best["id"])][role]
            )
    return selected


def select_v49(
    candidates: Sequence[dict[str, Any]], method: str, *, token_budget: int
) -> list[dict[str, Any]]:
    if method != LOW_CORE_DIVERGENCE_V49:
        return select_v46(list(candidates), method, token_budget=token_budget)
    details = low_core_divergence_details(candidates)
    return _target_rank_coverage_select(
        candidates,
        target=int(details["target_cardinality"]),
        token_budget=token_budget,
    )


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
        raise ValueError("Evidence Inference case has no alternative gold reference")
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
    return {
        "precision": best[2],
        "recall": best[3],
        "f1": best[0],
        "complete_recall": any(set(reference) <= selected_ids for reference in references),
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
        "complete_reference_recall": float(
            np.mean([item["complete_recall"] for item in metrics])
        ),
        "mean_selected_unit_count": float(
            np.mean([item["selected_unit_count"] for item in metrics])
        ),
        "mean_selected_token_cost": float(
            np.mean([item["selected_token_cost"] for item in metrics])
        ),
    }


def _comparison(point: float, values: np.ndarray) -> dict[str, float]:
    return {
        "point": round(float(point), 6),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
    }


def _paired_bootstrap(
    rows: Sequence[dict[str, Any]], *, resamples: int
) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    versus_frc = np.empty(resamples, dtype=float)
    versus_non_frc = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = [
            rows[int(value)] for value in rng.integers(0, len(rows), size=len(rows))
        ]
        candidate = _aggregate(sample, LOW_CORE_DIVERGENCE_V49)["evidence_macro_f1"]
        versus_frc[index] = candidate - max(
            _aggregate(sample, method)["evidence_macro_f1"] for method in FRC_CONTROLS
        )
        versus_non_frc[index] = candidate - max(
            _aggregate(sample, method)["evidence_macro_f1"]
            for method in NON_FRC_BASELINES
        )
    candidate = _aggregate(rows, LOW_CORE_DIVERGENCE_V49)["evidence_macro_f1"]
    return {
        "resamples": resamples,
        "seed": BOOTSTRAP_SEED,
        "v49_minus_strongest_frozen_frc_control": _comparison(
            candidate
            - max(
                _aggregate(rows, method)["evidence_macro_f1"]
                for method in FRC_CONTROLS
            ),
            versus_frc,
        ),
        "v49_minus_strongest_non_frc": _comparison(
            candidate
            - max(
                _aggregate(rows, method)["evidence_macro_f1"]
                for method in NON_FRC_BASELINES
            ),
            versus_non_frc,
        ),
    }


def _set_difference_rate(rows: Sequence[dict[str, Any]], reference: str) -> float:
    values = [
        set(
            row["configurations"][str(budget)]["methods"][LOW_CORE_DIVERGENCE_V49][
                "selected_ids"
            ]
        )
        != set(
            row["configurations"][str(budget)]["methods"][reference]["selected_ids"]
        )
        for row in rows
        for budget in BUDGETS
    ]
    return float(np.mean(values)) if values else 0.0


def _stratum_delta(rows: Sequence[dict[str, Any]]) -> tuple[str, float]:
    strongest = max(
        NON_FRC_BASELINES,
        key=lambda method: (_aggregate(rows, method)["evidence_macro_f1"], method),
    )
    delta = (
        _aggregate(rows, LOW_CORE_DIVERGENCE_V49)["evidence_macro_f1"]
        - _aggregate(rows, strongest)["evidence_macro_f1"]
    )
    return strongest, round(delta, 6)


def evaluate_evidence_inference(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    query_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not gold_rows or len(gold_rows) != len(scored_rows):
        raise ValueError("Evidence Inference gold and score rows are empty or misaligned")
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("Evidence Inference gold and scored case ids differ")
        candidates = merge_scored_candidates(dict(scored))
        details = low_core_divergence_details(candidates)
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v49(candidates, method, token_budget=budget)
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(selected, gold["gold_references"]),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "smallest_reference_count_group": str(
                    gold["smallest_reference_count_group"]
                ),
                "reference_count_group": str(gold["reference_count_group"]),
                "evidence_location": str(gold["evidence_location"]),
                "label_group": str(gold["label_group"]),
                "candidate_pool_quartile": str(gold["candidate_pool_quartile"]),
                "candidate_ceiling_complete": bool(gold["candidate_ceiling_complete"]),
                "candidate_unit_count": len(candidates),
                "selector": details,
                "expansion_group": (
                    "triggered" if details["expansion_triggered"] else "not_triggered"
                ),
                "configurations": configurations,
            }
        )
    aggregates = {method: _aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = max(
        NON_FRC_BASELINES,
        key=lambda method: (aggregates[method]["evidence_macro_f1"], method),
    )
    strongest_frc = max(
        FRC_CONTROLS,
        key=lambda method: (aggregates[method]["evidence_macro_f1"], method),
    )
    comparisons = _paired_bootstrap(evidence, resamples=resamples)
    trigger_rate = float(
        np.mean([bool(row["selector"]["expansion_triggered"]) for row in evidence])
    )
    difference_v43 = _set_difference_rate(evidence, ADAPTIVE_ARGMAX)
    difference_dynamic = _set_difference_rate(evidence, DYNAMIC_RANK)
    difference_v46 = _set_difference_rate(evidence, CONSENSUS_GUARDED_V46)
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        strongest = max(
            NON_FRC_BASELINES,
            key=lambda method: (
                _aggregate(evidence, method, (budget,))["evidence_macro_f1"],
                method,
            ),
        )
        budget_deltas[str(budget)] = round(
            _aggregate(evidence, LOW_CORE_DIVERGENCE_V49, (budget,))[
                "evidence_macro_f1"
            ]
            - _aggregate(evidence, strongest, (budget,))["evidence_macro_f1"],
            6,
        )
    strata: dict[str, Any] = {}
    for field, values in (
        ("smallest_reference_count_group", ("one", "multiple")),
        ("reference_count_group", ("one", "multiple")),
        ("evidence_location", ("all_abstract", "includes_non_abstract")),
        (
            "label_group",
            (
                "significantly increased",
                "no significant difference",
                "significantly decreased",
            ),
        ),
        ("candidate_pool_quartile", ("q1", "q2", "q3", "q4")),
        ("expansion_group", ("triggered", "not_triggered")),
    ):
        for value in values:
            subset = [row for row in evidence if row[field] == value]
            if len(subset) < MINIMUM_STRATUM_CASES:
                continue
            strongest, delta = _stratum_delta(subset)
            strata[f"{field}:{value}"] = {
                "cases": len(subset),
                "strongest_non_frc": strongest,
                "delta": delta,
            }
    ceiling = float(
        np.mean([bool(row["candidate_ceiling_complete"]) for row in evidence])
    )
    unit_reduction = (
        aggregates[DYNAMIC_RANK]["mean_selected_unit_count"]
        - aggregates[LOW_CORE_DIVERGENCE_V49]["mean_selected_unit_count"]
    )
    recall_improvement = (
        aggregates[LOW_CORE_DIVERGENCE_V49]["macro_recall"]
        - aggregates[ADAPTIVE_ARGMAX]["macro_recall"]
    )
    safety = [*budget_deltas.values(), *(float(item["delta"]) for item in strata.values())]
    frc = comparisons["v49_minus_strongest_frozen_frc_control"]
    non_frc = comparisons["v49_minus_strongest_non_frc"]
    checks = {
        "minimum_cases_met": len(evidence) >= MINIMUM_CASES,
        "candidate_ceiling_complete_rate_equals_1": ceiling == 1.0,
        "query_parser_fallback_rate_at_most_0_05": float(
            query_summary.get("fallback_rate", 1.0)
        )
        <= 0.05,
        "expansion_trigger_rate_at_least_0_05": trigger_rate >= 0.05,
        "expansion_trigger_rate_at_most_0_95": trigger_rate <= 0.95,
        "selection_set_difference_rate_vs_v43_at_least_0_05": difference_v43
        >= 0.05,
        "v49_minus_strongest_frc_point_at_least_0_005": frc["point"] >= 0.005,
        "v49_minus_strongest_frc_ci_low_above_0": frc["ci_low"] > 0,
        "v49_minus_strongest_non_frc_point_at_least_0_01": non_frc["point"]
        >= 0.01,
        "v49_minus_strongest_non_frc_ci_low_above_0": non_frc["ci_low"] > 0,
        "every_budget_and_supported_stratum_delta_at_least_minus_0_02": min(
            safety, default=-1.0
        )
        >= -0.02,
        "mean_selected_unit_reduction_at_least_1": unit_reduction >= 1.0,
        "macro_recall_improvement_vs_v43_at_least_0_01": recall_improvement
        >= 0.01,
    }
    if not (
        checks["minimum_cases_met"]
        and checks["candidate_ceiling_complete_rate_equals_1"]
    ):
        status = "EVIDENCE_INFERENCE_V2_VALIDATION_INCONCLUSIVE"
    elif all(checks.values()):
        status = "EVIDENCE_INFERENCE_LOW_CORE_DIVERGENCE_SUPPORT_ESTABLISHED_ON_VALIDATION"
    else:
        status = "EVIDENCE_INFERENCE_LOW_CORE_DIVERGENCE_SUPPORT_NOT_ESTABLISHED"
    core_sizes = Counter(int(row["selector"]["argmax_core_size"]) for row in evidence)
    surpluses = Counter(int(row["selector"]["runner_up_surplus"]) for row in evidence)
    targets = Counter(int(row["selector"]["target_cardinality"]) for row in evidence)
    report = {
        "schema_version": "frc-evidence-inference-low-core-divergence-report-v1",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "cases": len(evidence),
            "budgets": list(BUDGETS),
            "official_leaderboard_result": False,
            "eraser_result": False,
            "full_article_bounded_pool": True,
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
            "expansion_trigger_rate": round(trigger_rate, 6),
            "selection_set_difference_rate_vs_v43": round(difference_v43, 6),
            "selection_set_difference_rate_vs_dynamic": round(difference_dynamic, 6),
            "selection_set_difference_rate_vs_v46": round(difference_v46, 6),
            "mean_selected_unit_reduction_vs_dynamic": round(unit_reduction, 6),
            "macro_recall_improvement_vs_v43": round(recall_improvement, 6),
            "argmax_core_size_distribution": {
                str(key): core_sizes[key] for key in sorted(core_sizes)
            },
            "runner_up_surplus_distribution": {
                str(key): surpluses[key] for key in sorted(surpluses)
            },
            "target_cardinality_distribution": {
                str(key): targets[key] for key in sorted(targets)
            },
            "budget_deltas": budget_deltas,
            "supported_stratum_deltas": strata,
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "reuse_v49_cases_for_tuning_or_selection": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
        "limitations": [
            "Evidence Inference supplies an attached full article, not open-corpus retrieval.",
            "Character evidence spans are projected to a preregistered regex sentence split.",
            "The endpoint is evidence selection, not treatment-effect classification.",
            "This is not an official ERASER or Evidence Inference leaderboard result.",
            "This cannot establish SetR reproduction, flood-domain validity, or production readiness.",
        ],
    }
    return report, evidence


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
    json_path.write_text(
        json.dumps(rounded, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    analysis = rounded["analysis"]
    lines = [
        "# Evidence Inference 2.0 low-core-divergence evaluation (v49)",
        "",
        f"- Status: `{analysis['outcome']['status']}`",
        f"- Cases: {rounded['metadata']['cases']}",
        f"- Strongest frozen FRC control: `{analysis['strongest_frozen_frc_control']}`",
        f"- Strongest non-FRC: `{analysis['strongest_non_frc']}`",
        f"- Expansion-trigger rate: {analysis['expansion_trigger_rate']:.6f}",
        "- Gate 2: `NO-GO/SHADOW`",
        "",
        "## Aggregate methods",
        "",
        "| Method | F1 | Precision | Recall | Complete | Units | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in analysis["aggregates"].items():
        lines.append(
            f"| `{method}` | {values['evidence_macro_f1']:.6f} | "
            f"{values['macro_precision']:.6f} | {values['macro_recall']:.6f} | "
            f"{values['complete_reference_recall']:.6f} | "
            f"{values['mean_selected_unit_count']:.6f} | "
            f"{values['mean_selected_token_cost']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This is a one-shot Evidence Inference 2.0 validation full-article sentence evidence-selection experiment, not an ERASER or treatment-effect classification leaderboard result. Annotator references are alternatives, and v49 cases may not be reused for tuning or method selection.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_protocol(protocol_path: Path) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("Evidence Inference v49 protocol hash mismatch")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Evidence Inference v49 experiment mismatch")
    if value.get("methods", {}).get("candidate_method") != LOW_CORE_DIVERGENCE_V49:
        raise ValueError("Evidence Inference v49 candidate method mismatch")
    if value.get("methods", {}).get("token_budgets") != list(BUDGETS):
        raise ValueError("Evidence Inference v49 budget registration mismatch")
    boundary = value.get("development_boundary", {})
    if boundary.get("evidence_inference_train_split_permanently_excluded") is not True:
        raise ValueError("Evidence Inference v49 train exclusion is missing")
    if boundary.get("evidence_inference_test_split_permanently_excluded") is not True:
        raise ValueError("Evidence Inference v49 test exclusion is missing")
    return value


def validate_implementation_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    validate_protocol(protocol_path)
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != (
        "frc-evidence-inference-low-core-divergence-atomic-roles-implementation-v49"
    ):
        raise ValueError("Evidence Inference v49 implementation schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Evidence Inference v49 implementation experiment mismatch")
    if value.get("validation_archive_downloaded_before_registration") is not False:
        raise ValueError("Evidence Inference v49 crossed the validation boundary")
    expected = {
        "protocol_sha256": _sha256(protocol_path),
        "module_sha256": _sha256(module_path),
        "runner_sha256": _sha256(runner_path),
        "test_sha256": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("Evidence Inference v49 implementation hash mismatch")
    invariants = value.get("synthetic_invariants_verified")
    if not isinstance(invariants, dict) or not all(invariants.values()):
        raise ValueError("Evidence Inference v49 synthetic invariants are incomplete")
    if value.get("method_or_threshold_change_after_registration_forbidden") is not True:
        raise ValueError("Evidence Inference v49 implementation is not frozen")
    return value


def validate_execution_registration(
    execution_path: Path,
    *,
    implementation_path: Path,
    source_archive_path: Path,
    source_card_path: Path,
    prepared_path: Path,
    candidate_map_path: Path,
    census_path: Path,
    coverage_path: Path,
) -> dict[str, Any]:
    value = json.loads(execution_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != (
        "frc-evidence-inference-low-core-divergence-atomic-roles-execution-v49"
    ):
        raise ValueError("Evidence Inference v49 execution schema mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Evidence Inference v49 execution experiment mismatch")
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
        raise ValueError("Evidence Inference v49 execution artifact hash mismatch")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("minimum_cases_and_ceiling_checks_passed") is not True:
        raise ValueError("Evidence Inference v49 coverage gate is not open")
    for field in ("query_generation_started", "neural_scoring_started", "metrics_computed"):
        if value.get(field) is not False:
            raise ValueError(f"Evidence Inference v49 execution field changed: {field}")
    if value.get("negative_null_or_inconclusive_result_must_be_published") is not True:
        raise ValueError("Evidence Inference v49 publication boundary is missing")
    return value


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "EXPERIMENT_ID",
    "FrozenEvidenceInferenceScorer",
    "LOW_CORE_DIVERGENCE_V49",
    "OFFICIAL_ARCHIVE_BYTES",
    "OFFICIAL_REPOSITORY_REVISION",
    "BIGBIO_REPOSITORY_REVISION",
    "build_candidate_coverage",
    "build_gold_rows",
    "build_sentence_units",
    "evaluate_evidence_inference",
    "low_core_divergence_details",
    "prepare_blind_cases",
    "read_validation_source",
    "select_sample",
    "select_v49",
    "validate_execution_registration",
    "validate_implementation_registration",
    "validate_protocol",
    "write_report",
]
