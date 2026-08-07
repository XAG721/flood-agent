"""Prospective 2WikiMultiHopQA title-link support-path closure experiment (v74)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from research.frc_rag.public_evidence import evidence_metrics, select_precomputed
from research.frc_rag.twowiki_confirmation import FrozenBgeScorer


EXPERIMENT_ID = "FRC-2WIKI-SUPPORT-PATH-CLOSURE-V74"
SCHEMA_VERSION = "frc-twowiki-support-path-closure-v74"
QUESTION_TYPES = (
    "bridge_comparison",
    "comparison",
    "compositional",
    "inference",
)
CANDIDATE_METHOD = "soft_title_link_support_path_closure_v74"
CONTROL_METHODS = (
    "bm25_topk",
    "dense_topk",
    "hybrid_topk",
    "cross_encoder_topk",
    "coverage_greedy_proxy",
    "frc_select",
    "title_link_only_control",
    "cross_plus_title_link_without_source_diversity_control",
    "hard_title_link_chain_control",
    "alternating_anchor_link_control",
)
METHODS = (*CONTROL_METHODS, CANDIDATE_METHOD)
DEVELOPMENT_SALT = "FRC-2WIKI-V74-DEVELOPMENT|"
CONFIRMATION_SALT = "FRC-2WIKI-V74-CONFIRMATION|"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def write_jsonl_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0
        ) as archive:
            for row in rows:
                payload = (
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                archive.write(payload)


def read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _nested(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (dict, list, tuple)):
        converted = value.tolist()
        if converted is not value:
            return converted
    return value


def _token_count(text: str) -> int:
    return max(1, len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)))


def _order_key(salt: str, case_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{salt}{case_id}".encode("utf-8")).hexdigest()
    return digest, case_id


def load_history_ids(path: Path, *, expected_count: int = 1000) -> set[str]:
    rows = read_jsonl(path)
    ids = {str(row["id"]) for row in rows}
    if len(rows) != expected_count or len(ids) != expected_count:
        raise ValueError(
            f"history source requires {expected_count} unique ids, got {len(rows)}/{len(ids)}"
        )
    return ids


def select_stage_ids(
    metadata_rows: Iterable[dict[str, Any]],
    *,
    history_ids: set[str],
    stage: str,
    quota_per_type: int = 200,
) -> list[str]:
    if stage not in {"development", "confirmation"}:
        raise ValueError(f"unsupported v74 stage: {stage}")
    rows = [
        {"id": str(row.get("_id") or row.get("id") or ""), "type": str(row["type"])}
        for row in metadata_rows
    ]
    if any(not row["id"] for row in rows):
        raise ValueError("2Wiki metadata contains an empty case id")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("2Wiki metadata contains duplicate case ids")

    def choose(salt: str, excluded: set[str]) -> list[str]:
        chosen: list[str] = []
        for question_type in QUESTION_TYPES:
            eligible = [
                row["id"]
                for row in rows
                if row["type"] == question_type and row["id"] not in excluded
            ]
            eligible.sort(key=lambda case_id: _order_key(salt, case_id))
            if len(eligible) < quota_per_type:
                raise ValueError(
                    f"v74 lacks {quota_per_type} eligible {question_type} rows"
                )
            chosen.extend(eligible[:quota_per_type])
        return chosen

    development = choose(DEVELOPMENT_SALT, history_ids)
    if stage == "development":
        return development
    return choose(CONFIRMATION_SALT, history_ids | set(development))


def prepare_case(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = str(row.get("_id") or "").strip()
    question_type = str(row.get("type") or "").strip()
    if not case_id or question_type not in QUESTION_TYPES:
        raise ValueError(f"invalid 2Wiki v74 case metadata: {case_id!r}/{question_type!r}")
    context = _nested(row.get("context")) or []
    supporting_facts = _nested(row.get("supporting_facts")) or []
    gold_pairs = {
        (str(title), int(sentence_index))
        for title, sentence_index in supporting_facts
    }
    if len(gold_pairs) < 2:
        raise ValueError(f"2Wiki case {case_id} has fewer than two supporting facts")
    candidates: list[dict[str, Any]] = []
    matched: set[tuple[str, int]] = set()
    gold_ids: list[str] = []
    for paragraph_index, paragraph in enumerate(context):
        title, sentences = paragraph
        title = str(title)
        for sentence_index, sentence in enumerate(_nested(sentences) or []):
            candidate_id = (
                f"2wiki::{case_id}::p{paragraph_index}::s{sentence_index}"
            )
            pair = (title, sentence_index)
            if pair in gold_pairs:
                matched.add(pair)
                gold_ids.append(candidate_id)
            text = f"{title}. {str(sentence).strip()}".strip()
            candidates.append(
                {
                    "id": candidate_id,
                    "text": text,
                    "source": title,
                    "metadata": {
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "sentence_index": sentence_index,
                    },
                    "token_count": _token_count(text),
                }
            )
    if matched != gold_pairs:
        raise ValueError(
            f"2Wiki case {case_id} is missing supporting facts: {sorted(gold_pairs - matched)}"
        )
    if not candidates:
        raise ValueError(f"2Wiki case {case_id} has no candidates")
    blind = {
        "dataset": "2WikiMultiHopQA",
        "source": "official_dev_case_disjoint_v74",
        "id": case_id,
        "question": str(row.get("question") or ""),
        "question_type": question_type,
        "not_answerable": False,
        "required_roles": ["procedure", "answer"],
        "candidates": candidates,
    }
    gold = {
        "id": case_id,
        "question_type": question_type,
        "gold_evidence_ids": sorted(gold_ids),
    }
    return blind, gold


def prepare_stage(
    source_path: Path,
    history_path: Path,
    *,
    stage: str,
    blind_path: Path,
    gold_path: Path,
) -> dict[str, Any]:
    import pandas as pd
    import pyarrow.parquet as pq

    history_ids = load_history_ids(history_path)
    metadata = pq.read_table(source_path, columns=["_id", "type"]).to_pylist()
    selected_ids = select_stage_ids(
        metadata,
        history_ids=history_ids,
        stage=stage,
    )
    selected_set = set(selected_ids)
    frame = pd.read_parquet(source_path)
    rows_by_id = {
        str(row["_id"]): row
        for row in frame.to_dict(orient="records")
        if str(row["_id"]) in selected_set
    }
    if set(rows_by_id) != selected_set:
        raise ValueError("v74 selected ids are not all present in the registered source")
    blind_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for case_id in selected_ids:
        blind, gold = prepare_case(rows_by_id[case_id])
        blind_rows.append(blind)
        gold_rows.append(gold)
    write_jsonl(blind_path, blind_rows)
    write_jsonl(gold_path, gold_rows)
    counts = Counter(row["question_type"] for row in gold_rows)
    return {
        "stage": stage,
        "selected_cases": len(selected_ids),
        "question_type_counts": dict(sorted(counts.items())),
        "history_overlap": len(selected_set & history_ids),
        "selected_ids_sha256": canonical_json_sha256(selected_ids),
        "blind_sha256": sha256(blind_path),
        "gold_sha256": sha256(gold_path),
        "candidate_count": sum(len(row["candidates"]) for row in blind_rows),
    }


def _normalize_surface(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _contains_surface(text: str, surface: str) -> bool:
    return bool(surface) and f" {surface} " in f" {text} "


def build_title_graph(candidates: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    graph = {str(candidate["id"]): set() for candidate in candidates}
    normalized_text = {
        str(candidate["id"]): _normalize_surface(str(candidate["text"]))
        for candidate in candidates
    }
    normalized_source = {
        str(candidate["id"]): _normalize_surface(str(candidate["source"]))
        for candidate in candidates
    }
    for left in candidates:
        left_id = str(left["id"])
        for right in candidates:
            right_id = str(right["id"])
            if left_id == right_id or left["source"] == right["source"]:
                continue
            if _contains_surface(
                normalized_text[left_id], normalized_source[right_id]
            ):
                graph[left_id].add(right_id)
                graph[right_id].add(left_id)
    return graph


def _eligible_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    selected: Sequence[dict[str, Any]],
    token_budget: int,
) -> list[dict[str, Any]]:
    used = sum(int(candidate.get("token_count", 1)) for candidate in selected)
    selected_ids = {str(candidate["id"]) for candidate in selected}
    return [
        candidate
        for candidate in candidates
        if str(candidate["id"]) not in selected_ids
        and used + int(candidate.get("token_count", 1)) <= token_budget
    ]


def _cross(candidate: dict[str, Any]) -> float:
    return float(candidate.get("scores", {}).get("cross_encoder", 0.0))


def _soft_graph_select(
    row: dict[str, Any],
    *,
    link_bonus: float,
    new_source_bonus: float,
    k: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    candidates = list(row.get("candidates", []))
    graph = build_title_graph(candidates)
    selected: list[dict[str, Any]] = []
    while len(selected) < k:
        eligible = _eligible_candidates(
            candidates,
            selected=selected,
            token_budget=token_budget,
        )
        if not eligible:
            break
        selected_ids = {str(candidate["id"]) for candidate in selected}
        selected_sources = {str(candidate["source"]) for candidate in selected}

        def key(candidate: dict[str, Any]) -> tuple[float, float, str]:
            candidate_id = str(candidate["id"])
            linked = float(bool(selected) and bool(graph[candidate_id] & selected_ids))
            new_source = float(
                bool(selected) and str(candidate["source"]) not in selected_sources
            )
            cross = _cross(candidate)
            score = cross + link_bonus * linked + new_source_bonus * new_source
            return (-score, -cross, candidate_id)

        selected.append(min(eligible, key=key))
    return selected


def _title_link_only_select(
    row: dict[str, Any], *, k: int, token_budget: int
) -> list[dict[str, Any]]:
    candidates = list(row.get("candidates", []))
    graph = build_title_graph(candidates)
    selected: list[dict[str, Any]] = []
    while len(selected) < k:
        eligible = _eligible_candidates(
            candidates,
            selected=selected,
            token_budget=token_budget,
        )
        if not eligible:
            break
        selected_ids = {str(candidate["id"]) for candidate in selected}
        selected_sources = {str(candidate["source"]) for candidate in selected}

        def key(candidate: dict[str, Any]) -> tuple[int, int, int, str]:
            candidate_id = str(candidate["id"])
            linked = bool(selected) and bool(graph[candidate_id] & selected_ids)
            new_source = bool(selected) and str(candidate["source"]) not in selected_sources
            return (-int(linked), -int(new_source), -len(graph[candidate_id]), candidate_id)

        selected.append(min(eligible, key=key))
    return selected


def _hard_chain_select(
    row: dict[str, Any], *, k: int, token_budget: int
) -> list[dict[str, Any]]:
    candidates = list(row.get("candidates", []))
    graph = build_title_graph(candidates)
    selected: list[dict[str, Any]] = []
    while len(selected) < k:
        eligible = _eligible_candidates(
            candidates,
            selected=selected,
            token_budget=token_budget,
        )
        if not eligible:
            break
        selected_ids = {str(candidate["id"]) for candidate in selected}
        selected_sources = {str(candidate["source"]) for candidate in selected}

        def key(candidate: dict[str, Any]) -> tuple[int, int, float, str]:
            candidate_id = str(candidate["id"])
            linked = bool(selected) and bool(graph[candidate_id] & selected_ids)
            new_source = bool(selected) and str(candidate["source"]) not in selected_sources
            return (-int(linked), -int(new_source), -_cross(candidate), candidate_id)

        selected.append(min(eligible, key=key))
    return selected


def _alternating_anchor_select(
    row: dict[str, Any], *, k: int, token_budget: int
) -> list[dict[str, Any]]:
    candidates = list(row.get("candidates", []))
    graph = build_title_graph(candidates)
    selected: list[dict[str, Any]] = []
    while len(selected) < k:
        eligible = _eligible_candidates(
            candidates,
            selected=selected,
            token_budget=token_budget,
        )
        if not eligible:
            break
        anchor = min(eligible, key=lambda item: (-_cross(item), str(item["id"])))
        selected.append(anchor)
        if len(selected) >= k:
            break
        eligible = _eligible_candidates(
            candidates,
            selected=selected,
            token_budget=token_budget,
        )
        linked = [
            candidate
            for candidate in eligible
            if str(candidate["id"]) in graph[str(anchor["id"])]
            and candidate["source"] != anchor["source"]
        ]
        if linked:
            selected.append(
                min(linked, key=lambda item: (-_cross(item), str(item["id"])))
            )
    return selected


def select_method(
    row: dict[str, Any],
    method: str,
    *,
    k: int = 5,
    token_budget: int = 1500,
) -> list[dict[str, Any]]:
    if method in CONTROL_METHODS[:6]:
        return select_precomputed(row, method, k=k, budget=token_budget)
    if method == "title_link_only_control":
        return _title_link_only_select(row, k=k, token_budget=token_budget)
    if method == "cross_plus_title_link_without_source_diversity_control":
        return _soft_graph_select(
            row,
            link_bonus=1.0,
            new_source_bonus=0.0,
            k=k,
            token_budget=token_budget,
        )
    if method == "hard_title_link_chain_control":
        return _hard_chain_select(row, k=k, token_budget=token_budget)
    if method == "alternating_anchor_link_control":
        return _alternating_anchor_select(row, k=k, token_budget=token_budget)
    if method == CANDIDATE_METHOD:
        return _soft_graph_select(
            row,
            link_bonus=1.0,
            new_source_bonus=0.1,
            k=k,
            token_budget=token_budget,
        )
    raise ValueError(f"unsupported v74 method: {method}")


def write_selection_outputs(
    scored_rows: Sequence[dict[str, Any]],
    path: Path,
    *,
    k: int = 5,
    token_budget: int = 1500,
) -> list[dict[str, Any]]:
    forbidden = {"answer", "gold_evidence_ids", "gold", "gold_roles"}
    if any(forbidden & set(row) for row in scored_rows):
        raise ValueError("v74 scored model rows contain forbidden gold fields")
    outputs: list[dict[str, Any]] = []
    for row in scored_rows:
        methods: dict[str, Any] = {}
        for method in METHODS:
            selected = select_method(row, method, k=k, token_budget=token_budget)
            methods[method] = {
                "selected_ids": [str(candidate["id"]) for candidate in selected],
                "selected_tokens": sum(
                    int(candidate.get("token_count", 1)) for candidate in selected
                ),
            }
        outputs.append(
            {
                "case_id": str(row["id"]),
                "question_type": str(row["question_type"]),
                "candidate_ids": sorted(
                    str(candidate["id"]) for candidate in row["candidates"]
                ),
                "methods": methods,
            }
        )
    write_jsonl(path, outputs)
    return outputs


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _paired_bootstrap(
    candidate: Sequence[float],
    control: Sequence[float],
    *,
    seed: int,
    resamples: int = 10000,
) -> dict[str, Any]:
    candidate_values = np.asarray(candidate, dtype=float)
    control_values = np.asarray(control, dtype=float)
    if candidate_values.shape != control_values.shape or not candidate_values.size:
        raise ValueError("paired bootstrap inputs must be non-empty and aligned")
    difference = candidate_values - control_values
    generator = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for start in range(0, resamples, 250):
        batch = min(250, resamples - start)
        indices = generator.integers(
            0,
            difference.size,
            size=(batch, difference.size),
        )
        estimates.append(difference[indices].mean(axis=1))
    samples = np.concatenate(estimates)
    return {
        "point": round(float(difference.mean()), 6),
        "ci_low": round(float(np.quantile(samples, 0.025)), 6),
        "ci_high": round(float(np.quantile(samples, 0.975)), 6),
        "resamples": resamples,
        "seed": seed,
    }


def evaluate_stage(
    scored_path: Path,
    gold_path: Path,
    selection_path: Path,
    *,
    stage: str,
    seed: int,
    output_dir: Path,
    history_overlap: int,
    stage_overlap: int = 0,
) -> dict[str, Any]:
    scored_rows = read_jsonl(scored_path)
    selections = write_selection_outputs(scored_rows, selection_path)
    gold_rows = read_jsonl(gold_path)
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    selection_by_id = {str(row["case_id"]): row for row in selections}
    if set(gold_by_id) != set(selection_by_id):
        raise ValueError("v74 gold and selection case ids do not align")

    per_case: list[dict[str, Any]] = []
    invalid_outputs = 0
    for case_id in sorted(gold_by_id):
        gold = gold_by_id[case_id]
        selection = selection_by_id[case_id]
        candidate_ids = set(selection["candidate_ids"])
        method_rows: dict[str, Any] = {}
        for method in METHODS:
            method_selection = selection["methods"][method]
            ids = list(method_selection["selected_ids"])
            invalid = (
                len(ids) != len(set(ids))
                or len(ids) > 5
                or not set(ids) <= candidate_ids
                or int(method_selection["selected_tokens"]) > 1500
            )
            invalid_outputs += int(invalid)
            metrics = evidence_metrics(gold["gold_evidence_ids"], ids)
            method_rows[method] = {
                **metrics,
                "complete_evidence": float(
                    set(gold["gold_evidence_ids"]) <= set(ids)
                ),
                "selected_tokens": int(method_selection["selected_tokens"]),
                "invalid": invalid,
            }
        per_case.append(
            {
                "case_id": case_id,
                "question_type": str(gold["question_type"]),
                "gold_evidence_count": len(gold["gold_evidence_ids"]),
                "methods": method_rows,
            }
        )

    aggregates: dict[str, Any] = {}
    for method in METHODS:
        rows = [case["methods"][method] for case in per_case]
        aggregates[method] = {
            "cases": len(rows),
            "evidence_macro_f1": round(
                _mean([float(row["evidence_f1"]) for row in rows]), 6
            ),
            "evidence_macro_recall": round(
                _mean([float(row["evidence_recall"]) for row in rows]), 6
            ),
            "complete_evidence_recall": round(
                _mean([float(row["complete_evidence"]) for row in rows]), 6
            ),
            "mean_selected_tokens": round(
                _mean([float(row["selected_tokens"]) for row in rows]), 6
            ),
        }
    strongest_control = max(
        CONTROL_METHODS,
        key=lambda method: (
            aggregates[method]["evidence_macro_f1"],
            -CONTROL_METHODS.index(method),
        ),
    )
    comparisons = {
        method: _paired_bootstrap(
            [float(case["methods"][CANDIDATE_METHOD]["evidence_f1"]) for case in per_case],
            [float(case["methods"][method]["evidence_f1"]) for case in per_case],
            seed=seed + index,
        )
        for index, method in enumerate(CONTROL_METHODS)
    }
    type_deltas: dict[str, Any] = {}
    for question_type in QUESTION_TYPES:
        subset = [
            case for case in per_case if case["question_type"] == question_type
        ]
        candidate_value = _mean(
            [
                float(case["methods"][CANDIDATE_METHOD]["evidence_f1"])
                for case in subset
            ]
        )
        control_value = _mean(
            [
                float(case["methods"][strongest_control]["evidence_f1"])
                for case in subset
            ]
        )
        type_deltas[question_type] = {
            "cases": len(subset),
            "candidate_evidence_macro_f1": round(candidate_value, 6),
            "strongest_control_evidence_macro_f1": round(control_value, 6),
            "delta": round(candidate_value - control_value, 6),
        }
    candidate = aggregates[CANDIDATE_METHOD]
    strongest = aggregates[strongest_control]
    strongest_delta = comparisons[strongest_control]
    frc_delta = comparisons["frc_select"]
    token_ratio = float(candidate["mean_selected_tokens"]) / max(
        1e-12, float(strongest["mean_selected_tokens"])
    )
    type_counts = Counter(case["question_type"] for case in per_case)
    checks = {
        "exact_cases_equals_800": len(per_case) == 800,
        "exact_question_type_balance": type_counts
        == Counter({question_type: 200 for question_type in QUESTION_TYPES}),
        "history_or_stage_overlap_equals_0": history_overlap == 0
        and stage_overlap == 0,
        "invalid_selector_output_rate_equals_0": invalid_outputs == 0,
        "candidate_evidence_macro_f1_at_least_0_50": float(
            candidate["evidence_macro_f1"]
        )
        >= 0.50,
        "candidate_evidence_macro_recall_at_least_0_80": float(
            candidate["evidence_macro_recall"]
        )
        >= 0.80,
        "candidate_complete_evidence_recall_at_least_0_55": float(
            candidate["complete_evidence_recall"]
        )
        >= 0.55,
        "candidate_minus_strongest_control_f1_at_least_0_01": float(
            strongest_delta["point"]
        )
        >= 0.01,
        "candidate_minus_strongest_control_ci_low_above_0": float(
            strongest_delta["ci_low"]
        )
        > 0.0,
        "candidate_minus_frc_select_f1_at_least_0_03": float(
            frc_delta["point"]
        )
        >= 0.03,
        "candidate_minus_frc_select_ci_low_above_0": float(
            frc_delta["ci_low"]
        )
        > 0.0,
        "every_question_type_delta_at_least_minus_0_02": all(
            float(details["delta"]) >= -0.02 for details in type_deltas.values()
        ),
        "mean_selected_tokens_within_1_05_of_strongest_control": token_ratio
        <= 1.05,
    }
    passed = all(checks.values())
    if stage == "development":
        status = (
            "2WIKI_V74_SUPPORT_PATH_CLOSURE_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
            if passed
            else "2WIKI_V74_SUPPORT_PATH_CLOSURE_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "2WIKI_V74_SUPPORT_PATH_CLOSURE_CASE_DISJOINT_SUPPORT_ESTABLISHED"
            if passed
            else "2WIKI_V74_SUPPORT_PATH_CLOSURE_CONFIRMATION_SUPPORT_NOT_ESTABLISHED"
        )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "stage": stage,
            "cases": len(per_case),
            "question_type_counts": dict(sorted(type_counts.items())),
            "history_overlap": history_overlap,
            "stage_overlap": stage_overlap,
            "invalid_selector_output_count": invalid_outputs,
            "selection_output_sha256": sha256(selection_path),
            "gold_join_started_after_selection_output_written": True,
            "official_2wiki_leaderboard_result": False,
            "strict_independent_model_training_confirmation": False,
        },
        "analysis": {
            "methods": aggregates,
            "strongest_same_resource_control": {
                "name": strongest_control,
                **strongest,
            },
            "paired_f1_delta": comparisons,
            "per_question_type_delta_vs_strongest_control": type_deltas,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "stage_gate_passed": passed,
                "confirmation_open_authorized": stage == "development" and passed,
                "reuse_target_stage_for_method_weight_gate_or_selection": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    report_path = output_dir / "report.md"
    cases_path = output_dir / "cases.jsonl.gz"
    result_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(report), encoding="utf-8")
    write_jsonl_gzip(cases_path, per_case)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    analysis = report["analysis"]
    strongest = analysis["strongest_same_resource_control"]
    candidate = analysis["methods"][CANDIDATE_METHOD]
    comparison = analysis["paired_f1_delta"][strongest["name"]]
    lines = [
        f"# 2WikiMultiHopQA 支持路径闭包实验（v74 {metadata['stage']}）",
        "",
        f"- 案例：`{metadata['cases']}`",
        f"- 状态：`{analysis['outcome']['status']}`",
        f"- Gate 2：`{analysis['outcome']['gate_2']}`",
        "",
        "## 主要结果",
        "",
        "| 方法 | 证据 F1 | 证据召回 | 完整证据召回 | 平均 token |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        values = analysis["methods"][method]
        lines.append(
            f"| `{method}` | {values['evidence_macro_f1']:.6f} | "
            f"{values['evidence_macro_recall']:.6f} | "
            f"{values['complete_evidence_recall']:.6f} | "
            f"{values['mean_selected_tokens']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 冻结门槛",
            "",
            f"候选证据 F1 为 `{candidate['evidence_macro_f1']:.6f}`；最强同资源对照为 "
            f"`{strongest['name']}`（`{strongest['evidence_macro_f1']:.6f}`）。",
            f"配对差值 `{comparison['point']:+.6f}`，95% CI "
            f"[`{comparison['ci_low']:+.6f}`, `{comparison['ci_high']:+.6f}`]。",
            "",
        ]
    )
    for name, passed in analysis["support_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(
        [
            "",
            "该实验不是官方排行榜、独立数据集、自动分解、答案生成、洪水领域专家评测或生产放行证据；Gate 2 保持 `NO-GO/SHADOW`。",
            "",
        ]
    )
    return "\n".join(lines)


def validate_registered_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v74 experiment id")
    candidate = protocol["candidate"]
    if (
        float(candidate["link_bonus"]) != 1.0
        or float(candidate["new_source_bonus"]) != 0.1
    ):
        raise ValueError("v74 selector weights do not match the frozen protocol")
    if tuple(protocol["controls"]) != CONTROL_METHODS:
        raise ValueError("v74 control order does not match the frozen protocol")
    if protocol["stopping_and_outcomes"]["gate_2"] != "NO-GO/SHADOW":
        raise ValueError("v74 protocol must preserve Gate 2 NO-GO/SHADOW")


def validate_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(validate_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(validate_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


__all__ = [
    "CANDIDATE_METHOD",
    "CONTROL_METHODS",
    "EXPERIMENT_ID",
    "FrozenBgeScorer",
    "METHODS",
    "QUESTION_TYPES",
    "build_title_graph",
    "canonical_json_sha256",
    "evaluate_stage",
    "load_history_ids",
    "prepare_case",
    "prepare_stage",
    "read_json",
    "read_jsonl",
    "read_jsonl_gzip",
    "select_method",
    "select_stage_ids",
    "sha256",
    "validate_finite",
    "validate_registered_protocol",
    "write_jsonl",
    "write_selection_outputs",
]
