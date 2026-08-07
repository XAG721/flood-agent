"""Disjoint SQuAD 2.0 extractive support-gate experiment (v59)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import research.frc_rag.squad2_generative_answerability_gate as v58
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    FRC_CONTROLS,
    LOW_CORE_DIVERGENCE_V49,
    METHODS as V49_METHODS,
    _round_for_display,
    select_v49,
)
from research.frc_rag.feverous_adaptive_atomic_roles import ADAPTIVE_ARGMAX
from research.frc_rag.hover_dynamic_atomic_roles import (
    DYNAMIC_ROLES,
    merge_scored_candidates,
)
from research.frc_rag.rgb_cost_aware_frc import read_jsonl
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import NON_FRC_BASELINES


SCHEMA_VERSION = "frc-squad2-extractive-support-gate-v59"
EXPERIMENT_ID = "FRC-SQUAD2-EXTRACTIVE-SUPPORT-GATE-V59"
DATASET_ID = "squad2_disjoint_extractive_support_evidence_selection_v59"
CAPABILITY = "closed_paragraph_extractive_support_and_evidence_selection"
PROTOCOL_SHA256 = "d9c3350f9f54db473bbd1d521bfd1d052832712f9e554b31aed3e3e866b570f7"

STAGES = v58.STAGES
SOURCE_FILES = v58.SOURCE_FILES
SOURCE_URLS = v58.SOURCE_URLS
TARGET_CASES = 600
TARGET_PER_GROUP = 300
MAX_CASES_PER_PARAGRAPH = 2
MAX_CASES_PER_ARTICLE_PER_STATE = 12
MINIMUM_PARAGRAPHS = 250
MINIMUM_ARTICLES = {"development": 100, "confirmation": 25}
MINIMUM_STRATUM_CASES = 50
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260819
BUDGETS = v58.BUDGETS
SAMPLE_SALTS = {
    "development": "FRC-SQUAD2-V59-DISJOINT-DEVELOPMENT|",
    "confirmation": "FRC-SQUAD2-V59-CONFIRMATION|",
}

EXTRACTIVE_PROMPT = (
    "Copy the shortest contiguous span from the paragraph that directly answers "
    "the question. Return only that exact span and no explanation. If the "
    "paragraph does not contain a span that answers the requested fact, return "
    "exactly UNSUPPORTED. A related topic, a plausible guess, or an unstated "
    "assumption is not an answer. Do not use outside knowledge.\n\n"
    "Question: {question}\n\nParagraph:\n{context}\n\n"
    "Answer span or UNSUPPORTED:"
)

GATED_NON_FRC_BY_BASE = {
    method: f"qwen_span_supported_{method}_v59" for method in NON_FRC_BASELINES
}
FRC_SAFETY_BASES = tuple(
    dict.fromkeys(
        [
            *(method for method in FRC_CONTROLS if method != ADAPTIVE_ARGMAX),
            LOW_CORE_DIVERGENCE_V49,
        ]
    )
)
GATED_FRC_BY_BASE = {
    method: f"qwen_span_supported_{method}_v59" for method in FRC_SAFETY_BASES
}
GATED_NON_FRC_METHODS = tuple(GATED_NON_FRC_BY_BASE.values())
GATED_FRC_CONTROLS = tuple(GATED_FRC_BY_BASE.values())
CANDIDATE = "qwen_span_supported_adaptive_argmax_cardinality_frc_v43_transferred_v59"
EXACT_ANCHOR_GATED = GATED_NON_FRC_BY_BASE["cross_encoder_topk"]
EXACT_ANCHOR_UNGATED = "cross_encoder_topk"
METHODS = (*V49_METHODS, *GATED_NON_FRC_METHODS, *GATED_FRC_CONTROLS, CANDIDATE)
GATED_BASE_BY_METHOD = {
    **{value: key for key, value in GATED_NON_FRC_BY_BASE.items()},
    **{value: key for key, value in GATED_FRC_BY_BASE.items()},
    CANDIDATE: ADAPTIVE_ARGMAX,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def read_stage_source(source_root: Path, stage: str) -> dict[str, Any]:
    return v58.read_stage_source(source_root, stage)


def load_v58_excluded_commitments(path: Path) -> set[str]:
    rows = list(read_jsonl(path))
    values = {str(row["source_case_commitment"]) for row in rows}
    if len(rows) != v58.TARGET_CASES or len(values) != v58.TARGET_CASES:
        raise ValueError("SQuAD2 v59 v58 exclusion commitment cache is invalid")
    allowed_keys = {
        "schema_version",
        "id",
        "source_case_commitment",
        "article_commitment",
        "paragraph_commitment",
        "paragraph_length",
        "article_selected_case_count",
        "candidate_intervals",
    }
    if any(set(row) - allowed_keys for row in rows):
        raise ValueError("SQuAD2 v59 exclusion cache exposes an unexpected field")
    return values


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


def select_disjoint_balanced_sample(
    source: dict[str, Any],
    *,
    stage: str,
    excluded_commitments: set[str],
    target_per_group: int = TARGET_PER_GROUP,
    maximum_cases_per_paragraph: int = MAX_CASES_PER_PARAGRAPH,
    maximum_cases_per_article_per_state: int = MAX_CASES_PER_ARTICLE_PER_STATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported SQuAD2 v59 stage: {stage}")
    rows, schema = v58.extract_cases(source)
    overlap_eligible = sum(
        _hash(str(row["raw_id"])) in excluded_commitments for row in rows
    )
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
    if any(len(by_group[group]) < target_per_group for group in groups):
        raise ValueError("SQuAD2 v59 source lacks the disjoint answer-state quota")
    selected: list[dict[str, Any]] = []
    selected_by_group: Counter[str] = Counter()
    paragraph_counts: Counter[str] = Counter()
    article_state_counts: Counter[tuple[str, str]] = Counter()
    cursors = {group: 0 for group in groups}
    while any(selected_by_group[group] < target_per_group for group in groups):
        progressed = False
        for group in groups:
            if selected_by_group[group] >= target_per_group:
                continue
            values = by_group[group]
            while cursors[group] < len(values):
                row = values[cursors[group]]
                cursors[group] += 1
                paragraph_id = str(row["paragraph_id"])
                article_state = (str(row["article_id"]), group)
                if paragraph_counts[paragraph_id] >= maximum_cases_per_paragraph:
                    continue
                if (
                    article_state_counts[article_state]
                    >= maximum_cases_per_article_per_state
                ):
                    continue
                selected.append(row)
                selected_by_group[group] += 1
                paragraph_counts[paragraph_id] += 1
                article_state_counts[article_state] += 1
                progressed = True
                break
        if not progressed:
            raise ValueError("SQuAD2 v59 caps prevent the disjoint balanced sample")
    selected.sort(key=lambda row: (str(row["answer_state"]), _sample_key(row, stage)))
    selected_overlap = sum(
        _hash(str(row["raw_id"])) in excluded_commitments for row in selected
    )
    return selected, {
        "stage": stage,
        "target_cases": target_per_group * 2,
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
        "v58_excluded_commitment_count": len(excluded_commitments),
        "v58_commitments_present_in_source": overlap_eligible,
        "selected_v58_commitment_overlap": selected_overlap,
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
    prepared, maps, verifier_inputs, structural = v58.prepare_blind_cases(
        selected, tokenizer, stage=stage
    )
    for row in prepared:
        row["schema_version"] = SCHEMA_VERSION
        row["dataset_id"] = DATASET_ID
        row["capability"] = CAPABILITY
    for row in maps:
        row["schema_version"] = "frc-squad2-v59-candidate-map-v1"
    for row in verifier_inputs:
        row["schema_version"] = "frc-squad2-v59-verifier-input-v1"
        row["dataset_id"] = DATASET_ID
    structural = {
        **structural,
        "schema_version": "frc-squad2-v59-structural-census-v1",
        "inherited_candidate_construction": "frozen v58 exact implementation",
    }
    return prepared, maps, verifier_inputs, structural


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in prepared_rows:
        if v58._contains_forbidden_blind_key(row):
            raise ValueError("SQuAD2 v59 query builder received a forbidden field")
        question = str(row["query"])
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "stage": str(row["stage"]),
                "id": str(row["id"]),
                "atomic_queries": {
                    role: v58.QUERY_TEMPLATES[role].format(question=question)
                    for role in DYNAMIC_ROLES
                },
                "fallback_used": False,
                "fallback_reason": None,
                "gold_fields_visible_to_generator": False,
            }
        )
    return result


def validate_query_cache(
    prepared_rows: Sequence[dict[str, Any]], query_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    if [str(row["id"]) for row in prepared_rows] != [
        str(row["id"]) for row in query_rows
    ]:
        raise ValueError("SQuAD2 v59 query cache is incomplete or out of order")
    if list(query_rows) != build_deterministic_queries(prepared_rows):
        raise ValueError("SQuAD2 v59 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


class FrozenSquad2Scorer(v58.FrozenSquad2Scorer):
    """Frozen BGE scorer with v59 metadata labels."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def extractive_prompt(row: dict[str, Any]) -> str:
    if v58._contains_forbidden_blind_key(row):
        raise ValueError("SQuAD2 v59 extractive prompt received a forbidden field")
    return EXTRACTIVE_PROMPT.format(
        question=str(row["question"]), context=str(row["context"])
    )


def parse_extractive_support(text: str, context: str) -> str | None:
    value = text.strip()
    if value.upper() == "UNSUPPORTED":
        return "UNSUPPORTED"
    if value and value.casefold() in context.casefold():
        return "SUPPORTED_SPAN"
    return None


class LocalQwenExtractiveVerifier:
    """Frozen local deterministic extract-or-abstain verifier."""

    def __init__(
        self,
        *,
        model_path: Path,
        batch_size: int = 16,
        max_new_tokens: int = 48,
        maximum_input_tokens: int = 3072,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.maximum_input_tokens = maximum_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()

    def _chat_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def verify(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        output_path: Path,
        cache_key: str,
    ) -> list[dict[str, Any]]:
        existing = list(read_jsonl(output_path)) if output_path.is_file() else []
        expected_ids = [str(row["id"]) for row in rows]
        existing_ids = [str(row.get("id", "")) for row in existing]
        if existing_ids != expected_ids[: len(existing_ids)]:
            raise ValueError("SQuAD2 v59 verifier cache is not a valid prefix")
        if any(row.get("cache_key") != cache_key for row in existing):
            raise ValueError("SQuAD2 v59 verifier cache key changed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if existing else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
            for start in range(len(existing), len(rows), self.batch_size):
                batch = list(rows[start : start + self.batch_size])
                prompts = [self._chat_prompt(extractive_prompt(row)) for row in batch]
                encoded = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.maximum_input_tokens,
                ).to(self.model.device)
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **encoded,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                prompt_length = encoded.input_ids.shape[-1]
                decoded = self.tokenizer.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True
                )
                for source, raw in zip(batch, decoded, strict=True):
                    value = raw.strip()
                    decision = parse_extractive_support(value, str(source["context"]))
                    prediction = {
                        "schema_version": "frc-squad2-v59-extractive-decision-v1",
                        "experiment_id": EXPERIMENT_ID,
                        "stage": str(source["stage"]),
                        "id": str(source["id"]),
                        "cache_key": cache_key,
                        "raw_prediction": value,
                        "raw_prediction_sha256": hashlib.sha256(
                            value.encode("utf-8")
                        ).hexdigest(),
                        "raw_prediction_codepoints": len(value),
                        "decision": decision,
                        "support_passed": decision == "SUPPORTED_SPAN",
                        "invalid_fail_closed_used": decision is None,
                        "gold_fields_visible_to_verifier": False,
                    }
                    handle.write(
                        json.dumps(
                            prediction,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    existing.append(prediction)
                handle.flush()
                print(
                    f"verified {min(start + len(batch), len(rows))}/{len(rows)} "
                    "SQuAD2 v59 cases",
                    flush=True,
                )
        return existing


def validate_verifier_cache(
    verifier_inputs: Sequence[dict[str, Any]],
    decisions: Sequence[dict[str, Any]],
    *,
    cache_key: str,
) -> dict[str, Any]:
    expected = [str(row["id"]) for row in verifier_inputs]
    actual = [str(row["id"]) for row in decisions]
    if actual != expected:
        raise ValueError("SQuAD2 v59 verifier cache is incomplete or out of order")
    if any(row.get("cache_key") != cache_key for row in decisions):
        raise ValueError("SQuAD2 v59 verifier cache key changed")
    invalid = sum(bool(row.get("invalid_fail_closed_used")) for row in decisions)
    return {
        "rows": len(decisions),
        "invalid_output_count": invalid,
        "invalid_output_rate": invalid / len(decisions) if decisions else 1.0,
        "gold_fields_visible_to_verifier": False,
    }


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    structural_census: dict[str, Any],
) -> list[dict[str, Any]]:
    return v58.build_gold_rows(
        source,
        candidate_maps,
        scored_rows,
        structural_census=structural_census,
    )


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    value = v58.build_candidate_coverage(gold_rows, sampling)
    value["schema_version"] = "frc-squad2-v59-candidate-coverage-v1"
    value["experiment_id"] = EXPERIMENT_ID
    value["selected_v58_commitment_overlap"] = sampling[
        "selected_v58_commitment_overlap"
    ]
    return value


def select_v59(
    candidates: Sequence[dict[str, Any]],
    method: str,
    *,
    support_passed: bool,
    token_budget: int,
) -> list[dict[str, Any]]:
    values = list(candidates)
    if method in V49_METHODS:
        return select_v49(values, method, token_budget=token_budget)
    base = GATED_BASE_BY_METHOD.get(method)
    if base is None:
        raise ValueError(f"Unsupported SQuAD2 v59 method: {method}")
    if not support_passed:
        return []
    return select_v49(values, base, token_budget=token_budget)


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
        by_cluster[str(row["article_cluster"])].append(
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
            -v58._aggregate(rows, method)["answer_or_abstention_macro_f1"],
            method,
        ),
    )[0]


def _stratum_delta(rows: Sequence[dict[str, Any]]) -> tuple[str, float]:
    strongest = _strongest(rows, GATED_NON_FRC_METHODS)
    delta = float(np.mean([_case_utility(row, CANDIDATE) for row in rows])) - float(
        np.mean([_case_utility(row, strongest) for row in rows])
    )
    return strongest, round(delta, 6)


def evaluate_stage(
    gold_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    support_rows: Sequence[dict[str, Any]],
    query_summary: dict[str, Any],
    verifier_summary: dict[str, Any],
    source_artifacts: dict[str, Any],
    *,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        stage not in STAGES
        or len(gold_rows) != len(scored_rows)
        or len(gold_rows) != len(support_rows)
        or not gold_rows
    ):
        raise ValueError("SQuAD2 v59 gold, score and verifier caches differ")
    support_by_id = {str(row["id"]): row for row in support_rows}
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("SQuAD2 v59 gold and score ids differ")
        support = support_by_id.get(str(gold["case_id"]))
        if support is None:
            raise ValueError("SQuAD2 v59 support decision is missing")
        support_passed = bool(support["support_passed"])
        candidates = merge_scored_candidates(dict(scored))
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v59(
                    candidates,
                    method,
                    support_passed=support_passed,
                    token_budget=budget,
                )
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": v58._selection_metrics(
                        selected,
                        str(gold["answer_state"]),
                        gold["gold_candidate_ids"],
                    ),
                }
            configurations[str(budget)] = {"methods": methods}
        evidence.append(
            {
                "case_id": str(gold["case_id"]),
                "article_cluster": str(gold["article_cluster"]),
                "paragraph_cluster": str(gold["paragraph_cluster"]),
                "answer_state": str(gold["answer_state"]),
                "candidate_pool_quartile": str(gold["candidate_pool_quartile"]),
                "paragraph_length_quartile": str(gold["paragraph_length_quartile"]),
                "answer_length_quartile": str(gold["answer_length_quartile"]),
                "article_case_count_quartile": str(gold["article_case_count_quartile"]),
                "candidate_ceiling_complete": bool(gold["candidate_ceiling_complete"]),
                "support_verifier": {
                    "decision": support.get("decision"),
                    "support_passed": support_passed,
                    "invalid_fail_closed_used": bool(
                        support.get("invalid_fail_closed_used")
                    ),
                    "output_sha256": str(support["raw_prediction_sha256"]),
                    "output_codepoints": int(support["raw_prediction_codepoints"]),
                },
                "configurations": configurations,
            }
        )
    aggregates = {method: v58._aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = _strongest(evidence, GATED_NON_FRC_METHODS)
    strongest_same_gate_frc = _strongest(evidence, GATED_FRC_CONTROLS)
    comparisons = {
        "gated_exact_anchor_minus_ungated_exact_anchor": _cluster_bootstrap(
            evidence, EXACT_ANCHOR_GATED, EXACT_ANCHOR_UNGATED
        ),
        "candidate_minus_gated_exact_anchor": _cluster_bootstrap(
            evidence, CANDIDATE, EXACT_ANCHOR_GATED
        ),
        "candidate_minus_strongest_shared_gate_non_frc": _cluster_bootstrap(
            evidence, CANDIDATE, strongest_non_frc
        ),
        "candidate_minus_strongest_same_gate_frc": _cluster_bootstrap(
            evidence, CANDIDATE, strongest_same_gate_frc
        ),
    }
    answer_rows = [row for row in evidence if row["answer_state"] == "answer_bearing"]
    no_answer_rows = [row for row in evidence if row["answer_state"] == "no_answer"]
    answer_pass = float(
        np.mean([row["support_verifier"]["support_passed"] for row in answer_rows])
    )
    no_answer_reject = float(
        np.mean(
            [not row["support_verifier"]["support_passed"] for row in no_answer_rows]
        )
    )
    balanced_accuracy = (answer_pass + no_answer_reject) / 2.0
    candidate = aggregates[CANDIDATE]
    ungated_exact = aggregates[EXACT_ANCHOR_UNGATED]
    recall_drop = round(
        ungated_exact["answer_macro_recall"] - candidate["answer_macro_recall"], 6
    )
    budget_deltas: dict[str, float] = {}
    for budget in BUDGETS:
        subset: list[dict[str, Any]] = []
        for row in evidence:
            clone = dict(row)
            clone["configurations"] = {str(budget): row["configurations"][str(budget)]}
            subset.append(clone)
        _, budget_deltas[str(budget)] = _stratum_delta(subset)
    dimensions = {
        "answer_state": lambda row: row["answer_state"],
        "paragraph_length_quartile": lambda row: row["paragraph_length_quartile"],
        "candidate_pool_quartile": lambda row: row["candidate_pool_quartile"],
        "answer_length_quartile": lambda row: row["answer_length_quartile"],
        "article_case_count_quartile": lambda row: row["article_case_count_quartile"],
        "support_verifier_outcome": lambda row: (
            "supported" if row["support_verifier"]["support_passed"] else "unsupported"
        ),
    }
    strata: dict[str, dict[str, Any]] = {}
    for dimension, getter in dimensions.items():
        for value in sorted({str(getter(row)) for row in evidence}):
            subset = [row for row in evidence if str(getter(row)) == value]
            if len(subset) < MINIMUM_STRATUM_CASES:
                continue
            strongest, delta = _stratum_delta(subset)
            strata[f"{dimension}:{value}"] = {
                "cases": len(subset),
                "strongest_shared_gate_non_frc": strongest,
                "delta": delta,
            }
    minimum_delta = min(
        [*budget_deltas.values(), *(row["delta"] for row in strata.values())],
        default=-math.inf,
    )
    ceiling = float(np.mean([row["candidate_ceiling_complete"] for row in answer_rows]))
    state_counts = Counter(row["answer_state"] for row in evidence)
    articles = len({row["article_cluster"] for row in evidence})
    paragraphs = len({row["paragraph_cluster"] for row in evidence})
    sampling = source_artifacts["sampling"]
    exact_improvement = comparisons["gated_exact_anchor_minus_ungated_exact_anchor"]
    candidate_exact = comparisons["candidate_minus_gated_exact_anchor"]
    candidate_non_frc = comparisons["candidate_minus_strongest_shared_gate_non_frc"]
    candidate_frc = comparisons["candidate_minus_strongest_same_gate_frc"]
    checks = {
        "exact_cases_equals_600": len(evidence) == TARGET_CASES,
        "exact_answer_state_balance": state_counts
        == {"answer_bearing": TARGET_PER_GROUP, "no_answer": TARGET_PER_GROUP},
        "selected_v58_commitment_overlap_equals_0": int(
            sampling["selected_v58_commitment_overlap"]
        )
        == 0,
        "minimum_paragraphs_at_least_250": paragraphs >= MINIMUM_PARAGRAPHS,
        "minimum_articles_for_stage": articles >= MINIMUM_ARTICLES[stage],
        "paragraph_cap_at_most_2": int(sampling["maximum_cases_per_paragraph"])
        <= MAX_CASES_PER_PARAGRAPH,
        "article_state_cap_at_most_12": int(
            sampling["maximum_cases_per_article_per_answer_state"]
        )
        <= MAX_CASES_PER_ARTICLE_PER_STATE,
        "schema_exclusion_rate_at_most_0_01": float(sampling["schema_exclusion_rate"])
        <= 0.01,
        "candidate_ceiling_complete_rate_at_least_0_99": ceiling >= 0.99,
        "support_verifier_balanced_accuracy_at_least_0_75": balanced_accuracy >= 0.75,
        "answer_bearing_span_pass_rate_at_least_0_75": answer_pass >= 0.75,
        "no_answer_rejection_rate_at_least_0_65": no_answer_reject >= 0.65,
        "invalid_output_rate_at_most_0_02": float(
            verifier_summary["invalid_output_rate"]
        )
        <= 0.02,
        "gated_exact_anchor_minus_ungated_exact_anchor_point_at_least_0_1": (
            exact_improvement["point"] >= 0.1
        ),
        "gated_exact_anchor_minus_ungated_exact_anchor_ci_low_above_0": (
            exact_improvement["ci_low"] > 0.0
        ),
        "candidate_minus_gated_exact_anchor_point_at_least_0_05": (
            candidate_exact["point"] >= 0.05
        ),
        "candidate_minus_gated_exact_anchor_ci_low_above_0": (
            candidate_exact["ci_low"] > 0.0
        ),
        "candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_05": (
            candidate_non_frc["point"] >= 0.05
        ),
        "candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0": (
            candidate_non_frc["ci_low"] > 0.0
        ),
        "candidate_minus_strongest_same_gate_frc_point_at_least_minus_0_005": (
            candidate_frc["point"] >= -0.005
        ),
        "answer_recall_drop_vs_ungated_exact_anchor_at_most_0_1": recall_drop <= 0.1,
        "candidate_no_answer_abstention_accuracy_at_least_0_65": (
            candidate["no_answer_abstention_accuracy"] >= 0.65
        ),
        "candidate_abstention_rate_at_least_0_2": candidate["abstention_rate"] >= 0.2,
        "candidate_abstention_rate_at_most_0_7": candidate["abstention_rate"] <= 0.7,
        "every_budget_and_supported_stratum_delta_at_least_minus_0_02": (
            minimum_delta >= -0.02
        ),
        "deterministic_query_fallback_rate_equals_0": query_summary["fallback_rate"]
        == 0.0,
        "score_fallback_rate_equals_0": float(
            source_artifacts.get("score_fallback_rate", 0.0)
        )
        == 0.0,
    }
    supported = all(checks.values())
    if stage == "development":
        status = (
            "SQUAD2_V59_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
            if supported
            else "SQUAD2_V59_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "SQUAD2_V59_EXTRACTIVE_SUPPORT_FRC_FEASIBILITY_ESTABLISHED"
            if supported
            else "SQUAD2_V59_EXTRACTIVE_SUPPORT_FRC_FEASIBILITY_NOT_ESTABLISHED"
        )
    report = {
        "schema_version": "frc-squad2-extractive-support-report-v59",
        "experiment_id": EXPERIMENT_ID,
        "metadata": {
            "dataset_id": DATASET_ID,
            "stage": stage,
            "split": SOURCE_FILES[stage],
            "cases": len(evidence),
            "articles": articles,
            "paragraphs": paragraphs,
            "answer_state_counts": dict(state_counts),
            "budgets": list(BUDGETS),
            "official_squad2_answer_string_result": False,
            "balanced_mechanism_sample_not_natural_prevalence": True,
            "gold_joined_after_complete_score_and_generation_caches": True,
            "v58_case_level_artifact_reused": False,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "mechanism": (
                "one shared Qwen extract-or-UNSUPPORTED decision followed by "
                "frozen selectors; candidate is transferred v43 adaptive FRC"
            ),
            "support_verifier": {
                "balanced_accuracy": round(balanced_accuracy, 6),
                "answer_bearing_span_pass_rate": round(answer_pass, 6),
                "no_answer_rejection_rate": round(no_answer_reject, 6),
                **verifier_summary,
            },
            "aggregates": aggregates,
            "strongest_shared_gate_non_frc": strongest_non_frc,
            "strongest_same_gate_frc": strongest_same_gate_frc,
            "family_comparison": comparisons,
            "answer_recall_drop_vs_ungated_exact_anchor": recall_drop,
            "candidate_ceiling_complete_rate": round(ceiling, 6),
            "budget_deltas": budget_deltas,
            "supported_stratum_deltas": strata,
            "minimum_budget_or_supported_stratum_delta": round(minimum_delta, 6),
            "query_cache": query_summary,
            "support_checks": checks,
            "outcome": {
                "status": status,
                "support_established": supported,
                "confirmation_open_authorized": stage == "development" and supported,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "reuse_stage_for_tuning_or_selection": False,
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
    verifier = analysis["support_verifier"]
    lines = [
        f"# SQuAD 2.0 extractive support gate ({rounded['metadata']['stage']}, v59)",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases/articles/paragraphs: {rounded['metadata']['cases']}/{rounded['metadata']['articles']}/{rounded['metadata']['paragraphs']}",
        f"- Verifier balanced accuracy: {verifier['balanced_accuracy']:.6f}",
        f"- Answer-bearing span pass rate: {verifier['answer_bearing_span_pass_rate']:.6f}",
        f"- No-answer rejection rate: {verifier['no_answer_rejection_rate']:.6f}",
        f"- Invalid-output rate: {verifier['invalid_output_rate']:.6f}",
        f"- Candidate utility F1: {candidate['answer_or_abstention_macro_f1']:.6f}",
        f"- Candidate answer F1: {candidate['answer_bearing_macro_f1']:.6f}",
        f"- Candidate no-answer abstention: {candidate['no_answer_abstention_accuracy']:.6f}",
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
            "This is a balanced closed-paragraph SQuAD 2.0 mechanism experiment on a v58-disjoint sample. It is not official answer-string scoring, hidden-test evaluation, open-corpus retrieval, SetR reproduction, selector adoption, or flood-domain validation.",
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
        raise ValueError("SQuAD2 v59 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    if value["v58_boundary"]["v58_confirmation_dev_opened"] is not False:
        raise ValueError("SQuAD2 v59 dev boundary changed")
    if value["scope"]["v58_development_overlap"] != 0:
        raise ValueError("SQuAD2 v59 overlap rule changed")
    if value["frozen_extractive_prompt"] != EXTRACTIVE_PROMPT:
        raise ValueError("SQuAD2 v59 extractive prompt changed")
    return value


def validate_source_contract(
    source_registration_path: Path,
    *,
    protocol_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    if (
        _sha256(source_registration_path)
        != protocol["source_and_split"]["source_registration_sha256"]
    ):
        raise ValueError("SQuAD2 v59 source registration changed")
    return v58.validate_source_registration(
        source_registration_path,
        protocol_path=(
            protocol_path.resolve().parents[2]
            / "docs/progressive_upgrade/squad2_generative_answerability_gate_protocol_v58.json"
        ),
        source_root=source_root,
    )


def validate_implementation_registration(
    registration_path: Path,
    *,
    protocol_path: Path,
    source_registration_path: Path,
    module_path: Path,
    runner_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    expected = {
        "protocol": _sha256(protocol_path),
        "source_registration": _sha256(source_registration_path),
        "module": _sha256(module_path),
        "runner": _sha256(runner_path),
        "tests": _sha256(test_path),
    }
    if value.get("hashes") != expected:
        raise ValueError("SQuAD2 v59 implementation registration changed")
    if value.get("v59_sample_constructed_before_registration") is not False:
        raise ValueError("SQuAD2 v59 sample was constructed before freeze")
    if value.get("dev_content_read") is not False:
        raise ValueError("SQuAD2 v59 dev was opened before implementation freeze")
    return value


__all__ = [
    "BUDGETS",
    "CANDIDATE",
    "EXACT_ANCHOR_GATED",
    "EXACT_ANCHOR_UNGATED",
    "EXPERIMENT_ID",
    "EXTRACTIVE_PROMPT",
    "FrozenSquad2Scorer",
    "GATED_FRC_CONTROLS",
    "GATED_NON_FRC_METHODS",
    "LocalQwenExtractiveVerifier",
    "METHODS",
    "SOURCE_FILES",
    "STAGES",
    "TARGET_CASES",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "evaluate_stage",
    "extractive_prompt",
    "load_v58_excluded_commitments",
    "parse_extractive_support",
    "prepare_blind_cases",
    "read_stage_source",
    "select_disjoint_balanced_sample",
    "select_v59",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_contract",
    "validate_verifier_cache",
    "write_report",
]
