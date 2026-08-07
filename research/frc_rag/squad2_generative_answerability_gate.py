"""Prospective SQuAD 2.0 generative answerability gate experiment (v58)."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import research.frc_rag.doc2dial_document_contrastive_role_closure as v54
import research.frc_rag.quac_anchor_safe_consensus_slot as v57
from research.frc_rag.evidence_inference_low_core_divergence_atomic_roles import (
    FRC_CONTROLS,
    LOW_CORE_DIVERGENCE_V49,
    METHODS as V49_METHODS,
    FrozenEvidenceInferenceScorer,
    _round_for_display,
    select_v49,
)
from research.frc_rag.hover_dynamic_atomic_roles import (
    DYNAMIC_ROLES,
    merge_scored_candidates,
)
from research.frc_rag.rgb_cost_aware_frc import read_jsonl
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import NON_FRC_BASELINES


SCHEMA_VERSION = "frc-squad2-generative-answerability-gate-v58"
EXPERIMENT_ID = "FRC-SQUAD2-GENERATIVE-ANSWERABILITY-GATE-V58"
DATASET_ID = "squad2_balanced_answerability_evidence_selection_v58"
CAPABILITY = "closed_paragraph_answerability_and_evidence_selection"
PROTOCOL_SHA256 = "fc84cadc924ce091999ddfaa6419aad52d73fe420806310f65dae9a383f79b84"

STAGES = ("development", "confirmation")
SOURCE_FILES = {
    "development": "train-v2.0.json",
    "confirmation": "dev-v2.0.json",
}
SOURCE_URLS = {
    "development": (
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    ),
    "confirmation": (
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    ),
}
SAMPLE_SALTS = {
    "development": "FRC-SQUAD2-V58-DEVELOPMENT|",
    "confirmation": "FRC-SQUAD2-V58-CONFIRMATION|",
}
TARGET_CASES = 600
TARGET_PER_GROUP = 300
MAX_CASES_PER_PARAGRAPH = 2
MAX_CASES_PER_ARTICLE_PER_STATE = 12
MINIMUM_PARAGRAPHS = 250
MINIMUM_ARTICLES = {"development": 100, "confirmation": 25}
MINIMUM_STRATUM_CASES = 50
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260818
BUDGETS = (128, 256, 512)

SUPPORT_PROMPT = (
    "Determine whether the paragraph explicitly contains enough information to "
    "answer the question. Return exactly one token and no explanation: SUPPORTED "
    "or UNSUPPORTED. Choose SUPPORTED only when the answer is stated or can be "
    "directly resolved from the paragraph. Choose UNSUPPORTED when the paragraph "
    "is merely related, lacks the needed fact, or the question assumes something "
    "the paragraph does not establish. Do not use outside knowledge.\n\n"
    "Question: {question}\n\nParagraph:\n{context}\n\nDecision:"
)

QUERY_TEMPLATES = {
    "anchor": "{question}",
    "first_fact": (
        "Find the sentence that directly contains the answer to this question: "
        "{question}"
    ),
    "second_fact_or_bridge": (
        "Find context, definitions, entities, or relations needed to resolve this "
        "question: {question}"
    ),
    "counterevidence": (
        "Find negations, missing assumptions, contrasts, qualifications, dates, "
        "or limitations relevant to this question: {question}"
    ),
}

GATED_NON_FRC_BY_BASE = {
    method: f"qwen_supported_{method}_v58" for method in NON_FRC_BASELINES
}
GATED_FRC_BY_BASE = {method: f"qwen_supported_{method}_v58" for method in FRC_CONTROLS}
GATED_NON_FRC_METHODS = tuple(GATED_NON_FRC_BY_BASE.values())
GATED_FRC_CONTROLS = tuple(GATED_FRC_BY_BASE.values())
CANDIDATE = "qwen_supported_low_core_divergence_guarded_frc_v49_transferred_v58"
EXACT_ANCHOR_GATED = GATED_NON_FRC_BY_BASE["cross_encoder_topk"]
EXACT_ANCHOR_UNGATED = "cross_encoder_topk"
METHODS = (*V49_METHODS, *GATED_NON_FRC_METHODS, *GATED_FRC_CONTROLS, CANDIDATE)
GATED_BASE_BY_METHOD = {
    **{value: key for key, value in GATED_NON_FRC_BY_BASE.items()},
    **{value: key for key, value in GATED_FRC_BY_BASE.items()},
    CANDIDATE: LOW_CORE_DIVERGENCE_V49,
}

_FORBIDDEN_BLIND_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answer_start",
        "answer_end",
        "answer_state",
        "is_impossible",
        "plausible_answers",
        "title",
        "raw_id",
        "source_case_commitment",
    }
)


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


def _normalise(value: Any) -> str:
    return " ".join(str(value or "").split())


def _contains_forbidden_blind_key(value: Any) -> bool:
    if isinstance(value, dict):
        if _FORBIDDEN_BLIND_KEYS & {str(key).lower() for key in value}:
            return True
        return any(_contains_forbidden_blind_key(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_blind_key(item) for item in value)
    return False


def read_stage_source(source_root: Path, stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported SQuAD2 v58 stage: {stage}")
    value = json.loads(
        (source_root / SOURCE_FILES[stage]).read_text(encoding="utf-8-sig")
    )
    if not isinstance(value, dict):
        raise ValueError("SQuAD2 v58 source root must be an object")
    return value


def iter_paragraphs(source: dict[str, Any]) -> Iterable[dict[str, Any]]:
    data = source.get("data")
    if not isinstance(data, list):
        raise ValueError("SQuAD2 v58 data must be a list")
    seen_paragraphs: set[str] = set()
    for article_index, article in enumerate(data):
        if not isinstance(article, dict):
            raise ValueError("SQuAD2 v58 article must be an object")
        title = _normalise(article.get("title"))
        paragraphs = article.get("paragraphs")
        if not isinstance(paragraphs, list):
            raise ValueError("SQuAD2 v58 paragraphs must be a list")
        article_id = _hash("article", str(article_index), title)
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                raise ValueError("SQuAD2 v58 paragraph must be an object")
            context = str(paragraph.get("context", ""))
            paragraph_id = _hash("paragraph", article_id, str(paragraph_index), context)
            if paragraph_id in seen_paragraphs:
                raise ValueError("SQuAD2 v58 paragraph commitment is not unique")
            seen_paragraphs.add(paragraph_id)
            yield {
                "article_id": article_id,
                "paragraph_id": paragraph_id,
                "context": context,
                "qas": paragraph.get("qas"),
            }


def extract_cases(
    source: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    candidate_qas = 0
    seen_ids: set[str] = set()
    for paragraph in iter_paragraphs(source):
        context = str(paragraph["context"])
        qas = paragraph["qas"]
        if not context or not isinstance(qas, list):
            count = len(qas) if isinstance(qas, list) else 1
            candidate_qas += count
            exclusions["missing_context_or_qa_list"] += count
            continue
        for qa in qas:
            candidate_qas += 1
            if not isinstance(qa, dict):
                exclusions["qa_not_object"] += 1
                continue
            raw_id = _normalise(qa.get("id"))
            question = _normalise(qa.get("question"))
            impossible = qa.get("is_impossible")
            answers = qa.get("answers")
            if not raw_id:
                exclusions["empty_id"] += 1
                continue
            if raw_id in seen_ids:
                raise ValueError("SQuAD2 v58 QA id is not unique")
            seen_ids.add(raw_id)
            if not question:
                exclusions["empty_question"] += 1
                continue
            if not isinstance(impossible, bool):
                exclusions["invalid_is_impossible"] += 1
                continue
            if not isinstance(answers, list):
                exclusions["answers_not_list"] += 1
                continue
            intervals: list[tuple[int, int]] = []
            answer_lengths: list[int] = []
            if not impossible:
                if not answers:
                    exclusions["answerable_without_answers"] += 1
                    continue
                valid = True
                for answer in answers:
                    if not isinstance(answer, dict):
                        valid = False
                        break
                    text = str(answer.get("text", ""))
                    try:
                        start = int(answer.get("answer_start"))
                    except (TypeError, ValueError):
                        valid = False
                        break
                    end = start + len(text)
                    if not text or start < 0 or end > len(context):
                        valid = False
                        break
                    if context[start:end] != text:
                        valid = False
                        break
                    intervals.append((start, end))
                    answer_lengths.append(len(text))
                if not valid:
                    exclusions["invalid_answer_interval"] += 1
                    continue
            rows.append(
                {
                    "raw_id": raw_id,
                    "article_id": str(paragraph["article_id"]),
                    "paragraph_id": str(paragraph["paragraph_id"]),
                    "context": context,
                    "question": question,
                    "answer_state": "no_answer" if impossible else "answer_bearing",
                    "answer_intervals": intervals,
                    "answer_lengths": answer_lengths,
                }
            )
    excluded = sum(exclusions.values())
    return rows, {
        "candidate_qas": candidate_qas,
        "eligible_cases": len(rows),
        "schema_excluded_cases": excluded,
        "schema_exclusion_rate": excluded / candidate_qas if candidate_qas else 1.0,
        "schema_exclusion_reasons": dict(sorted(exclusions.items())),
    }


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
    target_per_group: int = TARGET_PER_GROUP,
    maximum_cases_per_paragraph: int = MAX_CASES_PER_PARAGRAPH,
    maximum_cases_per_article_per_state: int = MAX_CASES_PER_ARTICLE_PER_STATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported SQuAD2 v58 stage: {stage}")
    rows, schema = extract_cases(source)
    groups = ("answer_bearing", "no_answer")
    by_group = {
        group: sorted(
            [row for row in rows if row["answer_state"] == group],
            key=lambda row: _sample_key(row, stage),
        )
        for group in groups
    }
    if any(len(by_group[group]) < target_per_group for group in groups):
        raise ValueError("SQuAD2 v58 source lacks the answer-state quota")
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
            raise ValueError("SQuAD2 v58 caps prevent the balanced sample")
    selected.sort(key=lambda row: (str(row["answer_state"]), _sample_key(row, stage)))
    return selected, {
        "stage": stage,
        "target_cases": target_per_group * 2,
        "selected_cases": len(selected),
        "eligible_answer_state_counts": dict(
            Counter(row["answer_state"] for row in rows)
        ),
        "selected_answer_state_counts": dict(selected_by_group),
        "selected_paragraphs": len(paragraph_counts),
        "selected_articles": len({str(row["article_id"]) for row in selected}),
        "maximum_cases_per_paragraph": max(paragraph_counts.values(), default=0),
        "maximum_cases_per_article_per_answer_state": max(
            article_state_counts.values(), default=0
        ),
        **schema,
    }


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"minimum": 0, "mean": 0.0, "maximum": 0}
    return {
        "minimum": min(values),
        "mean": round(float(np.mean(values)), 6),
        "maximum": max(values),
    }


def _quartile(value: int, boundaries: Sequence[float]) -> str:
    if value <= float(boundaries[0]):
        return "q1"
    if value <= float(boundaries[1]):
        return "q2"
    if value <= float(boundaries[2]):
        return "q3"
    return "q4"


def canonical_candidate_rows(
    context: str, paragraph_id: str, tokenizer: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sentence_start, sentence_end in v57._sentence_intervals(context):
        for start, end in v57._window_interval(
            context, sentence_start, sentence_end, tokenizer
        ):
            text = context[start:end]
            if not text:
                continue
            identifier = _opaque("s58u", paragraph_id, str(start), str(end))
            rows.append(
                {
                    "id": identifier,
                    "source_id": identifier,
                    "text": text,
                    "token_count": v57._token_count(tokenizer, text),
                    "start": start,
                    "end": end,
                }
            )
    return rows


def prepare_blind_cases(
    selected: Sequence[dict[str, Any]], tokenizer: Any, *, stage: str
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    prepared: list[dict[str, Any]] = []
    maps: list[dict[str, Any]] = []
    verifier_inputs: list[dict[str, Any]] = []
    pool_sizes: list[int] = []
    token_costs: list[int] = []
    paragraph_lengths: list[int] = []
    article_case_counts = Counter(str(row["article_id"]) for row in selected)
    unit_cache: dict[str, list[dict[str, Any]]] = {}
    for source_row in selected:
        paragraph_id = str(source_row["paragraph_id"])
        candidates = unit_cache.get(paragraph_id)
        if candidates is None:
            candidates = canonical_candidate_rows(
                str(source_row["context"]), paragraph_id, tokenizer
            )
            if not candidates:
                raise ValueError("SQuAD2 v58 paragraph has no candidate units")
            unit_cache[paragraph_id] = candidates
        case_id = _opaque("s58c", stage, str(source_row["raw_id"]))
        blind = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "capability": CAPABILITY,
            "stage": stage,
            "id": case_id,
            "query": str(source_row["question"]),
            "candidates": [
                {
                    "id": str(row["id"]),
                    "source_id": str(row["source_id"]),
                    "text": str(row["text"]),
                    "token_count": int(row["token_count"]),
                }
                for row in candidates
            ],
            "gold_fields_visible_to_scorer": False,
        }
        verifier = {
            "schema_version": "frc-squad2-v58-verifier-input-v1",
            "dataset_id": DATASET_ID,
            "stage": stage,
            "id": case_id,
            "question": str(source_row["question"]),
            "context": str(source_row["context"]),
            "gold_fields_visible_to_verifier": False,
        }
        if _contains_forbidden_blind_key(blind) or _contains_forbidden_blind_key(
            verifier
        ):
            raise AssertionError("SQuAD2 v58 gold leaked into a blind cache")
        prepared.append(blind)
        verifier_inputs.append(verifier)
        maps.append(
            {
                "schema_version": "frc-squad2-v58-candidate-map-v1",
                "id": case_id,
                "source_case_commitment": _hash(str(source_row["raw_id"])),
                "article_commitment": _hash(str(source_row["article_id"])),
                "paragraph_commitment": _hash(str(source_row["paragraph_id"])),
                "paragraph_length": len(str(source_row["context"])),
                "article_selected_case_count": int(
                    article_case_counts[str(source_row["article_id"])]
                ),
                "candidate_intervals": [
                    {
                        "candidate_id": str(row["id"]),
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                    }
                    for row in candidates
                ],
            }
        )
        pool_sizes.append(len(candidates))
        token_costs.extend(int(row["token_count"]) for row in candidates)
        paragraph_lengths.append(len(str(source_row["context"])))
    pool_boundaries = [
        round(float(value), 6) for value in np.quantile(pool_sizes, [0.25, 0.5, 0.75])
    ]
    paragraph_boundaries = [
        round(float(value), 6)
        for value in np.quantile(paragraph_lengths, [0.25, 0.5, 0.75])
    ]
    article_count_boundaries = [
        round(float(value), 6)
        for value in np.quantile(list(article_case_counts.values()), [0.25, 0.5, 0.75])
    ]
    return (
        prepared,
        maps,
        verifier_inputs,
        {
            "stage": stage,
            "prepared_cases": len(prepared),
            "paragraphs": len(unit_cache),
            "articles": len(article_case_counts),
            "candidate_pool_size": _distribution(pool_sizes),
            "candidate_pool_quartile_boundaries": pool_boundaries,
            "candidate_token_cost": _distribution(token_costs),
            "paragraph_length_codepoints": _distribution(paragraph_lengths),
            "paragraph_length_quartile_boundaries": paragraph_boundaries,
            "article_selected_case_count": _distribution(
                list(article_case_counts.values())
            ),
            "article_case_count_quartile_boundaries": article_count_boundaries,
            "gold_fields_exported_to_blind_caches": False,
            "ready_for_query_generation_and_verification": (
                len(prepared) == len(selected) == len(verifier_inputs)
            ),
        },
    )


def build_deterministic_queries(
    prepared_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in prepared_rows:
        if _contains_forbidden_blind_key(row):
            raise ValueError("SQuAD2 v58 query builder received a forbidden field")
        question = str(row["query"])
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": DATASET_ID,
                "stage": str(row["stage"]),
                "id": str(row["id"]),
                "atomic_queries": {
                    role: QUERY_TEMPLATES[role].format(question=question)
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
        raise ValueError("SQuAD2 v58 query cache is incomplete or out of order")
    if list(query_rows) != build_deterministic_queries(prepared_rows):
        raise ValueError("SQuAD2 v58 deterministic query cache changed")
    return {
        "rows": len(query_rows),
        "fallback_count": 0,
        "fallback_rate": 0.0,
        "gold_fields_visible_to_generator": False,
        "deterministic_templates": True,
    }


class FrozenSquad2Scorer(FrozenEvidenceInferenceScorer):
    """Reuse the frozen v49 BGE stack with v58 metadata labels."""

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = super().score_cases(cases)
        for row in rows:
            row["schema_version"] = SCHEMA_VERSION
            row["dataset_id"] = DATASET_ID
            row["capability"] = CAPABILITY
        return rows


def support_prompt(row: dict[str, Any]) -> str:
    if _contains_forbidden_blind_key(row):
        raise ValueError("SQuAD2 v58 support prompt received a forbidden field")
    return SUPPORT_PROMPT.format(
        question=str(row["question"]), context=str(row["context"])
    )


def normalize_support_decision(text: str) -> str | None:
    normalized = text.strip().upper()
    return normalized if normalized in {"SUPPORTED", "UNSUPPORTED"} else None


class LocalQwenSupportVerifier:
    """Frozen local deterministic paragraph-support classifier."""

    def __init__(
        self,
        *,
        model_path: Path,
        batch_size: int = 16,
        max_new_tokens: int = 4,
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
            raise ValueError("SQuAD2 v58 verifier cache is not a valid prefix")
        if any(row.get("cache_key") != cache_key for row in existing):
            raise ValueError("SQuAD2 v58 verifier cache key changed")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if existing else "w"
        with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
            for start in range(len(existing), len(rows), self.batch_size):
                batch = list(rows[start : start + self.batch_size])
                prompts = [self._chat_prompt(support_prompt(row)) for row in batch]
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
                    decision = normalize_support_decision(raw)
                    prediction = {
                        "schema_version": "frc-squad2-v58-support-decision-v1",
                        "experiment_id": EXPERIMENT_ID,
                        "stage": str(source["stage"]),
                        "id": str(source["id"]),
                        "cache_key": cache_key,
                        "raw_prediction": raw.strip(),
                        "decision": decision,
                        "support_passed": decision == "SUPPORTED",
                        "malformed_fallback_used": decision is None,
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
                    "SQuAD2 cases",
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
        raise ValueError("SQuAD2 v58 verifier cache is incomplete or out of order")
    if any(row.get("cache_key") != cache_key for row in decisions):
        raise ValueError("SQuAD2 v58 verifier cache key changed")
    malformed = sum(bool(row.get("malformed_fallback_used")) for row in decisions)
    return {
        "rows": len(decisions),
        "malformed_or_fallback_count": malformed,
        "malformed_or_fallback_rate": malformed / len(decisions) if decisions else 1.0,
        "gold_fields_visible_to_verifier": False,
    }


def _source_case_index(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows, _ = extract_cases(source)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _hash(str(row["raw_id"]))
        if key in result:
            raise ValueError("SQuAD2 v58 source commitment is not unique")
        result[key] = row
    return result


def build_gold_rows(
    source: dict[str, Any],
    candidate_maps: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    structural_census: dict[str, Any],
) -> list[dict[str, Any]]:
    source_cases = _source_case_index(source)
    scored_by_id = {str(row["id"]): row for row in scored_rows}
    answer_lengths = [
        min(row["answer_lengths"])
        for row in source_cases.values()
        if row["answer_lengths"]
    ]
    answer_boundaries = [
        round(float(value), 6)
        for value in np.quantile(answer_lengths, [0.25, 0.5, 0.75])
    ]
    pool_boundaries = structural_census["candidate_pool_quartile_boundaries"]
    paragraph_boundaries = structural_census["paragraph_length_quartile_boundaries"]
    article_boundaries = structural_census["article_case_count_quartile_boundaries"]
    result: list[dict[str, Any]] = []
    for candidate_map in candidate_maps:
        source_row = source_cases.get(str(candidate_map["source_case_commitment"]))
        scored = scored_by_id.get(str(candidate_map["id"]))
        if source_row is None or scored is None:
            raise ValueError("SQuAD2 v58 gold join is incomplete")
        gold_ids: set[str] = set()
        for answer_start, answer_end in source_row["answer_intervals"]:
            for interval in candidate_map["candidate_intervals"]:
                if int(interval["start"]) < int(answer_end) and int(
                    interval["end"]
                ) > int(answer_start):
                    gold_ids.add(str(interval["candidate_id"]))
        available = {str(row["id"]) for row in scored["candidates"]}
        pool_size = len(available)
        answer_length = (
            min(source_row["answer_lengths"]) if source_row["answer_lengths"] else 0
        )
        result.append(
            {
                "case_id": str(candidate_map["id"]),
                "article_cluster": str(candidate_map["article_commitment"]),
                "paragraph_cluster": str(candidate_map["paragraph_commitment"]),
                "answer_state": str(source_row["answer_state"]),
                "gold_candidate_ids": sorted(gold_ids),
                "candidate_unit_count": pool_size,
                "candidate_pool_quartile": _quartile(pool_size, pool_boundaries),
                "paragraph_length_quartile": _quartile(
                    int(candidate_map["paragraph_length"]), paragraph_boundaries
                ),
                "answer_length_quartile": (
                    _quartile(answer_length, answer_boundaries)
                    if gold_ids
                    else "not_applicable"
                ),
                "article_case_count_quartile": _quartile(
                    int(candidate_map["article_selected_case_count"]),
                    article_boundaries,
                ),
                "candidate_ceiling_complete": (
                    bool(gold_ids and gold_ids <= available)
                    if source_row["answer_state"] == "answer_bearing"
                    else True
                ),
            }
        )
    return result


def build_candidate_coverage(
    gold_rows: Sequence[dict[str, Any]], sampling: dict[str, Any]
) -> dict[str, Any]:
    answer = [row for row in gold_rows if row["answer_state"] == "answer_bearing"]
    ceiling = (
        float(np.mean([row["candidate_ceiling_complete"] for row in answer]))
        if answer
        else 0.0
    )
    return {
        "schema_version": "frc-squad2-v58-candidate-coverage-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": sampling.get("stage"),
        "cases": len(gold_rows),
        "articles": len({row["article_cluster"] for row in gold_rows}),
        "paragraphs": len({row["paragraph_cluster"] for row in gold_rows}),
        "answer_bearing_cases": len(answer),
        "no_answer_cases": len(gold_rows) - len(answer),
        "candidate_ceiling_complete_rate_on_answer_bearing_cases": round(ceiling, 6),
        "sampling": sampling,
    }


def select_v58(
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
        raise ValueError(f"Unsupported SQuAD2 v58 method: {method}")
    if not support_passed:
        return []
    return select_v49(values, base, token_budget=token_budget)


def _selection_metrics(
    selected: Sequence[dict[str, Any]], answer_state: str, gold_ids: Sequence[str]
) -> dict[str, Any]:
    return v54._selection_metrics(selected, answer_state, gold_ids)


def _aggregate(rows: Sequence[dict[str, Any]], method: str) -> dict[str, float]:
    return v54._aggregate(rows, method)


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
            -_aggregate(rows, method)["answer_or_abstention_macro_f1"],
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
        raise ValueError("SQuAD2 v58 gold, score and support caches differ")
    support_by_id = {str(row["id"]): row for row in support_rows}
    evidence: list[dict[str, Any]] = []
    for gold, scored in zip(gold_rows, scored_rows, strict=True):
        if str(gold["case_id"]) != str(scored["id"]):
            raise ValueError("SQuAD2 v58 gold and score ids differ")
        support = support_by_id.get(str(gold["case_id"]))
        if support is None:
            raise ValueError("SQuAD2 v58 support decision is missing")
        support_passed = bool(support["support_passed"])
        candidates = merge_scored_candidates(dict(scored))
        configurations: dict[str, Any] = {}
        for budget in BUDGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                selected = select_v58(
                    candidates,
                    method,
                    support_passed=support_passed,
                    token_budget=budget,
                )
                methods[method] = {
                    "selected_ids": [str(item["id"]) for item in selected],
                    "metrics": _selection_metrics(
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
                    "malformed_fallback_used": bool(
                        support.get("malformed_fallback_used")
                    ),
                },
                "configurations": configurations,
            }
        )
    aggregates = {method: _aggregate(evidence, method) for method in METHODS}
    strongest_non_frc = _strongest(evidence, GATED_NON_FRC_METHODS)
    strongest_same_gate_frc = _strongest(evidence, GATED_FRC_CONTROLS)
    comparisons = {
        "gated_exact_anchor_minus_ungated_exact_anchor": _cluster_bootstrap(
            evidence, EXACT_ANCHOR_GATED, EXACT_ANCHOR_UNGATED
        ),
        "candidate_minus_exact_anchor_gated_ablation": _cluster_bootstrap(
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
    verifier_balanced_accuracy = (answer_pass + no_answer_reject) / 2.0
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
    candidate_exact = comparisons["candidate_minus_exact_anchor_gated_ablation"]
    candidate_non_frc = comparisons["candidate_minus_strongest_shared_gate_non_frc"]
    candidate_frc = comparisons["candidate_minus_strongest_same_gate_frc"]
    checks = {
        "exact_cases_equals_600": len(evidence) == TARGET_CASES,
        "exact_answer_state_balance": state_counts
        == {"answer_bearing": TARGET_PER_GROUP, "no_answer": TARGET_PER_GROUP},
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
        "support_verifier_balanced_accuracy_at_least_0_75": (
            verifier_balanced_accuracy >= 0.75
        ),
        "answer_bearing_support_pass_rate_at_least_0_75": answer_pass >= 0.75,
        "no_answer_support_rejection_rate_at_least_0_75": no_answer_reject >= 0.75,
        "malformed_or_fallback_rate_equals_0": float(
            verifier_summary["malformed_or_fallback_rate"]
        )
        == 0.0,
        "gated_exact_anchor_minus_ungated_exact_anchor_point_at_least_0_1": (
            exact_improvement["point"] >= 0.1
        ),
        "gated_exact_anchor_minus_ungated_exact_anchor_ci_low_above_0": (
            exact_improvement["ci_low"] > 0.0
        ),
        "candidate_minus_exact_anchor_gated_ablation_point_at_least_0_005": (
            candidate_exact["point"] >= 0.005
        ),
        "candidate_minus_exact_anchor_gated_ablation_ci_low_above_0": (
            candidate_exact["ci_low"] > 0.0
        ),
        "candidate_minus_strongest_shared_gate_non_frc_point_at_least_0_005": (
            candidate_non_frc["point"] >= 0.005
        ),
        "candidate_minus_strongest_shared_gate_non_frc_ci_low_above_0": (
            candidate_non_frc["ci_low"] > 0.0
        ),
        "candidate_minus_strongest_same_gate_frc_point_at_least_0_005": (
            candidate_frc["point"] >= 0.005
        ),
        "candidate_minus_strongest_same_gate_frc_ci_low_above_0": (
            candidate_frc["ci_low"] > 0.0
        ),
        "answer_recall_drop_vs_ungated_exact_anchor_at_most_0_1": (recall_drop <= 0.1),
        "candidate_no_answer_abstention_accuracy_at_least_0_75": (
            candidate["no_answer_abstention_accuracy"] >= 0.75
        ),
        "candidate_abstention_rate_at_least_0_25": candidate["abstention_rate"] >= 0.25,
        "candidate_abstention_rate_at_most_0_75": candidate["abstention_rate"] <= 0.75,
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
            "SQUAD2_V58_DEVELOPMENT_SUPPORT_ESTABLISHED_OPEN_CONFIRMATION"
            if supported
            else "SQUAD2_V58_DEVELOPMENT_SUPPORT_NOT_ESTABLISHED_STOP_BEFORE_CONFIRMATION"
        )
    else:
        status = (
            "SQUAD2_V58_GENERATIVE_ANSWERABILITY_GATE_SUPPORT_ESTABLISHED"
            if supported
            else "SQUAD2_V58_GENERATIVE_ANSWERABILITY_GATE_SUPPORT_NOT_ESTABLISHED"
        )
    report = {
        "schema_version": "frc-squad2-generative-answerability-report-v58",
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
            "v57_case_level_artifact_reused": False,
            "source_artifacts": source_artifacts,
        },
        "analysis": {
            "mechanism": (
                "one shared full-paragraph Qwen support decision followed by frozen "
                "selection methods; candidate is transferred v49 low-core FRC"
            ),
            "support_verifier": {
                "balanced_accuracy": round(verifier_balanced_accuracy, 6),
                "answer_bearing_pass_rate": round(answer_pass, 6),
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
        f"# SQuAD 2.0 generative answerability gate ({rounded['metadata']['stage']}, v58)",
        "",
        f"- Status: `{outcome['status']}`",
        f"- Cases/articles/paragraphs: {rounded['metadata']['cases']}/{rounded['metadata']['articles']}/{rounded['metadata']['paragraphs']}",
        f"- Verifier balanced accuracy: {verifier['balanced_accuracy']:.6f}",
        f"- Answer-bearing pass rate: {verifier['answer_bearing_pass_rate']:.6f}",
        f"- No-answer rejection rate: {verifier['no_answer_rejection_rate']:.6f}",
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
            "This is a balanced closed-paragraph SQuAD 2.0 mechanism experiment. It is not official answer-string scoring, hidden-test evaluation, open-corpus retrieval, SetR reproduction, selector adoption, or flood-domain validation.",
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
        raise ValueError("SQuAD2 v58 protocol hash changed")
    value = json.loads(protocol_path.read_text(encoding="utf-8"))
    boundary = value["prior_result_boundary"]
    if (
        boundary[
            "squad2_train_or_dev_downloaded_or_opened_before_protocol_registration"
        ]
        is not False
    ):
        raise ValueError("SQuAD2 v58 source was opened before registration")
    if value["scope"]["development_cases"] != TARGET_CASES:
        raise ValueError("SQuAD2 v58 development size changed")
    if value["methods"]["budgets"] != list(BUDGETS):
        raise ValueError("SQuAD2 v58 budgets changed")
    if value["frozen_support_prompt"] != SUPPORT_PROMPT:
        raise ValueError("SQuAD2 v58 support prompt changed")
    return value


def validate_source_registration(
    registration_path: Path, *, protocol_path: Path, source_root: Path
) -> dict[str, Any]:
    value = json.loads(registration_path.read_text(encoding="utf-8"))
    if value.get("protocol_sha256_before_download") != _sha256(protocol_path):
        raise ValueError("SQuAD2 v58 source registration protocol hash mismatch")
    registered = {str(row["path"]): str(row["sha256"]) for row in value["files"]}
    for name in SOURCE_FILES.values():
        path = source_root / name
        if registered.get(name) != _sha256(path):
            raise ValueError(f"SQuAD2 v58 source changed: {name}")
    if value.get("json_content_parsed_before_registration") is not False:
        raise ValueError("SQuAD2 v58 source was parsed before registration")
    if value.get("dev_content_read") is not False:
        raise ValueError("SQuAD2 v58 dev was opened before registration")
    return value


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
        raise ValueError("SQuAD2 v58 implementation registration changed")
    if value.get("train_content_read_before_registration") is not False:
        raise ValueError("SQuAD2 v58 train was opened before implementation freeze")
    if value.get("dev_content_read") is not False:
        raise ValueError("SQuAD2 v58 dev was opened before implementation freeze")
    return value


__all__ = [
    "BUDGETS",
    "CANDIDATE",
    "EXACT_ANCHOR_GATED",
    "EXACT_ANCHOR_UNGATED",
    "EXPERIMENT_ID",
    "GATED_FRC_CONTROLS",
    "GATED_NON_FRC_METHODS",
    "LocalQwenSupportVerifier",
    "METHODS",
    "SOURCE_FILES",
    "SOURCE_URLS",
    "SUPPORT_PROMPT",
    "TARGET_CASES",
    "FrozenSquad2Scorer",
    "build_candidate_coverage",
    "build_deterministic_queries",
    "build_gold_rows",
    "canonical_candidate_rows",
    "evaluate_stage",
    "extract_cases",
    "normalize_support_decision",
    "prepare_blind_cases",
    "read_stage_source",
    "select_balanced_sample",
    "select_v58",
    "support_prompt",
    "validate_implementation_registration",
    "validate_protocol",
    "validate_query_cache",
    "validate_source_registration",
    "validate_verifier_cache",
    "write_report",
]
