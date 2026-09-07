from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from research.frc_rag.conflicts_evaluation import (
    local_model_snapshot,
    minmax,
    read_jsonl,
    stable_hash,
    write_jsonl,
)
from research.frc_rag.public_evidence import paired_bootstrap


LAWSHIFT_ABLATION_SCHEMA = "frc-lawshift-temporal-applicability-ablation-v1"
LAWSHIFT_SOURCE_REPOSITORY = "triangularPeach/LawShift"
LAWSHIFT_SOURCE_REVISION = "0fce4f3821140bde29081ae0b20500e79aa065d5"
LAWSHIFT_SOURCE_LICENSE = "Apache-2.0"
DEFAULT_SEED = 20260713
DEFAULT_PAIRS_PER_REVISION = 2
DEFAULT_DISTRACTOR_ARTICLES = 3
DEFAULT_TOP_K = 1
DEFAULT_TOKEN_BUDGET = 512

LAWSHIFT_REVISION_TYPES = (
    "action_explicit_extend",
    "action_explicit_reduce",
    "action_implicit_expand",
    "action_implicit_reduce",
    "action_reallocated_expand",
    "action_reallocated_reduce",
    "objCon_addition",
    "objCon_explicit_extend",
    "objCon_implicit_expand",
    "objCon_implicit_reduce",
    "objCon_reallocated_expand",
    "objCon_reallocated_reduce",
    "objCon_removal",
    "object_explicit_extend",
    "object_explicit_reduce",
    "object_implicit_expand",
    "object_implicit_reduce",
    "object_reallocated_expand",
    "object_reallocated_reduce",
    "sbCon_addition",
    "sbCon_removal",
    "subject_explicit_extend",
    "subject_explicit_reduce",
    "subject_implicit_expand",
    "subject_implicit_reduce",
    "subject_reallocated_expand",
    "subject_reallocated_reduce",
    "term_down",
    "term_extremity_in",
    "term_extremity_out",
    "term_up",
)

LAWSHIFT_METHODS = (
    "char_bm25_top1",
    "cross_encoder_top1",
    "applicability_filtered_cross_encoder_top1",
    "frc_full",
    "w/o_applicability",
)

_SOURCE_FILES = (
    "articles_original.json",
    "articles_poisoned.json",
    "original.json",
    "poisoned.json",
)
_LATIN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_SEQUENCE_RE = re.compile(r"[\u3400-\u9fff]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_order(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _tokens(value: str) -> list[str]:
    tokens = [item.lower() for item in _LATIN_RE.findall(value)]
    for sequence in _CJK_SEQUENCE_RE.findall(value):
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def _token_cost(value: str) -> int:
    latin = len(_LATIN_RE.findall(value))
    cjk = sum(len(sequence) for sequence in _CJK_SEQUENCE_RE.findall(value))
    return max(1, latin + cjk)


def _bm25_scores(query: str, texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    query_terms = _tokens(query)
    documents = [_tokens(text) for text in texts]
    if not documents:
        return []
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))
    average_length = sum(map(len, documents)) / max(1, len(documents))
    raw_scores = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1.0 - b + b * len(document) / max(1.0, average_length)
            )
            score += inverse * frequency * (k1 + 1.0) / denominator
        raw_scores.append(score)
    low = min(raw_scores)
    high = max(raw_scores)
    if math.isclose(low, high):
        return [1.0 if high > 0 else 0.0 for _ in raw_scores]
    return [(score - low) / (high - low) for score in raw_scores]


def build_source_manifest(
    root: Path,
    *,
    revision_types: Iterable[str] = LAWSHIFT_REVISION_TYPES,
) -> dict[str, Any]:
    files = []
    for revision_type in revision_types:
        for name in _SOURCE_FILES:
            path = root / revision_type / name
            if not path.is_file():
                raise FileNotFoundError(f"missing LawShift source file: {path}")
            files.append(
                {
                    "file": f"{revision_type}/{name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return {
        "repository": LAWSHIFT_SOURCE_REPOSITORY,
        "revision": LAWSHIFT_SOURCE_REVISION,
        "license": LAWSHIFT_SOURCE_LICENSE,
        "files": files,
        "sha256": stable_hash(files),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(
    *,
    revision_type: str,
    article_id: str,
    version: str,
    text: str,
) -> dict[str, Any]:
    return {
        "id": f"lawshift:{revision_type}:{article_id}:{version}",
        "article_id": article_id,
        "version": version,
        "text": text,
        "citation": f"刑法 {article_id} ({version})",
        "token_count": _token_cost(text),
        "metadata": {
            "revision_type": revision_type,
            "article_id": article_id,
            "version": version,
            "valid_from": "revision-boundary" if version == "revised" else None,
            "valid_to": "revision-boundary" if version == "original" else None,
            "source_dataset": LAWSHIFT_SOURCE_REPOSITORY,
        },
    }


def build_lawshift_temporal_cases(
    root: Path,
    *,
    revision_types: Iterable[str] = LAWSHIFT_REVISION_TYPES,
    pairs_per_revision: int = DEFAULT_PAIRS_PER_REVISION,
    distractor_articles: int = DEFAULT_DISTRACTOR_ARTICLES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Build balanced before/after retrieval cases from expert-reviewed LawShift pairs."""

    revision_types = tuple(revision_types)
    if pairs_per_revision <= 0 or distractor_articles <= 0:
        raise ValueError("LawShift pair and distractor counts must be positive")
    cases: list[dict[str, Any]] = []
    for revision_type in revision_types:
        source = root / revision_type
        original_articles = _load_json(source / "articles_original.json")
        revised_articles = _load_json(source / "articles_poisoned.json")
        original_cases = _load_json(source / "original.json")
        revised_cases = _load_json(source / "poisoned.json")
        if not isinstance(original_articles, dict) or not isinstance(revised_articles, dict):
            raise ValueError(f"invalid LawShift article maps: {revision_type}")
        if set(original_articles) != set(revised_articles):
            raise ValueError(f"LawShift article identifiers differ: {revision_type}")
        changed_articles = sorted(
            article_id
            for article_id in original_articles
            if original_articles[article_id] != revised_articles[article_id]
        )
        if len(changed_articles) != 1:
            raise ValueError(f"expected one revised article for {revision_type}")
        target_article = changed_articles[0]
        if len(original_cases) != len(revised_cases):
            raise ValueError(f"LawShift paired case counts differ: {revision_type}")
        paired_indices = []
        for index, (original_case, revised_case) in enumerate(
            zip(original_cases, revised_cases, strict=True)
        ):
            original_relevant = list(original_case.get("relevant_articles", []))
            revised_relevant = list(revised_case.get("relevant_articles", []))
            if original_relevant != revised_relevant or target_article not in original_relevant:
                raise ValueError(f"LawShift paired labels differ: {revision_type}/{index}")
            paired_indices.append(index)
        paired_indices.sort(
            key=lambda index: (
                _stable_order(
                    seed,
                    f"{revision_type}|{index}|{original_cases[index]['fact']}|"
                    f"{revised_cases[index]['fact']}",
                ),
                index,
            )
        )
        if len(paired_indices) < pairs_per_revision:
            raise ValueError(f"insufficient LawShift pairs: {revision_type}")
        for pair_rank, source_index in enumerate(
            paired_indices[:pairs_per_revision], start=1
        ):
            for version, source_case, article_map in (
                ("original", original_cases[source_index], original_articles),
                ("revised", revised_cases[source_index], revised_articles),
            ):
                fact = str(source_case["fact"]).strip()
                query = (
                    f"时间口径：{'法条修订前' if version == 'original' else '法条修订后'}。"
                    f"根据以下案件事实检索直接适用的刑法法条：\n{fact}"
                )
                distractor_ids = [
                    article_id for article_id in article_map if article_id != target_article
                ]
                distractor_texts = [str(article_map[article_id]) for article_id in distractor_ids]
                lexical = _bm25_scores(query, distractor_texts)
                ranked_distractors = sorted(
                    range(len(distractor_ids)),
                    key=lambda index: (
                        -lexical[index],
                        _stable_order(
                            seed,
                            f"distractor|{revision_type}|{source_index}|{version}|"
                            f"{distractor_ids[index]}",
                        ),
                        distractor_ids[index],
                    ),
                )[:distractor_articles]
                article_ids = [target_article, *[distractor_ids[index] for index in ranked_distractors]]
                candidates = []
                for article_id in article_ids:
                    candidates.extend(
                        (
                            _candidate(
                                revision_type=revision_type,
                                article_id=article_id,
                                version="original",
                                text=str(original_articles[article_id]),
                            ),
                            _candidate(
                                revision_type=revision_type,
                                article_id=article_id,
                                version="revised",
                                text=str(revised_articles[article_id]),
                            ),
                        )
                    )
                case_id = f"lawshift:{revision_type}:{pair_rank:02d}:{version}"
                cases.append(
                    {
                        "id": case_id,
                        "dataset": "lawshift",
                        "revision_type": revision_type,
                        "source_case_index": source_index,
                        "as_of_version": version,
                        "question": query,
                        "charge": str(source_case.get("charge", "")),
                        "prison_time": source_case.get("prison_time"),
                        "gold_article_id": target_article,
                        "gold_evidence_id": (
                            f"lawshift:{revision_type}:{target_article}:{version}"
                        ),
                        "candidates": sorted(candidates, key=lambda item: item["id"]),
                    }
                )
    expected = len(revision_types) * pairs_per_revision * 2
    if len(cases) != expected:
        raise ValueError(f"built {len(cases)} of {expected} LawShift temporal cases")
    return cases


class LawShiftRealScorer:
    def __init__(
        self,
        *,
        reranker_model: str,
        device: str,
        hf_home: Path,
        rerank_batch_size: int = 16,
    ) -> None:
        import os

        os.environ["HF_HOME"] = str(hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import CrossEncoder

        self.rerank_batch_size = rerank_batch_size
        self.reranker = CrossEncoder(
            str(local_model_snapshot(hf_home, reranker_model)), device=device
        )

    def score_cases(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs = [
            (str(case["question"]), str(candidate["text"]))
            for case in cases
            for candidate in case["candidates"]
        ]
        raw = self.reranker.predict(
            pairs,
            batch_size=self.rerank_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        output = []
        offset = 0
        for case in cases:
            candidates = case["candidates"]
            texts = [str(candidate["text"]) for candidate in candidates]
            lexical = _bm25_scores(str(case["question"]), texts)
            cross = minmax(raw[offset : offset + len(candidates)])
            offset += len(candidates)
            output.append(
                {
                    **case,
                    "candidates": [
                        {
                            **candidate,
                            "scores": {
                                "char_bm25": float(lexical[index]),
                                "cross_encoder": float(cross[index]),
                            },
                        }
                        for index, candidate in enumerate(candidates)
                    ],
                }
            )
        print(f"reranked {len(pairs)} LawShift temporal candidate pairs", flush=True)
        return output


def score_lawshift_cases(
    cases: list[dict[str, Any]],
    *,
    output_path: Path,
    metadata_path: Path,
    source_manifest_sha256: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    case_signature = stable_hash(
        [
            {
                "id": case["id"],
                "question": case["question"],
                "gold_evidence_id": case["gold_evidence_id"],
                "candidates": [
                    {
                        "id": item["id"],
                        "text": item["text"],
                        "token_count": item["token_count"],
                    }
                    for item in case["candidates"]
                ],
            }
            for case in cases
        ]
    )
    expected = {
        "source_manifest_sha256": source_manifest_sha256,
        "case_signature": case_signature,
        "config": config,
        "cases": len(cases),
    }
    if output_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == expected:
            cached = list(read_jsonl(output_path))
            if len(cached) == len(cases):
                return cached
    scorer = LawShiftRealScorer(
        reranker_model=str(config["reranker_model"]),
        device=str(config["device"]),
        hf_home=Path(str(config["hf_home"])),
        rerank_batch_size=int(config["rerank_batch_size"]),
    )
    scored = scorer.score_cases(cases)
    write_jsonl(output_path, scored)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return scored


def _is_applicable(case: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return candidate.get("version") == case.get("as_of_version")


def select_lawshift_evidence(
    case: dict[str, Any],
    method: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    if method not in LAWSHIFT_METHODS:
        raise ValueError(f"unsupported LawShift method: {method}")
    candidates = list(case["candidates"])
    if method in {"applicability_filtered_cross_encoder_top1", "frc_full"}:
        candidates = [item for item in candidates if _is_applicable(case, item)]
    score_name = "char_bm25" if method == "char_bm25_top1" else "cross_encoder"
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["scores"][score_name]),
            str(item["id"]),
        ),
    )
    selected = []
    cost = 0
    for candidate in ranked:
        candidate_cost = int(candidate.get("token_count", 1))
        if cost + candidate_cost > token_budget:
            continue
        selected.append(candidate)
        cost += candidate_cost
        if len(selected) == top_k:
            break
    return selected


def select_lawshift_methods(
    cases: Iterable[dict[str, Any]],
    *,
    methods: Iterable[str] = LAWSHIFT_METHODS,
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for method in methods:
            selected = select_lawshift_evidence(
                case,
                method,
                top_k=top_k,
                token_budget=token_budget,
            )
            rows.append(
                {
                    "case_id": case["id"],
                    "method": method,
                    "revision_type": case["revision_type"],
                    "source_case_index": case["source_case_index"],
                    "as_of_version": case["as_of_version"],
                    "gold_article_id": case["gold_article_id"],
                    "gold_evidence_id": case["gold_evidence_id"],
                    "candidate_count": len(case["candidates"]),
                    "selected_ids": [item["id"] for item in selected],
                    "selected_evidence": selected,
                }
            )
    return rows


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    selected = row["selected_evidence"]
    selected_first = selected[0] if selected else {}
    article_correct = float(selected_first.get("article_id") == row["gold_article_id"])
    version_correct = float(selected_first.get("version") == row["as_of_version"])
    exact = float(selected_first.get("id") == row["gold_evidence_id"])
    return {
        "case_id": row["case_id"],
        "method": row["method"],
        "revision_type": row["revision_type"],
        "source_case_index": row["source_case_index"],
        "as_of_version": row["as_of_version"],
        "gold_evidence_id": row["gold_evidence_id"],
        "selected_ids": row["selected_ids"],
        "metrics": {
            "article_recall_at_1": article_correct,
            "version_accuracy": version_correct,
            "exact_evidence_accuracy": exact,
            "invalid_applicability_rate": 1.0 - version_correct,
            "token_cost": sum(
                int(item.get("token_count", 1)) for item in row["selected_evidence"]
            ),
        },
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    metric_names = tuple(rows[0]["metrics"]) if rows else ()
    return {
        "cases": len(rows),
        **{
            name: round(
                sum(float(row["metrics"][name]) for row in rows) / len(rows),
                6,
            )
            for name in metric_names
        },
    }


def _paired_comparison(
    by_method: dict[str, list[dict[str, Any]]], left: str, right: str
) -> dict[str, Any]:
    left_rows = {row["case_id"]: row for row in by_method[left]}
    right_rows = {row["case_id"]: row for row in by_method[right]}
    if set(left_rows) != set(right_rows):
        raise ValueError(f"paired LawShift cases differ for {left} and {right}")
    metrics = (
        "article_recall_at_1",
        "version_accuracy",
        "exact_evidence_accuracy",
        "invalid_applicability_rate",
    )
    return {
        "left": left,
        "right": right,
        "direction": "left_minus_right",
        "metrics": {
            metric: paired_bootstrap(
                [
                    float(left_rows[case_id]["metrics"][metric])
                    - float(right_rows[case_id]["metrics"][metric])
                    for case_id in sorted(left_rows)
                ],
                seed=DEFAULT_SEED,
            )
            for metric in metrics
        },
    }


def build_lawshift_ablation_report(
    *,
    cases: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    config: dict[str, Any],
    source_manifest: dict[str, Any],
    scored_path: Path,
) -> dict[str, Any]:
    results = [_score_row(row) for row in selected_rows]
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_method[row["method"]].append(row)
    if set(by_method) != set(LAWSHIFT_METHODS):
        raise ValueError("LawShift method coverage is incomplete")
    aggregates = {method: _aggregate(rows) for method, rows in by_method.items()}
    baseline_methods = (
        "char_bm25_top1",
        "cross_encoder_top1",
        "applicability_filtered_cross_encoder_top1",
    )
    strongest_baseline = max(
        baseline_methods,
        key=lambda method: (
            float(aggregates[method]["exact_evidence_accuracy"]),
            float(aggregates[method]["article_recall_at_1"]),
            method,
        ),
    )
    paired = {
        "full_minus_w_o_applicability": _paired_comparison(
            by_method, "frc_full", "w/o_applicability"
        ),
        "full_minus_strongest_baseline": _paired_comparison(
            by_method, "frc_full", strongest_baseline
        ),
    }
    full_by_case = {row["case_id"]: row for row in by_method["frc_full"]}
    without_by_case = {
        row["case_id"]: row for row in by_method["w/o_applicability"]
    }
    selection_changed = sum(
        full_by_case[case_id]["selected_ids"] != without_by_case[case_id]["selected_ids"]
        for case_id in full_by_case
    )
    by_revision_type = []
    for revision_type in sorted({case["revision_type"] for case in cases}):
        revision_rows = {
            method: [row for row in rows if row["revision_type"] == revision_type]
            for method, rows in by_method.items()
        }
        by_revision_type.append(
            {
                "revision_type": revision_type,
                "cases": len(revision_rows["frc_full"]),
                "frc_full_exact_evidence_accuracy": _aggregate(
                    revision_rows["frc_full"]
                )["exact_evidence_accuracy"],
                "w_o_applicability_exact_evidence_accuracy": _aggregate(
                    revision_rows["w/o_applicability"]
                )["exact_evidence_accuracy"],
            }
        )
    baseline_difference = paired["full_minus_strongest_baseline"]["metrics"][
        "exact_evidence_accuracy"
    ]
    return {
        "schema_version": LAWSHIFT_ABLATION_SCHEMA,
        "dataset": LAWSHIFT_SOURCE_REPOSITORY,
        "metadata": {
            "public_dataset": True,
            "expert_annotated_revisions": True,
            "real_model_scores": True,
            "source_repository": LAWSHIFT_SOURCE_REPOSITORY,
            "source_revision": LAWSHIFT_SOURCE_REVISION,
            "source_license": LAWSHIFT_SOURCE_LICENSE,
            "source_file_count": len(source_manifest["files"]),
            "source_manifest_sha256": source_manifest["sha256"],
            "scored_cases_sha256": sha256(scored_path),
            "case_count": len(cases),
            "paired_case_count": len(cases) // 2,
            "revision_type_count": len({case["revision_type"] for case in cases}),
            "original_case_count": sum(
                case["as_of_version"] == "original" for case in cases
            ),
            "revised_case_count": sum(
                case["as_of_version"] == "revised" for case in cases
            ),
            "candidate_occurrences": sum(len(case["candidates"]) for case in cases),
            "methods": list(LAWSHIFT_METHODS),
            "models": {"reranker": config["reranker_model"]},
            "selection_parameters": {
                "top_k": config["top_k"],
                "token_budget": config["token_budget"],
                "pairs_per_revision": config["pairs_per_revision"],
                "distractor_articles": config["distractor_articles"],
                "seed": config["seed"],
            },
            "frozen_protocol": (
                "two deterministic paired case indices per each of 31 expert-reviewed revision "
                "types; both original and revised snapshots are evaluated; candidate pools contain "
                "both versions of the gold article and three lexical hard-negative article pairs"
            ),
            "gold_usage": (
                "relevant_articles identify the gold evidence only after retrieval; facts and "
                "before/after time scope are scorer inputs, while charge and prison_time are not"
            ),
        },
        "coverage": {
            "version_replacement": "RUN_PUBLIC_EXPERT_REVIEWED_REVISION_REAL_MODEL",
            "original_and_revised_snapshots": "IDENTIFIABLE",
            "effective_or_expiry_dates": "NOT_IDENTIFIABLE_NO_EFFECTIVE_DATES",
        },
        "aggregates": aggregates,
        "by_revision_type": by_revision_type,
        "strongest_baseline_by_exact_evidence_accuracy": strongest_baseline,
        "paired_comparisons": paired,
        "selection_changed_cases": selection_changed,
        "decision": {
            "full_strictly_better_than_w_o_applicability": (
                paired["full_minus_w_o_applicability"]["metrics"][
                    "exact_evidence_accuracy"
                ]["mean_difference"]
                > 0.0
            ),
            "full_exact_gain_over_strongest_baseline_at_least_0_05": (
                float(baseline_difference["mean_difference"]) >= 0.05
            ),
            "gate_2": "NO-GO",
            "reason": (
                "LawShift identifies expert-reviewed before/after statutory replacement, but the "
                "fair applicability-filtered Cross-Encoder baseline uses the same validity metadata; "
                "the dataset has no effective/expiry dates and is not a flood-response benchmark."
            ),
        },
        "limitations": [
            "LawShift evaluates Chinese criminal-law judgment adaptation, not flood-response or district-government evidence retrieval.",
            "The revised statutes are expert-reviewed hypothetical revisions rather than enacted historical versions with authoritative effective dates.",
            "The benchmark identifies before/after version replacement but cannot test exact effective or expiry dates.",
            "The frozen candidate pool is a retrieval stress slice built from the labeled article pair plus lexical hard negatives, not the paper's original legal-judgment-prediction protocol.",
            "Only evidence selection is evaluated; charge and sentence generation are not scored in this retrieval ablation.",
        ],
        "case_results": results,
    }


def render_lawshift_ablation_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# LawShift 跨版本适用性真实模型消融",
        "",
        f"- 固定来源：`{metadata['source_repository']}@{metadata['source_revision']}`",
        f"- 许可：`{metadata['source_license']}`",
        f"- 用例：{metadata['case_count']}（{metadata['revision_type_count']} 类修订，修订前/后各 {metadata['original_case_count']}）",
        f"- Reranker：`{metadata['models']['reranker']}`",
        "- 范围：专家审阅的假设修订版本替换；不含法定生效/失效日期。",
        "",
        "| 方法 | Article Recall@1 | Version Accuracy | Exact Evidence Accuracy | Invalid Applicability |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in metadata["methods"]:
        values = report["aggregates"][method]
        lines.append(
            f"| {method} | {values['article_recall_at_1']:.6f} | "
            f"{values['version_accuracy']:.6f} | {values['exact_evidence_accuracy']:.6f} | "
            f"{values['invalid_applicability_rate']:.6f} |"
        )
    comparison = report["paired_comparisons"]["full_minus_w_o_applicability"][
        "metrics"
    ]["exact_evidence_accuracy"]
    baseline = report["paired_comparisons"]["full_minus_strongest_baseline"][
        "metrics"
    ]["exact_evidence_accuracy"]
    lines.extend(
        [
            "",
            f"Full−w/o Applicability 的 Exact Evidence Accuracy 差值为 {comparison['mean_difference']:+.6f}，"
            f"95% CI=[{comparison['ci_low']:+.6f}, {comparison['ci_high']:+.6f}]。",
            f"最强公平基线为 `{report['strongest_baseline_by_exact_evidence_accuracy']}`；"
            f"Full 相对其差值为 {baseline['mean_difference']:+.6f}，Gate 2 保持 `NO-GO`。",
            "",
            "## 边界",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"
