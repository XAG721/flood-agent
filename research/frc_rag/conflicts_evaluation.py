from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from research.frc_rag.public_evidence import paired_bootstrap, select_precomputed


CONFLICT_LABELS = (
    "No conflict",
    "Complementary information",
    "Conflicting opinions and research outcomes",
    "Conflict due to outdated information",
    "Conflict due to misinformation",
)

CONFLICT_ROLES = {
    "answer_claim": "Evidence that directly answers the question or states the central factual claim.",
    "source_attribution": "Evidence identifying the source, authority, study, or speaker behind a claim.",
    "alternative_claim": "Evidence presenting a different, conflicting, or complementary answer or viewpoint.",
    "temporal_validity": "Evidence containing dates, current status, supersession, expiry, or time-sensitive validity.",
}

CONFLICT_METHODS = (
    "bm25_topk",
    "dense_topk",
    "hybrid_topk",
    "cross_encoder_topk",
    "coverage_greedy_proxy",
    "frc_select",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def normalized_text(text: str) -> str:
    return " ".join(tokens(text))


def approximate_token_count(text: str) -> int:
    return max(1, len(tokens(text)))


def minmax(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=float)
    if not len(array):
        return array
    low = float(array.min())
    high = float(array.max())
    if math.isclose(low, high):
        return np.ones_like(array) if high > 0 else np.zeros_like(array)
    return (array - low) / (high - low)


def rank_desc(values: Any) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-float(values[index]), index))


def rrf_scores(rankings: list[list[int]], size: int, *, rrf_k: int = 60) -> Any:
    import numpy as np

    scores = np.zeros(size, dtype=float)
    for ranking in rankings:
        for position, index in enumerate(ranking, start=1):
            scores[index] += 1.0 / (rrf_k + position)
    return minmax(scores)


def bm25_scores(query: str, texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> Any:
    import numpy as np

    query_terms = tokens(query)
    documents = [tokens(text) for text in texts]
    if not documents:
        return np.array([], dtype=float)
    document_frequency = Counter()
    for document in documents:
        document_frequency.update(set(document))
    average_length = sum(map(len, documents)) / max(1, len(documents))
    scores = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * len(document) / max(1.0, average_length))
            score += inverse * frequency * (k1 + 1.0) / denominator
        scores.append(score)
    return minmax(np.asarray(scores, dtype=float))


def load_conflicts_cases(path: Path, *, sample_size: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(read_jsonl(path)):
        if sample_size is not None and len(cases) >= sample_size:
            break
        candidates = []
        for result_index, result in enumerate(row.get("search_results", [])):
            text = str(result.get("short_text") or result.get("snippet") or "").strip()
            if not text:
                continue
            candidates.append(
                {
                    "id": f"conflicts-{index:04d}-result-{result_index:02d}",
                    "text": text,
                    "title": str(result.get("title") or ""),
                    "url": str(result.get("url") or ""),
                    "date": str(result.get("date") or ""),
                    "token_count": approximate_token_count(text),
                }
            )
        cases.append(
            {
                "id": f"conflicts-{index:04d}",
                "source": str(row.get("source") or ""),
                "question": str(row.get("question") or ""),
                "conflict_type": str(row.get("conflict_type") or ""),
                "correct_answer": str(row.get("correct_answer") or "").strip(),
                "required_roles": list(CONFLICT_ROLES),
                "candidates": candidates,
            }
        )
    return cases


def local_model_snapshot(hf_home: Path, repository_id: str) -> Path:
    repository = hf_home / "hub" / f"models--{repository_id.replace('/', '--')}"
    reference = repository / "refs" / "main"
    if not reference.is_file():
        raise FileNotFoundError(f"missing local model reference: {reference}")
    revision = reference.read_text(encoding="utf-8").strip()
    snapshot = repository / "snapshots" / revision
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"incomplete local model snapshot: {snapshot}")
    return snapshot


class ConflictRealScorer:
    def __init__(
        self,
        *,
        embedding_model: str,
        reranker_model: str,
        device: str,
        hf_home: Path,
        embedding_batch_size: int = 32,
        rerank_batch_size: int = 16,
        role_relevance_mix: float = 0.15,
    ) -> None:
        import os

        os.environ["HF_HOME"] = str(hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.device = device
        self.embedding_batch_size = embedding_batch_size
        self.rerank_batch_size = rerank_batch_size
        self.role_relevance_mix = role_relevance_mix
        embedding_path = local_model_snapshot(hf_home, embedding_model)
        reranker_path = local_model_snapshot(hf_home, reranker_model)
        self.embedder = SentenceTransformer(str(embedding_path), device=device)
        self.reranker = CrossEncoder(str(reranker_path), device=device)

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        return np.asarray(
            self.embedder.encode(
                texts,
                batch_size=self.embedding_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def predict(self, pairs: list[tuple[str, str]]) -> Any:
        import numpy as np

        return np.asarray(
            self.reranker.predict(
                pairs,
                batch_size=self.rerank_batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )

    def score_case(self, case: dict[str, Any]) -> dict[str, Any]:
        candidates = case["candidates"]
        texts = [candidate["text"] for candidate in candidates]
        question = case["question"]
        if not texts:
            return case
        vectors = self.encode(texts)
        query_vector = self.encode([question])[0]
        bm25 = bm25_scores(question, texts)
        dense = minmax(vectors @ query_vector)
        hybrid = rrf_scores([rank_desc(bm25), rank_desc(dense)], len(texts))
        cross = minmax(self.predict([(question, text) for text in texts]))
        role_values: dict[str, Any] = {}
        for role, instruction in CONFLICT_ROLES.items():
            role_query = f"{instruction} Question: {question}"
            calibrated = minmax(self.predict([(role_query, text) for text in texts]))
            role_values[role] = (1.0 - self.role_relevance_mix) * calibrated + self.role_relevance_mix * cross
        scored_candidates = []
        for index, candidate in enumerate(candidates):
            scored_candidates.append(
                {
                    **candidate,
                    "scores": {
                        "bm25": float(bm25[index]),
                        "dense": float(dense[index]),
                        "hybrid": float(hybrid[index]),
                        "cross_encoder": float(cross[index]),
                    },
                    "role_scores": {
                        role: float(values[index]) for role, values in role_values.items()
                    },
                }
            )
        return {**case, "candidates": scored_candidates}


def score_conflicts_cases(
    cases: list[dict[str, Any]],
    *,
    output_path: Path,
    metadata_path: Path,
    input_hash: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = {"input_sha256": input_hash, "config": config, "cases": len(cases)}
    if output_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == expected:
            return list(read_jsonl(output_path))
    scorer = ConflictRealScorer(
        embedding_model=config["embedding_model"],
        reranker_model=config["reranker_model"],
        device=config["device"],
        hf_home=Path(config["hf_home"]),
        embedding_batch_size=config["embedding_batch_size"],
        rerank_batch_size=config["rerank_batch_size"],
        role_relevance_mix=config["role_relevance_mix"],
    )
    rows = []
    for index, case in enumerate(cases, start=1):
        rows.append(scorer.score_case(case))
        if index % 25 == 0 or index == len(cases):
            print(f"scored {index}/{len(cases)} CONFLICTS cases", flush=True)
    write_jsonl(output_path, rows)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def select_conflicts_evidence(
    cases: list[dict[str, Any]],
    *,
    top_k: int = 5,
    token_budget: int = 1500,
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for method in CONFLICT_METHODS:
            selected = select_precomputed(case, method, k=top_k, budget=token_budget)
            rows.append(
                {
                    "case_id": case["id"],
                    "source": case["source"],
                    "question": case["question"],
                    "conflict_type": case["conflict_type"],
                    "correct_answer": case["correct_answer"],
                    "method": method,
                    "candidate_count": len(case["candidates"]),
                    "selected_ids": [item["id"] for item in selected],
                    "selected_evidence": selected,
                }
            )
    return rows


def conflict_classification_prompt(row: dict[str, Any]) -> str:
    evidence = "\n\n".join(
        f"[{index}] title={item.get('title', '')}; date={item.get('date', '')}; "
        f"url={item.get('url', '')}\n{item.get('text', '')}"
        for index, item in enumerate(row["selected_evidence"], start=1)
    )
    labels = "\n".join(f"- {label}" for label in CONFLICT_LABELS)
    return (
        "Classify the relationship among the retrieved sources for the question.\n"
        "Return exactly one label from the list and no explanation.\n"
        "Use dates to distinguish outdated information from other disagreement.\n\n"
        f"Labels:\n{labels}\n\nQuestion: {row['question']}\n\nSources:\n{evidence}\n\nLabel:"
    )


def normalize_conflict_label(text: str) -> str | None:
    normalized = normalized_text(text)
    for label in CONFLICT_LABELS:
        if normalized == normalized_text(label):
            return label
    matches = [label for label in CONFLICT_LABELS if normalized_text(label) in normalized]
    return max(matches, key=len) if matches else None


def selection_signature(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return row["case_id"], tuple(row["selected_ids"])


class LocalConflictClassifier:
    def __init__(self, *, model_path: Path, batch_size: int = 16, max_new_tokens: int = 24) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
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
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def classify(self, rows: list[dict[str, Any]], *, output_path: Path, cache_key: str) -> list[dict[str, Any]]:
        completed: dict[tuple[str, str], dict[str, Any]] = {}
        if output_path.is_file():
            for row in read_jsonl(output_path):
                if row.get("cache_key") == cache_key:
                    completed[(row["case_id"], row["method"])] = row
        output_path.parent.mkdir(parents=True, exist_ok=True)
        signature_predictions: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        for source in rows:
            prediction = completed.get((source["case_id"], source["method"]))
            if prediction is not None:
                signature_predictions.setdefault(selection_signature(source), prediction)
        grouped_pending: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
        reused_rows = []
        for source in rows:
            key = (source["case_id"], source["method"])
            if key in completed:
                continue
            signature = selection_signature(source)
            equivalent = signature_predictions.get(signature)
            if equivalent is not None:
                prediction = {
                    **equivalent,
                    "method": source["method"],
                    "reused_from_equivalent_selection": True,
                }
                completed[key] = prediction
                reused_rows.append(prediction)
            else:
                grouped_pending[signature].append(source)
        if reused_rows:
            with output_path.open("a", encoding="utf-8", newline="\n") as handle:
                for prediction in reused_rows:
                    handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
        representatives = [group[0] for group in grouped_pending.values()]
        for start in range(0, len(representatives), self.batch_size):
            batch = representatives[start : start + self.batch_size]
            prompts = [self._chat_prompt(conflict_classification_prompt(row)) for row in batch]
            encoded = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=3072,
            ).to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            prompt_length = encoded.input_ids.shape[-1]
            decoded = self.tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)
            new_rows = []
            for source, raw in zip(batch, decoded, strict=True):
                signature = selection_signature(source)
                group = grouped_pending[signature]
                for group_index, equivalent_source in enumerate(group):
                    prediction = {
                        "cache_key": cache_key,
                        "case_id": equivalent_source["case_id"],
                        "method": equivalent_source["method"],
                        "gold_label": equivalent_source["conflict_type"],
                        "raw_prediction": raw.strip(),
                        "predicted_label": normalize_conflict_label(raw),
                        "reused_from_equivalent_selection": group_index > 0,
                    }
                    completed[(equivalent_source["case_id"], equivalent_source["method"])] = prediction
                    new_rows.append(prediction)
            with output_path.open("a", encoding="utf-8", newline="\n") as handle:
                for prediction in new_rows:
                    handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(
                f"classified {min(start + len(batch), len(representatives))}/{len(representatives)} "
                "unique pending selections",
                flush=True,
            )
        return [completed[(row["case_id"], row["method"])] for row in rows]


def parse_date(value: str) -> datetime | None:
    text = value.strip()
    if not text or text.lower() in {"na", "none", "null", "unknown"}:
        return None
    normalized = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text, flags=re.IGNORECASE)
    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %Y",
        "%B %Y",
        "%Y",
    )
    for date_format in formats:
        try:
            return datetime.strptime(normalized, date_format)
        except ValueError:
            continue
    match = re.search(r"\b(20\d{2}|19\d{2})\b", normalized)
    return datetime(int(match.group(1)), 1, 1) if match else None


def domain(url: str) -> str:
    value = urlparse(url).netloc.lower()
    return value[4:] if value.startswith("www.") else value


def lexical_diversity(selected: list[dict[str, Any]]) -> float:
    token_sets = [set(tokens(item.get("text", ""))) for item in selected]
    similarities = []
    for left_index, left in enumerate(token_sets):
        for right in token_sets[left_index + 1 :]:
            similarities.append(len(left & right) / max(1, len(left | right)))
    return 1.0 - sum(similarities) / len(similarities) if similarities else 0.0


def answer_support(answer: str, selected: list[dict[str, Any]]) -> tuple[float, float]:
    if not answer.strip():
        return 0.0, 0.0
    selected_text = normalized_text(" ".join(item.get("text", "") for item in selected))
    normalized_answer = normalized_text(answer)
    answer_tokens = set(tokens(answer))
    selected_tokens = set(tokens(selected_text))
    return (
        float(bool(normalized_answer) and normalized_answer in selected_text),
        len(answer_tokens & selected_tokens) / max(1, len(answer_tokens)),
    )


def selection_diagnostics(row: dict[str, Any], case: dict[str, Any]) -> dict[str, float]:
    selected = row["selected_evidence"]
    selected_domains = {domain(item.get("url", "")) for item in selected if domain(item.get("url", ""))}
    candidate_domains = {
        domain(item.get("url", "")) for item in case["candidates"] if domain(item.get("url", ""))
    }
    exact_support, token_recall = answer_support(row["correct_answer"], selected)
    candidate_dates = [(item["id"], parse_date(item.get("date", ""))) for item in case["candidates"]]
    dated_candidates = [(item_id, value) for item_id, value in candidate_dates if value is not None]
    selected_ids = set(row["selected_ids"])
    newest_retention = 0.0
    endpoint_coverage = 0.0
    if dated_candidates:
        earliest = min(value for _, value in dated_candidates)
        latest = max(value for _, value in dated_candidates)
        earliest_ids = {item_id for item_id, value in dated_candidates if value == earliest}
        latest_ids = {item_id for item_id, value in dated_candidates if value == latest}
        newest_retention = float(bool(selected_ids & latest_ids))
        endpoint_coverage = (
            float(bool(selected_ids & earliest_ids)) + float(bool(selected_ids & latest_ids))
        ) / 2.0
    return {
        "domain_coverage": len(selected_domains) / max(1, min(len(selected), len(candidate_domains))),
        "lexical_diversity": lexical_diversity(selected),
        "token_cost": float(sum(int(item.get("token_count", 0)) for item in selected)),
        "exact_answer_support": exact_support,
        "answer_token_recall": token_recall,
        "newest_date_retention": newest_retention,
        "temporal_endpoint_coverage": endpoint_coverage,
    }


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {label: Counter() for label in CONFLICT_LABELS}
    gold_support = Counter(row["gold_label"] for row in rows)
    parsed = 0
    correct = 0
    for row in rows:
        predicted = row.get("predicted_label")
        gold = row["gold_label"]
        if predicted:
            parsed += 1
            confusion[gold][predicted] += 1
            correct += int(predicted == gold)
    per_type = {}
    f1_values = []
    for label in CONFLICT_LABELS:
        true_positive = confusion[label][label]
        false_negative = gold_support[label] - true_positive
        false_positive = sum(confusion[other][label] for other in CONFLICT_LABELS if other != label)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1_values.append(f1)
        per_type[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": gold_support[label],
        }
    return {
        "cases": len(rows),
        "parsed_predictions": parsed,
        "parse_rate": round(parsed / max(1, len(rows)), 6),
        "accuracy": round(correct / max(1, len(rows)), 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "per_type": per_type,
    }


def aggregate_diagnostics(
    selected_rows: list[dict[str, Any]], cases: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, float]]]] = defaultdict(list)
    for row in selected_rows:
        grouped[row["method"]].append((row, selection_diagnostics(row, cases[row["case_id"]])))
    output = {}
    for method, values in grouped.items():
        answer_values = [metrics for row, metrics in values if row["correct_answer"]]
        outdated_values = [
            metrics
            for row, metrics in values
            if row["conflict_type"] == "Conflict due to outdated information"
        ]
        output[method] = {
            "cases": len(values),
            "domain_coverage": round(sum(item["domain_coverage"] for _, item in values) / len(values), 6),
            "lexical_diversity": round(
                sum(item["lexical_diversity"] for _, item in values) / len(values), 6
            ),
            "mean_token_cost": round(sum(item["token_cost"] for _, item in values) / len(values), 3),
            "answer_cases": len(answer_values),
            "exact_answer_support": round(
                sum(item["exact_answer_support"] for item in answer_values) / max(1, len(answer_values)), 6
            ),
            "answer_token_recall": round(
                sum(item["answer_token_recall"] for item in answer_values) / max(1, len(answer_values)), 6
            ),
            "outdated_cases": len(outdated_values),
            "newest_date_retention": round(
                sum(item["newest_date_retention"] for item in outdated_values)
                / max(1, len(outdated_values)),
                6,
            ),
            "temporal_endpoint_coverage": round(
                sum(item["temporal_endpoint_coverage"] for item in outdated_values)
                / max(1, len(outdated_values)),
                6,
            ),
        }
    return output


def build_conflicts_report(
    *,
    dataset_path: Path,
    scored_path: Path,
    selected_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    cases = {row["id"]: row for row in read_jsonl(scored_path)}
    diagnostics = aggregate_diagnostics(selected_rows, cases)
    prediction_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        prediction_groups[row["method"]].append(row)
    classification = {
        method: classification_metrics(prediction_groups[method]) for method in CONFLICT_METHODS
    }
    baseline_methods = [method for method in CONFLICT_METHODS if method != "frc_select"]
    strongest = max(
        baseline_methods,
        key=lambda method: (classification[method]["accuracy"], classification[method]["macro_f1"], method),
    )
    by_key = {(row["case_id"], row["method"]): row for row in predictions}
    differences = []
    for case_id in cases:
        frc = by_key[(case_id, "frc_select")]
        baseline = by_key[(case_id, strongest)]
        differences.append(
            float(frc.get("predicted_label") == frc["gold_label"])
            - float(baseline.get("predicted_label") == baseline["gold_label"])
        )
    paired = paired_bootstrap(differences, resamples=2000)
    paired["wins"] = sum(value > 0 for value in differences)
    paired["ties"] = sum(value == 0 for value in differences)
    paired["losses"] = sum(value < 0 for value in differences)
    label_counts = Counter(case["conflict_type"] for case in cases.values())
    answer_cases = sum(bool(case["correct_answer"]) for case in cases.values())
    superiority = paired["ci_low"] > 0
    return {
        "metadata": {
            "name": "Google CONFLICTS FRC retrieval and conflict-classification audit",
            "status": "RUN",
            "dataset_sha256": sha256(dataset_path),
            "scored_cache_sha256": sha256(scored_path),
            "license": "Apache-2.0",
            "cases": len(cases),
            "answer_annotated_cases": answer_cases,
            "conflict_type_counts": dict(label_counts),
            **config,
        },
        "selection_diagnostics": diagnostics,
        "classification_metrics": classification,
        "strongest_reproducible_baseline_by_accuracy": strongest,
        "paired_frc_minus_baseline_accuracy": paired,
        "decision": {
            "pipeline_feasible": True,
            "frc_accuracy_superiority": superiority,
            "gate_2": "NO-GO",
            "reason": (
                "CONFLICTS retrieval and label classification are reproducible, but the paper's expected-behavior "
                "adherence evaluation and independent human judging are not reproduced; classification alone cannot "
                "authorize CANARY or DEFAULT."
            ),
        },
        "limitations": [
            "This is a retrieval-selection and conflict-type classification audit, not the paper's official expected-behavior adherence reproduction.",
            "Only 237 of 458 public cases have nonblank correct_answer values; answer-support diagnostics exclude the rest.",
            "Date retention is a provenance diagnostic and does not imply that the newest source is factually correct.",
            "The local Qwen2.5-7B-Instruct-GPTQ-Int4 classifier is shared by all methods and is not an expert judge.",
            "coverage_greedy_proxy is not SetR and must not be reported as a SetR reproduction.",
        ],
    }


def render_conflicts_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "# Google CONFLICTS：FRC 检索与冲突分类审计",
        "",
        f"- 状态：`{metadata['status']}`",
        f"- 用例：{metadata['cases']}；有非空正确答案：{metadata['answer_annotated_cases']}",
        f"- 数据 SHA-256：`{metadata['dataset_sha256']}`",
        f"- 统一预算：Top-K={metadata['top_k']}，{metadata['token_budget']} tokens",
        f"- 模型：`{metadata['embedding_model']}` / `{metadata['reranker_model']}` / `{metadata['generator_model']}`",
        "- `coverage_greedy_proxy` 是覆盖贪心代理，不是 SetR 复现。",
        "",
        "## 冲突类型",
        "",
        "| 类型 | 用例 |",
        "|---|---:|",
    ]
    for label in CONFLICT_LABELS:
        lines.append(f"| {label} | {metadata['conflict_type_counts'].get(label, 0)} |")
    lines.extend(
        [
            "",
            "## 方法对比",
            "",
            "| 方法 | Accuracy | Macro-F1 | 过时类型 Recall | 最新日期保留 | 时间端点覆盖 | 答案 Token Recall |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in CONFLICT_METHODS:
        classification = report["classification_metrics"][method]
        diagnostics = report["selection_diagnostics"][method]
        outdated = classification["per_type"]["Conflict due to outdated information"]
        lines.append(
            f"| {method} | {classification['accuracy']:.6f} | {classification['macro_f1']:.6f} | "
            f"{outdated['recall']:.6f} | {diagnostics['newest_date_retention']:.6f} | "
            f"{diagnostics['temporal_endpoint_coverage']:.6f} | {diagnostics['answer_token_recall']:.6f} |"
        )
    strongest = report["strongest_reproducible_baseline_by_accuracy"]
    paired = report["paired_frc_minus_baseline_accuracy"]
    lines.extend(
        [
            "",
            "## 配对判定",
            "",
            f"- 最强可复现分类基线：`{strongest}`",
            f"- FRC - 基线 Accuracy：{paired['mean_difference']:+.6f}",
            f"- 95% CI：[{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}]",
            f"- 胜/平/负：{paired['wins']}/{paired['ties']}/{paired['losses']}",
            f"- Gate 2：`{report['decision']['gate_2']}`",
            f"- 说明：{report['decision']['reason']}",
            "",
            "## 限制",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"
