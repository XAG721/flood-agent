from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)


def terms(text: str) -> list[str]:
    raw = TOKEN_PATTERN.findall(text.lower())
    chinese = [item for item in raw if "\u4e00" <= item <= "\u9fff"]
    bigrams = [chinese[index] + chinese[index + 1] for index in range(len(chinese) - 1)]
    return raw + bigrams


class BM25Index:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.document_terms = [terms(f"{item['title']} {item['content']}") for item in documents]
        self.term_counts = [Counter(values) for values in self.document_terms]
        self.average_length = sum(map(len, self.document_terms)) / max(1, len(self.document_terms))
        self.document_frequency = Counter(token for values in self.document_terms for token in set(values))

    def rank(self, query: str) -> list[str]:
        query_terms = terms(query)
        scored = []
        for document, values, counts in zip(self.documents, self.document_terms, self.term_counts):
            score = 0.0
            for token in query_terms:
                frequency = counts[token]
                if not frequency:
                    continue
                idf = math.log(
                    1 + (len(self.documents) - self.document_frequency[token] + 0.5)
                    / (self.document_frequency[token] + 0.5)
                )
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * len(values) / max(1, self.average_length)
                )
                score += idf * frequency * 2.5 / denominator
            scored.append((score, document["doc_id"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[1] for item in scored]


def bm25_ranking(documents: list[dict], query: str) -> list[str]:
    return BM25Index(documents).rank(query)


def reciprocal_rank_fusion(*rankings: list[str]) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (60 + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def score_selection(case: dict, selected: list[str], documents_by_id: dict[str, dict]) -> dict[str, float]:
    expected = set(case["relevant_doc_ids"])
    selected_set = set(selected)
    true_positive = len(expected & selected_set)
    recall = true_positive / max(1, len(expected))
    precision = true_positive / max(1, len(selected_set))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    selected_roles = {
        role
        for doc_id in selected
        for role in documents_by_id[doc_id].get("metadata", {}).get("evidence_roles", [])
    }
    required_roles = set(case.get("required_roles", []))
    role_coverage = len(selected_roles & required_roles) / max(1, len(required_roles))
    return {
        "evidence_recall": round(recall, 4),
        "role_coverage": round(role_coverage, 4),
        "answer_f1": round(f1, 4),
        "task_element_completeness": round(role_coverage, 4),
        "citation_precision": round(precision, 4),
        "unsupported_evidence_ratio": round(1 - precision, 4),
    }


def load_encoder(model_id: str, *, batch_size: int = 16):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=False)
    model.eval()
    def encode(texts: list[str], *, query: bool):
        instruction = (
            "为这个句子生成表示以用于检索相关文章："
            if "-zh" in model_id.lower()
            else "Represent this sentence for searching relevant passages: "
        )
        prepared = [f"{instruction}{text}" for text in texts] if query else texts
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(prepared), batch_size):
                inputs = tokenizer(
                    prepared[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                output = model(**inputs).last_hidden_state
                pooled = output[:, 0]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.cpu().numpy())
        return vectors

    return encode, model.config.hidden_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real neural Dense/Hybrid RAG evaluation.")
    parser.add_argument("--benchmark", type=Path, default=Path("flood_system/rag_benchmarks/district_policy_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/rag_evaluation"))
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    import numpy as np
    import torch
    import transformers

    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    documents = payload["documents"]
    cases = payload["cases"]
    document_texts = [f"{item['title']}。{item['content']}" for item in documents]
    encoder, dimension = load_encoder(args.model)
    index_started = time.perf_counter()
    document_vectors = encoder(document_texts, query=False)
    index_ms = (time.perf_counter() - index_started) * 1000
    matrix = np.asarray(document_vectors)
    doc_ids = [item["doc_id"] for item in documents]
    documents_by_id = {item["doc_id"]: item for item in documents}
    rows = []
    for case in cases:
        started = time.perf_counter()
        query_vectors = encoder([case["query"]], query=True)
        similarities = matrix @ np.asarray(query_vectors[0])
        dense_ranking = [doc_ids[index] for index in np.argsort(-similarities)]
        dense_latency = (time.perf_counter() - started) * 1000
        bm25 = bm25_ranking(documents, case["query"])
        hybrid = reciprocal_rank_fusion(bm25, dense_ranking)
        for method, ranking in (("neural_dense_top_k", dense_ranking), ("neural_hybrid_rrf", hybrid)):
            selected = ranking[: args.top_k]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "method": method,
                    "selected_doc_ids": selected,
                    **score_selection(case, selected, documents_by_id),
                    "query_latency_ms": round(dense_latency, 4),
                }
            )
    aggregates = {}
    for method in ("neural_dense_top_k", "neural_hybrid_rrf"):
        method_rows = [item for item in rows if item["method"] == method]
        aggregates[method] = {
            key: round(sum(item[key] for item in method_rows) / len(method_rows), 4)
            for key in (
                "evidence_recall",
                "role_coverage",
                "answer_f1",
                "task_element_completeness",
                "citation_precision",
                "unsupported_evidence_ratio",
                "query_latency_ms",
            )
        }
    report = {
        "model": args.model,
        "backend": "transformers.AutoModel CLS pooling",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "embedding_dimension": dimension,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "case_count": len(cases),
        "top_k": args.top_k,
        "index_build_ms": round(index_ms, 4),
        "aggregates": aggregates,
        "cases": rows,
        "limitations": [
            "本结果使用仓库内小规模人工构造工程验收集，不代表公开 benchmark 或真实业务标注集结论。",
            "查询延迟在模型常驻后测量，不包含首次模型加载与下载时间。",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "neural_rag_evaluation_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# V3 神经向量 RAG 评测报告",
        "",
        f"- 模型：`{args.model}`",
        f"- 后端：{report['backend']}；设备：{report['device']}；向量维度：{dimension}",
        f"- PyTorch：{torch.__version__}；Transformers：{transformers.__version__}",
        f"- 用例数：{len(cases)}；Top-K：{args.top_k}；索引构建：{index_ms:.2f} ms",
        "",
        "| 方法 | Evidence Recall | Role Coverage | Answer F1 | 任务要素完整率 | 引用正确率 | 无依据证据比例 | 查询延迟 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in aggregates.items():
        lines.append(
            f"| {method} | {row['evidence_recall']:.4f} | {row['role_coverage']:.4f} | {row['answer_f1']:.4f} | "
            f"{row['task_element_completeness']:.4f} | {row['citation_precision']:.4f} | "
            f"{row['unsupported_evidence_ratio']:.4f} | {row['query_latency_ms']:.4f} |"
        )
    lines.extend(["", "## 限制", "", *[f"- {item}" for item in report["limitations"]]])
    (args.output_dir / "neural_rag_evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "aggregates": aggregates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
