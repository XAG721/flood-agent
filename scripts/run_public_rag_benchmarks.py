from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np

try:
    from scripts.run_neural_rag_evaluation import BM25Index, bm25_ranking, load_encoder, reciprocal_rank_fusion
except ModuleNotFoundError:  # Direct script execution adds scripts/ rather than the repository root.
    from run_neural_rag_evaluation import BM25Index, bm25_ranking, load_encoder, reciprocal_rank_fusion


TAG_PATTERN = re.compile(r"<[^>]+>")


def normalized_html(value: str) -> str:
    return re.sub(r"\s+", " ", TAG_PATTERN.sub(" ", value)).strip().lower()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    git_dir = path / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        if not head.startswith("ref: "):
            return head
        ref_name = head.removeprefix("ref: ")
        loose_ref = git_dir / ref_name
        if loose_ref.is_file():
            return loose_ref.read_text(encoding="ascii").strip()
        packed_refs = git_dir / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text(encoding="ascii").splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == ref_name:
                        return commit
    except OSError:
        pass
    return "unknown"


def rank_dense(matrix: np.ndarray, query_vector: np.ndarray, ids: list[str]) -> list[str]:
    similarities = matrix @ query_vector
    return [ids[index] for index in np.argsort(-similarities)]


def retrieval_metrics(expected: set[str], selected: list[str]) -> dict[str, float]:
    selected_set = set(selected)
    recall = len(expected & selected_set) / max(1, len(expected))
    first_rank = next((index for index, item in enumerate(selected, start=1) if item in expected), None)
    return {
        "evidence_recall": round(recall, 4),
        "complete_evidence_set": float(expected <= selected_set),
        "reciprocal_rank": round(1 / first_rank, 4) if first_rank else 0.0,
    }


def aggregate(rows: list[dict]) -> dict:
    output = {}
    for method in ("neural_dense_top_k", "neural_hybrid_rrf"):
        selected = [item for item in rows if item["method"] == method]
        output[method] = {
            key: round(sum(item[key] for item in selected) / max(1, len(selected)), 4)
            for key in ("evidence_recall", "complete_evidence_set", "reciprocal_rank", "query_latency_ms")
        }
    return output


def evaluate_global(
    *,
    name: str,
    documents: list[dict],
    cases: list[dict],
    encoder,
    top_k: int,
) -> dict:
    document_ids = [item["doc_id"] for item in documents]
    index_started = time.perf_counter()
    matrix = np.asarray(encoder([item["text"] for item in documents], query=False))
    index_ms = (time.perf_counter() - index_started) * 1000
    query_started = time.perf_counter()
    query_vectors = np.asarray(encoder([case["query"] for case in cases], query=True))
    per_query_latency = (time.perf_counter() - query_started) * 1000 / max(1, len(cases))
    rows = []
    lexical_docs = [
        {"doc_id": item["doc_id"], "title": item.get("title", ""), "content": item["text"]}
        for item in documents
    ]
    bm25_index = BM25Index(lexical_docs)
    for case, query_vector in zip(cases, query_vectors):
        dense = rank_dense(matrix, query_vector, document_ids)
        hybrid = reciprocal_rank_fusion(bm25_index.rank(case["query"]), dense)
        expected = set(case["expected_ids"])
        for method, ranking in (("neural_dense_top_k", dense), ("neural_hybrid_rrf", hybrid)):
            selected = ranking[:top_k]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "method": method,
                    "expected_count": len(expected),
                    "selected_ids": selected,
                    **retrieval_metrics(expected, selected),
                    "query_latency_ms": round(per_query_latency, 4),
                }
            )
    return {
        "dataset": name,
        "case_count": len(cases),
        "candidate_count": len(documents),
        "index_build_ms": round(index_ms, 4),
        "aggregates": aggregate(rows),
        "cases": rows,
    }


def multihop_rag_data(root: Path, sample_size: int, seed: int) -> tuple[list[dict], list[dict], dict]:
    corpus_path = root / "dataset" / "corpus.json"
    query_path = root / "dataset" / "MultiHopRAG.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    queries = json.loads(query_path.read_text(encoding="utf-8"))
    documents = [
        {"doc_id": item["url"], "title": item["title"], "text": f"{item['title']}. {item['body']}"}
        for item in corpus
    ]
    known = {item["doc_id"] for item in documents}
    valid = []
    for index, item in enumerate(queries):
        expected = list(dict.fromkeys(evidence["url"] for evidence in item["evidence_list"] if evidence["url"] in known))
        if len(expected) >= 2:
            valid.append({"case_id": f"multihop-{index}", "query": item["query"], "expected_ids": expected})
    evaluable_count = len(valid)
    random.Random(seed).shuffle(valid)
    return documents, valid[:sample_size], {
        "repository_revision": git_revision(root),
        "corpus_sha256": sha256(corpus_path),
        "queries_sha256": sha256(query_path),
        "full_query_count": len(queries),
        "evaluable_query_count": evaluable_count,
        "excluded_query_count": len(queries) - evaluable_count,
        "exclusion_rule": "fewer than two evidence URLs map to the released corpus",
    }


def conditionalqa_data(root: Path, sample_size: int, seed: int) -> tuple[list[dict], list[dict], dict]:
    documents_path = root / "v1_0" / "documents.json"
    dev_path = root / "v1_0" / "dev.json"
    source_documents = json.loads(documents_path.read_text(encoding="utf-8"))
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    by_url = {item["url"]: item for item in source_documents}
    eligible = [item for item in dev if not item["not_answerable"] and item["evidences"] and item["url"] in by_url]
    normalized_contents = {
        item["url"]: {normalized_html(content) for content in item["contents"]}
        for item in source_documents
    }
    full_evaluable_count = sum(
        any(normalized_html(evidence) in normalized_contents[item["url"]] for evidence in item["evidences"])
        for item in eligible
    )
    random.Random(seed).shuffle(eligible)
    sampled = eligible[:sample_size]
    selected_urls = {item["url"] for item in sampled}
    distractors = [item["url"] for item in source_documents if item["url"] not in selected_urls][:100]
    selected_urls.update(distractors)
    documents = []
    normalized_to_id: dict[tuple[str, str], str] = {}
    for url in sorted(selected_urls):
        source = by_url[url]
        for index, content in enumerate(source["contents"]):
            doc_id = f"{url}#chunk-{index}"
            clean = normalized_html(content)
            documents.append({"doc_id": doc_id, "title": source["title"], "text": clean})
            normalized_to_id[(url, clean)] = doc_id
    cases = []
    for item in sampled:
        expected = [
            normalized_to_id[(item["url"], normalized_html(evidence))]
            for evidence in item["evidences"]
            if (item["url"], normalized_html(evidence)) in normalized_to_id
        ]
        expected = list(dict.fromkeys(expected))
        if expected:
            cases.append(
                {
                    "case_id": item["id"],
                    # Keep the actual question before the potentially long scenario so truncation
                    # cannot remove the retrieval intent; scenario conditions remain available.
                    "query": f"{item['question']} [SCENARIO] {item['scenario']}",
                    "expected_ids": expected,
                }
            )
    return documents, cases, {
        "repository_revision": git_revision(root),
        "documents_sha256": sha256(documents_path),
        "dev_sha256": sha256(dev_path),
        "full_dev_count": len(dev),
        "eligible_answerable_count": len(eligible),
        "evaluable_query_count": full_evaluable_count,
        "excluded_query_count": len(dev) - full_evaluable_count,
        "exclusion_rule": "not answerable, missing evidence, or evidence paragraph cannot be mapped",
        "sampled_document_count": len(selected_urls),
    }


def hotpotqa_evaluate(path: Path, sample_size: int, seed: int, encoder, top_k: int) -> tuple[dict, dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    random.Random(seed).shuffle(rows)
    rows = rows[: min(sample_size, len(rows))]
    flattened_paragraphs: list[str] = []
    offsets: list[tuple[int, int]] = []
    for item in rows:
        paragraphs = [" ".join(sentences) for sentences in item["context"]["sentences"]]
        start = len(flattened_paragraphs)
        flattened_paragraphs.extend(paragraphs)
        offsets.append((start, len(flattened_paragraphs)))
    encoding_started = time.perf_counter()
    paragraph_vectors = np.asarray(encoder(flattened_paragraphs, query=False))
    query_vectors = np.asarray(encoder([item["question"] for item in rows], query=True))
    per_query_latency = (time.perf_counter() - encoding_started) * 1000 / max(1, len(rows))
    output = []
    for item, query_vector, (start, end) in zip(rows, query_vectors, offsets):
        titles = list(item["context"]["title"])
        paragraphs = [" ".join(sentences) for sentences in item["context"]["sentences"]]
        dense = rank_dense(paragraph_vectors[start:end], query_vector, titles)
        lexical = [
            {"doc_id": title, "title": title, "content": paragraph}
            for title, paragraph in zip(titles, paragraphs)
        ]
        hybrid = reciprocal_rank_fusion(bm25_ranking(lexical, item["question"]), dense)
        expected = set(item["supporting_facts"]["title"])
        for method, ranking in (("neural_dense_top_k", dense), ("neural_hybrid_rrf", hybrid)):
            selected = ranking[:top_k]
            output.append(
                {
                    "case_id": item["id"],
                    "method": method,
                    "expected_count": len(expected),
                    "selected_ids": selected,
                    **retrieval_metrics(expected, selected),
                    "query_latency_ms": round(per_query_latency, 4),
                }
            )
    return {
        "dataset": "HotpotQA distractor validation",
        "case_count": len(rows),
        "candidate_count": 10,
        "candidate_scope": "per-query distractor context",
        "aggregates": aggregate(output),
        "cases": output,
    }, {"parquet_sha256": sha256(path), "full_validation_count": pq.ParquetFile(path).metadata.num_rows}


def render_markdown(report: dict) -> str:
    lines = [
        "# V3 公开 RAG Benchmark " + ("全量可评测集报告" if report["scope"] == "full_evaluable" else "固定子集报告"),
        "",
        f"- 模型：`{report['model']}`；设备：{report['device']}；Top-K：{report['top_k']}",
        f"- 固定随机种子：{report['seed']}；单数据集样本上限：{report['sample_size']}",
        "- 范围：" + (
            "官方发布数据中的全部可映射证据查询；仍不是官方 leaderboard 提交。"
            if report["scope"] == "full_evaluable"
            else "固定子集工程复现；不是官方全量 leaderboard 成绩。"
        ),
        "",
        "| 数据集 | 用例 | 候选规模 | 方法 | Evidence Recall | 完整证据集命中率 | MRR | 延迟 ms |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for dataset in report["datasets"]:
        for method, row in dataset["aggregates"].items():
            lines.append(
                f"| {dataset['dataset']} | {dataset['case_count']} | {dataset['candidate_count']} | {method} | "
                f"{row['evidence_recall']:.4f} | {row['complete_evidence_set']:.4f} | "
                f"{row['reciprocal_rank']:.4f} | {row['query_latency_ms']:.4f} |"
            )
    lines.extend(["", "## 数据溯源", "", "```json", json.dumps(report["sources"], ensure_ascii=False, indent=2), "```", "", "## 限制", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed official public RAG benchmark subsets.")
    parser.add_argument("--multihop-root", type=Path, default=Path(".cache/benchmarks/MultiHop-RAG"))
    parser.add_argument("--conditionalqa-root", type=Path, default=Path(".cache/benchmarks/ConditionalQA"))
    parser.add_argument("--hotpot-path", type=Path, default=Path(".cache/benchmarks/hotpot_hf/distractor/validation-00000-of-00001.parquet"))
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output-dir", type=Path, default=Path("output/rag_evaluation"))
    parser.add_argument("--refresh-existing", type=Path, default=None)
    args = parser.parse_args()

    if args.refresh_existing is not None:
        report_path = args.refresh_existing
        report = json.loads(report_path.read_text(encoding="utf-8"))
        datasets = {item["dataset"]: item for item in report["datasets"]}
        multihop_cases = datasets["MultiHop-RAG"]["case_count"]
        conditional_cases = datasets["ConditionalQA"]["case_count"]
        hotpot_cases = datasets["HotpotQA distractor validation"]["case_count"]
        sources = report["sources"]
        sources["MultiHop-RAG"].update(
            {
                "evaluable_query_count": multihop_cases,
                "excluded_query_count": sources["MultiHop-RAG"]["full_query_count"] - multihop_cases,
                "exclusion_rule": "fewer than two evidence URLs map to the released corpus",
            }
        )
        sources["ConditionalQA"].update(
            {
                "eligible_answerable_count": conditional_cases,
                "evaluable_query_count": conditional_cases,
                "excluded_query_count": sources["ConditionalQA"]["full_dev_count"] - conditional_cases,
                "exclusion_rule": "not answerable, missing evidence, or evidence paragraph cannot be mapped",
            }
        )
        report["scope"] = (
            "full_evaluable"
            if hotpot_cases == sources["HotpotQA"]["full_validation_count"]
            else "fixed_subset"
        )
        report["limitations"] = [
            "覆盖官方发布数据中的全部可评测查询，但使用自定义检索实现与指标，不能替代官方 leaderboard 提交。",
            f"MultiHop-RAG 排除 {sources['MultiHop-RAG']['excluded_query_count']} 条无法将至少两份证据 URL 映射回发布语料的查询。",
            f"ConditionalQA 排除 {sources['ConditionalQA']['excluded_query_count']} 条不可回答、缺少证据或证据段无法映射的开发样本。",
            "ConditionalQA 候选池由样本关联文档加 100 个固定干扰文档组成；HotpotQA 使用官方 per-query distractor context。",
            "本报告只评估证据检索，不使用生成模型计算最终答案准确率。",
            "MultiHop-RAG 和 ConditionalQA 延迟只含常驻索引后的查询编码；HotpotQA 延迟还包含每题 10 段上下文编码，两者不可直接横向比较。",
            "文档输入受模型 512 token 上限截断，长文档结果可能低估分块索引后的最佳性能。",
        ]
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
        print(f"Refreshed report metadata: {report_path}")
        return

    import torch

    encoder, dimension = load_encoder(args.model, batch_size=32)
    multihop_docs, multihop_cases, multihop_source = multihop_rag_data(args.multihop_root, args.sample_size, args.seed)
    conditional_docs, conditional_cases, conditional_source = conditionalqa_data(args.conditionalqa_root, args.sample_size, args.seed)
    datasets = [
        evaluate_global(name="MultiHop-RAG", documents=multihop_docs, cases=multihop_cases, encoder=encoder, top_k=args.top_k),
        evaluate_global(name="ConditionalQA", documents=conditional_docs, cases=conditional_cases, encoder=encoder, top_k=args.top_k),
    ]
    hotpot, hotpot_source = hotpotqa_evaluate(args.hotpot_path, args.sample_size, args.seed, encoder, args.top_k)
    datasets.append(hotpot)
    full_evaluable = (
        datasets[0]["case_count"] == multihop_source["evaluable_query_count"]
        and datasets[1]["case_count"] == conditional_source["evaluable_query_count"]
        and datasets[2]["case_count"] == hotpot_source["full_validation_count"]
    )
    report = {
        "model": args.model,
        "backend": "transformers.AutoModel CLS pooling",
        "embedding_dimension": dimension,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "top_k": args.top_k,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "scope": "full_evaluable" if full_evaluable else "fixed_subset",
        "datasets": datasets,
        "sources": {
            "MultiHop-RAG": multihop_source,
            "ConditionalQA": conditional_source,
            "HotpotQA": hotpot_source,
        },
        "limitations": [
            (
                "覆盖官方发布数据中的全部可评测查询，但使用自定义检索实现与指标，不能替代官方 leaderboard 提交。"
                if full_evaluable
                else "每个数据集使用固定随机子集，不能替代官方全量评测或 leaderboard 提交。"
            ),
            f"MultiHop-RAG 排除 {multihop_source['excluded_query_count']} 条无法将至少两份证据 URL 映射回发布语料的查询。",
            f"ConditionalQA 排除 {conditional_source['excluded_query_count']} 条不可回答、缺少证据或证据段无法映射的开发样本。",
            "ConditionalQA 候选池由样本关联文档加 100 个固定干扰文档组成；HotpotQA 使用官方 per-query distractor context。",
            "本报告只评估证据检索，不使用生成模型计算最终答案准确率。",
            "MultiHop-RAG 和 ConditionalQA 延迟只含常驻索引后的查询编码；HotpotQA 延迟还包含每题 10 段上下文编码，两者不可直接横向比较。",
            "文档输入受模型 512 token 上限截断，长文档结果可能低估分块索引后的最佳性能。",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "public_benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "public_benchmark_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({item["dataset"]: item["aggregates"] for item in datasets}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
