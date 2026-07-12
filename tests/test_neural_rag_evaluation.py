from __future__ import annotations

from scripts.run_neural_rag_evaluation import (
    bm25_ranking,
    reciprocal_rank_fusion,
    score_selection,
    terms,
)
from scripts.run_public_rag_benchmarks import evaluate_global, git_revision, normalized_html, retrieval_metrics


def test_neural_evaluation_helpers_rank_and_fuse_deterministically() -> None:
    documents = [
        {"doc_id": "a", "title": "下穿通道处置", "content": "橙色暴雨预警触发道路封控"},
        {"doc_id": "b", "title": "学校处置", "content": "学校通知家长"},
    ]
    assert "下穿" in terms("下穿通道")
    assert bm25_ranking(documents, "下穿通道橙色暴雨")[0] == "a"
    assert reciprocal_rank_fusion(["a", "b"], ["b", "a"]) == ["a", "b"]


def test_neural_evaluation_metrics_include_role_coverage_and_unsupported_ratio() -> None:
    case = {
        "relevant_doc_ids": ["a", "b"],
        "required_roles": ["condition", "procedure", "exception"],
    }
    documents = {
        "a": {"metadata": {"evidence_roles": ["condition", "procedure"]}},
        "noise": {"metadata": {"evidence_roles": []}},
    }

    score = score_selection(case, ["a", "noise"], documents)

    assert score["evidence_recall"] == 0.5
    assert score["citation_precision"] == 0.5
    assert score["unsupported_evidence_ratio"] == 0.5
    assert score["role_coverage"] == 0.6667
    assert score["task_element_completeness"] == score["role_coverage"]


def test_public_benchmark_helpers_normalize_evidence_and_score_complete_sets() -> None:
    assert normalized_html("<p>  Apply <strong>within 10 days</strong>. </p>") == "apply within 10 days ."
    complete = retrieval_metrics({"a", "b"}, ["noise", "a", "b"])
    partial = retrieval_metrics({"a", "b"}, ["a", "noise"])

    assert complete["evidence_recall"] == 1.0
    assert complete["complete_evidence_set"] == 1.0
    assert complete["reciprocal_rank"] == 0.5
    assert partial["evidence_recall"] == 0.5
    assert partial["complete_evidence_set"] == 0.0


def test_public_benchmark_git_revision_does_not_require_safe_directory(tmp_path) -> None:
    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    (git_dir / "refs" / "heads" / "main").write_text("a" * 40 + "\n", encoding="ascii")

    assert git_revision(tmp_path) == "a" * 40


def test_public_benchmark_global_evaluation_batches_document_and_query_encoding() -> None:
    calls: list[tuple[int, bool]] = []

    def encoder(texts: list[str], *, query: bool):
        calls.append((len(texts), query))
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]

    report = evaluate_global(
        name="fixture",
        documents=[
            {"doc_id": "a", "title": "Alpha", "text": "alpha evidence"},
            {"doc_id": "b", "title": "Beta", "text": "beta evidence"},
        ],
        cases=[
            {"case_id": "q1", "query": "alpha", "expected_ids": ["a"]},
            {"case_id": "q2", "query": "beta", "expected_ids": ["b"]},
        ],
        encoder=encoder,
        top_k=1,
    )

    assert calls == [(2, False), (2, True)]
    assert report["case_count"] == 2
    assert report["aggregates"]["neural_dense_top_k"]["evidence_recall"] == 1.0
