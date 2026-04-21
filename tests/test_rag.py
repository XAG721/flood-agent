from __future__ import annotations

from datetime import UTC, datetime, timedelta

from flood_system.models import CorpusType, RAGDocument
from flood_system.rag import SimpleRAGStore


def build_store(documents: list[RAGDocument]) -> SimpleRAGStore:
    return SimpleRAGStore(documents)


def test_query_supports_chinese_terms_and_ngrams() -> None:
    store = build_store(
        [
            RAGDocument(
                doc_id="profile_elder",
                corpus=CorpusType.PROFILE,
                title="低洼区老人家庭画像",
                content="该片区老人家庭集中，行动不便，需要优先转移。",
                metadata={"region": "西安市碑林区", "village": "建设里片区"},
            ),
            RAGDocument(
                doc_id="profile_school",
                corpus=CorpusType.PROFILE,
                title="学校接送区画像",
                content="放学高峰容易产生车辆滞留。",
                metadata={"region": "西安市碑林区", "village": "五一路片区"},
            ),
        ]
    )

    results = store.query(
        CorpusType.PROFILE,
        "这个水位对低洼区老人意味着什么",
        filters={"region": "西安市碑林区"},
        top_k=2,
    )

    assert results
    assert results[0].doc_id == "profile_elder"
    explain = store.explain(results[0])
    assert "低洼区老人" in explain["matched_fragments"]
    assert explain["matched_filters"] == {"region": "西安市碑林区"}


def test_query_prefers_title_and_phrase_matches() -> None:
    store = build_store(
        [
            RAGDocument(
                doc_id="policy_title_hit",
                corpus=CorpusType.POLICY,
                title="学校停课与接送区疏导规则",
                content="积水达到阈值后应尽快发布停课指令。",
                metadata={"region": "西安市碑林区"},
            ),
            RAGDocument(
                doc_id="policy_content_hit",
                corpus=CorpusType.POLICY,
                title="校园交通组织要求",
                content="学校停课与接送区疏导规则需要由指挥席确认后执行。",
                metadata={"region": "西安市碑林区"},
            ),
        ]
    )

    results = store.query(CorpusType.POLICY, "学校停课接送区", filters={"region": "西安市碑林区"}, top_k=2)

    assert [item.doc_id for item in results] == ["policy_title_hit", "policy_content_hit"]


def test_query_uses_time_decay_to_prefer_recent_documents() -> None:
    recent_time = datetime.now(UTC) - timedelta(hours=6)
    old_time = datetime.now(UTC) - timedelta(days=420)
    store = build_store(
        [
            RAGDocument(
                doc_id="case_recent",
                corpus=CorpusType.CASE,
                title="工厂库存转移案例",
                content="冷链库存上移后损失显著下降。",
                metadata={"region": "西安市碑林区", "updated_at": recent_time.isoformat()},
            ),
            RAGDocument(
                doc_id="case_old",
                corpus=CorpusType.CASE,
                title="工厂库存转移案例",
                content="冷链库存上移后损失显著下降。",
                metadata={"region": "西安市碑林区", "updated_at": old_time.isoformat()},
            ),
        ]
    )

    results = store.query(CorpusType.CASE, "工厂库存转移", filters={"region": "西安市碑林区"}, top_k=2)

    assert [item.doc_id for item in results] == ["case_recent", "case_old"]
    assert store.explain(results[0])["time_decay"] > store.explain(results[1])["time_decay"]


def test_cite_exposes_retrieval_explain() -> None:
    store = build_store(
        [
            RAGDocument(
                doc_id="policy_factory",
                corpus=CorpusType.POLICY,
                title="工厂停工审批规则",
                content="涉及库存和危化风险时必须先审批再执行。",
                metadata={"region": "西安市碑林区", "updated_at": "2026-04-01T08:00:00+00:00"},
            )
        ]
    )

    result = store.query(CorpusType.POLICY, "工厂停工审批", filters={"region": "西安市碑林区"}, top_k=1)[0]
    citation = store.cite(result)

    assert citation.doc_id == "policy_factory"
    assert citation.retrieval_explain["final_score"] > 0
    assert citation.retrieval_explain["matched_terms"]["title"]
