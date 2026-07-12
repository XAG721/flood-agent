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


def test_query_evidence_set_selects_complementary_slots_under_budget() -> None:
    store = build_store(
        [
            RAGDocument(
                doc_id="a_elder_primary",
                corpus=CorpusType.POLICY,
                title="Elder evacuation plan",
                content="Elder evacuation uses staffed buses and assisted transfer.",
                metadata={"region": "beilin", "source_tier": "official", "token_cost": 80},
            ),
            RAGDocument(
                doc_id="b_elder_duplicate",
                corpus=CorpusType.POLICY,
                title="Elder evacuation route",
                content="Elder evacuation uses the same staffed bus transfer route.",
                metadata={"region": "beilin", "source_tier": "official", "token_cost": 80},
            ),
            RAGDocument(
                doc_id="c_school_closure",
                corpus=CorpusType.POLICY,
                title="School closure pickup plan",
                content="School closure starts when pickup routes become congested.",
                metadata={"region": "beilin", "source_tier": "official", "token_cost": 80},
            ),
        ]
    )

    results = store.query_evidence_set(
        CorpusType.POLICY,
        "elder evacuation school closure",
        filters={"region": "beilin"},
        top_k=2,
        candidate_k=3,
        token_budget=180,
        slots=["elder evacuation", "school closure"],
    )

    assert [item.doc_id for item in results] == ["a_elder_primary", "c_school_closure"]
    first_selection = store.explain(results[0])["evidence_selection"]
    second_selection = store.explain(results[1])["evidence_selection"]
    assert first_selection["covered_slots"][0]["text"] == "elder evacuation"
    assert second_selection["covered_slots"][0]["text"] == "school closure"
    assert second_selection["selected_token_cost"] == 160


def test_query_evidence_set_respects_token_budget_and_records_ledger() -> None:
    store = build_store(
        [
            RAGDocument(
                doc_id="oversized_all_in_one",
                corpus=CorpusType.POLICY,
                title="Shelter route water pump plan",
                content="Shelter route water pump plan covers every operational slot.",
                metadata={"region": "beilin", "source_tier": "official", "token_cost": 180},
            ),
            RAGDocument(
                doc_id="shelter_short",
                corpus=CorpusType.POLICY,
                title="Shelter opening plan",
                content="Shelter opening covers safe indoor capacity.",
                metadata={"region": "beilin", "source_tier": "official", "token_cost": 70},
            ),
            RAGDocument(
                doc_id="route_short",
                corpus=CorpusType.POLICY,
                title="Route control plan",
                content="Route control keeps evacuation lanes accessible.",
                metadata={"region": "beilin", "source_tier": "official", "token_cost": 80},
            ),
        ]
    )

    results = store.query_evidence_set(
        CorpusType.POLICY,
        "shelter route",
        filters={"region": "beilin"},
        top_k=2,
        candidate_k=3,
        token_budget=150,
        slots=["shelter", "route"],
    )

    assert [item.doc_id for item in results] == ["shelter_short", "route_short"]
    explain = store.explain(results[0])
    selection = explain["evidence_selection"]
    assert selection["mode"] == "budget_aware_trustworthy_set_selection"
    assert selection["token_budget"] == 150
    assert selection["selected_token_cost"] == 150
    assert all(item.doc_id != "oversized_all_in_one" for item in results)
    assert store.cite(results[0]).retrieval_explain["evidence_selection"]["selection_rank"] == 1


def test_query_evidence_set_penalizes_low_reliability_and_conflict() -> None:
    store = build_store(
        [
            RAGDocument(
                doc_id="school_closure_misinfo",
                corpus=CorpusType.POLICY,
                title="School closure status",
                content="School closure is cancelled and no assisted pickup is required.",
                metadata={
                    "region": "beilin",
                    "source_label": "misinfo",
                    "stance": "cancel",
                    "token_cost": 60,
                },
            ),
            RAGDocument(
                doc_id="school_closure_official",
                corpus=CorpusType.POLICY,
                title="School closure official notice",
                content="School closure is active and assisted pickup starts on the north route.",
                metadata={
                    "region": "beilin",
                    "source_label": "correct",
                    "stance": "activate",
                    "conflicts_with": ["school_closure_misinfo"],
                    "token_cost": 70,
                },
            ),
        ]
    )

    results = store.query_evidence_set(
        CorpusType.POLICY,
        "school closure assisted pickup",
        filters={"region": "beilin"},
        top_k=1,
        candidate_k=2,
        token_budget=100,
        slots=["school closure", "assisted pickup"],
    )

    assert [item.doc_id for item in results] == ["school_closure_official"]
    selection = store.explain(results[0])["evidence_selection"]
    assert selection["trust_score"] > 0.85
    assert selection["source"]["source_label"] == "correct"
