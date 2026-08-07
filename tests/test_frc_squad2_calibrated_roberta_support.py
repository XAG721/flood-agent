from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import research.frc_rag.squad2_calibrated_roberta_support as v64


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    REPO_ROOT
    / "docs/progressive_upgrade/squad2_calibrated_roberta_support_protocol_v64.json"
)
SOURCE_REGISTRATION = REPO_ROOT / "docs/progressive_upgrade/squad2_source_registration_v58.json"
MODEL_REGISTRATION = (
    REPO_ROOT
    / "docs/progressive_upgrade/quac_roberta_qa_support_transfer_model_registration_v62.json"
)
IMPLEMENTATION_ERRATUM = (
    REPO_ROOT
    / "docs/progressive_upgrade/squad2_calibrated_roberta_support_implementation_erratum_v64.json"
)
LATEST_IMPLEMENTATION_ERRATUM = (
    REPO_ROOT
    / "docs/progressive_upgrade/squad2_calibrated_roberta_support_result_label_erratum_v64.json"
)


def _source(article_count: int = 8) -> dict:
    data = []
    for article in range(article_count):
        context = f"Article {article} states that the answer is value {article}."
        answer = f"value {article}"
        start = context.index(answer)
        data.append(
            {
                "title": f"Article {article}",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": f"a-{article}",
                                "question": f"What is the value for article {article}?",
                                "is_impossible": False,
                                "answers": [{"text": answer, "answer_start": start}],
                            },
                            {
                                "id": f"n-{article}",
                                "question": f"Who founded article {article}?",
                                "is_impossible": True,
                                "answers": [],
                            },
                        ],
                    }
                ],
            }
        )
    return {"version": "2.0", "data": data}


def _support_row(
    case_id: str,
    article: str,
    state: str,
    margin: float,
) -> dict:
    return {
        "case_id": case_id,
        "article_cluster": article,
        "paragraph_cluster": f"p-{case_id}",
        "answer_state": state,
        "score_margin": margin,
        "invalid_fail_closed_used": False,
        "raw_prediction_sha256": f"sha-{case_id}",
    }


def test_protocol_is_hash_locked_before_dev_access() -> None:
    value = v64.validate_protocol(PROTOCOL)
    assert value["experiment_id"] == v64.EXPERIMENT_ID
    assert value["prior_boundary"]["dev_v2_content_opened_or_parsed"] is False
    assert value["threshold_calibration"]["learned_parameter_count"] == 1


def test_selection_is_balanced_deterministic_and_excludes_commitments() -> None:
    source = _source(12)
    rows, _ = v64.v58.extract_cases(source)
    excluded = {v64._hash(str(rows[0]["raw_id"]))}
    first, summary = v64.select_balanced_sample(
        source,
        stage="calibration",
        excluded_commitments=excluded,
        target_per_group=4,
        maximum_cases_per_article_per_state=1,
    )
    second, _ = v64.select_balanced_sample(
        source,
        stage="calibration",
        excluded_commitments=excluded,
        target_per_group=4,
        maximum_cases_per_article_per_state=1,
    )
    assert [row["raw_id"] for row in first] == [row["raw_id"] for row in second]
    assert summary["selected_answer_state_counts"] == {
        "answer_bearing": 4,
        "no_answer": 4,
    }
    assert summary["selected_prior_commitment_overlap"] == 0


def test_blind_preparation_exports_no_gold_fields() -> None:
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return list(range(len(text.split())))

    selected, _ = v64.select_balanced_sample(
        _source(6),
        stage="confirmation",
        excluded_commitments=set(),
        target_per_group=2,
        maximum_cases_per_article_per_state=1,
    )
    prepared, maps, qa_inputs, structural = v64.prepare_blind_cases(
        selected,
        Tokenizer(),
        stage="confirmation",
    )
    forbidden = {"answer_state", "answer_text", "answers", "gold_candidate_ids"}
    assert len(prepared) == len(maps) == len(qa_inputs) == 4
    assert not any(forbidden & set(row) for row in prepared)
    assert not any(forbidden & set(row) for row in qa_inputs)
    assert all(row["id"].startswith("s64c") for row in prepared)
    assert structural["gold_fields_exported_to_blind_caches"] is False


def test_threshold_rule_is_strict_and_deterministic() -> None:
    rows = [
        _support_row("a1", "x", "answer_bearing", 2.0),
        _support_row("a2", "y", "answer_bearing", 3.0),
        _support_row("n1", "z", "no_answer", -2.0),
        _support_row("n2", "w", "no_answer", -1.0),
    ]
    fitted = v64.choose_threshold(rows)
    assert fitted == v64.choose_threshold(rows)
    assert fitted["metrics"]["balanced_accuracy"] == 1.0
    assert v64.apply_threshold_decision(rows[0], 2.0) is False
    assert v64.apply_threshold_decision(rows[0], 1.999) is True


def test_article_disjoint_oof_calibration_has_no_group_leakage() -> None:
    articles_by_fold: dict[int, list[str]] = {fold: [] for fold in range(5)}
    index = 0
    while any(len(values) < 2 for values in articles_by_fold.values()):
        article = f"article-{index}"
        fold = v64.fold_for_article(article)
        if len(articles_by_fold[fold]) < 2:
            articles_by_fold[fold].append(article)
        index += 1
    rows = []
    for articles in articles_by_fold.values():
        for article in articles:
            rows.append(_support_row(f"a-{article}", article, "answer_bearing", 4.0))
            rows.append(_support_row(f"n-{article}", article, "no_answer", -4.0))
    result = v64.calibrate_article_disjoint_oof(rows)
    assert result["article_disjoint"] is True
    assert result["oof_metrics"]["balanced_accuracy"] == 1.0
    assert all(item["validation_articles"] == 2 for item in result["folds"])


def test_calibration_gate_locks_one_threshold_and_opens_dev() -> None:
    rows = []
    for index in range(1000):
        rows.append(
            _support_row(f"a-{index}", f"a-{index % 250}", "answer_bearing", 5.0)
        )
        rows.append(_support_row(f"n-{index}", f"n-{index % 250}", "no_answer", -5.0))
    report, evidence = v64.evaluate_calibration_gate(
        rows,
        {
            "rows": 2000,
            "invalid_output_count": 0,
            "invalid_output_rate": 0.0,
            "gold_fields_visible_to_verifier": False,
        },
        {
            "selected_prior_commitment_overlap": 0,
            "schema_exclusion_rate": 0.0,
            "selected_paragraphs": 1000,
            "selected_articles": 500,
        },
        {"query_or_retrieval_scoring_started": False},
    )
    outcome = report["analysis"]["outcome"]
    assert len(evidence) == 2000
    assert outcome["calibration_gate_passed"] is True
    assert outcome["confirmation_dev_open_authorized"] is True
    assert isinstance(outcome["locked_threshold"], float)


def test_confirmation_support_gate_stops_retrieval_on_failure() -> None:
    rows = [
        *(
            _support_row(f"a-{index}", f"a-{index}", "answer_bearing", -1.0)
            for index in range(300)
        ),
        *(
            _support_row(f"n-{index}", f"n-{index}", "no_answer", 1.0)
            for index in range(300)
        ),
    ]
    report, _ = v64.evaluate_confirmation_support_gate(
        rows,
        {
            "rows": 600,
            "invalid_output_count": 0,
            "invalid_output_rate": 0.0,
            "gold_fields_visible_to_verifier": False,
        },
        {
            "schema_exclusion_rate": 0.0,
            "selected_paragraphs": 600,
            "selected_articles": 600,
        },
        {"query_or_retrieval_scoring_started": False},
        locked_threshold=0.0,
    )
    outcome = report["analysis"]["outcome"]
    assert outcome["support_gate_passed"] is False
    assert outcome["retrieval_scoring_open_authorized"] is False
    assert outcome["gate_2"] == "NO-GO/SHADOW"


def test_thresholded_support_preserves_raw_margin_and_hashes_decision() -> None:
    raw = {
        "id": "case-1",
        "score_margin": 0.5,
        "span_sha256": "span",
        "invalid_fail_closed_used": False,
    }
    supported = v64.threshold_support_rows([raw], 0.4)[0]
    rejected = v64.threshold_support_rows([raw], 0.6)[0]
    assert supported["support_passed"] is True
    assert rejected["support_passed"] is False
    assert supported["score_margin"] == rejected["score_margin"] == 0.5
    assert supported["raw_prediction_sha256"] != rejected["raw_prediction_sha256"]


def test_report_evidence_archive_is_deterministic(tmp_path: Path) -> None:
    report = {
        "schema_version": "test",
        "experiment_id": v64.EXPERIMENT_ID,
        "metadata": {"cases": 1},
        "analysis": {
            "outcome": {
                "status": "TEST",
                "selector_adoption_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            }
        },
    }
    archives = []
    for index in range(2):
        archive = tmp_path / f"cases-{index}.jsonl.gz"
        v64.write_report(
            report,
            [{"case_id": "one"}],
            tmp_path / f"result-{index}.json",
            tmp_path / f"report-{index}.md",
            archive,
            title="test",
        )
        archives.append(archive)
    assert archives[0].read_bytes() == archives[1].read_bytes()
    with gzip.open(archives[0], "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline()) == {"case_id": "one"}


def test_prior_exclusion_union_requires_four_disjoint_600_row_maps(
    tmp_path: Path,
) -> None:
    paths = []
    for group in range(4):
        path = tmp_path / f"map-{group}.jsonl"
        rows = [
            {
                "schema_version": "test",
                "id": f"case-{group}-{index}",
                "source_case_commitment": f"commitment-{group}-{index}",
                "article_commitment": f"article-{group}-{index}",
                "paragraph_commitment": f"paragraph-{group}-{index}",
                "paragraph_length": 1,
                "article_selected_case_count": 1,
                "candidate_intervals": [],
            }
            for index in range(600)
        ]
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        paths.append(path)
    assert len(v64.load_prior_exclusion_union(paths)) == 2400
    with pytest.raises(ValueError, match="exactly four"):
        v64.load_prior_exclusion_union(paths[:3])


def test_candidate_coverage_adapts_union_overlap_to_legacy_audit_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def inherited_coverage(gold_rows: list[dict], sampling: dict) -> dict:
        observed["gold_rows"] = gold_rows
        observed["sampling"] = sampling
        return {"selected_v58_commitment_overlap": sampling["selected_v58_commitment_overlap"]}

    monkeypatch.setattr(v64.v61, "build_candidate_coverage", inherited_coverage)
    coverage = v64.build_candidate_coverage(
        [{"id": "case-1"}],
        {"selected_prior_commitment_overlap": 0},
    )
    assert observed["gold_rows"] == [{"id": "case-1"}]
    assert observed["sampling"]["selected_v58_commitment_overlap"] == 0
    assert coverage["selected_v58_commitment_overlap"] == 0
    assert coverage["schema_version"] == "frc-squad2-v64-candidate-coverage-v1"


def test_successor_erratum_hash_chain_accepts_current_implementation() -> None:
    value = v64.validate_implementation_registration(
        IMPLEMENTATION_ERRATUM,
        protocol_path=PROTOCOL,
        source_registration_path=SOURCE_REGISTRATION,
        model_registration_path=MODEL_REGISTRATION,
        module_path=REPO_ROOT
        / "research/frc_rag/squad2_calibrated_roberta_support.py",
        runner_path=REPO_ROOT
        / "scripts/run_squad2_calibrated_roberta_support.py",
        test_path=Path(__file__),
        successor_erratum_path=LATEST_IMPLEMENTATION_ERRATUM,
    )
    assert value["schema_version"] == "frc-squad2-v64-implementation-erratum-v1"


def test_method_renaming_removes_inherited_qwen_v59_labels() -> None:
    report = {
        "analysis": {
            "aggregates": {"qwen_span_supported_bm25_topk_v59": {}},
            "strongest_shared_gate_non_frc": "qwen_span_supported_bm25_topk_v59",
            "strongest_same_gate_frc": "qwen_span_supported_frc_v43_v59",
            "mechanism": "legacy",
            "supported_stratum_deltas": {
                "answer_state:no_answer": {
                    "strongest_shared_gate_non_frc": "qwen_span_supported_bm25_topk_v59"
                }
            },
        }
    }
    evidence = [
        {
            "configurations": {
                "128": {"methods": {"qwen_span_supported_bm25_topk_v59": {}}}
            }
        }
    ]
    v64._rename_methods(report, evidence)
    assert "Qwen" not in report["analysis"]["mechanism"]
    assert report["analysis"]["strongest_shared_gate_non_frc"].endswith("_v64")
    stratum = report["analysis"]["supported_stratum_deltas"][
        "answer_state:no_answer"
    ]
    assert stratum["strongest_shared_gate_non_frc"].endswith("_v64")
