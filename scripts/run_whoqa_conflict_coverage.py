from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally live outside the runtime wheel.

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.whoqa_conflict_coverage import (
    BASELINES,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DATASET_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    EXPECTED_CASES,
    ALLOWED_TEMPLATE_COUNTS,
    FRC_METHOD,
    METHODS,
    REFERENCE_BASELINE,
    RERANKER_MODEL,
    RERANKER_REVISION,
    ROLE_NAMES,
    ROLE_QUERIES,
    ROLE_RELEVANCE_MIX,
    ROLE_THRESHOLD,
    RRF_K,
    SOURCE_FILENAME,
    SOURCE_LICENSE,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    SOURCE_SHA256,
    TOKEN_BUDGET,
    TOP_K,
    FrozenWhoQAScorer,
    evaluate_scored_cases,
    load_raw,
    preparation_summary,
    prepare_cases,
    read_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
    write_report,
)


PROTOCOL_SHA256 = "7a92b637521a8b0ee5d9620b2972c1dfb95a5f0b6c246974171749f7b53c6bb9"
EXECUTION_SHA256 = "ba9df61c8a146901df35a1af6216f694bd1a3b7bcbe9477ecb37c55ee02b57aa"
EXECUTION_ERRATUM_SHA256 = "1e7ac8fe9bf6008fba27c63ed4895d9ff976f6efec407bd682307984bf7212a2"
EXECUTION_ERRATUM2_SHA256 = "8e39f2cddb2e73add09b8c1785ae539403f87c16aac5cbc65c62d783f2e4d86f"
EXECUTION_ERRATUM3_SHA256 = "2027903c6a71e9b06d1bc8e9c592e37730fdb2cdc15fd652f060a4e92ccfa7f7"
EXECUTION_ERRATUM4_SHA256 = "d4638618477cc0956e33d77d6ef299bee4a15bc7649bf4b18664721dabf2856a"
EXECUTION_ERRATUM5_SHA256 = "25364bb774cced00c2507d87daa31e74ab7e5cc9c4df2a1a266ba9c88a060209"
POST_RESULT_CORRECTION_SHA256 = "5106dea12cad7b26a4f16d973eac770a85d81b6bf02b5a79c4a53d1b4b372927"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_registration(
    protocol_path: Path,
    execution_path: Path,
    erratum_path: Path,
    erratum2_path: Path,
    erratum3_path: Path,
    erratum4_path: Path,
    erratum5_path: Path,
    correction_path: Path,
) -> None:
    if sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("WhoQA parent protocol changed after registration")
    protocol = _load_json(protocol_path)
    execution = _load_json(execution_path)
    erratum = _load_json(erratum_path)
    erratum2 = _load_json(erratum2_path)
    erratum3 = _load_json(erratum3_path)
    erratum4 = _load_json(erratum4_path)
    erratum5 = _load_json(erratum5_path)
    correction = _load_json(correction_path)
    if protocol.get("schema_version") != "frc-whoqa-conflict-coverage-protocol-v1":
        raise ValueError("unsupported WhoQA parent protocol")
    if execution.get("schema_version") != "frc-whoqa-conflict-coverage-execution-v1":
        raise ValueError("unsupported WhoQA execution registration")
    if (
        erratum.get("schema_version")
        != "frc-whoqa-conflict-coverage-execution-erratum-v1"
    ):
        raise ValueError("unsupported WhoQA execution erratum")
    if (
        erratum2.get("schema_version")
        != "frc-whoqa-conflict-coverage-execution-erratum-v2"
    ):
        raise ValueError("unsupported WhoQA second execution erratum")
    if (
        erratum3.get("schema_version")
        != "frc-whoqa-conflict-coverage-execution-erratum-v3"
    ):
        raise ValueError("unsupported WhoQA third execution erratum")
    if (
        erratum4.get("schema_version")
        != "frc-whoqa-conflict-coverage-execution-erratum-v4"
    ):
        raise ValueError("unsupported WhoQA fourth execution erratum")
    if (
        erratum5.get("schema_version")
        != "frc-whoqa-conflict-coverage-execution-erratum-v5"
    ):
        raise ValueError("unsupported WhoQA fifth execution erratum")
    if (
        correction.get("schema_version")
        != "frc-whoqa-conflict-coverage-post-result-correction-v1"
    ):
        raise ValueError("unsupported WhoQA post-result correction")
    if execution["parent_protocol"]["sha256"] != PROTOCOL_SHA256:
        raise ValueError("WhoQA execution registration points to another protocol")
    if sha256(execution_path) != EXECUTION_SHA256:
        raise ValueError("WhoQA execution registration changed after the erratum")
    if erratum["parent_files"] != {
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_sha256": EXECUTION_SHA256,
    }:
        raise ValueError("WhoQA erratum parent hashes do not match")
    if sha256(erratum_path) != EXECUTION_ERRATUM_SHA256:
        raise ValueError("WhoQA first erratum changed after second registration")
    if erratum2["parent_files"] != {
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_sha256": EXECUTION_SHA256,
        "execution_erratum_sha256": EXECUTION_ERRATUM_SHA256,
    }:
        raise ValueError("WhoQA second erratum parent hashes do not match")
    if sha256(erratum2_path) != EXECUTION_ERRATUM2_SHA256:
        raise ValueError("WhoQA second erratum changed after third registration")
    if erratum3["parent_files"] != {
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_sha256": EXECUTION_SHA256,
        "execution_erratum_sha256": EXECUTION_ERRATUM_SHA256,
        "execution_erratum2_sha256": EXECUTION_ERRATUM2_SHA256,
    }:
        raise ValueError("WhoQA third erratum parent hashes do not match")
    if sha256(erratum3_path) != EXECUTION_ERRATUM3_SHA256:
        raise ValueError("WhoQA third erratum changed after registration")
    if erratum4["parent_files"] != {
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_sha256": EXECUTION_SHA256,
        "execution_erratum_sha256": EXECUTION_ERRATUM_SHA256,
        "execution_erratum2_sha256": EXECUTION_ERRATUM2_SHA256,
        "execution_erratum3_sha256": EXECUTION_ERRATUM3_SHA256,
    }:
        raise ValueError("WhoQA fourth erratum parent hashes do not match")
    if sha256(erratum4_path) != EXECUTION_ERRATUM4_SHA256:
        raise ValueError("WhoQA fourth erratum changed after registration")
    if erratum5["parent_files"] != {
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_sha256": EXECUTION_SHA256,
        "execution_erratum_sha256": EXECUTION_ERRATUM_SHA256,
        "execution_erratum2_sha256": EXECUTION_ERRATUM2_SHA256,
        "execution_erratum3_sha256": EXECUTION_ERRATUM3_SHA256,
        "execution_erratum4_sha256": EXECUTION_ERRATUM4_SHA256,
    }:
        raise ValueError("WhoQA fifth erratum parent hashes do not match")
    if sha256(erratum5_path) != EXECUTION_ERRATUM5_SHA256:
        raise ValueError("WhoQA fifth erratum changed after registration")
    if sha256(correction_path) != POST_RESULT_CORRECTION_SHA256:
        raise ValueError("WhoQA post-result correction changed after registration")
    if (
        correction["initial_result_disclosed"]["status"]
        != "WHOQA_SELECTION_SUPPORT_NOT_ESTABLISHED"
        or correction["frozen_correction_before_recalculation"][
            "primary_metric_or_threshold_changed"
        ]
        is not False
    ):
        raise ValueError("WhoQA post-result correction boundary mismatch")

    source = protocol["frozen_source"]
    expected_source = {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "path": SOURCE_FILENAME,
        "license": SOURCE_LICENSE,
        "expected_evaluation_question_count": EXPECTED_CASES,
    }
    if any(source[key] != value for key, value in expected_source.items()):
        raise ValueError("WhoQA frozen source does not match implementation")
    observed = execution["frozen_source_observation"]
    if (
        observed["sha256"] != SOURCE_SHA256
        or int(observed["record_count"]) != EXPECTED_CASES
    ):
        raise ValueError("WhoQA source observation does not match implementation")

    scoring = protocol["frozen_scoring"]
    expected_scoring = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "reranker_model": RERANKER_MODEL,
        "reranker_revision": RERANKER_REVISION,
        "score_calibration": "per_case_minmax",
        "rrf_k": RRF_K,
        "role_relevance_mix": ROLE_RELEVANCE_MIX,
        "gold_fields_visible_to_scorer": False,
        "role_queries": ROLE_QUERIES,
    }
    if any(scoring[key] != value for key, value in expected_scoring.items()):
        raise ValueError("WhoQA frozen scoring contract does not match implementation")
    selection = protocol["frozen_selection"]
    if (
        selection["methods"] != list(METHODS)
        or selection["reproducible_baselines"] != list(BASELINES)
        or selection["frc_method"] != FRC_METHOD
        or selection["predeclared_reference_baseline"] != REFERENCE_BASELINE
        or int(selection["top_k"]) != TOP_K
        or int(selection["token_budget"]) != TOKEN_BUDGET
        or float(selection["role_threshold"]) != ROLE_THRESHOLD
        or selection["required_roles"] != list(ROLE_NAMES)
        or selection["gold_fields_visible_to_selector"] is not False
    ):
        raise ValueError("WhoQA frozen selector contract does not match implementation")
    bootstrap = protocol["frozen_metrics"]["paired_bootstrap"]
    if (
        int(bootstrap["resamples"]) != BOOTSTRAP_RESAMPLES
        or int(bootstrap["seed"]) != BOOTSTRAP_SEED
        or bootstrap["resample_unit"] != "case"
    ):
        raise ValueError("WhoQA frozen bootstrap contract does not match implementation")
    templates = erratum["corrected_frozen_template_handling"]
    if (
        templates["score_all_non_empty_public_templates"] is not True
        or tuple(templates["allowed_templates_per_q_id"])
        != ALLOWED_TEMPLATE_COUNTS
        or templates["statistical_unit"] != "q_id"
    ):
        raise ValueError("WhoQA template execution contract does not match implementation")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen blind WhoQA conflict-viewpoint selection audit."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(".cache/benchmarks/whoqa/WhoQA.json"),
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path(".cache/benchmarks/whoqa/prepared_blind.jsonl"),
    )
    parser.add_argument(
        "--scored",
        type=Path,
        default=Path(".cache/benchmarks/whoqa/scored_blind.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".cache/benchmarks/whoqa/provenance.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/whoqa_conflict_coverage_protocol.json"
        ),
    )
    parser.add_argument(
        "--execution",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/whoqa_conflict_coverage_execution.json"
        ),
    )
    parser.add_argument(
        "--execution-erratum",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/whoqa_conflict_coverage_execution_erratum.json"
        ),
    )
    parser.add_argument(
        "--execution-erratum2",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/whoqa_conflict_coverage_execution_erratum2.json"
        ),
    )
    parser.add_argument(
        "--execution-erratum3",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/whoqa_conflict_coverage_execution_erratum3.json"
        ),
    )
    parser.add_argument(
        "--execution-erratum4",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/whoqa_conflict_coverage_execution_erratum4.json"
        ),
    )
    parser.add_argument(
        "--execution-erratum5",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/whoqa_conflict_coverage_execution_erratum5.json"
        ),
    )
    parser.add_argument(
        "--post-result-correction",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/whoqa_conflict_coverage_post_result_correction.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/whoqa_conflict_coverage"),
    )
    parser.add_argument("--hf-home", type=Path, default=Path("D:/RAG_test/.hf_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--rerank-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    stage_flags = sum((args.prepare_only, args.score_only, args.evaluate_only))
    if stage_flags > 1:
        parser.error("choose at most one stage-only flag")
    for path in (
        args.input,
        args.protocol,
        args.execution,
        args.execution_erratum,
        args.execution_erratum2,
        args.execution_erratum3,
        args.execution_erratum4,
        args.execution_erratum5,
        args.post_result_correction,
    ):
        if not path.is_file():
            parser.error(f"required frozen input does not exist: {path}")
    _validate_registration(
        args.protocol,
        args.execution,
        args.execution_erratum,
        args.execution_erratum2,
        args.execution_erratum3,
        args.execution_erratum4,
        args.execution_erratum5,
        args.post_result_correction,
    )

    if args.evaluate_only:
        if not args.prepared.is_file() or not args.scored.is_file():
            parser.error("evaluation requires prepared and scored blind caches")
        summary = preparation_summary(list(read_jsonl(args.prepared)))
    else:
        if args.force_prepare or not args.prepared.is_file():
            raw = load_raw(args.input)
            prepared = prepare_cases(raw)
            write_jsonl(args.prepared, prepared)
        else:
            if sha256(args.input) != SOURCE_SHA256:
                parser.error("WhoQA raw source hash changed")
            prepared = list(read_jsonl(args.prepared))
            if len(prepared) != EXPECTED_CASES:
                parser.error("WhoQA prepared cache has invalid case coverage")
        summary = preparation_summary(prepared)

    manifest = {
        "schema_version": "frc-whoqa-conflict-coverage-provenance-v1",
        "dataset": DATASET_NAME,
        "registration": {
            "protocol_path_label": args.protocol.name,
            "protocol_sha256": sha256(args.protocol),
            "execution_path_label": args.execution.name,
            "execution_sha256": sha256(args.execution),
            "execution_erratum_path_label": args.execution_erratum.name,
            "execution_erratum_sha256": sha256(args.execution_erratum),
            "execution_erratum2_path_label": args.execution_erratum2.name,
            "execution_erratum2_sha256": sha256(args.execution_erratum2),
            "execution_erratum3_path_label": args.execution_erratum3.name,
            "execution_erratum3_sha256": sha256(args.execution_erratum3),
            "execution_erratum4_path_label": args.execution_erratum4.name,
            "execution_erratum4_sha256": sha256(args.execution_erratum4),
            "execution_erratum5_path_label": args.execution_erratum5.name,
            "execution_erratum5_sha256": sha256(args.execution_erratum5),
            "post_result_correction_path_label": args.post_result_correction.name,
            "post_result_correction_sha256": sha256(args.post_result_correction),
        },
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "path_label": args.input.name,
            "sha256": sha256(args.input),
            "expected_sha256": SOURCE_SHA256,
            "license": SOURCE_LICENSE,
        },
        "preparation": {
            **summary,
            "path_label": args.prepared.name,
            "sha256": sha256(args.prepared),
        },
        "scoring": {
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "reranker_model": RERANKER_MODEL,
            "reranker_revision": RERANKER_REVISION,
            "score_calibration": "per_case_minmax",
            "all_public_templates": True,
            "gold_fields_visible_to_scorer": False,
            "floating_point_dtype": "float16",
            "embedding_batch_size": args.embedding_batch_size,
            "reranker_batch_size": args.rerank_batch_size,
            "case_batch_size": args.case_batch_size,
            "status": "NOT_RUN",
        },
        "evaluation": {"status": "NOT_RUN"},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if args.prepare_only:
        print(args.prepared)
        print(args.manifest)
        return 0

    if not args.evaluate_only:
        started = time.perf_counter()

        def progress(index: int, total: int, case_id: str) -> None:
            if index == 1 or index % 25 == 0 or index == total:
                print(f"scored {index}/{total}: {case_id}", flush=True)

        scorer = FrozenWhoQAScorer(
            hf_home=args.hf_home,
            device=args.device,
            embedding_batch_size=args.embedding_batch_size,
            rerank_batch_size=args.rerank_batch_size,
        )
        scored_count = score_cases_resumable(
            args.prepared,
            args.scored,
            scorer=scorer,
            case_batch_size=args.case_batch_size,
            progress=progress,
        )
        manifest["scoring"].update(
            {
                "status": "COMPLETE",
                "case_count": scored_count,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "path_label": args.scored.name,
                "sha256": sha256(args.scored),
            }
        )
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        manifest["scoring"].update(
            {
                "status": "REUSED_COMPLETE",
                "case_count": EXPECTED_CASES,
                "path_label": args.scored.name,
                "sha256": sha256(args.scored),
            }
        )
    if args.score_only:
        print(args.scored)
        print(args.manifest)
        return 0

    raw = load_raw(args.input)
    report, evidence = evaluate_scored_cases(raw, read_jsonl(args.scored))
    paths = write_report(
        report,
        evidence,
        args.output_dir,
        source_paths={
            "protocol": args.protocol,
            "execution": args.execution,
            "execution_erratum": args.execution_erratum,
            "execution_erratum2": args.execution_erratum2,
            "execution_erratum3": args.execution_erratum3,
            "execution_erratum4": args.execution_erratum4,
            "execution_erratum5": args.execution_erratum5,
            "post_result_correction": args.post_result_correction,
            "raw_source": args.input,
            "prepared_blind": args.prepared,
            "scored_blind": args.scored,
        },
    )
    manifest["evaluation"] = {
        "status": "COMPLETE",
        "outcome": report["analysis"]["outcome"]["status"],
        "artifacts": {
            name: {"path_label": path.name, "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
