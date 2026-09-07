from __future__ import annotations

import argparse
import json
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.contractnli_dev_calibrated_robust_consensus import (  # noqa: E402
    EXPERIMENT_ID,
    FrozenDevelopmentScorer,
    build_candidate_coverage,
    build_deterministic_queries,
    build_development_gold_rows,
    build_development_records,
    calibrate_development,
    prepare_blind_cases,
    read_split_source,
    select_balanced_sample,
    validate_protocol,
    validate_query_cache,
    write_development_report,
)
from research.frc_rag.rgb_cost_aware_frc import (  # noqa: E402
    load_frozen_tokenizer,
    read_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)


MODULE_PATH = (
    REPO_ROOT / "research/frc_rag/contractnli_dev_calibrated_robust_consensus.py"
)
RUNNER_PATH = REPO_ROOT / "scripts/run_contractnli_dev_calibrated_robust_consensus.py"
TEST_PATH = (
    REPO_ROOT / "tests/test_frc_contractnli_dev_calibrated_robust_consensus.py"
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _source_manifest(archive_path: Path, license_path: Path) -> dict[str, Any]:
    if not archive_path.is_file() or not license_path.is_file():
        raise FileNotFoundError("ContractNLI archive or official license is missing")
    license_text = license_path.read_text(encoding="utf-8-sig", errors="replace")
    if "Creative Commons Attribution 4.0" not in license_text:
        raise ValueError("ContractNLI official license is not CC BY 4.0")
    with zipfile.ZipFile(archive_path) as archive:
        regular = [item for item in archive.infolist() if not item.is_dir()]
        members: dict[str, dict[str, Any]] = {}
        for split in ("train", "dev", "test"):
            matched = [
                item
                for item in regular
                if Path(item.filename).name.lower() == f"{split}.json"
            ]
            if len(matched) != 1:
                raise ValueError(f"ContractNLI archive lacks one {split}.json")
            members[split] = {
                "path": matched[0].filename,
                "bytes": matched[0].file_size,
            }
    return {
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "license_sha256": sha256(license_path),
        "license_identifier": "CC-BY-4.0",
        "regular_files": len(regular),
        "split_members": members,
        "dev_member_opened": False,
        "train_member_opened": False,
        "test_member_opened_by_v51": False,
    }


def _runtime_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "embedding_batch_size": args.embedding_batch_size,
        "reranker_batch_size": args.reranker_batch_size,
        "case_batch_size": args.case_batch_size,
        "fp16": not args.no_fp16,
        "offline_model_loading": True,
        "query_generator": "none",
        "token_budgets": [256, 512, 1024],
        "development_folds": 5,
    }


def _runtime_environment(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    import transformers

    gpu = None
    memory = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        memory = int(torch.cuda.get_device_properties(0).total_memory / (1024**2))
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "gpu_memory_mib": memory,
        "parameters": _runtime_parameters(args),
    }


def _validate_implementation(args: argparse.Namespace) -> dict[str, Any]:
    validate_protocol(args.protocol)
    original = json.loads(args.implementation.read_text(encoding="utf-8"))
    erratum = json.loads(args.implementation_erratum.read_text(encoding="utf-8"))
    if sha256(args.implementation) != erratum.get(
        "original_implementation_registration_sha256"
    ):
        raise ValueError("ContractNLI v51 original implementation changed")
    if original.get("hashes") != erratum.get("superseded_hashes"):
        raise ValueError("ContractNLI v51 superseded implementation hashes changed")
    expected = {
        "protocol_sha256": sha256(args.protocol),
        "module_sha256": sha256(MODULE_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "test_sha256": sha256(TEST_PATH),
    }
    if erratum.get("corrected_hashes") != expected:
        raise ValueError("ContractNLI v51 implementation erratum hash mismatch")
    boundary = erratum.get("boundary_at_erratum_registration", {})
    required = {
        "dev_opened_only_for_registered_sampling_and_blind_preparation": True,
        "dev_gold_joined_for_coverage_or_metrics": False,
        "dev_queries_generated": False,
        "dev_neural_scoring_started": False,
        "train_member_opened": False,
        "test_member_opened_by_v51": False,
    }
    if boundary != required:
        raise ValueError("ContractNLI v51 erratum boundary changed")
    return {"original": original, "erratum": erratum}


def _validate_execution(args: argparse.Namespace) -> dict[str, Any]:
    _validate_implementation(args)
    value = json.loads(args.dev_execution.read_text(encoding="utf-8"))
    expected = {
        "original_implementation_registration_sha256": sha256(
            args.implementation
        ),
        "implementation_erratum_sha256": sha256(args.implementation_erratum),
        "source_archive_sha256": sha256(args.source_archive),
        "prepared_blind_sha256": sha256(args.dev_prepared),
        "candidate_map_sha256": sha256(args.dev_candidate_map),
        "blind_census_sha256": sha256(args.dev_census),
    }
    if value.get("hashes") != expected:
        raise ValueError("ContractNLI v51 development execution hash mismatch")
    if value.get("dev_query_generation_started") is not False:
        raise ValueError("ContractNLI v51 dev execution was registered too late")
    if value.get("candidate_coverage_pending_until_after_complete_scoring") is not True:
        raise ValueError("ContractNLI v51 late-gold boundary changed")
    if value.get("train_member_opened") is not False:
        raise ValueError("ContractNLI v51 train was opened before development")
    return value


def _assert_before_dev_execution(args: argparse.Namespace) -> None:
    if args.dev_execution.exists():
        raise ValueError("ContractNLI v51 dev preparation is frozen")
    existing = [
        path
        for path in (
            args.dev_prepared,
            args.dev_candidate_map,
            args.dev_census,
            args.dev_coverage,
            args.dev_queries,
            args.dev_scored,
        )
        if path.exists()
    ]
    if existing:
        raise ValueError(
            "ContractNLI v51 preparation cannot run after neural caches exist: "
            + ", ".join(str(path) for path in existing)
        )


def prepare_dev(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _assert_before_dev_execution(args)
    manifest = _source_manifest(args.source_archive, args.license_file)
    source = read_split_source(args.source_archive, "dev")
    selected, sampling = select_balanced_sample(source)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared, maps, structural = prepare_blind_cases(selected, tokenizer)
    write_jsonl(args.dev_prepared, prepared)
    write_jsonl(args.dev_candidate_map, maps)
    census = {
        "schema_version": "frc-contractnli-v51-dev-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "source_manifest": {
            **manifest,
            "dev_member_opened": True,
            "train_member_opened": False,
            "test_member_opened_by_v51": False,
        },
        "sampling": sampling,
        "structural_census": structural,
        "prepared_blind_sha256": sha256(args.dev_prepared),
        "candidate_map_sha256": sha256(args.dev_candidate_map),
        "labels_document_ids_urls_offsets_folds_or_gold_exported_to_blind_cache": False,
        "train_member_opened": False,
        "test_member_opened_by_v51": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }
    _write_json(args.dev_census, census)
    print(
        json.dumps(
            {
                "sampling": sampling,
                "structural_census": structural,
                "hashes": {
                    "prepared": sha256(args.dev_prepared),
                    "candidate_map": sha256(args.dev_candidate_map),
                    "census": sha256(args.dev_census),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _dev_gold(args: argparse.Namespace) -> list[dict[str, Any]]:
    source = read_split_source(args.source_archive, "dev")
    maps = list(read_jsonl(args.dev_candidate_map))
    census = json.loads(args.dev_census.read_text(encoding="utf-8"))
    structural = census["structural_census"]
    return build_development_gold_rows(
        source,
        maps,
        pool_quartile_boundaries=structural["candidate_pool_quartile_boundaries"],
        document_length_quartile_boundaries=structural[
            "document_length_quartile_boundaries"
        ],
    )


def coverage_dev(args: argparse.Namespace) -> None:
    execution = _validate_execution(args)
    census = json.loads(args.dev_census.read_text(encoding="utf-8"))
    if sha256(args.dev_prepared) != census["prepared_blind_sha256"] or sha256(
        args.dev_candidate_map
    ) != census["candidate_map_sha256"]:
        raise ValueError("ContractNLI v51 dev blind artifacts changed")
    prepared = list(read_jsonl(args.dev_prepared))
    queries = list(read_jsonl(args.dev_queries))
    scored = list(read_jsonl(args.dev_scored))
    expected_ids = [str(row["id"]) for row in prepared]
    if len(expected_ids) != 240:
        raise ValueError("ContractNLI v51 dev blind cache is incomplete")
    for name, rows in (("query", queries), ("score", scored)):
        if [str(row["id"]) for row in rows] != expected_ids:
            raise ValueError(
                f"ContractNLI v51 dev {name} cache must be complete before gold join"
            )
    gold_rows = _dev_gold(args)
    report = build_candidate_coverage(gold_rows, census["sampling"])
    report["gold_joined_after_complete_dev_score_cache"] = True
    report["queries_sha256"] = sha256(args.dev_queries)
    report["scored_sha256"] = sha256(args.dev_scored)
    report["development_execution_sha256"] = sha256(args.dev_execution)
    report["development_execution"] = execution
    _write_json(args.dev_coverage, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def register_dev(args: argparse.Namespace) -> None:
    implementation = _validate_implementation(args)
    if args.dev_execution.exists():
        raise FileExistsError("ContractNLI v51 dev execution already exists")
    if args.dev_queries.exists() or args.dev_scored.exists():
        raise ValueError("ContractNLI v51 dev neural work started too early")
    if args.dev_coverage.exists():
        raise ValueError("ContractNLI v51 gold coverage was joined before scoring")
    runtime = _runtime_environment(args)
    if args.device == "cuda" and runtime["cuda_available"] is not True:
        raise ValueError("ContractNLI v51 frozen CUDA runtime is unavailable")
    value = {
        "schema_version": "frc-contractnli-v51-development-execution-v1",
        "experiment_id": EXPERIMENT_ID,
        "registered_at": args.registered_at,
        "hashes": {
            "original_implementation_registration_sha256": sha256(
                args.implementation
            ),
            "implementation_erratum_sha256": sha256(
                args.implementation_erratum
            ),
            "source_archive_sha256": sha256(args.source_archive),
            "prepared_blind_sha256": sha256(args.dev_prepared),
            "candidate_map_sha256": sha256(args.dev_candidate_map),
            "blind_census_sha256": sha256(args.dev_census),
        },
        "source_artifacts": {
            **_source_manifest(args.source_archive, args.license_file),
            "dev_member_opened_for_structural_preparation": True,
        },
        "runtime": runtime,
        "dev_query_generation_started": False,
        "dev_neural_scoring_started": False,
        "dev_metrics_computed": False,
        "candidate_coverage_pending_until_after_complete_scoring": True,
        "gold_joined_for_coverage_or_metrics": False,
        "train_member_opened": False,
        "test_member_opened_by_v51": False,
        "implementation_registration": implementation,
        "threshold_selection_rule_change_after_registration_forbidden": True,
    }
    _write_json(args.dev_execution, value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def generate_dev(args: argparse.Namespace) -> None:
    _validate_execution(args)
    if args.dev_scored.exists():
        raise ValueError("ContractNLI v51 score cache exists before queries")
    prepared = list(read_jsonl(args.dev_prepared))
    rows = build_deterministic_queries(prepared)
    validate_query_cache(prepared, rows)
    write_jsonl(args.dev_queries, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "queries_sha256": sha256(args.dev_queries),
                "fallback_count": 0,
                "fallback_rate": 0.0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def score_dev(args: argparse.Namespace) -> None:
    _validate_execution(args)
    prepared = list(read_jsonl(args.dev_prepared))
    queries = list(read_jsonl(args.dev_queries))
    validate_query_cache(prepared, queries)
    scorer = FrozenDevelopmentScorer(
        query_rows=queries,
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        use_fp16=not args.no_fp16,
    )
    scored = score_cases_resumable(
        prepared,
        scorer,
        args.dev_scored,
        case_batch_size=args.case_batch_size,
    )
    print(
        json.dumps(
            {"rows": len(scored), "scored_sha256": sha256(args.dev_scored)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def calibrate_dev(args: argparse.Namespace) -> None:
    execution = _validate_execution(args)
    coverage = json.loads(args.dev_coverage.read_text(encoding="utf-8"))
    if coverage.get("minimum_cases_and_ceiling_checks_passed") is not True:
        raise ValueError("ContractNLI v51 post-score candidate coverage gate is closed")
    if coverage.get("gold_joined_after_complete_dev_score_cache") is not True:
        raise ValueError("ContractNLI v51 gold was not joined after complete scoring")
    prepared = list(read_jsonl(args.dev_prepared))
    maps = list(read_jsonl(args.dev_candidate_map))
    queries = list(read_jsonl(args.dev_queries))
    scored = list(read_jsonl(args.dev_scored))
    expected = [str(row["id"]) for row in prepared]
    for name, rows in (("map", maps), ("query", queries), ("score", scored)):
        if [str(row["id"]) for row in rows] != expected:
            raise ValueError(f"ContractNLI v51 dev {name} cache is incomplete")
    query_summary = validate_query_cache(prepared, queries)
    gold = _dev_gold(args)
    records = build_development_records(gold, scored)
    source_artifacts = {
        "development_execution_sha256": sha256(args.dev_execution),
        "development_execution": execution,
        "candidate_coverage_sha256": sha256(args.dev_coverage),
        "prepared_blind_sha256": sha256(args.dev_prepared),
        "candidate_map_sha256": sha256(args.dev_candidate_map),
        "queries_sha256": sha256(args.dev_queries),
        "scored_sha256": sha256(args.dev_scored),
        "module_sha256": sha256(MODULE_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "test_sha256": sha256(TEST_PATH),
    }
    report, evidence = calibrate_development(
        records, query_summary, source_artifacts
    )
    write_development_report(
        report,
        evidence,
        args.dev_result,
        args.dev_report,
        args.dev_evidence,
    )
    first = {
        "json": sha256(args.dev_result),
        "markdown": sha256(args.dev_report),
        "evidence": sha256(args.dev_evidence),
    }
    write_development_report(
        report,
        evidence,
        args.dev_result,
        args.dev_report,
        args.dev_evidence,
    )
    second = {
        "json": sha256(args.dev_result),
        "markdown": sha256(args.dev_report),
        "evidence": sha256(args.dev_evidence),
    }
    if first != second:
        raise AssertionError("ContractNLI v51 dev report is not deterministic")
    print(
        json.dumps(
            {
                "status": report["analysis"]["outcome"]["status"],
                "train_open_authorized": report["analysis"]["outcome"][
                    "train_open_authorized"
                ],
                "final_threshold_hex": report["analysis"]["outcome"][
                    "final_threshold_hex"
                ],
                "oof_utility_gain": report["analysis"][
                    "oof_utility_gain_vs_ungated_v49"
                ],
                "checks": report["analysis"]["development_checks"],
                "output_hashes": second,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def runtime(args: argparse.Namespace) -> None:
    print(json.dumps(_runtime_environment(args), ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered ContractNLI v51 development calibration."
    )
    parser.add_argument(
        "command",
        choices=(
            "prepare-dev",
            "coverage-dev",
            "register-dev",
            "generate-dev",
            "score-dev",
            "calibrate-dev",
            "runtime",
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_dev_calibrated_robust_consensus_protocol_v51.json"
        ),
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_dev_calibrated_robust_consensus_implementation_v51.json"
        ),
    )
    parser.add_argument(
        "--implementation-erratum",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_dev_calibrated_robust_consensus_implementation_erratum_v51.json"
        ),
    )
    parser.add_argument(
        "--dev-execution",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_dev_calibrated_robust_consensus_development_execution_v51.json"
        ),
    )
    root = Path(".cache/benchmarks/contractnli")
    cache = root / "v51"
    parser.add_argument(
        "--source-archive", type=Path, default=root / "contract-nli.zip"
    )
    parser.add_argument("--license-file", type=Path, default=root / "LICENSE")
    parser.add_argument(
        "--dev-prepared", type=Path, default=cache / "dev_prepared_blind.jsonl"
    )
    parser.add_argument(
        "--dev-candidate-map", type=Path, default=cache / "dev_candidate_map.jsonl"
    )
    parser.add_argument("--dev-census", type=Path, default=cache / "dev_census.json")
    parser.add_argument(
        "--dev-coverage", type=Path, default=cache / "dev_coverage.json"
    )
    parser.add_argument(
        "--dev-queries", type=Path, default=cache / "dev_queries.jsonl"
    )
    parser.add_argument("--dev-scored", type=Path, default=cache / "dev_scored.jsonl")
    output = Path(
        "output/rag_evaluation/contractnli_dev_calibrated_robust_consensus"
    )
    parser.add_argument(
        "--dev-result",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_dev_calibrated_robust_consensus_development_result_v51.json"
        ),
    )
    parser.add_argument("--dev-report", type=Path, default=output / "development.md")
    parser.add_argument(
        "--dev-evidence", type=Path, default=output / "development_cases.jsonl.gz"
    )
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=4)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--registered-at", default="2026-08-02T20:00:00+08:00")
    args = parser.parse_args()
    for name in (
        "protocol",
        "implementation",
        "implementation_erratum",
        "dev_execution",
        "source_archive",
        "license_file",
        "dev_prepared",
        "dev_candidate_map",
        "dev_census",
        "dev_coverage",
        "dev_queries",
        "dev_scored",
        "dev_result",
        "dev_report",
        "dev_evidence",
        "hf_home",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    commands = {
        "prepare-dev": prepare_dev,
        "coverage-dev": coverage_dev,
        "register-dev": register_dev,
        "generate-dev": generate_dev,
        "score-dev": score_dev,
        "calibrate-dev": calibrate_dev,
        "runtime": runtime,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
