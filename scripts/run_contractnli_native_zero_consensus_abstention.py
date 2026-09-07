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

from research.frc_rag.contractnli_native_zero_consensus_abstention import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXPERIMENT_ID,
    FrozenContractNliScorer,
    OFFICIAL_REPOSITORY_REVISION,
    build_candidate_coverage,
    build_deterministic_queries,
    build_gold_rows,
    evaluate_contractnli,
    prepare_blind_cases,
    read_test_source,
    select_sample,
    validate_execution_registration,
    validate_implementation_registration,
    validate_protocol,
    validate_query_cache,
    write_report,
)
from research.frc_rag.rgb_cost_aware_frc import (  # noqa: E402
    load_frozen_tokenizer,
    read_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)


MODULE_PATH = (
    REPO_ROOT / "research/frc_rag/contractnli_native_zero_consensus_abstention.py"
)
RUNNER_PATH = REPO_ROOT / "scripts/run_contractnli_native_zero_consensus_abstention.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_contractnli_native_zero_consensus_abstention.py"


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
        test = [
            item for item in regular if Path(item.filename).name.lower() == "test.json"
        ]
        if len(test) != 1:
            raise ValueError("ContractNLI archive lacks one unambiguous test.json")
        train = [
            item for item in regular if Path(item.filename).name.lower() == "train.json"
        ]
        dev = [
            item for item in regular if Path(item.filename).name.lower() == "dev.json"
        ]
        if len(train) != 1 or len(dev) != 1:
            raise ValueError("ContractNLI archive split member census changed")
        member_manifest = {
            "regular_files": len(regular),
            "test_member": {"path": test[0].filename, "bytes": test[0].file_size},
            "train_member_name_seen_but_content_opened": False,
            "dev_member_name_seen_but_content_opened": False,
        }
    return {
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "license_sha256": sha256(license_path),
        "license_identifier": "CC-BY-4.0",
        "dataset_repository_revision": OFFICIAL_REPOSITORY_REVISION,
        "archive_members": member_manifest,
        "test_member_opened": False,
        "train_member_opened": False,
        "dev_member_opened": False,
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
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
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
    return validate_implementation_registration(
        args.implementation,
        protocol_path=args.protocol,
        module_path=MODULE_PATH,
        runner_path=RUNNER_PATH,
        test_path=TEST_PATH,
    )


def _validate_execution(args: argparse.Namespace) -> dict[str, Any]:
    _validate_implementation(args)
    return validate_execution_registration(
        args.execution,
        implementation_path=args.implementation,
        source_archive=args.source_archive,
        prepared_path=args.prepared,
        candidate_map_path=args.candidate_map,
        census_path=args.census,
        coverage_path=args.coverage,
    )


def _assert_before_neural_work(args: argparse.Namespace) -> None:
    if args.execution.exists():
        raise ValueError("ContractNLI preparation is frozen after execution registration")
    existing = [path for path in (args.queries, args.scored) if path.exists()]
    if existing:
        raise ValueError(
            "ContractNLI preparation cannot run after neural caches exist: "
            + ", ".join(str(path) for path in existing)
        )


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _assert_before_neural_work(args)
    source_manifest = _source_manifest(args.source_archive, args.license_file)
    source = read_test_source(args.source_archive)
    selected, sampling = select_sample(source)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared, maps, structural = prepare_blind_cases(selected, tokenizer)
    write_jsonl(args.prepared, prepared)
    write_jsonl(args.candidate_map, maps)
    census = {
        "schema_version": "frc-contractnli-v50-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "source_manifest": {
            **source_manifest,
            "test_member_opened": True,
            "train_member_opened": False,
            "dev_member_opened": False,
        },
        "sampling": sampling,
        "structural_census": structural,
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "labels_document_ids_urls_offsets_or_gold_exported_to_blind_cache": False,
        "train_or_dev_content_opened": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }
    _write_json(args.census, census)
    print(
        json.dumps(
            {
                "sampling": sampling,
                "structural_census": structural,
                "hashes": {
                    "prepared": sha256(args.prepared),
                    "candidate_map": sha256(args.candidate_map),
                    "census": sha256(args.census),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def coverage(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _assert_before_neural_work(args)
    census = json.loads(args.census.read_text(encoding="utf-8"))
    if sha256(args.prepared) != census["prepared_blind_sha256"] or sha256(
        args.candidate_map
    ) != census["candidate_map_sha256"]:
        raise ValueError("ContractNLI blind artifacts changed after census")
    source = read_test_source(args.source_archive)
    maps = list(read_jsonl(args.candidate_map))
    structural = census["structural_census"]
    gold_rows = build_gold_rows(
        source,
        maps,
        pool_quartile_boundaries=structural[
            "candidate_pool_quartile_boundaries"
        ],
        document_length_quartile_boundaries=structural[
            "document_length_quartile_boundaries"
        ],
    )
    report = build_candidate_coverage(gold_rows, census["sampling"])
    _write_json(args.coverage, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def register_execution(args: argparse.Namespace) -> None:
    implementation = _validate_implementation(args)
    if args.execution.exists():
        raise FileExistsError("ContractNLI execution registration already exists")
    if args.queries.exists() or args.scored.exists():
        raise ValueError("ContractNLI neural work started before execution registration")
    coverage_report = json.loads(args.coverage.read_text(encoding="utf-8"))
    if coverage_report.get("minimum_cases_and_ceiling_checks_passed") is not True:
        raise ValueError("ContractNLI candidate coverage gate is not open")
    runtime = _runtime_environment(args)
    if args.device == "cuda" and runtime["cuda_available"] is not True:
        raise ValueError("ContractNLI frozen CUDA runtime is unavailable")
    source_manifest = _source_manifest(args.source_archive, args.license_file)
    value = {
        "schema_version": "frc-contractnli-native-zero-consensus-execution-v50",
        "experiment_id": EXPERIMENT_ID,
        "registered_at": args.registered_at,
        "hashes": {
            "implementation_registration_sha256": sha256(args.implementation),
            "source_archive_sha256": sha256(args.source_archive),
            "prepared_blind_sha256": sha256(args.prepared),
            "candidate_map_sha256": sha256(args.candidate_map),
            "blind_census_sha256": sha256(args.census),
            "candidate_coverage_sha256": sha256(args.coverage),
        },
        "source_artifacts": source_manifest,
        "runtime": runtime,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
        "method_or_threshold_change_after_registration_forbidden": True,
        "negative_null_or_inconclusive_result_must_be_published": True,
        "implementation_registration": implementation,
    }
    _write_json(args.execution, value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def generate(args: argparse.Namespace) -> None:
    _validate_execution(args)
    if args.scored.exists():
        raise ValueError("ContractNLI score cache exists before query generation")
    prepared_rows = list(read_jsonl(args.prepared))
    rows = build_deterministic_queries(prepared_rows)
    validate_query_cache(prepared_rows, rows)
    write_jsonl(args.queries, rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "queries_sha256": sha256(args.queries),
                "fallback_count": 0,
                "fallback_rate": 0.0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def score(args: argparse.Namespace) -> None:
    _validate_execution(args)
    prepared_rows = list(read_jsonl(args.prepared))
    query_rows = list(read_jsonl(args.queries))
    validate_query_cache(prepared_rows, query_rows)
    scorer = FrozenContractNliScorer(
        query_rows=query_rows,
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        use_fp16=not args.no_fp16,
    )
    scored = score_cases_resumable(
        prepared_rows,
        scorer,
        args.scored,
        case_batch_size=args.case_batch_size,
    )
    print(
        json.dumps(
            {"rows": len(scored), "scored_sha256": sha256(args.scored)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    execution = _validate_execution(args)
    prepared_rows = list(read_jsonl(args.prepared))
    candidate_maps = list(read_jsonl(args.candidate_map))
    query_rows = list(read_jsonl(args.queries))
    scored_rows = list(read_jsonl(args.scored))
    expected_ids = [str(row["id"]) for row in prepared_rows]
    for name, rows in (
        ("candidate map", candidate_maps),
        ("query", query_rows),
        ("score", scored_rows),
    ):
        if [str(row["id"]) for row in rows] != expected_ids:
            raise ValueError(f"ContractNLI {name} cache is incomplete or out of order")
    query_summary = validate_query_cache(prepared_rows, query_rows)
    census = json.loads(args.census.read_text(encoding="utf-8"))
    source = read_test_source(args.source_archive)
    structural = census["structural_census"]
    gold_rows = build_gold_rows(
        source,
        candidate_maps,
        pool_quartile_boundaries=structural[
            "candidate_pool_quartile_boundaries"
        ],
        document_length_quartile_boundaries=structural[
            "document_length_quartile_boundaries"
        ],
    )
    source_artifacts = {
        "execution_registration_sha256": sha256(args.execution),
        "execution_registration": execution,
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "queries_sha256": sha256(args.queries),
        "scored_sha256": sha256(args.scored),
        "module_sha256": sha256(MODULE_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "test_sha256": sha256(TEST_PATH),
    }
    report, evidence = evaluate_contractnli(
        gold_rows, scored_rows, query_summary, source_artifacts
    )
    write_report(
        report,
        evidence,
        args.report_json,
        args.report_markdown,
        args.evidence,
    )
    first = {
        "json": sha256(args.report_json),
        "markdown": sha256(args.report_markdown),
        "evidence": sha256(args.evidence),
    }
    write_report(
        report,
        evidence,
        args.report_json,
        args.report_markdown,
        args.evidence,
    )
    second = {
        "json": sha256(args.report_json),
        "markdown": sha256(args.report_markdown),
        "evidence": sha256(args.evidence),
    }
    if first != second:
        raise AssertionError("ContractNLI v50 report is not byte deterministic")
    print(
        json.dumps(
            {
                "status": report["analysis"]["outcome"]["status"],
                "comparison": report["analysis"]["family_comparison"],
                "checks": report["analysis"]["support_checks"],
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
        description="Run the preregistered ContractNLI v50 abstention experiment."
    )
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "coverage",
            "register-execution",
            "generate",
            "score",
            "evaluate",
            "runtime",
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_native_zero_consensus_abstention_protocol_v50.json"
        ),
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_native_zero_consensus_abstention_implementation_v50.json"
        ),
    )
    parser.add_argument(
        "--execution",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_native_zero_consensus_abstention_execution_v50.json"
        ),
    )
    root = Path(".cache/benchmarks/contractnli")
    cache = root / "v50"
    parser.add_argument(
        "--source-archive", type=Path, default=root / "contract-nli.zip"
    )
    parser.add_argument("--license-file", type=Path, default=root / "LICENSE")
    parser.add_argument("--prepared", type=Path, default=cache / "prepared_blind.jsonl")
    parser.add_argument(
        "--candidate-map", type=Path, default=cache / "candidate_map.jsonl"
    )
    parser.add_argument("--census", type=Path, default=cache / "blind_census.json")
    parser.add_argument(
        "--coverage", type=Path, default=cache / "candidate_coverage.json"
    )
    parser.add_argument("--queries", type=Path, default=cache / "queries.jsonl")
    parser.add_argument("--scored", type=Path, default=cache / "scored.jsonl")
    output = Path(
        "output/rag_evaluation/contractnli_native_zero_consensus_abstention"
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_native_zero_consensus_abstention_result_v50.json"
        ),
    )
    parser.add_argument("--report-markdown", type=Path, default=output / "report.md")
    parser.add_argument("--evidence", type=Path, default=output / "cases.jsonl.gz")
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=4)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--registered-at", default="2026-08-02T19:00:00+08:00")
    args = parser.parse_args()
    for name in (
        "protocol",
        "implementation",
        "execution",
        "source_archive",
        "license_file",
        "prepared",
        "candidate_map",
        "census",
        "coverage",
        "queries",
        "scored",
        "report_json",
        "report_markdown",
        "evidence",
        "hf_home",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    commands = {
        "prepare": prepare,
        "coverage": coverage,
        "register-execution": register_execution,
        "generate": generate,
        "score": score,
        "evaluate": evaluate,
        "runtime": runtime,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
