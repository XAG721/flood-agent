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

from research.frc_rag.contractnli_rank_concurrence_confirmation import (  # noqa: E402
    EXPERIMENT_ID,
    FrozenRankConcurrenceScorer,
    build_candidate_coverage,
    build_deterministic_queries,
    build_gold_rows,
    evaluate_confirmation,
    prepare_blind_cases,
    read_train_source,
    select_balanced_sample,
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
    REPO_ROOT / "research/frc_rag/contractnli_rank_concurrence_confirmation.py"
)
RUNNER_PATH = REPO_ROOT / "scripts/run_contractnli_rank_concurrence_confirmation.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_contractnli_rank_concurrence_confirmation.py"


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
        "train_member_opened_by_v52": False,
        "dev_member_opened_by_v52": False,
        "test_member_opened_by_v52": False,
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
        "learned_threshold": None,
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
    )


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    existing = [
        path
        for path in (
            args.prepared,
            args.candidate_map,
            args.census,
            args.execution,
            args.queries,
            args.scored,
            args.coverage,
            args.result,
        )
        if path.exists()
    ]
    if existing:
        raise ValueError(
            "ContractNLI v52 preparation is frozen: "
            + ", ".join(str(path) for path in existing)
        )
    manifest = _source_manifest(args.source_archive, args.license_file)
    source = read_train_source(args.source_archive)
    selected, sampling = select_balanced_sample(source)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared, maps, structural = prepare_blind_cases(selected, tokenizer)
    write_jsonl(args.prepared, prepared)
    write_jsonl(args.candidate_map, maps)
    census = {
        "schema_version": "frc-contractnli-v52-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "source_manifest": {
            **manifest,
            "train_member_opened_by_v52": True,
            "train_open_purpose": "registered balanced sampling and blind preparation only",
        },
        "sampling": sampling,
        "structural_census": structural,
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "labels_document_ids_urls_offsets_or_gold_exported_to_blind_cache": False,
        "gold_joined_for_coverage_or_metrics": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
        "dev_or_test_member_opened_by_v52": False,
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


def register(args: argparse.Namespace) -> None:
    implementation = _validate_implementation(args)
    if args.execution.exists():
        raise FileExistsError("ContractNLI v52 execution already exists")
    if args.queries.exists() or args.scored.exists() or args.coverage.exists():
        raise ValueError("ContractNLI v52 outcome-bearing work started too early")
    census = json.loads(args.census.read_text(encoding="utf-8"))
    if sha256(args.prepared) != census["prepared_blind_sha256"] or sha256(
        args.candidate_map
    ) != census["candidate_map_sha256"]:
        raise ValueError("ContractNLI v52 blind artifacts changed")
    runtime = _runtime_environment(args)
    if args.device == "cuda" and runtime["cuda_available"] is not True:
        raise ValueError("ContractNLI v52 frozen CUDA runtime is unavailable")
    value = {
        "schema_version": "frc-contractnli-v52-execution-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "registered_at": args.registered_at,
        "hashes": {
            "implementation_registration_sha256": sha256(args.implementation),
            "source_archive_sha256": sha256(args.source_archive),
            "prepared_blind_sha256": sha256(args.prepared),
            "candidate_map_sha256": sha256(args.candidate_map),
            "blind_census_sha256": sha256(args.census),
        },
        "source_artifacts": {
            **_source_manifest(args.source_archive, args.license_file),
            "train_member_opened_for_registered_blind_preparation": True,
            "dev_or_test_member_opened_by_v52": False,
        },
        "runtime": runtime,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "gold_joined_for_coverage_or_metrics": False,
        "metrics_computed": False,
        "implementation_registration": implementation,
        "rank_gate_formula_change_after_registration_forbidden": True,
        "confirmation_reuse_forbidden": True,
    }
    _write_json(args.execution, value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def generate(args: argparse.Namespace) -> None:
    _validate_execution(args)
    if args.scored.exists() or args.coverage.exists():
        raise ValueError("ContractNLI v52 scoring or gold join started before queries")
    prepared = list(read_jsonl(args.prepared))
    rows = build_deterministic_queries(prepared)
    validate_query_cache(prepared, rows)
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
    if args.coverage.exists():
        raise ValueError("ContractNLI v52 gold was joined before scoring")
    prepared = list(read_jsonl(args.prepared))
    queries = list(read_jsonl(args.queries))
    validate_query_cache(prepared, queries)
    scorer = FrozenRankConcurrenceScorer(
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


def _train_gold(args: argparse.Namespace) -> list[dict[str, Any]]:
    source = read_train_source(args.source_archive)
    maps = list(read_jsonl(args.candidate_map))
    census = json.loads(args.census.read_text(encoding="utf-8"))
    structural = census["structural_census"]
    return build_gold_rows(
        source,
        maps,
        pool_quartile_boundaries=structural["candidate_pool_quartile_boundaries"],
        document_length_quartile_boundaries=structural[
            "document_length_quartile_boundaries"
        ],
    )


def evaluate(args: argparse.Namespace) -> None:
    execution = _validate_execution(args)
    if args.result.exists() or args.coverage.exists():
        raise ValueError("ContractNLI v52 confirmation evaluation is frozen")
    prepared = list(read_jsonl(args.prepared))
    maps = list(read_jsonl(args.candidate_map))
    queries = list(read_jsonl(args.queries))
    scored = list(read_jsonl(args.scored))
    expected = [str(row["id"]) for row in prepared]
    if len(expected) != 600:
        raise ValueError("ContractNLI v52 blind cache is incomplete")
    for name, rows in (("map", maps), ("query", queries), ("score", scored)):
        if [str(row["id"]) for row in rows] != expected:
            raise ValueError(
                f"ContractNLI v52 {name} cache must be complete before gold join"
            )
    query_summary = validate_query_cache(prepared, queries)
    gold = _train_gold(args)
    census = json.loads(args.census.read_text(encoding="utf-8"))
    coverage = build_candidate_coverage(gold, census["sampling"])
    coverage["gold_joined_after_complete_score_cache"] = True
    coverage["queries_sha256"] = sha256(args.queries)
    coverage["scored_sha256"] = sha256(args.scored)
    coverage["execution_registration_sha256"] = sha256(args.execution)
    _write_json(args.coverage, coverage)
    if coverage["all_checks_passed"] is not True:
        raise ValueError("ContractNLI v52 candidate coverage gate is closed")
    source_artifacts = {
        "execution_registration_sha256": sha256(args.execution),
        "execution_registration": execution,
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "blind_census_sha256": sha256(args.census),
        "candidate_coverage_sha256": sha256(args.coverage),
        "queries_sha256": sha256(args.queries),
        "scored_sha256": sha256(args.scored),
        "module_sha256": sha256(MODULE_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "test_sha256": sha256(TEST_PATH),
    }
    report, evidence = evaluate_confirmation(
        gold, scored, query_summary, source_artifacts
    )
    write_report(
        report,
        evidence,
        args.result,
        args.report,
        args.evidence,
    )
    first = {
        "json": sha256(args.result),
        "markdown": sha256(args.report),
        "evidence": sha256(args.evidence),
    }
    write_report(
        report,
        evidence,
        args.result,
        args.report,
        args.evidence,
    )
    second = {
        "json": sha256(args.result),
        "markdown": sha256(args.report),
        "evidence": sha256(args.evidence),
    }
    if first != second:
        raise AssertionError("ContractNLI v52 report is not deterministic")
    print(
        json.dumps(
            {
                "status": report["analysis"]["outcome"]["status"],
                "support_established": report["analysis"]["outcome"][
                    "support_established"
                ],
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
        description="Run the preregistered ContractNLI v52 rank confirmation."
    )
    parser.add_argument(
        "command",
        choices=("prepare", "register", "generate", "score", "evaluate", "runtime"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_rank_concurrence_confirmation_protocol_v52.json"
        ),
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_rank_concurrence_confirmation_implementation_v52.json"
        ),
    )
    parser.add_argument(
        "--execution",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_rank_concurrence_confirmation_execution_v52.json"
        ),
    )
    root = Path(".cache/benchmarks/contractnli")
    cache = root / "v52"
    parser.add_argument(
        "--source-archive", type=Path, default=root / "contract-nli.zip"
    )
    parser.add_argument("--license-file", type=Path, default=root / "LICENSE")
    parser.add_argument("--prepared", type=Path, default=cache / "prepared_blind.jsonl")
    parser.add_argument(
        "--candidate-map", type=Path, default=cache / "candidate_map.jsonl"
    )
    parser.add_argument("--census", type=Path, default=cache / "blind_census.json")
    parser.add_argument("--queries", type=Path, default=cache / "queries.jsonl")
    parser.add_argument("--scored", type=Path, default=cache / "scored.jsonl")
    parser.add_argument("--coverage", type=Path, default=cache / "coverage.json")
    output = Path("output/rag_evaluation/contractnli_rank_concurrence_confirmation")
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/contractnli_rank_concurrence_confirmation_result_v52.json"
        ),
    )
    parser.add_argument("--report", type=Path, default=output / "report.md")
    parser.add_argument(
        "--evidence", type=Path, default=output / "cases.jsonl.gz"
    )
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=4)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--registered-at", default="2026-08-02T20:30:00+08:00")
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
        "queries",
        "scored",
        "coverage",
        "result",
        "report",
        "evidence",
        "hf_home",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    commands = {
        "prepare": prepare,
        "register": register,
        "generate": generate,
        "score": score,
        "evaluate": evaluate,
        "runtime": runtime,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
