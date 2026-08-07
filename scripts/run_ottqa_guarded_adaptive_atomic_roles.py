from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.hover_dynamic_atomic_roles import (  # noqa: E402
    LocalQwenAtomicQueryGenerator,
    generate_queries_resumable,
    validate_query_cache,
)
from research.frc_rag.ottqa_guarded_adaptive_atomic_roles import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXPERIMENT_ID,
    MAXIMUM_POOL_SIZE,
    FrozenOttqaScorer,
    archive_directory_git_manifest,
    archive_git_blob_sha1,
    build_candidate_coverage,
    build_gold_rows,
    evaluate_ottqa,
    prepare_blind_cases,
    read_archive_json,
    validate_execution_registration,
    validate_implementation_registration,
    write_report,
)
from research.frc_rag.rgb_cost_aware_frc import (  # noqa: E402
    load_frozen_tokenizer,
    read_jsonl as read_cache_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)


MODULE_PATH = REPO_ROOT / "research/frc_rag/ottqa_guarded_adaptive_atomic_roles.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_ottqa_guarded_adaptive_atomic_roles.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_ottqa_guarded_adaptive_atomic_roles.py"
INHERITED_MODULE_PATH = REPO_ROOT / "research/frc_rag/hover_dynamic_atomic_roles.py"

EXPECTED_LINKED_BLOB = "3454a0dea9fd3a284ec9cd376fc21bcb63ec9ca1"
EXPECTED_LICENSE_BLOB = "6cce7fb48f73fe3e9cf469ebefa2d29293c6f056"
EXPECTED_TABLE_MANIFEST = {
    "files": 8891,
    "bytes": 65523616,
    "manifest_sha256": "bb8bfda9233ad197cf2c21c8a2657e1428065d268d79a939e15602f46bd42e69",
}
EXPECTED_PASSAGE_MANIFEST = {
    "files": 8891,
    "bytes": 255477273,
    "manifest_sha256": "8f13044a1fe5ec93f0daf60c6bab13d5e8edfba3d5c9ca29530b4ad88eeafe96",
}


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _runtime_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "fp16": args.device.startswith("cuda"),
        "embedding_batch_size": args.embedding_batch_size,
        "reranker_batch_size": args.reranker_batch_size,
        "case_batch_size": args.case_batch_size,
        "generator_batch_size": args.generator_batch_size,
        "generator_max_input_tokens": 512,
        "generator_max_new_tokens": 160,
        "chunk_window_tokens": 192,
        "chunk_stride_tokens": 160,
        "lexical_pool_size": 96,
        "maximum_pool_size": MAXIMUM_POOL_SIZE,
        "bootstrap_resamples": args.resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "offline_model_loading": True,
    }


def _implementation_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "protocol_path": args.protocol,
        "protocol_erratum_path": args.protocol_erratum,
        "protocol_erratum2_path": args.protocol_erratum2,
        "protocol_erratum3_path": args.protocol_erratum3,
        "module_path": MODULE_PATH,
        "runner_path": RUNNER_PATH,
        "test_path": TEST_PATH,
        "inherited_module_path": INHERITED_MODULE_PATH,
        "erratum_path": args.implementation_erratum,
        "erratum2_path": args.implementation_erratum2,
    }


def _validate_implementation(args: argparse.Namespace) -> dict[str, Any]:
    return validate_implementation_registration(
        args.implementation,
        **_implementation_kwargs(args),
    )


def _validate_execution(args: argparse.Namespace) -> dict[str, Any]:
    _validate_implementation(args)
    return validate_execution_registration(
        args.execution,
        implementation_path=args.implementation,
        implementation_erratum_path=args.implementation_erratum,
        implementation_erratum2_path=args.implementation_erratum2,
        protocol_path=args.protocol,
        protocol_erratum_path=args.protocol_erratum,
        protocol_erratum2_path=args.protocol_erratum2,
        protocol_erratum3_path=args.protocol_erratum3,
        source_archive_path=args.source_archive,
        prepared_path=args.prepared,
        candidate_map_path=args.candidate_map,
        census_path=args.census,
        coverage_path=args.coverage,
        runtime_parameters=_runtime_parameters(args),
    )


def _load_linked(path: Path) -> list[dict[str, Any]]:
    value = read_archive_json(path, "preprocessed_data/dev_linked.json")
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("OTT-QA linked development file must be a JSON object list")
    return [dict(row) for row in value]


def prepare(args: argparse.Namespace) -> None:
    implementation = _validate_implementation(args)
    if not args.source_archive.is_file():
        raise FileNotFoundError(
            f"OTT-QA source archive is missing: {args.source_archive}"
        )

    linked_blob = archive_git_blob_sha1(
        args.source_archive, "preprocessed_data/dev_linked.json"
    )
    license_blob = archive_git_blob_sha1(args.source_archive, "LICENSE")
    table_manifest = archive_directory_git_manifest(
        args.source_archive, "data/traindev_tables_tok"
    )
    passage_manifest = archive_directory_git_manifest(
        args.source_archive, "data/traindev_request_tok"
    )
    if linked_blob != EXPECTED_LINKED_BLOB:
        raise ValueError("OTT-QA linked development Git blob mismatch")
    if license_blob != EXPECTED_LICENSE_BLOB:
        raise ValueError("OTT-QA license Git blob mismatch")
    if table_manifest != EXPECTED_TABLE_MANIFEST:
        raise ValueError("OTT-QA table-directory manifest mismatch")
    if passage_manifest != EXPECTED_PASSAGE_MANIFEST:
        raise ValueError("OTT-QA passage-directory manifest mismatch")

    linked_rows = _load_linked(args.source_archive)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared_rows, candidate_maps, structural = prepare_blind_cases(
        linked_rows,
        None,
        None,
        tokenizer,
        source_archive=args.source_archive,
    )
    write_jsonl(args.prepared, prepared_rows)
    write_jsonl(args.candidate_map, candidate_maps)
    census = {
        "schema_version": "frc-ottqa-v44-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "implementation_registration_sha256": sha256(args.implementation),
        "implementation_erratum_sha256": sha256(args.implementation_erratum),
        "implementation_erratum2_sha256": sha256(args.implementation_erratum2),
        "protocol_sha256": sha256(args.protocol),
        "protocol_erratum_sha256": sha256(args.protocol_erratum),
        "protocol_erratum2_sha256": sha256(args.protocol_erratum2),
        "protocol_erratum3_sha256": sha256(args.protocol_erratum3),
        "implementation_registration": implementation,
        "source_artifacts": {
            "source_archive_sha256": sha256(args.source_archive),
            "source_archive_bytes": args.source_archive.stat().st_size,
            "linked_development_git_blob": linked_blob,
            "linked_development_rows": len(linked_rows),
            "license_git_blob": license_blob,
            "table_manifest": table_manifest,
            "passage_manifest": passage_manifest,
        },
        "structural_census": structural,
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "content_fields_used": [
            "question_id, question, and table_id",
            "answer-node type for sealed deterministic sampling only",
            "official tf-idf, string-overlap, and links nodes for gold-free anchors",
            "official oracle table and linked passages for candidate construction",
        ],
        "answer_or_gold_identifiers_exported_to_blind_cache": False,
        "source_mode_exported_to_generator_or_scorer": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }
    _write_json(args.census, census)
    print(json.dumps(census, ensure_ascii=False, sort_keys=True))


def coverage(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    if args.queries.exists() or args.scored.exists():
        raise ValueError(
            "OTT-QA candidate coverage must be registered before query or score caches"
        )
    linked_rows = _load_linked(args.source_archive)
    candidate_maps = list(read_cache_jsonl(args.candidate_map))
    census = json.loads(args.census.read_text(encoding="utf-8"))
    gold_rows = build_gold_rows(linked_rows, candidate_maps)
    report = build_candidate_coverage(gold_rows, census["structural_census"])
    report["hashes"] = {
        "implementation_registration_sha256": sha256(args.implementation),
        "implementation_erratum_sha256": sha256(args.implementation_erratum),
        "implementation_erratum2_sha256": sha256(args.implementation_erratum2),
        "protocol_sha256": sha256(args.protocol),
        "protocol_erratum_sha256": sha256(args.protocol_erratum),
        "protocol_erratum2_sha256": sha256(args.protocol_erratum2),
        "protocol_erratum3_sha256": sha256(args.protocol_erratum3),
        "source_archive_sha256": sha256(args.source_archive),
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "blind_census_sha256": sha256(args.census),
    }
    _write_json(args.coverage, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def generate(args: argparse.Namespace) -> None:
    _validate_execution(args)
    prepared_rows = list(read_cache_jsonl(args.prepared))
    generator = LocalQwenAtomicQueryGenerator(
        model_path=args.generator_model,
        batch_size=args.generator_batch_size,
        max_input_tokens=512,
        max_new_tokens=160,
    )
    rows = generate_queries_resumable(
        prepared_rows,
        generator,
        args.queries,
        case_batch_size=args.generator_batch_size,
    )
    summary = validate_query_cache(prepared_rows, rows)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def score(args: argparse.Namespace) -> None:
    _validate_execution(args)
    prepared_rows = list(read_cache_jsonl(args.prepared))
    query_rows = list(read_cache_jsonl(args.queries))
    validate_query_cache(prepared_rows, query_rows)
    scorer = FrozenOttqaScorer(
        query_rows=query_rows,
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        use_fp16=args.device.startswith("cuda"),
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
    prepared_rows = list(read_cache_jsonl(args.prepared))
    candidate_maps = list(read_cache_jsonl(args.candidate_map))
    query_rows = list(read_cache_jsonl(args.queries))
    scored_rows = list(read_cache_jsonl(args.scored))
    expected_ids = [str(row.get("id")) for row in prepared_rows]
    for name, rows in (
        ("candidate map", candidate_maps),
        ("query", query_rows),
        ("score", scored_rows),
    ):
        if [str(row.get("id")) for row in rows] != expected_ids:
            raise ValueError(f"OTT-QA {name} cache is incomplete or out of order")
    query_summary = validate_query_cache(prepared_rows, query_rows)

    # Gold is read only after all complete ordered blind caches pass validation.
    linked_rows = _load_linked(args.source_archive)
    gold_rows = build_gold_rows(linked_rows, candidate_maps)
    report, evidence = evaluate_ottqa(
        gold_rows,
        scored_rows,
        query_summary,
        resamples=args.resamples,
    )
    report["metadata"]["source_artifacts"] = {
        "execution_registration_sha256": sha256(args.execution),
        "execution_registration": execution,
        "candidate_coverage_sha256": sha256(args.coverage),
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "queries_sha256": sha256(args.queries),
        "scored_sha256": sha256(args.scored),
        "module_sha256": sha256(MODULE_PATH),
        "inherited_v41_module_sha256": sha256(INHERITED_MODULE_PATH),
    }
    paths = write_report(report, evidence, args.output_directory)
    first = {name: sha256(path) for name, path in paths.items()}
    paths = write_report(report, evidence, args.output_directory)
    second = {name: sha256(path) for name, path in paths.items()}
    if first != second:
        raise AssertionError("OTT-QA v44 report is not byte deterministic")
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
    _validate_implementation(args)
    import numpy
    import torch
    import transformers

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_mib": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**2))
            if torch.cuda.is_available()
            else None
        ),
        "runtime_parameters": _runtime_parameters(args),
        "module_sha256": sha256(MODULE_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "test_sha256": sha256(TEST_PATH),
        "implementation_registration_sha256": sha256(args.implementation),
        "implementation_erratum_sha256": sha256(args.implementation_erratum),
        "implementation_erratum2_sha256": sha256(args.implementation_erratum2),
        "protocol_sha256": sha256(args.protocol),
        "protocol_erratum_sha256": sha256(args.protocol_erratum),
        "protocol_erratum2_sha256": sha256(args.protocol_erratum2),
        "protocol_erratum3_sha256": sha256(args.protocol_erratum3),
        "inherited_v41_module_sha256": sha256(INHERITED_MODULE_PATH),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered OTT-QA v44 bounded-pool experiment."
    )
    parser.add_argument(
        "command",
        choices=("prepare", "coverage", "generate", "score", "evaluate", "runtime"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/ottqa_guarded_adaptive_atomic_roles_protocol_v44.json"
        ),
    )
    parser.add_argument(
        "--protocol-erratum",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "ottqa_guarded_adaptive_atomic_roles_protocol_erratum_v44.json"
        ),
    )
    parser.add_argument(
        "--protocol-erratum2",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "ottqa_guarded_adaptive_atomic_roles_protocol_erratum2_v44.json"
        ),
    )
    parser.add_argument(
        "--protocol-erratum3",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "ottqa_guarded_adaptive_atomic_roles_protocol_erratum3_v44.json"
        ),
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/ottqa_guarded_adaptive_atomic_roles_implementation_v44.json"
        ),
    )
    parser.add_argument(
        "--implementation-erratum",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "ottqa_guarded_adaptive_atomic_roles_implementation_erratum_v44.json"
        ),
    )
    parser.add_argument(
        "--implementation-erratum2",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "ottqa_guarded_adaptive_atomic_roles_implementation_erratum2_v44.json"
        ),
    )
    parser.add_argument(
        "--execution",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/ottqa_guarded_adaptive_atomic_roles_execution_v44.json"
        ),
    )
    root = Path(".cache/benchmarks/ottqa")
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=root / "ottqa_b289bcca_registered_sources.tar",
    )
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument(
        "--generator-model",
        type=Path,
        default=Path(
            r"D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4"
        ),
    )
    cache = root / "v44"
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
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/rag_evaluation/ottqa_guarded_adaptive_atomic_roles"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=4)
    parser.add_argument("--generator-batch-size", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, _resolve(value))
    commands = {
        "prepare": prepare,
        "coverage": coverage,
        "generate": generate,
        "score": score,
        "evaluate": evaluate,
        "runtime": runtime,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
