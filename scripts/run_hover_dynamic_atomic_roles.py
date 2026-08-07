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
    FrozenDynamicHoVerScorer,
    LocalQwenAtomicQueryGenerator,
    PROTOCOL_SHA256,
    build_confirmation_gold,
    build_partitions,
    evaluate_confirmation,
    evaluate_mechanism_gate,
    generate_queries_resumable,
    inspect_database,
    load_articles,
    load_frozen_tokenizer,
    partition_commitments,
    prepare_blind_partition,
    read_jsonl,
    requested_titles_for_partition,
    require_pilot_pass,
    score_cases_resumable,
    sha256,
    validate_query_cache,
    validate_registration,
    write_confirmation_report,
    write_jsonl,
    write_mechanism_report,
)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected JSON object array: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_sources(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_registration(
        args.protocol,
        args.v40_closure,
        args.dataset,
        args.retrieval,
        args.database,
    )
    return _read_array(args.dataset), _read_array(args.retrieval)


def _validate_alignment(
    dataset_rows: list[dict[str, Any]], retrieval_rows: list[dict[str, Any]]
) -> None:
    dataset_ids = [str(row.get("uid", "")).strip() for row in dataset_rows]
    retrieval_ids = [str(row.get("id", "")).strip() for row in retrieval_rows]
    if (
        any(not value for value in dataset_ids)
        or any(not value for value in retrieval_ids)
        or len(dataset_ids) != len(set(dataset_ids))
        or len(retrieval_ids) != len(set(retrieval_ids))
        or set(dataset_ids) != set(retrieval_ids)
    ):
        raise ValueError("HoVer train dataset/TF-IDF ID alignment failed")


def partition(args: argparse.Namespace) -> None:
    dataset_rows, retrieval_rows = _load_sources(args)
    _validate_alignment(dataset_rows, retrieval_rows)
    partitions = build_partitions(dataset_rows)
    result = {
        "schema_version": "frc-hover-v41-partition-commitments-v1",
        "experiment_id": "FRC-HOVER-DYNAMIC-ATOMIC-ROLES-V41",
        "protocol_sha256": PROTOCOL_SHA256,
        "dataset_rows": len(dataset_rows),
        "tfidf_rows": len(retrieval_rows),
        "commitments": partition_commitments(partitions),
        "content_or_gold_statistics_computed": False,
    }
    _write_json(args.partition_output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def _prepare(
    args: argparse.Namespace,
    *,
    partition_name: str,
    prepared_path: Path,
    census_path: Path,
) -> dict[str, Any]:
    dataset_rows, retrieval_rows = _load_sources(args)
    _validate_alignment(dataset_rows, retrieval_rows)
    partitions = build_partitions(dataset_rows)
    ordered_ids = partitions[partition_name]
    titles = requested_titles_for_partition(retrieval_rows, ordered_ids)
    database_schema = inspect_database(args.database)
    if database_schema["quick_check"] != "ok":
        raise ValueError("HoVer database quick_check failed")
    articles = load_articles(args.database, titles, schema=database_schema)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared, summary = prepare_blind_partition(
        dataset_rows,
        retrieval_rows,
        ordered_ids,
        articles,
        tokenizer,
    )
    write_jsonl(prepared_path, prepared)
    census = {
        "schema_version": "frc-hover-v41-blind-census-v1",
        "experiment_id": "FRC-HOVER-DYNAMIC-ATOMIC-ROLES-V41",
        "partition": partition_name,
        "protocol_sha256": PROTOCOL_SHA256,
        "partition_commitments": partition_commitments(partitions),
        "database": {
            "schema": database_schema,
            "requested_unique_titles": len(titles),
            "resolved_unique_titles": len(articles),
        },
        "structural_census": summary,
        "prepared_blind_sha256": sha256(prepared_path),
        "source_hashes": {
            "dataset": sha256(args.dataset),
            "official_tfidf_candidates": sha256(args.retrieval),
            "official_wikipedia_database": sha256(args.database),
        },
        "content_fields_used": ["uid", "claim", "official_tfidf_top20", "article_text"],
        "gold_fields_used": [],
    }
    _write_json(census_path, census)
    return census


def prepare_pilot(args: argparse.Namespace) -> None:
    census = _prepare(
        args,
        partition_name="pilot",
        prepared_path=args.pilot_prepared,
        census_path=args.pilot_census,
    )
    print(json.dumps(census, ensure_ascii=False, sort_keys=True))


def generate_pilot(args: argparse.Namespace) -> None:
    _load_sources(args)
    prepared = list(read_jsonl(args.pilot_prepared))
    generator = LocalQwenAtomicQueryGenerator(
        model_path=args.generator_model,
        batch_size=args.generator_batch_size,
        max_input_tokens=512,
        max_new_tokens=160,
    )
    rows = generate_queries_resumable(
        prepared,
        generator,
        args.pilot_queries,
        case_batch_size=args.generator_batch_size,
    )
    summary = validate_query_cache(prepared, rows)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def score_pilot(args: argparse.Namespace) -> None:
    _load_sources(args)
    prepared = list(read_jsonl(args.pilot_prepared))
    queries = list(read_jsonl(args.pilot_queries))
    validate_query_cache(prepared, queries)
    scorer = FrozenDynamicHoVerScorer(
        query_rows=queries,
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        use_fp16=args.device.startswith("cuda"),
    )
    scored = score_cases_resumable(
        prepared,
        scorer,
        args.pilot_scored,
        case_batch_size=args.case_batch_size,
    )
    print(
        json.dumps(
            {"rows": len(scored), "scored_sha256": sha256(args.pilot_scored)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def gate_pilot(args: argparse.Namespace) -> None:
    _load_sources(args)
    prepared = list(read_jsonl(args.pilot_prepared))
    queries = list(read_jsonl(args.pilot_queries))
    scored = list(read_jsonl(args.pilot_scored))
    if [row.get("id") for row in scored] != [row.get("id") for row in prepared]:
        raise ValueError("v41 pilot score coverage/order mismatch")
    query_summary = validate_query_cache(prepared, queries)
    report = evaluate_mechanism_gate(scored, query_summary)
    report["metadata"]["source_artifacts"] = {
        "pilot_prepared_sha256": sha256(args.pilot_prepared),
        "pilot_queries_sha256": sha256(args.pilot_queries),
        "pilot_scored_sha256": sha256(args.pilot_scored),
        "implementation_sha256": sha256(
            REPO_ROOT / "research/frc_rag/hover_dynamic_atomic_roles.py"
        ),
    }
    paths = write_mechanism_report(report, args.output_dir)
    first = {name: sha256(path) for name, path in paths.items()}
    paths = write_mechanism_report(report, args.output_dir)
    second = {name: sha256(path) for name, path in paths.items()}
    if first != second:
        raise AssertionError("v41 pilot report is not byte deterministic")
    print(
        json.dumps(
            {
                "status": report["outcome"]["status"],
                "checks": report["checks"],
                "mechanism": report["mechanism"],
                "output_hashes": second,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def prepare_confirmation(args: argparse.Namespace) -> None:
    require_pilot_pass(args.pilot_report)
    _validate_confirmation_open(args)
    census = _prepare(
        args,
        partition_name="confirmation",
        prepared_path=args.confirmation_prepared,
        census_path=args.confirmation_census,
    )
    census["pilot_report_sha256"] = sha256(args.pilot_report)
    census["confirmation_generation_or_scoring_started"] = False
    _write_json(args.confirmation_census, census)
    print(json.dumps(census, ensure_ascii=False, sort_keys=True))


def _validate_confirmation_open(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirmation_open.is_file():
        raise FileNotFoundError(
            "v41 confirmation-open registration is required before preparation"
        )
    registration = json.loads(
        args.confirmation_open.read_text(encoding="utf-8")
    )
    if registration.get("experiment_id") != (
        "FRC-HOVER-DYNAMIC-ATOMIC-ROLES-V41"
    ):
        raise ValueError("v41 confirmation-open experiment mismatch")
    if registration.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("v41 confirmation-open protocol mismatch")
    expected = {
        "pilot_report": sha256(args.pilot_report),
        "pilot_prepared": sha256(args.pilot_prepared),
        "pilot_queries": sha256(args.pilot_queries),
        "pilot_scored": sha256(args.pilot_scored),
        "implementation": sha256(
            REPO_ROOT / "research/frc_rag/hover_dynamic_atomic_roles.py"
        ),
        "runner": sha256(
            REPO_ROOT / "scripts/run_hover_dynamic_atomic_roles.py"
        ),
    }
    if registration.get("artifact_sha256") != expected:
        raise ValueError("v41 confirmation-open artifacts changed")
    if registration.get("confirmation_preparation_authorized") is not True:
        raise RuntimeError("v41 confirmation blind preparation is not authorized")
    boundary = registration.get("registration_boundary", {})
    if boundary.get("confirmation_content_accessed_before_registration") is not False:
        raise ValueError("v41 confirmation-open access boundary is invalid")
    return registration


def _validate_execution(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execution.is_file():
        raise FileNotFoundError("v41 confirmation execution registration is missing")
    execution = json.loads(args.execution.read_text(encoding="utf-8"))
    if execution.get("experiment_id") != "FRC-HOVER-DYNAMIC-ATOMIC-ROLES-V41":
        raise ValueError("v41 execution experiment mismatch")
    if execution.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("v41 execution protocol mismatch")
    if execution.get("confirmation_open_sha256") != sha256(
        args.confirmation_open
    ):
        raise ValueError("v41 execution references another confirmation-open record")
    _validate_confirmation_open(args)
    expected = {
        "pilot_report": sha256(args.pilot_report),
        "pilot_prepared": sha256(args.pilot_prepared),
        "pilot_queries": sha256(args.pilot_queries),
        "pilot_scored": sha256(args.pilot_scored),
        "confirmation_census": sha256(args.confirmation_census),
        "confirmation_prepared": sha256(args.confirmation_prepared),
        "implementation": sha256(
            REPO_ROOT / "research/frc_rag/hover_dynamic_atomic_roles.py"
        ),
        "runner": sha256(REPO_ROOT / "scripts/run_hover_dynamic_atomic_roles.py"),
    }
    if execution.get("artifact_sha256") != expected:
        raise ValueError("v41 execution artifact fingerprints changed")
    require_pilot_pass(args.pilot_report)
    return execution


def generate_confirmation(args: argparse.Namespace) -> None:
    _load_sources(args)
    _validate_execution(args)
    prepared = list(read_jsonl(args.confirmation_prepared))
    generator = LocalQwenAtomicQueryGenerator(
        model_path=args.generator_model,
        batch_size=args.generator_batch_size,
        max_input_tokens=512,
        max_new_tokens=160,
    )
    rows = generate_queries_resumable(
        prepared,
        generator,
        args.confirmation_queries,
        case_batch_size=args.generator_batch_size,
    )
    print(
        json.dumps(
            validate_query_cache(prepared, rows),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def score_confirmation(args: argparse.Namespace) -> None:
    _load_sources(args)
    _validate_execution(args)
    prepared = list(read_jsonl(args.confirmation_prepared))
    queries = list(read_jsonl(args.confirmation_queries))
    validate_query_cache(prepared, queries)
    scorer = FrozenDynamicHoVerScorer(
        query_rows=queries,
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        use_fp16=args.device.startswith("cuda"),
    )
    scored = score_cases_resumable(
        prepared,
        scorer,
        args.confirmation_scored,
        case_batch_size=args.case_batch_size,
    )
    print(
        json.dumps(
            {"rows": len(scored), "scored_sha256": sha256(args.confirmation_scored)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    dataset_rows, retrieval_rows = _load_sources(args)
    _validate_execution(args)
    prepared = list(read_jsonl(args.confirmation_prepared))
    queries = list(read_jsonl(args.confirmation_queries))
    scored = list(read_jsonl(args.confirmation_scored))
    validate_query_cache(prepared, queries)
    if [row.get("id") for row in scored] != [row.get("id") for row in prepared]:
        raise ValueError("v41 confirmation score coverage/order mismatch")
    partitions = build_partitions(dataset_rows)
    titles = requested_titles_for_partition(
        retrieval_rows, partitions["confirmation"]
    )
    schema = inspect_database(args.database)
    articles = load_articles(args.database, titles, schema=schema)
    gold = build_confirmation_gold(
        dataset_rows,
        retrieval_rows,
        partitions["confirmation"],
        prepared,
        set(articles),
    )
    report, evidence = evaluate_confirmation(
        gold, scored, resamples=args.resamples
    )
    report["metadata"]["source_artifacts"] = {
        "execution_sha256": sha256(args.execution),
        "confirmation_prepared_sha256": sha256(args.confirmation_prepared),
        "confirmation_queries_sha256": sha256(args.confirmation_queries),
        "confirmation_scored_sha256": sha256(args.confirmation_scored),
        "implementation_sha256": sha256(
            REPO_ROOT / "research/frc_rag/hover_dynamic_atomic_roles.py"
        ),
    }
    paths = write_confirmation_report(report, evidence, args.output_dir)
    first = {name: sha256(path) for name, path in paths.items()}
    paths = write_confirmation_report(report, evidence, args.output_dir)
    second = {name: sha256(path) for name, path in paths.items()}
    if first != second:
        raise AssertionError("v41 confirmation report is not byte deterministic")
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
        "module_sha256": sha256(
            REPO_ROOT / "research/frc_rag/hover_dynamic_atomic_roles.py"
        ),
        "runner_sha256": sha256(REPO_ROOT / "scripts/run_hover_dynamic_atomic_roles.py"),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen v41 HoVer dynamic atomic-role experiment."
    )
    parser.add_argument(
        "command",
        choices=(
            "partition",
            "prepare-pilot",
            "generate-pilot",
            "score-pilot",
            "gate-pilot",
            "prepare-confirmation",
            "generate-confirmation",
            "score-confirmation",
            "evaluate",
            "runtime",
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/hover_dynamic_atomic_roles_protocol_v41.json"
        ),
    )
    parser.add_argument(
        "--v40-closure",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/hover_dynamic_atomic_roles_v40_closure.json"
        ),
    )
    parser.add_argument(
        "--execution",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/hover_dynamic_atomic_roles_execution_v41.json"
        ),
    )
    parser.add_argument(
        "--confirmation-open",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "hover_dynamic_atomic_roles_confirmation_open_v41_v2.json"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(".cache/benchmarks/hover/hover_train_release_v1.1.json"),
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=Path(
            ".cache/benchmarks/hover/train_tfidf_doc_retrieval_results.json"
        ),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(".cache/benchmarks/hover/wiki_wo_links.db"),
    )
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument(
        "--generator-model",
        type=Path,
        default=Path(
            r"D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4"
        ),
    )
    cache = Path(".cache/benchmarks/hover/v41")
    parser.add_argument("--partition-output", type=Path, default=cache / "partitions.json")
    parser.add_argument("--pilot-prepared", type=Path, default=cache / "pilot_prepared.jsonl")
    parser.add_argument("--pilot-census", type=Path, default=cache / "pilot_census.json")
    parser.add_argument("--pilot-queries", type=Path, default=cache / "pilot_queries.jsonl")
    parser.add_argument("--pilot-scored", type=Path, default=cache / "pilot_scored.jsonl")
    parser.add_argument(
        "--confirmation-prepared", type=Path, default=cache / "confirmation_prepared.jsonl"
    )
    parser.add_argument(
        "--confirmation-census", type=Path, default=cache / "confirmation_census.json"
    )
    parser.add_argument(
        "--confirmation-queries", type=Path, default=cache / "confirmation_queries.jsonl"
    )
    parser.add_argument(
        "--confirmation-scored", type=Path, default=cache / "confirmation_scored.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/hover_dynamic_atomic_roles"),
    )
    parser.add_argument(
        "--pilot-report",
        type=Path,
        default=Path(
            "output/rag_evaluation/hover_dynamic_atomic_roles/"
            "hover_dynamic_atomic_roles_pilot.json"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=8)
    parser.add_argument("--generator-batch-size", type=int, default=8)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, _resolve(value))
    commands = {
        "partition": partition,
        "prepare-pilot": prepare_pilot,
        "generate-pilot": generate_pilot,
        "score-pilot": score_pilot,
        "gate-pilot": gate_pilot,
        "prepare-confirmation": prepare_confirmation,
        "generate-confirmation": generate_confirmation,
        "score-confirmation": score_confirmation,
        "evaluate": evaluate,
        "runtime": runtime,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
