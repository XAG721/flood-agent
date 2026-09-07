from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tarfile
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
from research.frc_rag.rgb_cost_aware_frc import (  # noqa: E402
    load_frozen_tokenizer,
    read_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)
from research.frc_rag.tatqa_consensus_guarded_atomic_roles import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXPERIMENT_ID,
    OFFICIAL_REPOSITORY_REVISION,
    FrozenTatqaScorer,
    build_candidate_coverage,
    build_gold_rows,
    evaluate_tatqa,
    prepare_blind_cases,
    read_source_contexts,
    validate_execution_registration,
    validate_implementation_registration,
    write_report,
)


MODULE_PATH = REPO_ROOT / "research/frc_rag/tatqa_consensus_guarded_atomic_roles.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_tatqa_consensus_guarded_atomic_roles.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_tatqa_consensus_guarded_atomic_roles.py"
ARCHIVE_ROOT = f"TAT-QA-{OFFICIAL_REPOSITORY_REVISION}"
ARCHIVE_MEMBERS = {
    "development": "dataset_raw/tatqa_dataset_dev.json",
    "readme": "README.md",
    "license": "LICENSE",
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


def _member_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"TAT-QA source archive is missing: {path}")
    expected_paths = {
        name: f"{ARCHIVE_ROOT}/{relative}" for name, relative in ARCHIVE_MEMBERS.items()
    }
    artifacts: dict[str, Any] = {}
    license_raw = b""
    with tarfile.open(path, mode="r:gz") as archive:
        roots = {
            member.name.split("/", 1)[0]
            for member in archive.getmembers()
            if member.name
        }
        if roots != {ARCHIVE_ROOT}:
            raise ValueError("TAT-QA archive root does not match the frozen revision")
        members = {member.name: member for member in archive.getmembers()}
        for name, member_path in expected_paths.items():
            member = members.get(member_path)
            if member is None or not member.isfile():
                raise ValueError(f"TAT-QA archive member is missing: {member_path}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"TAT-QA archive member is unreadable: {member_path}")
            raw = handle.read()
            artifacts[name] = {
                "path": ARCHIVE_MEMBERS[name],
                "bytes": len(raw),
                "sha256": _member_sha256(raw),
            }
            if name == "license":
                license_raw = raw
    license_text = license_raw.decode("utf-8", errors="replace").lower()
    recognized = next(
        (
            name
            for name, marker in (
                ("MIT", "mit license"),
                ("Apache-2.0", "apache license"),
                ("CC", "creative commons"),
            )
            if marker in license_text
        ),
        None,
    )
    if recognized is None:
        raise ValueError("TAT-QA frozen LICENSE is not a recognized permissive license")
    return {
        "repository_revision": OFFICIAL_REPOSITORY_REVISION,
        "archive_root": ARCHIVE_ROOT,
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256(path),
        "license_identifier_detected": recognized,
        "license_checked_before_data_member_opened": True,
        "members": artifacts,
    }


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
        "bootstrap_resamples": args.resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "offline_model_loading": True,
    }


def _runtime_environment(args: argparse.Namespace) -> dict[str, Any]:
    import numpy
    import torch
    import transformers

    return {
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
        "parameters": _runtime_parameters(args),
    }


def _validate_implementation(args: argparse.Namespace) -> dict[str, Any]:
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
        source_archive_path=args.source_archive,
        prepared_path=args.prepared,
        candidate_map_path=args.candidate_map,
        census_path=args.census,
        coverage_path=args.coverage,
    )


def _assert_before_neural_work(args: argparse.Namespace) -> None:
    existing = [path for path in (args.queries, args.scored) if path.exists()]
    if existing:
        raise ValueError(
            "TAT-QA preparation cannot run after neural caches exist: "
            + ", ".join(str(path) for path in existing)
        )


def prepare(args: argparse.Namespace) -> None:
    implementation = _validate_implementation(args)
    _assert_before_neural_work(args)
    if args.execution.exists():
        raise ValueError("TAT-QA preparation is frozen after execution registration")
    source_manifest = _source_manifest(args.source_archive)
    source_contexts = read_source_contexts(args.source_archive)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared_rows, candidate_maps, structural = prepare_blind_cases(
        source_contexts, tokenizer
    )
    write_jsonl(args.prepared, prepared_rows)
    write_jsonl(args.candidate_map, candidate_maps)
    census = {
        "schema_version": "frc-tatqa-v46-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "implementation_registration_sha256": sha256(args.implementation),
        "implementation_registration": implementation,
        "protocol_sha256": sha256(args.protocol),
        "source_artifacts": source_manifest,
        "structural_census": structural,
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "raw_question_uids_exported_to_blind_or_candidate_map": False,
        "answer_derivation_mapping_or_gold_exported_to_blind_cache": False,
        "source_mode_or_answer_type_exported_to_generator_or_scorer": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }
    _write_json(args.census, census)
    print(
        json.dumps(
            {
                "schema_version": census["schema_version"],
                "source_artifacts": source_manifest,
                "structural_census": structural,
                "prepared_blind_sha256": census["prepared_blind_sha256"],
                "candidate_map_sha256": census["candidate_map_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def coverage(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _assert_before_neural_work(args)
    if args.execution.exists():
        raise ValueError("TAT-QA coverage is frozen after execution registration")
    source_contexts = read_source_contexts(args.source_archive)
    candidate_maps = list(read_jsonl(args.candidate_map))
    census = json.loads(args.census.read_text(encoding="utf-8"))
    if sha256(args.prepared) != census.get("prepared_blind_sha256"):
        raise ValueError("TAT-QA prepared cache changed after blind census")
    if sha256(args.candidate_map) != census.get("candidate_map_sha256"):
        raise ValueError("TAT-QA candidate map changed after blind census")
    gold_rows = build_gold_rows(source_contexts, candidate_maps)
    report = build_candidate_coverage(
        gold_rows, dict(census["structural_census"]["sampling"])
    )
    report["hashes"] = {
        "implementation_registration_sha256": sha256(args.implementation),
        "source_archive_sha256": sha256(args.source_archive),
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "blind_census_sha256": sha256(args.census),
    }
    _write_json(args.coverage, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def register_execution(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _assert_before_neural_work(args)
    if args.execution.exists():
        value = _validate_execution(args)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return
    coverage_report = json.loads(args.coverage.read_text(encoding="utf-8"))
    if coverage_report.get("minimum_cases_and_ceiling_checks_passed") is not True:
        raise ValueError("TAT-QA candidate coverage gate is not open")
    value = {
        "schema_version": "frc-tatqa-consensus-guarded-atomic-roles-execution-v46",
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
        "source_artifacts": _source_manifest(args.source_archive),
        "runtime": _runtime_environment(args),
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
        "method_or_threshold_change_after_registration_forbidden": True,
        "negative_null_or_inconclusive_result_must_be_published": True,
    }
    _write_json(args.execution, value)
    _validate_execution(args)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def generate(args: argparse.Namespace) -> None:
    _validate_execution(args)
    prepared_rows = list(read_jsonl(args.prepared))
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
    prepared_rows = list(read_jsonl(args.prepared))
    query_rows = list(read_jsonl(args.queries))
    validate_query_cache(prepared_rows, query_rows)
    scorer = FrozenTatqaScorer(
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
    prepared_rows = list(read_jsonl(args.prepared))
    candidate_maps = list(read_jsonl(args.candidate_map))
    query_rows = list(read_jsonl(args.queries))
    scored_rows = list(read_jsonl(args.scored))
    expected_ids = [str(row.get("id")) for row in prepared_rows]
    for name, rows in (
        ("candidate map", candidate_maps),
        ("query", query_rows),
        ("score", scored_rows),
    ):
        if [str(row.get("id")) for row in rows] != expected_ids:
            raise ValueError(f"TAT-QA {name} cache is incomplete or out of order")
    query_summary = validate_query_cache(prepared_rows, query_rows)

    source_contexts = read_source_contexts(args.source_archive)
    gold_rows = build_gold_rows(source_contexts, candidate_maps)
    source_artifacts = {
        "execution_registration_sha256": sha256(args.execution),
        "execution_registration": execution,
        "candidate_coverage_sha256": sha256(args.coverage),
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "queries_sha256": sha256(args.queries),
        "scored_sha256": sha256(args.scored),
        "module_sha256": sha256(MODULE_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "test_sha256": sha256(TEST_PATH),
    }
    report, evidence = evaluate_tatqa(
        gold_rows,
        scored_rows,
        query_summary=query_summary,
        source_artifacts=source_artifacts,
        resamples=args.resamples,
    )
    write_report(
        report,
        evidence,
        json_path=args.report_json,
        markdown_path=args.report_markdown,
        evidence_path=args.evidence,
    )
    first = {
        "json": sha256(args.report_json),
        "markdown": sha256(args.report_markdown),
        "evidence": sha256(args.evidence),
    }
    write_report(
        report,
        evidence,
        json_path=args.report_json,
        markdown_path=args.report_markdown,
        evidence_path=args.evidence,
    )
    second = {
        "json": sha256(args.report_json),
        "markdown": sha256(args.report_markdown),
        "evidence": sha256(args.evidence),
    }
    if first != second:
        raise AssertionError("TAT-QA v46 report is not byte deterministic")
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
    result = {
        **_runtime_environment(args),
        "module_sha256": sha256(MODULE_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "test_sha256": sha256(TEST_PATH),
        "implementation_registration_sha256": sha256(args.implementation),
        "protocol_sha256": sha256(args.protocol),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered TAT-QA v46 full-context experiment."
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
            "docs/progressive_upgrade/tatqa_consensus_guarded_atomic_roles_protocol_v46.json"
        ),
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "tatqa_consensus_guarded_atomic_roles_implementation_v46.json"
        ),
    )
    parser.add_argument(
        "--execution",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "tatqa_consensus_guarded_atomic_roles_execution_v46.json"
        ),
    )
    root = Path(".cache/benchmarks/tatqa")
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=root / f"TAT-QA-{OFFICIAL_REPOSITORY_REVISION}.tar.gz",
    )
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument(
        "--generator-model",
        type=Path,
        default=Path(
            r"D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4"
        ),
    )
    cache = root / "v46"
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
    output = Path("output/rag_evaluation/tatqa_consensus_guarded_atomic_roles")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/tatqa_consensus_guarded_atomic_roles_result_v46.json"
        ),
    )
    parser.add_argument(
        "--report-markdown", type=Path, default=output / "report.md"
    )
    parser.add_argument("--evidence", type=Path, default=output / "cases.jsonl.gz")
    parser.add_argument("--registered-at", default="2026-08-02T09:00:00+08:00")
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
