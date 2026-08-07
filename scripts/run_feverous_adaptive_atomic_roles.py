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

from research.frc_rag.feverous_adaptive_atomic_roles import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXPERIMENT_ID,
    FrozenFeverousScorer,
    build_gold_rows,
    evaluate_feverous,
    prepare_blind_cases,
    read_jsonl,
    validate_execution_registration,
    validate_implementation_registration,
    write_report,
)
from research.frc_rag.hover_dynamic_atomic_roles import (  # noqa: E402
    LocalQwenAtomicQueryGenerator,
    generate_queries_resumable,
    validate_query_cache,
)
from research.frc_rag.rgb_cost_aware_frc import (  # noqa: E402
    load_frozen_tokenizer,
    read_jsonl as read_cache_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)


MODULE_PATH = REPO_ROOT / "research/frc_rag/feverous_adaptive_atomic_roles.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_feverous_adaptive_atomic_roles.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_feverous_adaptive_atomic_roles.py"
INHERITED_MODULE_PATH = REPO_ROOT / "research/frc_rag/hover_dynamic_atomic_roles.py"


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _implementation_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "protocol_path": args.protocol,
        "protocol_erratum_path": args.protocol_erratum,
        "module_path": MODULE_PATH,
        "runner_path": RUNNER_PATH,
        "test_path": TEST_PATH,
        "inherited_module_path": INHERITED_MODULE_PATH,
        "erratum_path": args.implementation_erratum,
        "erratum2_path": args.implementation_erratum2,
        "erratum3_path": args.implementation_erratum3,
        "erratum4_path": args.implementation_erratum4,
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
        "lexical_pool_size": 80,
        "maximum_pool_size": 189,
        "cases_per_challenge": 40,
        "bootstrap_resamples": args.resamples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "offline_model_loading": True,
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
        implementation_erratum3_path=args.implementation_erratum3,
        implementation_erratum4_path=args.implementation_erratum4,
        protocol_path=args.protocol,
        protocol_erratum_path=args.protocol_erratum,
        development_path=args.development,
        baseline_path=args.baseline,
        database_archive_path=args.database_archive,
        database_path=args.database,
        prepared_path=args.prepared,
        candidate_map_path=args.candidate_map,
        census_path=args.census,
        query_path=args.queries,
        scored_path=args.scored,
        execution_erratum_path=args.execution_erratum,
        runtime_parameters=_runtime_parameters(args),
    )


def prepare(args: argparse.Namespace) -> None:
    implementation = _validate_implementation(args)
    for path in (
        args.development,
        args.baseline,
        args.database_archive,
        args.database,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"FEVEROUS source artifact is missing: {path}")
    development_rows = read_jsonl(args.development)
    baseline_rows = read_jsonl(args.baseline)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared, candidate_maps, structural = prepare_blind_cases(
        development_rows,
        baseline_rows,
        args.database,
        tokenizer,
    )
    write_jsonl(args.prepared, prepared)
    write_jsonl(args.candidate_map, candidate_maps)
    census = {
        "schema_version": "frc-feverous-v43-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "implementation_registration_sha256": sha256(args.implementation),
        "implementation_erratum_sha256": sha256(args.implementation_erratum),
        "implementation_erratum2_sha256": sha256(args.implementation_erratum2),
        "implementation_erratum3_sha256": sha256(args.implementation_erratum3),
        "protocol_sha256": sha256(args.protocol),
        "protocol_erratum_sha256": sha256(args.protocol_erratum),
        "implementation_registration": implementation,
        "source_artifacts": {
            "development_sha256": sha256(args.development),
            "development_bytes": args.development.stat().st_size,
            "development_rows": len(development_rows),
            "baseline_prediction_sha256": sha256(args.baseline),
            "baseline_prediction_bytes": args.baseline.stat().st_size,
            "baseline_prediction_rows": len(baseline_rows),
            "database_archive_sha256": sha256(args.database_archive),
            "database_archive_bytes": args.database_archive.stat().st_size,
            "database_sha256": sha256(args.database),
            "database_bytes": args.database.stat().st_size,
        },
        "structural_census": structural,
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "content_fields_used": [
            "id",
            "claim",
            "challenge for deterministic stratified sampling only",
            "evidence only for nonempty eligibility before blind export",
            "official predicted_evidence for candidate pages and seed units",
            "Wikipedia page content for gold-free candidate unit construction",
        ],
        "gold_identifiers_or_labels_exported_to_blind_cache": False,
        "challenge_exported_to_generator_or_scorer": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }
    _write_json(args.census, census)
    print(json.dumps(census, ensure_ascii=False, sort_keys=True))


def generate(args: argparse.Namespace) -> None:
    _validate_execution(args)
    prepared = list(read_cache_jsonl(args.prepared))
    generator = LocalQwenAtomicQueryGenerator(
        model_path=args.generator_model,
        batch_size=args.generator_batch_size,
        max_input_tokens=512,
        max_new_tokens=160,
    )
    rows = generate_queries_resumable(
        prepared,
        generator,
        args.queries,
        case_batch_size=args.generator_batch_size,
    )
    summary = validate_query_cache(prepared, rows)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def score(args: argparse.Namespace) -> None:
    _validate_execution(args)
    prepared = list(read_cache_jsonl(args.prepared))
    queries = list(read_cache_jsonl(args.queries))
    validate_query_cache(prepared, queries)
    scorer = FrozenFeverousScorer(
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
    prepared = list(read_cache_jsonl(args.prepared))
    candidate_maps = list(read_cache_jsonl(args.candidate_map))
    queries = list(read_cache_jsonl(args.queries))
    scored = list(read_cache_jsonl(args.scored))
    expected_ids = [str(row.get("id")) for row in prepared]
    if [str(row.get("id")) for row in candidate_maps] != expected_ids:
        raise ValueError("FEVEROUS candidate map is incomplete or out of order")
    if [str(row.get("id")) for row in queries] != expected_ids:
        raise ValueError("FEVEROUS query cache is incomplete or out of order")
    if [str(row.get("id")) for row in scored] != expected_ids:
        raise ValueError("FEVEROUS score cache is incomplete or out of order")
    query_summary = validate_query_cache(prepared, queries)

    # Gold is deliberately read and joined only after complete blind caches pass.
    development_rows = read_jsonl(args.development)
    gold = build_gold_rows(development_rows, candidate_maps)
    report, evidence = evaluate_feverous(
        gold,
        scored,
        query_summary,
        resamples=args.resamples,
    )
    report["metadata"]["source_artifacts"] = {
        "execution_registration_sha256": sha256(args.execution),
        "execution_erratum_sha256": sha256(args.execution_erratum),
        "execution_registration": execution,
        "prepared_blind_sha256": sha256(args.prepared),
        "candidate_map_sha256": sha256(args.candidate_map),
        "queries_sha256": sha256(args.queries),
        "scored_sha256": sha256(args.scored),
        "module_sha256": sha256(MODULE_PATH),
        "inherited_v41_module_sha256": sha256(INHERITED_MODULE_PATH),
    }
    paths = write_report(report, evidence, args.output_dir)
    first = {name: sha256(path) for name, path in paths.items()}
    paths = write_report(report, evidence, args.output_dir)
    second = {name: sha256(path) for name, path in paths.items()}
    if first != second:
        raise AssertionError("FEVEROUS v43 report is not byte deterministic")
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
        "runtime_parameters": _runtime_parameters(args),
        "module_sha256": sha256(MODULE_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "test_sha256": sha256(TEST_PATH),
        "implementation_registration_sha256": sha256(args.implementation),
        "implementation_erratum_sha256": sha256(args.implementation_erratum),
        "implementation_erratum2_sha256": sha256(args.implementation_erratum2),
        "implementation_erratum3_sha256": sha256(args.implementation_erratum3),
        "implementation_erratum4_sha256": sha256(args.implementation_erratum4),
        "execution_erratum_sha256": sha256(args.execution_erratum),
        "protocol_sha256": sha256(args.protocol),
        "protocol_erratum_sha256": sha256(args.protocol_erratum),
        "inherited_v41_module_sha256": sha256(INHERITED_MODULE_PATH),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered FEVEROUS v43 bounded-pool experiment."
    )
    parser.add_argument(
        "command", choices=("prepare", "generate", "score", "evaluate", "runtime")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/feverous_adaptive_atomic_roles_protocol_v43.json"
        ),
    )
    parser.add_argument(
        "--protocol-erratum",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "feverous_adaptive_atomic_roles_protocol_erratum_v43.json"
        ),
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "feverous_adaptive_atomic_roles_implementation_v43.json"
        ),
    )
    parser.add_argument(
        "--execution",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/feverous_adaptive_atomic_roles_execution_v43.json"
        ),
    )
    parser.add_argument(
        "--execution-erratum",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "feverous_adaptive_atomic_roles_execution_erratum_v43.json"
        ),
    )
    parser.add_argument(
        "--implementation-erratum",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "feverous_adaptive_atomic_roles_implementation_erratum_v43.json"
        ),
    )
    parser.add_argument(
        "--implementation-erratum2",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "feverous_adaptive_atomic_roles_implementation_erratum2_v43.json"
        ),
    )
    root = Path(".cache/benchmarks/feverous")
    parser.add_argument(
        "--implementation-erratum3",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "feverous_adaptive_atomic_roles_implementation_erratum3_v43.json"
        ),
    )
    parser.add_argument(
        "--implementation-erratum4",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "feverous_adaptive_atomic_roles_implementation_erratum4_v43.json"
        ),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=root / "feverous_dev_challenges.jsonl",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=(
            root
            / "official_repo/baseline_output/"
            "dev.combined.not_precomputed.p5.s5.t3.cells.verdict.jsonl"
        ),
    )
    parser.add_argument(
        "--database-archive",
        type=Path,
        default=root / "feverous-wiki-pages-db.zip",
    )
    parser.add_argument(
        "--database", type=Path, default=root / "feverous_wikiv1.db"
    )
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument(
        "--generator-model",
        type=Path,
        default=Path(
            r"D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4"
        ),
    )
    cache = root / "v43"
    parser.add_argument("--prepared", type=Path, default=cache / "prepared_blind.jsonl")
    parser.add_argument(
        "--candidate-map", type=Path, default=cache / "candidate_map.jsonl"
    )
    parser.add_argument("--census", type=Path, default=cache / "blind_census.json")
    parser.add_argument("--queries", type=Path, default=cache / "queries.jsonl")
    parser.add_argument("--scored", type=Path, default=cache / "scored.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/feverous_adaptive_atomic_roles"),
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
        "generate": generate,
        "score": score,
        "evaluate": evaluate,
        "runtime": runtime,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
