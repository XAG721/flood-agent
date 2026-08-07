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
from research.frc_rag.rgb_cost_aware_frc import (  # noqa: E402
    load_frozen_tokenizer,
    read_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)
from research.frc_rag.scifact_dynamic_atomic_roles import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    EXPERIMENT_ID,
    FrozenSciFactScorer,
    build_gold_rows,
    evaluate_scifact,
    prepare_blind_cases,
    read_source_jsonl,
    validate_execution_registration,
    validate_implementation_registration,
    write_report,
)


MODULE_PATH = REPO_ROOT / "research/frc_rag/scifact_dynamic_atomic_roles.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_scifact_dynamic_atomic_roles.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_scifact_dynamic_atomic_roles.py"
INHERITED_MODULE_PATH = (
    REPO_ROOT / "research/frc_rag/hover_dynamic_atomic_roles.py"
)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _find_one(root: Path, filename: str) -> Path:
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {filename} below {root}, found {len(matches)}"
        )
    return matches[0]


def _source_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    return (
        _find_one(args.source_root, "corpus.jsonl"),
        _find_one(args.source_root, "claims_dev.jsonl"),
    )


def _implementation_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "protocol_path": args.protocol,
        "module_path": MODULE_PATH,
        "runner_path": RUNNER_PATH,
        "test_path": TEST_PATH,
        "inherited_module_path": INHERITED_MODULE_PATH,
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
        "bootstrap_seed": 20260802,
        "offline_model_loading": True,
    }


def _validate_implementation(args: argparse.Namespace) -> dict[str, Any]:
    return validate_implementation_registration(
        args.implementation,
        **_implementation_kwargs(args),
    )


def _validate_execution(args: argparse.Namespace) -> dict[str, Any]:
    _validate_implementation(args)
    corpus_path, claims_path = _source_paths(args)
    return validate_execution_registration(
        args.execution,
        implementation_path=args.implementation,
        protocol_path=args.protocol,
        archive_path=args.archive,
        corpus_path=corpus_path,
        claims_path=claims_path,
        prepared_path=args.prepared,
        census_path=args.census,
        module_path=MODULE_PATH,
        runner_path=RUNNER_PATH,
        test_path=TEST_PATH,
        inherited_module_path=INHERITED_MODULE_PATH,
        runtime_parameters=_runtime_parameters(args),
    )


def prepare(args: argparse.Namespace) -> None:
    implementation = _validate_implementation(args)
    if not args.archive.is_file():
        raise FileNotFoundError(f"SciFact release archive is missing: {args.archive}")
    corpus_path, claims_path = _source_paths(args)
    corpus_rows = read_source_jsonl(corpus_path)
    claim_rows = read_source_jsonl(claims_path)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared, structural = prepare_blind_cases(
        corpus_rows,
        claim_rows,
        tokenizer,
    )
    write_jsonl(args.prepared, prepared)
    census = {
        "schema_version": "frc-scifact-v42-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "implementation_registration_sha256": sha256(args.implementation),
        "implementation_registration": implementation,
        "source_artifacts": {
            "archive_sha256": sha256(args.archive),
            "archive_bytes": args.archive.stat().st_size,
            "corpus_sha256": sha256(corpus_path),
            "corpus_bytes": corpus_path.stat().st_size,
            "corpus_rows": len(corpus_rows),
            "claims_dev_sha256": sha256(claims_path),
            "claims_dev_bytes": claims_path.stat().st_size,
            "claims_dev_rows": len(claim_rows),
        },
        "structural_census": structural,
        "prepared_blind_sha256": sha256(args.prepared),
        "content_fields_used": ["id", "claim", "doc_id", "title", "abstract"],
        "evidence_used_only_for_eligibility_and_missing_document_validation": True,
        "gold_identifiers_or_labels_exported": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }
    _write_json(args.census, census)
    print(json.dumps(census, ensure_ascii=False, sort_keys=True))


def generate(args: argparse.Namespace) -> None:
    _validate_execution(args)
    prepared = list(read_jsonl(args.prepared))
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
    prepared = list(read_jsonl(args.prepared))
    queries = list(read_jsonl(args.queries))
    validate_query_cache(prepared, queries)
    scorer = FrozenSciFactScorer(
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
    prepared = list(read_jsonl(args.prepared))
    queries = list(read_jsonl(args.queries))
    scored = list(read_jsonl(args.scored))
    expected_ids = [str(row.get("id")) for row in prepared]
    if [str(row.get("id")) for row in queries] != expected_ids:
        raise ValueError("SciFact query cache is incomplete or out of order")
    if [str(row.get("id")) for row in scored] != expected_ids:
        raise ValueError("SciFact score cache is incomplete or out of order")
    query_summary = validate_query_cache(prepared, queries)

    # Gold is deliberately read and joined only after complete blind caches pass.
    corpus_path, claims_path = _source_paths(args)
    corpus_rows = read_source_jsonl(corpus_path)
    claim_rows = read_source_jsonl(claims_path)
    gold = build_gold_rows(corpus_rows, claim_rows, prepared)
    report, evidence = evaluate_scifact(
        gold,
        scored,
        query_summary,
        resamples=args.resamples,
    )
    report["metadata"]["source_artifacts"] = {
        "execution_registration_sha256": sha256(args.execution),
        "execution_registration": execution,
        "prepared_blind_sha256": sha256(args.prepared),
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
        raise AssertionError("SciFact v42 report is not byte deterministic")
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
        "inherited_v41_module_sha256": sha256(INHERITED_MODULE_PATH),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered SciFact v42 transfer experiment."
    )
    parser.add_argument("command", choices=("prepare", "generate", "score", "evaluate", "runtime"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/scifact_dynamic_atomic_roles_protocol_v42.json"
        ),
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "scifact_dynamic_atomic_roles_implementation_v42.json"
        ),
    )
    parser.add_argument(
        "--execution",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "scifact_dynamic_atomic_roles_execution_v42.json"
        ),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path(".cache/benchmarks/scifact/data.tar.gz"),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(".cache/benchmarks/scifact/data"),
    )
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument(
        "--generator-model",
        type=Path,
        default=Path(
            r"D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4"
        ),
    )
    cache = Path(".cache/benchmarks/scifact/v42")
    parser.add_argument("--prepared", type=Path, default=cache / "prepared_blind.jsonl")
    parser.add_argument("--census", type=Path, default=cache / "blind_census.json")
    parser.add_argument("--queries", type=Path, default=cache / "queries.jsonl")
    parser.add_argument("--scored", type=Path, default=cache / "scored.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/scifact_dynamic_atomic_roles"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=8)
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
