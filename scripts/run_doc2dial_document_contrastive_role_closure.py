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

from research.frc_rag.doc2dial_document_contrastive_role_closure import (  # noqa: E402
    BUDGETS,
    EXPERIMENT_ID,
    SOURCE_FILES,
    SOURCE_SHA256,
    TARGET_CASES,
    FrozenDoc2DialScorer,
    build_candidate_coverage,
    build_deterministic_queries,
    build_gold_rows,
    evaluate_stage,
    prepare_blind_cases,
    read_stage_sources,
    select_balanced_sample,
    validate_implementation_registration,
    validate_protocol,
    validate_query_cache,
    validate_source_registration,
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
    REPO_ROOT / "research/frc_rag/doc2dial_document_contrastive_role_closure.py"
)
RUNNER_PATH = REPO_ROOT / "scripts/run_doc2dial_document_contrastive_role_closure.py"
TEST_PATH = (
    REPO_ROOT / "tests/test_frc_doc2dial_document_contrastive_role_closure.py"
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


def _source_manifest(source_root: Path) -> dict[str, Any]:
    files = {}
    for name, expected in SOURCE_SHA256.items():
        path = source_root / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Doc2Dial v54 fixed source changed: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": actual}
    return {
        "source_root": str(source_root),
        "files": files,
        "dataset_version": "v1.0.1",
        "repository_revision": "7584ac0c1c617496d644f1ea34a6b812a5771539",
        "license_status": "official fixed repository has no explicit license file; HF card says cc-by-3.0",
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
        "token_budgets": list(BUDGETS),
        "learned_threshold": None,
        "contrastive_documents": 8,
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
    validate_source_registration(
        args.source_registration,
        protocol_path=args.protocol,
        source_root=args.source_root,
    )
    return validate_implementation_registration(
        args.implementation,
        protocol_path=args.protocol,
        source_registration_path=args.source_registration,
        module_path=MODULE_PATH,
        runner_path=RUNNER_PATH,
        test_path=TEST_PATH,
    )


def _stage_paths(args: argparse.Namespace) -> dict[str, Path]:
    cache = args.cache_root / args.stage
    output = args.output_root / args.stage
    return {
        "prepared": cache / "prepared_blind.jsonl",
        "candidate_map": cache / "candidate_map.jsonl",
        "document_store": cache / "blind_document_store.jsonl",
        "census": cache / "blind_census.json",
        "queries": cache / "queries.jsonl",
        "scored": cache / "scored.jsonl",
        "coverage": cache / "coverage.json",
        "execution": args.execution_root
        / f"doc2dial_document_contrastive_role_closure_{args.stage}_execution_v54.json",
        "result": args.result_root
        / f"doc2dial_document_contrastive_role_closure_{args.stage}_result_v54.json",
        "report": output / "report.md",
        "evidence": output / "cases.jsonl.gz",
    }


def _validate_confirmation_open(args: argparse.Namespace) -> None:
    if args.stage != "confirmation":
        return
    development = args.result_root / (
        "doc2dial_document_contrastive_role_closure_development_result_v54.json"
    )
    if not development.is_file():
        raise ValueError(
            "Doc2Dial v54 confirmation cannot open before development result"
        )
    value = json.loads(development.read_text(encoding="utf-8"))
    outcome = value["analysis"]["outcome"]
    if (
        outcome["support_established"] is not True
        or outcome["validation_open_authorized"] is not True
    ):
        raise ValueError(
            "Doc2Dial v54 development gate did not authorize validation opening"
        )


def _validate_execution(
    args: argparse.Namespace, paths: dict[str, Path]
) -> dict[str, Any]:
    _validate_implementation(args)
    value = json.loads(paths["execution"].read_text(encoding="utf-8"))
    expected = {
        "implementation_registration_sha256": sha256(args.implementation),
        "document_source_sha256": sha256(
            args.source_root / SOURCE_FILES["documents"]
        ),
        "dialogue_source_sha256": sha256(args.source_root / SOURCE_FILES[args.stage]),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "blind_census_sha256": sha256(paths["census"]),
    }
    if value.get("hashes") != expected:
        raise ValueError("Doc2Dial v54 execution registration hash mismatch")
    if value.get("stage") != args.stage:
        raise ValueError("Doc2Dial v54 execution stage changed")
    if value.get("query_generation_started") is not False:
        raise ValueError("Doc2Dial v54 execution was registered too late")
    if value.get("gold_joined_for_coverage_or_metrics") is not False:
        raise ValueError("Doc2Dial v54 gold was joined before scoring")
    return value


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _validate_confirmation_open(args)
    paths = _stage_paths(args)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise ValueError(
            "Doc2Dial v54 stage preparation is frozen: "
            + ", ".join(str(path) for path in existing)
        )
    documents, dialogues = read_stage_sources(args.source_root, args.stage)
    selected, sampling = select_balanced_sample(
        documents, dialogues, stage=args.stage
    )
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared_rows, maps, document_store, structural = prepare_blind_cases(
        selected, documents, tokenizer, stage=args.stage
    )
    write_jsonl(paths["prepared"], prepared_rows)
    write_jsonl(paths["candidate_map"], maps)
    write_jsonl(paths["document_store"], document_store)
    manifest = _source_manifest(args.source_root)
    census = {
        "schema_version": "frc-doc2dial-v54-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "source_manifest": {
            **manifest,
            "opened_document_file": SOURCE_FILES["documents"],
            "opened_dialogue_file": SOURCE_FILES[args.stage],
            "other_stage_dialogue_file_opened": False,
        },
        "sampling": sampling,
        "structural_census": structural,
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "references_raw_ids_dialogue_acts_or_gold_exported_to_blind_cache": False,
        "gold_joined_for_coverage_or_metrics": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }
    _write_json(paths["census"], census)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "sampling": sampling,
                "structural_census": structural,
                "hashes": {
                    "prepared": sha256(paths["prepared"]),
                    "candidate_map": sha256(paths["candidate_map"]),
                    "document_store": sha256(paths["document_store"]),
                    "census": sha256(paths["census"]),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def register(args: argparse.Namespace) -> None:
    implementation = _validate_implementation(args)
    _validate_confirmation_open(args)
    paths = _stage_paths(args)
    if paths["execution"].exists():
        raise FileExistsError("Doc2Dial v54 execution registration already exists")
    if any(
        paths[name].exists() for name in ("queries", "scored", "coverage", "result")
    ):
        raise ValueError(
            "Doc2Dial v54 outcome-bearing work started before execution registration"
        )
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    expected = {
        "prepared": census["prepared_blind_sha256"],
        "candidate_map": census["candidate_map_sha256"],
        "document_store": census["blind_document_store_sha256"],
    }
    actual = {
        "prepared": sha256(paths["prepared"]),
        "candidate_map": sha256(paths["candidate_map"]),
        "document_store": sha256(paths["document_store"]),
    }
    if actual != expected:
        raise ValueError("Doc2Dial v54 blind artifacts changed")
    runtime = _runtime_environment(args)
    if args.device == "cuda" and runtime["cuda_available"] is not True:
        raise ValueError("Doc2Dial v54 frozen CUDA runtime is unavailable")
    value = {
        "schema_version": "frc-doc2dial-v54-execution-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "registered_at": args.registered_at,
        "hashes": {
            "implementation_registration_sha256": sha256(args.implementation),
            "document_source_sha256": sha256(
                args.source_root / SOURCE_FILES["documents"]
            ),
            "dialogue_source_sha256": sha256(
                args.source_root / SOURCE_FILES[args.stage]
            ),
            "prepared_blind_sha256": sha256(paths["prepared"]),
            "candidate_map_sha256": sha256(paths["candidate_map"]),
            "blind_document_store_sha256": sha256(paths["document_store"]),
            "blind_census_sha256": sha256(paths["census"]),
        },
        "source_artifacts": _source_manifest(args.source_root),
        "runtime": runtime,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "gold_joined_for_coverage_or_metrics": False,
        "metrics_computed": False,
        "implementation_registration": implementation,
        "method_formula_change_after_registration_forbidden": True,
        "stage_reuse_for_tuning_or_selection_forbidden": True,
    }
    _write_json(paths["execution"], value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def generate(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    _validate_execution(args, paths)
    if paths["scored"].exists() or paths["coverage"].exists():
        raise ValueError("Doc2Dial v54 scoring or gold join started before queries")
    prepared_rows = list(read_jsonl(paths["prepared"]))
    rows = build_deterministic_queries(prepared_rows)
    summary = validate_query_cache(prepared_rows, rows)
    write_jsonl(paths["queries"], rows)
    print(
        json.dumps(
            {**summary, "queries_sha256": sha256(paths["queries"])},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def score(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    _validate_execution(args, paths)
    if paths["coverage"].exists():
        raise ValueError("Doc2Dial v54 gold was joined before complete scoring")
    prepared_rows = list(read_jsonl(paths["prepared"]))
    queries = list(read_jsonl(paths["queries"]))
    document_store = list(read_jsonl(paths["document_store"]))
    validate_query_cache(prepared_rows, queries)
    scorer = FrozenDoc2DialScorer(
        query_rows=queries,
        document_store=document_store,
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        use_fp16=not args.no_fp16,
    )
    scored = score_cases_resumable(
        prepared_rows,
        scorer,
        paths["scored"],
        case_batch_size=args.case_batch_size,
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "rows": len(scored),
                "scored_sha256": sha256(paths["scored"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    execution = _validate_execution(args, paths)
    if paths["result"].exists() or paths["coverage"].exists():
        raise ValueError("Doc2Dial v54 stage evaluation is frozen")
    prepared_rows = list(read_jsonl(paths["prepared"]))
    maps = list(read_jsonl(paths["candidate_map"]))
    queries = list(read_jsonl(paths["queries"]))
    scored = list(read_jsonl(paths["scored"]))
    expected = [str(row["id"]) for row in prepared_rows]
    if len(expected) != TARGET_CASES:
        raise ValueError("Doc2Dial v54 blind cache is incomplete")
    for name, rows in (("map", maps), ("query", queries), ("score", scored)):
        if [str(row["id"]) for row in rows] != expected:
            raise ValueError(
                f"Doc2Dial v54 {name} cache is incomplete before gold join"
            )
    query_summary = validate_query_cache(prepared_rows, queries)
    documents, dialogues = read_stage_sources(args.source_root, args.stage)
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    structural = census["structural_census"]
    gold = build_gold_rows(
        documents,
        dialogues,
        maps,
        scored,
        document_length_quartile_boundaries=structural[
            "document_length_quartile_boundaries"
        ],
        turn_position_quartile_boundaries=structural[
            "dialogue_turn_position_quartile_boundaries"
        ],
    )
    coverage = build_candidate_coverage(gold, census["sampling"])
    coverage["gold_joined_after_complete_score_cache"] = True
    coverage["queries_sha256"] = sha256(paths["queries"])
    coverage["scored_sha256"] = sha256(paths["scored"])
    coverage["execution_registration_sha256"] = sha256(paths["execution"])
    _write_json(paths["coverage"], coverage)
    source_artifacts = {
        "execution_registration_sha256": sha256(paths["execution"]),
        "execution_registration": execution,
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "blind_census_sha256": sha256(paths["census"]),
        "candidate_coverage_sha256": sha256(paths["coverage"]),
        "queries_sha256": sha256(paths["queries"]),
        "scored_sha256": sha256(paths["scored"]),
        "module_sha256": sha256(MODULE_PATH),
        "runner_sha256": sha256(RUNNER_PATH),
        "test_sha256": sha256(TEST_PATH),
        "sampling": census["sampling"],
        "structural_census": structural,
        "score_fallback_rate": 0.0,
    }
    report, evidence = evaluate_stage(
        gold,
        scored,
        query_summary,
        source_artifacts,
        stage=args.stage,
    )
    write_report(report, evidence, paths["result"], paths["report"], paths["evidence"])
    first = {
        "json": sha256(paths["result"]),
        "markdown": sha256(paths["report"]),
        "evidence": sha256(paths["evidence"]),
    }
    write_report(report, evidence, paths["result"], paths["report"], paths["evidence"])
    second = {
        "json": sha256(paths["result"]),
        "markdown": sha256(paths["report"]),
        "evidence": sha256(paths["evidence"]),
    }
    if first != second:
        raise AssertionError("Doc2Dial v54 report is not deterministic")
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": report["analysis"]["outcome"]["status"],
                "support_established": report["analysis"]["outcome"][
                    "support_established"
                ],
                "validation_open_authorized": report["analysis"]["outcome"][
                    "validation_open_authorized"
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
        description="Run the preregistered Doc2Dial v54 contrastive experiment."
    )
    parser.add_argument(
        "command",
        choices=("prepare", "register", "generate", "score", "evaluate", "runtime"),
    )
    parser.add_argument(
        "--stage", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/doc2dial_document_contrastive_role_closure_protocol_v54.json"
        ),
    )
    parser.add_argument(
        "--source-registration",
        type=Path,
        default=Path("docs/progressive_upgrade/doc2dial_source_registration_v54.json"),
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/doc2dial_document_contrastive_role_closure_implementation_v54.json"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(".cache/benchmarks/doc2dial/v54_source"),
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path(".cache/benchmarks/doc2dial/v54")
    )
    parser.add_argument(
        "--execution-root", type=Path, default=Path("docs/progressive_upgrade")
    )
    parser.add_argument(
        "--result-root", type=Path, default=Path("docs/progressive_upgrade")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "output/rag_evaluation/doc2dial_document_contrastive_role_closure"
        ),
    )
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=2)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--registered-at", default="2026-08-02T23:45:00+08:00")
    args = parser.parse_args()
    for name in (
        "protocol",
        "source_registration",
        "implementation",
        "source_root",
        "cache_root",
        "execution_root",
        "result_root",
        "output_root",
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
