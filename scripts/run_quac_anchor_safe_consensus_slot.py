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

from research.frc_rag.quac_anchor_safe_consensus_slot import (  # noqa: E402
    BUDGETS,
    EXPERIMENT_ID,
    SOURCE_FILES,
    TARGET_CASES,
    FrozenQuacScorer,
    build_candidate_coverage,
    build_deterministic_queries,
    build_gold_rows,
    evaluate_stage,
    prepare_blind_cases,
    read_stage_source,
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


MODULE_PATH = REPO_ROOT / "research/frc_rag/quac_anchor_safe_consensus_slot.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_quac_anchor_safe_consensus_slot.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_quac_anchor_safe_consensus_slot.py"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _source_manifest(args: argparse.Namespace) -> dict[str, Any]:
    registration = validate_source_registration(
        args.source_registration,
        protocol_path=args.protocol,
        source_root=args.source_root,
    )
    return {
        "source_root": str(args.source_root),
        "files": {
            name: {
                "bytes": (args.source_root / name).stat().st_size,
                "sha256": sha256(args.source_root / name),
            }
            for name in SOURCE_FILES.values()
        },
        "dataset_version": "QuAC v0.2",
        "license": "CC BY-SA 4.0 as stated on the official QuAC homepage",
        "source_registration_sha256": sha256(args.source_registration),
        "source_registration": registration,
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
        "learned_parameter": None,
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
        / f"quac_anchor_safe_consensus_slot_{args.stage}_execution_v57.json",
        "result": args.result_root
        / f"quac_anchor_safe_consensus_slot_{args.stage}_result_v57.json",
        "report": output / "report.md",
        "evidence": output / "cases.jsonl.gz",
    }


def _validate_confirmation_open(args: argparse.Namespace) -> None:
    if args.stage != "confirmation":
        return
    development = args.result_root / (
        "quac_anchor_safe_consensus_slot_development_result_v57.json"
    )
    if not development.is_file():
        raise ValueError("QuAC v57 confirmation cannot open before development result")
    value = json.loads(development.read_text(encoding="utf-8"))
    outcome = value["analysis"]["outcome"]
    if (
        outcome["support_established"] is not True
        or outcome["validation_open_authorized"] is not True
    ):
        raise ValueError("QuAC v57 development gate did not authorize validation opening")


def _validate_execution(
    args: argparse.Namespace, paths: dict[str, Path]
) -> dict[str, Any]:
    _validate_implementation(args)
    value = json.loads(paths["execution"].read_text(encoding="utf-8"))
    expected = {
        "implementation_registration_sha256": sha256(args.implementation),
        "stage_source_sha256": sha256(args.source_root / SOURCE_FILES[args.stage]),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "blind_census_sha256": sha256(paths["census"]),
    }
    if value.get("hashes") != expected:
        raise ValueError("QuAC v57 execution registration hash mismatch")
    if value.get("stage") != args.stage:
        raise ValueError("QuAC v57 execution stage changed")
    if value.get("query_generation_started") is not False:
        raise ValueError("QuAC v57 execution was registered too late")
    if value.get("gold_joined_for_coverage_or_metrics") is not False:
        raise ValueError("QuAC v57 gold was joined before scoring")
    return value


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _validate_confirmation_open(args)
    paths = _stage_paths(args)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise ValueError(
            "QuAC v57 stage preparation is frozen: "
            + ", ".join(str(path) for path in existing)
        )
    source = read_stage_source(args.source_root, args.stage)
    selected, sampling = select_balanced_sample(source, stage=args.stage)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared_rows, maps, document_store, structural = prepare_blind_cases(
        selected, source, tokenizer, stage=args.stage
    )
    write_jsonl(paths["prepared"], prepared_rows)
    write_jsonl(paths["candidate_map"], maps)
    write_jsonl(paths["document_store"], document_store)
    census = {
        "schema_version": "frc-quac-v57-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "source_manifest": {
            **_source_manifest(args),
            "opened_source_file": SOURCE_FILES[args.stage],
            "other_stage_source_file_opened": False,
        },
        "sampling": sampling,
        "structural_census": structural,
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "current_answer_state_offset_raw_ids_or_gold_exported_to_blind_cache": False,
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
        raise FileExistsError("QuAC v57 execution registration already exists")
    if any(paths[name].exists() for name in ("queries", "scored", "coverage", "result")):
        raise ValueError("QuAC v57 outcome-bearing work started before execution registration")
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
        raise ValueError("QuAC v57 blind artifacts changed")
    runtime = _runtime_environment(args)
    if args.device == "cuda" and runtime["cuda_available"] is not True:
        raise ValueError("QuAC v57 frozen CUDA runtime is unavailable")
    value = {
        "schema_version": "frc-quac-v57-execution-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "registered_at": args.registered_at,
        "hashes": {
            "implementation_registration_sha256": sha256(args.implementation),
            "stage_source_sha256": sha256(args.source_root / SOURCE_FILES[args.stage]),
            "prepared_blind_sha256": sha256(paths["prepared"]),
            "candidate_map_sha256": sha256(paths["candidate_map"]),
            "blind_document_store_sha256": sha256(paths["document_store"]),
            "blind_census_sha256": sha256(paths["census"]),
        },
        "source_artifacts": _source_manifest(args),
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
        raise ValueError("QuAC v57 scoring or gold join started before queries")
    prepared_rows = list(read_jsonl(paths["prepared"]))
    rows = build_deterministic_queries(prepared_rows)
    summary = validate_query_cache(prepared_rows, rows)
    write_jsonl(paths["queries"], rows)
    print(json.dumps({**summary, "queries_sha256": sha256(paths["queries"])}, sort_keys=True))


def score(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    _validate_execution(args, paths)
    if paths["coverage"].exists():
        raise ValueError("QuAC v57 gold was joined before complete scoring")
    prepared_rows = list(read_jsonl(paths["prepared"]))
    queries = list(read_jsonl(paths["queries"]))
    document_store = list(read_jsonl(paths["document_store"]))
    validate_query_cache(prepared_rows, queries)
    scorer = FrozenQuacScorer(
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
            {"stage": args.stage, "rows": len(scored), "scored_sha256": sha256(paths["scored"])},
            sort_keys=True,
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    execution = _validate_execution(args, paths)
    if paths["result"].exists() or paths["coverage"].exists():
        raise ValueError("QuAC v57 stage evaluation is frozen")
    prepared_rows = list(read_jsonl(paths["prepared"]))
    maps = list(read_jsonl(paths["candidate_map"]))
    queries = list(read_jsonl(paths["queries"]))
    scored = list(read_jsonl(paths["scored"]))
    expected = [str(row["id"]) for row in prepared_rows]
    if len(expected) != TARGET_CASES:
        raise ValueError("QuAC v57 blind cache is incomplete")
    for name, rows in (("map", maps), ("query", queries), ("score", scored)):
        if [str(row["id"]) for row in rows] != expected:
            raise ValueError(f"QuAC v57 {name} cache is incomplete before gold join")
    query_summary = validate_query_cache(prepared_rows, queries)
    source = read_stage_source(args.source_root, args.stage)
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    structural = census["structural_census"]
    gold = build_gold_rows(
        source,
        maps,
        scored,
        document_length_quartile_boundaries=structural["document_length_quartile_boundaries"],
        turn_position_quartile_boundaries=structural["dialogue_turn_position_quartile_boundaries"],
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
        gold, scored, query_summary, source_artifacts, stage=args.stage
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
        raise AssertionError("QuAC v57 report is not deterministic")
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": report["analysis"]["outcome"]["status"],
                "support_established": report["analysis"]["outcome"]["support_established"],
                "validation_open_authorized": report["analysis"]["outcome"]["validation_open_authorized"],
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
    parser = argparse.ArgumentParser(description="Run the preregistered QuAC v57 experiment.")
    parser.add_argument("command", choices=("prepare", "register", "generate", "score", "evaluate", "runtime"))
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/progressive_upgrade/quac_anchor_safe_consensus_slot_protocol_v57.json"),
    )
    parser.add_argument(
        "--source-registration",
        type=Path,
        default=Path("docs/progressive_upgrade/quac_source_registration_v57.json"),
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path("docs/progressive_upgrade/quac_anchor_safe_consensus_slot_implementation_v57.json"),
    )
    parser.add_argument("--source-root", type=Path, default=Path(".cache/benchmarks/quac/v57_source"))
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/benchmarks/quac/v57"))
    parser.add_argument("--execution-root", type=Path, default=Path("docs/progressive_upgrade"))
    parser.add_argument("--result-root", type=Path, default=Path("docs/progressive_upgrade"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/rag_evaluation/quac_anchor_safe_consensus_slot"),
    )
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=2)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--registered-at", default="2026-08-03T02:10:00+08:00")
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
