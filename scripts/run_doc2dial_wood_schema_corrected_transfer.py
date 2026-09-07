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

from research.frc_rag.doc2dial_wood_schema_corrected_transfer import (  # noqa: E402
    BUDGETS,
    EXPERIMENT_ID,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_MEMBERS,
    TARGET_CASES,
    FrozenDoc2DialWoodSchemaScorer,
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


MODULE_PATH = REPO_ROOT / "research/frc_rag/doc2dial_wood_schema_corrected_transfer.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_doc2dial_wood_schema_corrected_transfer.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_doc2dial_wood_schema_corrected_transfer.py"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _source_manifest(archive_path: Path) -> dict[str, Any]:
    if sha256(archive_path) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("Doc2Dial wOOD v56 source archive changed")
    with zipfile.ZipFile(archive_path) as archive:
        regular = {item.filename: item for item in archive.infolist() if not item.is_dir()}
        members = {}
        for key, name in SOURCE_MEMBERS.items():
            item = regular.get(name)
            if item is None:
                raise ValueError(f"Doc2Dial wOOD v56 archive lacks {name}")
            members[key] = {
                "path": name,
                "uncompressed_bytes": item.file_size,
                "compressed_bytes": item.compress_size,
            }
    return {
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "resolved_members": members,
        "dataset_version": "v0.9",
        "pre_outcome_schema_access_disclosed": True,
        "confirmation_opened": False,
        "license_status": "official v0.9 source has no explicit license statement; no source text is published",
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
        "parameters": {
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
            "retrieval_gate_selector_unchanged": True,
        },
    }


def _validate_implementation(args: argparse.Namespace) -> dict[str, Any]:
    validate_protocol(args.protocol)
    validate_source_registration(
        args.source_registration,
        protocol_path=args.protocol,
        source_archive=args.source_archive,
    )
    return validate_implementation_registration(
        args.implementation,
        protocol_path=args.protocol,
        source_registration_path=args.source_registration,
        module_path=MODULE_PATH,
        runner_path=RUNNER_PATH,
        test_path=TEST_PATH,
    )


def _paths(args: argparse.Namespace) -> dict[str, Path]:
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
        / f"doc2dial_wood_schema_corrected_transfer_{args.stage}_execution_v56.json",
        "result": args.result_root
        / f"doc2dial_wood_schema_corrected_transfer_{args.stage}_result_v56.json",
        "report": output / "report.md",
        "evidence": output / "cases.jsonl.gz",
    }


def _validate_confirmation(args: argparse.Namespace) -> None:
    if args.stage != "confirmation":
        return
    path = args.result_root / (
        "doc2dial_wood_schema_corrected_transfer_development_result_v56.json"
    )
    if not path.is_file():
        raise ValueError("Doc2Dial wOOD v56 confirmation lacks development result")
    outcome = json.loads(path.read_text(encoding="utf-8"))["analysis"]["outcome"]
    if not outcome["support_established"] or not outcome[
        "confirmation_open_authorized"
    ]:
        raise ValueError("Doc2Dial wOOD v56 gates forbid confirmation opening")


def _validate_execution(args: argparse.Namespace, paths: dict[str, Path]) -> dict:
    _validate_implementation(args)
    value = json.loads(paths["execution"].read_text(encoding="utf-8"))
    expected = {
        "implementation_registration_sha256": sha256(args.implementation),
        "source_archive_sha256": sha256(args.source_archive),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "blind_census_sha256": sha256(paths["census"]),
    }
    if value.get("hashes") != expected or value.get("stage") != args.stage:
        raise ValueError("Doc2Dial wOOD v56 execution registration changed")
    if value.get("query_generation_started") is not False or value.get(
        "gold_joined_for_coverage_or_metrics"
    ) is not False:
        raise ValueError("Doc2Dial wOOD v56 execution registered too late")
    return value


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _validate_confirmation(args)
    paths = _paths(args)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise ValueError("Doc2Dial wOOD v56 preparation is frozen: " + ", ".join(map(str, existing)))
    documents, dialogues = read_stage_sources(args.source_archive, args.stage)
    selected, sampling = select_balanced_sample(documents, dialogues, stage=args.stage)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared_rows, maps, store, structural = prepare_blind_cases(
        selected, documents, tokenizer, stage=args.stage
    )
    write_jsonl(paths["prepared"], prepared_rows)
    write_jsonl(paths["candidate_map"], maps)
    write_jsonl(paths["document_store"], store)
    census = {
        "schema_version": "frc-doc2dial-wood-v56-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "source_manifest": {
            **_source_manifest(args.source_archive),
            "opened_document_member": SOURCE_MEMBERS["documents"],
            "opened_dialogue_member": SOURCE_MEMBERS[args.stage],
            "other_stage_dialogue_member_opened": False,
            "woood_sibling_members_opened": False,
        },
        "sampling": sampling,
        "structural_census": structural,
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "gold_or_dialogue_acts_exported_to_blind_cache": False,
        "gold_joined_for_coverage_or_metrics": False,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "metrics_computed": False,
    }
    _write_json(paths["census"], census)
    print(json.dumps({"stage": args.stage, "sampling": sampling, "structural_census": structural, "hashes": {"prepared": sha256(paths["prepared"]), "candidate_map": sha256(paths["candidate_map"]), "document_store": sha256(paths["document_store"]), "census": sha256(paths["census"])}}, ensure_ascii=False, sort_keys=True))


def register(args: argparse.Namespace) -> None:
    implementation = _validate_implementation(args)
    _validate_confirmation(args)
    paths = _paths(args)
    if paths["execution"].exists():
        raise FileExistsError("Doc2Dial wOOD v56 execution already exists")
    if any(paths[name].exists() for name in ("queries", "scored", "coverage", "result")):
        raise ValueError("Doc2Dial wOOD v56 outcome work started before registration")
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    actual = {
        "prepared": sha256(paths["prepared"]),
        "candidate_map": sha256(paths["candidate_map"]),
        "document_store": sha256(paths["document_store"]),
    }
    expected = {
        "prepared": census["prepared_blind_sha256"],
        "candidate_map": census["candidate_map_sha256"],
        "document_store": census["blind_document_store_sha256"],
    }
    if actual != expected:
        raise ValueError("Doc2Dial wOOD v56 blind artifacts changed")
    runtime = _runtime_environment(args)
    if args.device == "cuda" and not runtime["cuda_available"]:
        raise ValueError("Doc2Dial wOOD v56 CUDA is unavailable")
    value = {
        "schema_version": "frc-doc2dial-wood-v56-execution-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "registered_at": args.registered_at,
        "hashes": {
            "implementation_registration_sha256": sha256(args.implementation),
            "source_archive_sha256": sha256(args.source_archive),
            "prepared_blind_sha256": sha256(paths["prepared"]),
            "candidate_map_sha256": sha256(paths["candidate_map"]),
            "blind_document_store_sha256": sha256(paths["document_store"]),
            "blind_census_sha256": sha256(paths["census"]),
        },
        "source_artifacts": _source_manifest(args.source_archive),
        "runtime": runtime,
        "query_generation_started": False,
        "neural_scoring_started": False,
        "gold_joined_for_coverage_or_metrics": False,
        "metrics_computed": False,
        "implementation_registration": implementation,
        "retrieval_gate_selector_changed": False,
        "method_formula_change_after_registration_forbidden": True,
        "stage_reuse_for_tuning_or_selection_forbidden": True,
    }
    _write_json(paths["execution"], value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def generate(args: argparse.Namespace) -> None:
    paths = _paths(args)
    _validate_execution(args, paths)
    if paths["scored"].exists() or paths["coverage"].exists():
        raise ValueError("Doc2Dial wOOD v56 scoring started before queries")
    prepared = list(read_jsonl(paths["prepared"]))
    rows = build_deterministic_queries(prepared)
    summary = validate_query_cache(prepared, rows)
    write_jsonl(paths["queries"], rows)
    print(json.dumps({**summary, "queries_sha256": sha256(paths["queries"])}, ensure_ascii=False, sort_keys=True))


def score(args: argparse.Namespace) -> None:
    paths = _paths(args)
    _validate_execution(args, paths)
    if paths["coverage"].exists():
        raise ValueError("Doc2Dial wOOD v56 gold joined before scores")
    prepared = list(read_jsonl(paths["prepared"]))
    queries = list(read_jsonl(paths["queries"]))
    store = list(read_jsonl(paths["document_store"]))
    validate_query_cache(prepared, queries)
    scorer = FrozenDoc2DialWoodSchemaScorer(
        query_rows=queries,
        document_store=store,
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        use_fp16=not args.no_fp16,
    )
    rows = score_cases_resumable(prepared, scorer, paths["scored"], case_batch_size=args.case_batch_size)
    print(json.dumps({"stage": args.stage, "rows": len(rows), "scored_sha256": sha256(paths["scored"])}, ensure_ascii=False, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    if paths["result"].exists() or paths["coverage"].exists():
        raise ValueError("Doc2Dial wOOD v56 evaluation is frozen")
    prepared = list(read_jsonl(paths["prepared"]))
    maps = list(read_jsonl(paths["candidate_map"]))
    queries = list(read_jsonl(paths["queries"]))
    scored = list(read_jsonl(paths["scored"]))
    expected = [str(row["id"]) for row in prepared]
    if len(expected) != TARGET_CASES:
        raise ValueError("Doc2Dial wOOD v56 blind cache is incomplete")
    for name, rows in (("map", maps), ("query", queries), ("score", scored)):
        if [str(row["id"]) for row in rows] != expected:
            raise ValueError(f"Doc2Dial wOOD v56 {name} cache is incomplete")
    query_summary = validate_query_cache(prepared, queries)
    documents, dialogues = read_stage_sources(args.source_archive, args.stage)
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    structural = census["structural_census"]
    gold = build_gold_rows(
        documents,
        dialogues,
        maps,
        scored,
        document_length_quartile_boundaries=structural["document_length_quartile_boundaries"],
        turn_position_quartile_boundaries=structural["dialogue_turn_position_quartile_boundaries"],
    )
    coverage = build_candidate_coverage(gold, census["sampling"])
    coverage.update({"gold_joined_after_complete_score_cache": True, "queries_sha256": sha256(paths["queries"]), "scored_sha256": sha256(paths["scored"]), "execution_registration_sha256": sha256(paths["execution"])})
    _write_json(paths["coverage"], coverage)
    artifacts = {
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
    report, evidence = evaluate_stage(gold, scored, query_summary, artifacts, stage=args.stage)
    write_report(report, evidence, paths["result"], paths["report"], paths["evidence"])
    first = {"json": sha256(paths["result"]), "markdown": sha256(paths["report"]), "evidence": sha256(paths["evidence"])}
    write_report(report, evidence, paths["result"], paths["report"], paths["evidence"])
    second = {"json": sha256(paths["result"]), "markdown": sha256(paths["report"]), "evidence": sha256(paths["evidence"])}
    if first != second:
        raise AssertionError("Doc2Dial wOOD v56 report is not deterministic")
    outcome = report["analysis"]["outcome"]
    print(json.dumps({"stage": args.stage, "status": outcome["status"], "support_established": outcome["support_established"], "confirmation_open_authorized": outcome["confirmation_open_authorized"], "checks": report["analysis"]["support_checks"], "output_hashes": second}, ensure_ascii=False, sort_keys=True))


def runtime(args: argparse.Namespace) -> None:
    print(json.dumps(_runtime_environment(args), ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Doc2Dial wOOD v56 schema-corrected transfer.")
    parser.add_argument("command", choices=("prepare", "register", "generate", "score", "evaluate", "runtime"))
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--protocol", type=Path, default=Path("docs/progressive_upgrade/doc2dial_wood_schema_corrected_transfer_protocol_v56.json"))
    parser.add_argument("--source-registration", type=Path, default=Path("docs/progressive_upgrade/doc2dial_wood_source_registration_v56.json"))
    parser.add_argument("--implementation", type=Path, default=Path("docs/progressive_upgrade/doc2dial_wood_schema_corrected_transfer_implementation_v56.json"))
    parser.add_argument("--source-archive", type=Path, default=Path(".cache/benchmarks/doc2dial/v55_source/doc2dial_v0.9.zip"))
    parser.add_argument("--cache-root", type=Path, default=Path(".cache/benchmarks/doc2dial/v56"))
    parser.add_argument("--execution-root", type=Path, default=Path("docs/progressive_upgrade"))
    parser.add_argument("--result-root", type=Path, default=Path("docs/progressive_upgrade"))
    parser.add_argument("--output-root", type=Path, default=Path("output/rag_evaluation/doc2dial_wood_schema_corrected_transfer"))
    parser.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--reranker-batch-size", type=int, default=256)
    parser.add_argument("--case-batch-size", type=int, default=2)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--registered-at", default="2026-08-03T00:45:00+08:00")
    args = parser.parse_args()
    for name in ("protocol", "source_registration", "implementation", "source_archive", "cache_root", "execution_root", "result_root", "output_root", "hf_home"):
        setattr(args, name, _resolve(getattr(args, name)))
    {"prepare": prepare, "register": register, "generate": generate, "score": score, "evaluate": evaluate, "runtime": runtime}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
