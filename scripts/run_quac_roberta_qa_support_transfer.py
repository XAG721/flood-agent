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

from research.frc_rag.quac_roberta_qa_support_transfer import (  # noqa: E402
    BUDGETS,
    EXPERIMENT_ID,
    MODEL_REVISION,
    SOURCE_FILES,
    TARGET_CASES,
    FrozenQuacV62Scorer,
    LocalRobertaQASupportVerifier,
    build_candidate_coverage,
    build_deterministic_queries,
    build_gold_rows,
    evaluate_stage,
    load_v57_excluded_commitments,
    prepare_blind_cases,
    read_stage_source,
    select_disjoint_balanced_sample,
    validate_implementation_registration,
    validate_model_registration,
    validate_protocol,
    validate_query_cache,
    validate_source_contract,
    validate_support_cache,
    write_report,
)
from research.frc_rag.rgb_cost_aware_frc import (  # noqa: E402
    load_frozen_tokenizer,
    read_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)


MODULE_PATH = REPO_ROOT / "research/frc_rag/quac_roberta_qa_support_transfer.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_quac_roberta_qa_support_transfer.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_quac_roberta_qa_support_transfer.py"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _stage_paths(args: argparse.Namespace) -> dict[str, Path]:
    cache = args.cache_root / args.stage
    output = args.output_root / args.stage
    return {
        "prepared": cache / "prepared_blind.jsonl",
        "candidate_map": cache / "candidate_map.jsonl",
        "document_store": cache / "blind_document_store.jsonl",
        "qa_input": cache / "qa_input.jsonl",
        "census": cache / "blind_census.json",
        "queries": cache / "queries.jsonl",
        "scored": cache / "scored.jsonl",
        "support": cache / "qa_support_decisions.jsonl",
        "coverage": cache / "coverage.json",
        "execution": args.execution_root
        / f"quac_roberta_qa_support_transfer_{args.stage}_execution_v62.json",
        "result": args.result_root
        / f"quac_roberta_qa_support_transfer_{args.stage}_result_v62.json",
        "closure": args.result_root
        / f"quac_roberta_qa_support_transfer_{args.stage}_closure_v62.json",
        "report": output / "report.md",
        "evidence": output / "cases.jsonl.gz",
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
            "qa_batch_size": args.qa_batch_size,
            "qa_max_length": 384,
            "qa_doc_stride": 128,
            "qa_maximum_answer_tokens": 30,
            "fp16": not args.no_fp16,
            "offline_model_loading": True,
            "query_generator": "deterministic_templates",
            "token_budgets": list(BUDGETS),
            "learned_parameter": None,
        },
    }


def _validate_implementation(args: argparse.Namespace) -> dict[str, Any]:
    validate_protocol(args.protocol)
    validate_source_contract(
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


def _validate_confirmation_open(args: argparse.Namespace) -> None:
    if args.stage != "confirmation":
        return
    development = args.result_root / (
        "quac_roberta_qa_support_transfer_development_result_v62.json"
    )
    if not development.is_file():
        raise ValueError("QuAC v62 validation cannot open before development result")
    result = json.loads(development.read_text(encoding="utf-8"))
    outcome = result["analysis"]["outcome"]
    if (
        outcome["support_established"] is not True
        or outcome["confirmation_open_authorized"] is not True
    ):
        raise ValueError("QuAC v62 development gate did not authorize validation")


def _qa_cache_key(args: argparse.Namespace, execution: dict[str, Any]) -> str:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "model_registration_sha256": sha256(args.model_registration),
        "qa_input_sha256": execution["hashes"]["qa_input_sha256"],
        "max_length": 384,
        "doc_stride": 128,
        "maximum_answer_tokens": 30,
        "support_rule": "best_span_score_strictly_greater_than_minimum_cls_null_score",
    }
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _validate_execution(
    args: argparse.Namespace, paths: dict[str, Path]
) -> dict[str, Any]:
    _validate_implementation(args)
    validate_model_registration(
        args.model_registration,
        protocol_path=args.protocol,
        model_root=args.qa_model,
    )
    value = json.loads(paths["execution"].read_text(encoding="utf-8"))
    expected = {
        "implementation_registration_sha256": sha256(args.implementation),
        "model_registration_sha256": sha256(args.model_registration),
        "stage_source_sha256": sha256(args.source_root / SOURCE_FILES[args.stage]),
        "v57_exclusion_map_sha256": sha256(args.v57_exclusion_map),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "blind_census_sha256": sha256(paths["census"]),
    }
    if value.get("hashes") != expected:
        raise ValueError("QuAC v62 execution registration hash mismatch")
    if value.get("query_scoring_or_qa_started") is not False:
        raise ValueError("QuAC v62 execution was registered too late")
    if value.get("gold_joined_for_coverage_or_metrics") is not False:
        raise ValueError("QuAC v62 gold was joined before blind execution")
    return value


def model_manifest(args: argparse.Namespace) -> None:
    validate_protocol(args.protocol)
    if args.model_registration.exists():
        raise FileExistsError("QuAC v62 model registration already exists")
    files = {
        path.name: sha256(path)
        for path in sorted(args.qa_model.iterdir())
        if path.is_file()
    }
    required = {"config.json", "model.safetensors", "tokenizer.json"}
    if not required <= set(files):
        raise ValueError(f"QuAC v62 model is incomplete: {sorted(required - set(files))}")
    value = {
        "schema_version": "frc-quac-v62-model-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "registered_at": args.registered_at,
        "protocol_sha256": sha256(args.protocol),
        "repository": "deepset/roberta-base-squad2",
        "revision": MODEL_REVISION,
        "license": "CC-BY-4.0",
        "local_path": str(args.qa_model),
        "files": files,
        "quac_content_used_for_model_training_or_finetuning": False,
        "local_offline_loading_required": True,
    }
    _write_json(args.model_registration, value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _validate_confirmation_open(args)
    paths = _stage_paths(args)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise ValueError(
            "QuAC v62 preparation is frozen: "
            + ", ".join(str(path) for path in existing)
        )
    source = read_stage_source(args.source_root, args.stage)
    excluded = load_v57_excluded_commitments(args.v57_exclusion_map)
    selected, sampling = select_disjoint_balanced_sample(
        source,
        stage=args.stage,
        excluded_commitments=excluded,
    )
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared_rows, maps, documents, qa_inputs, structural = prepare_blind_cases(
        selected,
        source,
        tokenizer,
        stage=args.stage,
    )
    write_jsonl(paths["prepared"], prepared_rows)
    write_jsonl(paths["candidate_map"], maps)
    write_jsonl(paths["document_store"], documents)
    write_jsonl(paths["qa_input"], qa_inputs)
    census = {
        "schema_version": "frc-quac-v62-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "sampling": sampling,
        "structural_census": structural,
        "source_manifest": {
            "opened_source_file": SOURCE_FILES[args.stage],
            "opened_source_sha256": sha256(
                args.source_root / SOURCE_FILES[args.stage]
            ),
            "other_stage_source_file_opened": False,
        },
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "gold_joined_for_coverage_or_metrics": False,
        "query_scoring_or_qa_started": False,
        "metrics_computed": False,
    }
    _write_json(paths["census"], census)
    print(json.dumps(census, ensure_ascii=False, sort_keys=True))


def register(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    validate_model_registration(
        args.model_registration,
        protocol_path=args.protocol,
        model_root=args.qa_model,
    )
    _validate_confirmation_open(args)
    paths = _stage_paths(args)
    if paths["execution"].exists():
        raise FileExistsError("QuAC v62 execution registration already exists")
    if any(paths[name].exists() for name in ("queries", "scored", "support", "result")):
        raise ValueError("QuAC v62 outcome-bearing work started before registration")
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    runtime = _runtime_environment(args)
    if args.device == "cuda" and runtime["cuda_available"] is not True:
        raise ValueError("QuAC v62 frozen CUDA runtime is unavailable")
    value = {
        "schema_version": "frc-quac-v62-execution-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "registered_at": args.registered_at,
        "hashes": {
            "implementation_registration_sha256": sha256(args.implementation),
            "model_registration_sha256": sha256(args.model_registration),
            "stage_source_sha256": sha256(
                args.source_root / SOURCE_FILES[args.stage]
            ),
            "v57_exclusion_map_sha256": sha256(args.v57_exclusion_map),
            "prepared_blind_sha256": sha256(paths["prepared"]),
            "candidate_map_sha256": sha256(paths["candidate_map"]),
            "blind_document_store_sha256": sha256(paths["document_store"]),
            "qa_input_sha256": sha256(paths["qa_input"]),
            "blind_census_sha256": sha256(paths["census"]),
        },
        "sampling": census["sampling"],
        "runtime": runtime,
        "query_scoring_or_qa_started": False,
        "gold_joined_for_coverage_or_metrics": False,
        "other_stage_source_content_read": False,
    }
    _write_json(paths["execution"], value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def queries(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    _validate_execution(args, paths)
    if any(paths[name].exists() for name in ("scored", "support", "coverage")):
        raise ValueError("QuAC v62 blind caches started before queries")
    prepared = list(read_jsonl(paths["prepared"]))
    rows = build_deterministic_queries(prepared)
    summary = validate_query_cache(prepared, rows)
    write_jsonl(paths["queries"], rows)
    print(json.dumps({**summary, "sha256": sha256(paths["queries"])}, sort_keys=True))


def score(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    _validate_execution(args, paths)
    if paths["coverage"].exists():
        raise ValueError("QuAC v62 gold was joined before scoring")
    prepared = list(read_jsonl(paths["prepared"]))
    query_rows = list(read_jsonl(paths["queries"]))
    documents = list(read_jsonl(paths["document_store"]))
    validate_query_cache(prepared, query_rows)
    scorer = FrozenQuacV62Scorer(
        query_rows=query_rows,
        document_store=documents,
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        use_fp16=not args.no_fp16,
    )
    rows = score_cases_resumable(
        prepared,
        scorer,
        paths["scored"],
        case_batch_size=args.case_batch_size,
    )
    print(json.dumps({"rows": len(rows), "sha256": sha256(paths["scored"])}, sort_keys=True))


def verify(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    execution = _validate_execution(args, paths)
    if paths["support"].exists():
        raise FileExistsError("QuAC v62 QA support cache already exists")
    if paths["coverage"].exists():
        raise ValueError("QuAC v62 gold was joined before QA support inference")
    qa_inputs = list(read_jsonl(paths["qa_input"]))
    cache_key = _qa_cache_key(args, execution)
    verifier = LocalRobertaQASupportVerifier(
        args.qa_model,
        device=args.device,
        batch_size=args.qa_batch_size,
        fp16=not args.no_fp16,
    )
    rows = verifier.verify(qa_inputs, cache_key=cache_key)
    write_jsonl(paths["support"], rows)
    summary = validate_support_cache(qa_inputs, rows, cache_key=cache_key)
    print(json.dumps({**summary, "sha256": sha256(paths["support"])}, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    execution = _validate_execution(args, paths)
    if paths["result"].exists() or paths["report"].exists():
        raise ValueError("QuAC v62 result is already frozen")
    prepared = list(read_jsonl(paths["prepared"]))
    maps = list(read_jsonl(paths["candidate_map"]))
    query_rows = list(read_jsonl(paths["queries"]))
    scored_rows = list(read_jsonl(paths["scored"]))
    qa_inputs = list(read_jsonl(paths["qa_input"]))
    support_rows = list(read_jsonl(paths["support"]))
    expected = [str(row["id"]) for row in prepared]
    if len(expected) != TARGET_CASES:
        raise ValueError("QuAC v62 blind cache is incomplete")
    for name, rows in (
        ("candidate map", maps),
        ("queries", query_rows),
        ("scores", scored_rows),
        ("QA inputs", qa_inputs),
        ("QA decisions", support_rows),
    ):
        if [str(row["id"]) for row in rows] != expected:
            raise ValueError(f"QuAC v62 {name} cache is incomplete before gold join")
    query_summary = validate_query_cache(prepared, query_rows)
    verifier_summary = validate_support_cache(
        qa_inputs,
        support_rows,
        cache_key=_qa_cache_key(args, execution),
    )
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    structural = census["structural_census"]
    source = read_stage_source(args.source_root, args.stage)
    gold_rows = build_gold_rows(
        source,
        maps,
        scored_rows,
        document_length_quartile_boundaries=structural[
            "document_length_quartile_boundaries"
        ],
        turn_position_quartile_boundaries=structural[
            "dialogue_turn_position_quartile_boundaries"
        ],
    )
    coverage = build_candidate_coverage(gold_rows, census["sampling"])
    coverage["gold_joined_after_complete_score_and_qa_caches"] = True
    _write_json(paths["coverage"], coverage)
    source_artifacts = {
        "protocol_sha256": sha256(args.protocol),
        "source_registration_sha256": sha256(args.source_registration),
        "implementation_registration_sha256": sha256(args.implementation),
        "model_registration_sha256": sha256(args.model_registration),
        "execution_registration_sha256": sha256(paths["execution"]),
        "v57_exclusion_map_sha256": sha256(args.v57_exclusion_map),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "query_cache_sha256": sha256(paths["queries"]),
        "score_cache_sha256": sha256(paths["scored"]),
        "qa_support_cache_sha256": sha256(paths["support"]),
        "coverage_sha256": sha256(paths["coverage"]),
        "sampling": census["sampling"],
        "structural_census": structural,
        "score_fallback_rate": 0.0,
    }
    report, evidence = evaluate_stage(
        gold_rows,
        scored_rows,
        support_rows,
        query_summary,
        verifier_summary,
        source_artifacts,
        stage=args.stage,
    )
    write_report(report, evidence, paths["result"], paths["report"], paths["evidence"])
    outcome = report["analysis"]["outcome"]
    if args.stage == "development" and not outcome["support_established"]:
        _write_json(
            paths["closure"],
            {
                "schema_version": "frc-quac-v62-development-closure-v1",
                "experiment_id": EXPERIMENT_ID,
                "status": outcome["status"],
                "development_result_sha256": sha256(paths["result"]),
                "quac_validation_content_opened_or_parsed": False,
                "confirmation_execution_started": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        )
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))


def runtime(args: argparse.Namespace) -> None:
    print(json.dumps(_runtime_environment(args), ensure_ascii=False, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "command",
        choices=(
            "model-manifest",
            "prepare",
            "register",
            "queries",
            "score",
            "verify",
            "evaluate",
            "runtime",
        ),
    )
    result.add_argument("--stage", choices=("development", "confirmation"), default="development")
    result.add_argument(
        "--protocol",
        type=Path,
        default=Path("docs/progressive_upgrade/quac_roberta_qa_support_transfer_protocol_v62.json"),
    )
    result.add_argument(
        "--source-registration",
        type=Path,
        default=Path("docs/progressive_upgrade/quac_source_registration_v57.json"),
    )
    result.add_argument(
        "--implementation",
        type=Path,
        default=Path("docs/progressive_upgrade/quac_roberta_qa_support_transfer_implementation_v62.json"),
    )
    result.add_argument(
        "--model-registration",
        type=Path,
        default=Path("docs/progressive_upgrade/quac_roberta_qa_support_transfer_model_registration_v62.json"),
    )
    result.add_argument(
        "--source-root", type=Path, default=Path(".cache/benchmarks/quac/v57_source")
    )
    result.add_argument(
        "--v57-exclusion-map",
        type=Path,
        default=Path(".cache/benchmarks/quac/v57/development/candidate_map.jsonl"),
    )
    result.add_argument("--cache-root", type=Path, default=Path(".cache/benchmarks/quac/v62"))
    result.add_argument("--execution-root", type=Path, default=Path("docs/progressive_upgrade"))
    result.add_argument("--result-root", type=Path, default=Path("docs/progressive_upgrade"))
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/rag_evaluation/quac_roberta_qa_support_transfer"),
    )
    result.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    result.add_argument(
        "--qa-model",
        type=Path,
        default=Path(".cache/benchmarks/models/deepset_roberta_base_squad2_adc3b06"),
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--embedding-batch-size", type=int, default=64)
    result.add_argument("--reranker-batch-size", type=int, default=256)
    result.add_argument("--case-batch-size", type=int, default=2)
    result.add_argument("--qa-batch-size", type=int, default=16)
    result.add_argument("--no-fp16", action="store_true")
    result.add_argument("--registered-at", default="2026-08-03T19:45:00+08:00")
    return result


def main() -> int:
    args = parser().parse_args()
    for name in (
        "protocol",
        "source_registration",
        "implementation",
        "model_registration",
        "source_root",
        "v57_exclusion_map",
        "cache_root",
        "execution_root",
        "result_root",
        "output_root",
        "hf_home",
        "qa_model",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    commands = {
        "model-manifest": model_manifest,
        "prepare": prepare,
        "register": register,
        "queries": queries,
        "score": score,
        "verify": verify,
        "evaluate": evaluate,
        "runtime": runtime,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
