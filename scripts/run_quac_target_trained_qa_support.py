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

from research.frc_rag.quac_target_trained_qa_support import (  # noqa: E402
    BUDGETS,
    EXPERIMENT_ID,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    SOURCE_FILE,
    TARGET_CASES,
    FrozenQuacV63Scorer,
    LocalSciBertQuacSupportVerifier,
    build_candidate_coverage,
    build_deterministic_queries,
    build_gold_rows,
    build_support_gold_rows,
    evaluate_final_method,
    evaluate_support_open_gate,
    prepare_blind_cases,
    read_validation_source,
    select_balanced_validation_sample,
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


MODULE_PATH = REPO_ROOT / "research/frc_rag/quac_target_trained_qa_support.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_quac_target_trained_qa_support.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_quac_target_trained_qa_support.py"


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    cache = args.cache_root / "validation"
    support_output = args.output_root / "support_gate"
    final_output = args.output_root / "final"
    return {
        "prepared": cache / "prepared_blind.jsonl",
        "candidate_map": cache / "candidate_map.jsonl",
        "document_store": cache / "blind_document_store.jsonl",
        "qa_input": cache / "qa_input.jsonl",
        "census": cache / "blind_census.json",
        "support": cache / "qa_support_decisions.jsonl",
        "queries": cache / "queries.jsonl",
        "scored": cache / "scored.jsonl",
        "coverage": cache / "coverage.json",
        "execution": args.execution_root
        / "quac_target_trained_qa_support_validation_execution_v63.json",
        "support_result": args.result_root
        / "quac_target_trained_qa_support_validation_support_result_v63.json",
        "support_open": args.result_root
        / "quac_target_trained_qa_support_retrieval_open_v63.json",
        "support_closure": args.result_root
        / "quac_target_trained_qa_support_validation_support_closure_v63.json",
        "support_report": support_output / "report.md",
        "support_evidence": support_output / "cases.jsonl.gz",
        "final_result": args.result_root
        / "quac_target_trained_qa_support_validation_result_v63.json",
        "final_closure": args.result_root
        / "quac_target_trained_qa_support_validation_closure_v63.json",
        "final_report": final_output / "report.md",
        "final_evidence": final_output / "cases.jsonl.gz",
    }


def _runtime_environment(args: argparse.Namespace) -> dict[str, Any]:
    import numpy
    import torch
    import transformers

    gpu_name = None
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = int(torch.cuda.get_device_properties(0).total_memory / 1024**2)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu_name,
        "gpu_memory_mib": gpu_memory,
        "parameters": {
            "device": args.device,
            "qa_batch_size": args.qa_batch_size,
            "qa_max_length": 384,
            "qa_doc_stride": 128,
            "qa_maximum_answer_tokens": 30,
            "embedding_batch_size": args.embedding_batch_size,
            "reranker_batch_size": args.reranker_batch_size,
            "case_batch_size": args.case_batch_size,
            "fp16": not args.no_fp16,
            "token_budgets": list(BUDGETS),
            "learned_parameter": None,
            "offline_model_loading": True,
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


def _qa_cache_key(args: argparse.Namespace, execution: dict[str, Any]) -> str:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "validation",
        "model_registration_sha256": sha256(args.model_registration),
        "qa_input_sha256": execution["hashes"]["qa_input_sha256"],
        "max_length": 384,
        "doc_stride": 128,
        "maximum_answer_tokens": 30,
        "batch_size": args.qa_batch_size,
        "fp16": not args.no_fp16,
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
    value = json.loads(paths["execution"].read_text(encoding="utf-8"))
    expected = {
        "implementation_registration_sha256": sha256(args.implementation),
        "model_registration_sha256": sha256(args.model_registration),
        "stage_source_sha256": sha256(args.source_root / SOURCE_FILE),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "blind_document_store_sha256": sha256(paths["document_store"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "blind_census_sha256": sha256(paths["census"]),
    }
    if value.get("hashes") != expected:
        raise ValueError("QuAC v63 execution registration hash mismatch")
    if value.get("query_scoring_or_qa_started") is not False:
        raise ValueError("QuAC v63 execution was registered too late")
    if value.get("gold_joined_for_coverage_or_metrics") is not False:
        raise ValueError("QuAC v63 gold was joined before blind execution")
    return value


def _validate_support_open(paths: dict[str, Path]) -> dict[str, Any]:
    if not paths["support_result"].is_file() or not paths["support_open"].is_file():
        raise ValueError("QuAC v63 retrieval scoring is not authorized")
    result = json.loads(paths["support_result"].read_text(encoding="utf-8"))
    opened = json.loads(paths["support_open"].read_text(encoding="utf-8"))
    outcome = result["analysis"]["outcome"]
    if outcome["retrieval_scoring_open_authorized"] is not True:
        raise ValueError("QuAC v63 support gate failed")
    if opened["support_result_sha256"] != sha256(paths["support_result"]):
        raise ValueError("QuAC v63 support-open registration changed")
    return result


def model_manifest(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    if args.model_registration.exists():
        raise FileExistsError("QuAC v63 model registration already exists")
    files = {
        path.name: sha256(path)
        for path in sorted(args.qa_model.iterdir())
        if path.is_file()
    }
    required = {"config.json", "vocab.txt"}
    if not required.issubset(files) or not (
        {"model.safetensors", "pytorch_model.bin"} & files.keys()
    ):
        raise ValueError(f"QuAC v63 model is incomplete: {sorted(files)}")
    value = {
        "schema_version": "frc-quac-v63-model-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "registered_at": args.registered_at,
        "protocol_sha256": sha256(args.protocol),
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "license_on_model_card": "NOT_DECLARED",
        "usage_boundary": "research evaluation only; redistribution and production prohibited pending verification",
        "local_path": str(args.qa_model),
        "local_offline_loading_required": True,
        "local_finetuning_calibration_or_threshold_fitting": False,
        "exact_quac_training_split_and_checkpoint_selection_disclosed": False,
        "strict_independent_confirmation_claimed": False,
        "files": files,
    }
    _write_json(args.model_registration, value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    validate_model_registration(
        args.model_registration,
        protocol_path=args.protocol,
        model_root=args.qa_model,
    )
    paths = _paths(args)
    existing = [
        path
        for key, path in paths.items()
        if key in {"prepared", "candidate_map", "document_store", "qa_input", "census"}
        and path.exists()
    ]
    if existing:
        raise ValueError(
            "QuAC v63 preparation is frozen: " + ", ".join(map(str, existing))
        )
    source = read_validation_source(args.source_root)
    selected, sampling = select_balanced_validation_sample(source)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared_rows, maps, documents, qa_inputs, structural = prepare_blind_cases(
        selected,
        source,
        tokenizer,
    )
    write_jsonl(paths["prepared"], prepared_rows)
    write_jsonl(paths["candidate_map"], maps)
    write_jsonl(paths["document_store"], documents)
    write_jsonl(paths["qa_input"], qa_inputs)
    census = {
        "schema_version": "frc-quac-v63-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": "validation",
        "sampling": sampling,
        "structural_census": structural,
        "source_manifest": {
            "opened_source_file": SOURCE_FILE,
            "opened_source_sha256": sha256(args.source_root / SOURCE_FILE),
            "opened_only_after_protocol_implementation_and_model_registration": True,
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
    paths = _paths(args)
    if paths["execution"].exists():
        raise FileExistsError("QuAC v63 execution registration already exists")
    if any(
        paths[name].exists()
        for name in ("support", "queries", "scored", "support_result", "final_result")
    ):
        raise ValueError("QuAC v63 outcome-bearing work started before registration")
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    runtime = _runtime_environment(args)
    if args.device == "cuda" and runtime["cuda_available"] is not True:
        raise ValueError("QuAC v63 frozen CUDA runtime is unavailable")
    value = {
        "schema_version": "frc-quac-v63-execution-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": "validation",
        "registered_at": args.registered_at,
        "hashes": {
            "implementation_registration_sha256": sha256(args.implementation),
            "model_registration_sha256": sha256(args.model_registration),
            "stage_source_sha256": sha256(args.source_root / SOURCE_FILE),
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
        "other_source_content_read": False,
    }
    _write_json(paths["execution"], value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def verify(args: argparse.Namespace) -> None:
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    if paths["support"].exists():
        raise FileExistsError("QuAC v63 QA support cache already exists")
    if any(
        paths[name].exists()
        for name in ("support_result", "queries", "scored", "coverage")
    ):
        raise ValueError("QuAC v63 blind support inference order changed")
    qa_inputs = list(read_jsonl(paths["qa_input"]))
    cache_key = _qa_cache_key(args, execution)
    verifier = LocalSciBertQuacSupportVerifier(
        args.qa_model,
        device=args.device,
        batch_size=args.qa_batch_size,
        fp16=not args.no_fp16,
    )
    rows = verifier.verify(qa_inputs, cache_key=cache_key)
    write_jsonl(paths["support"], rows)
    summary = validate_support_cache(qa_inputs, rows, cache_key=cache_key)
    print(json.dumps({**summary, "sha256": sha256(paths["support"])}, sort_keys=True))


def support_evaluate(args: argparse.Namespace) -> None:
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    if paths["support_result"].exists():
        raise FileExistsError("QuAC v63 support result is already frozen")
    if any(paths[name].exists() for name in ("queries", "scored", "coverage")):
        raise ValueError("QuAC v63 retrieval work started before support gate")
    maps = list(read_jsonl(paths["candidate_map"]))
    qa_inputs = list(read_jsonl(paths["qa_input"]))
    support_rows = list(read_jsonl(paths["support"]))
    expected = [str(row["id"]) for row in maps]
    if len(expected) != TARGET_CASES:
        raise ValueError("QuAC v63 candidate map is incomplete")
    if [str(row["id"]) for row in qa_inputs] != expected or [
        str(row["id"]) for row in support_rows
    ] != expected:
        raise ValueError("QuAC v63 QA cache is incomplete before gold join")
    verifier_summary = validate_support_cache(
        qa_inputs,
        support_rows,
        cache_key=_qa_cache_key(args, execution),
    )
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    source = read_validation_source(args.source_root)
    support_gold = build_support_gold_rows(source, maps, support_rows)
    source_artifacts = {
        "protocol_sha256": sha256(args.protocol),
        "source_registration_sha256": sha256(args.source_registration),
        "implementation_registration_sha256": sha256(args.implementation),
        "model_registration_sha256": sha256(args.model_registration),
        "execution_registration_sha256": sha256(paths["execution"]),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "qa_support_cache_sha256": sha256(paths["support"]),
        "sampling": census["sampling"],
        "gold_joined_after_complete_qa_cache": True,
        "query_or_retrieval_scoring_started": False,
    }
    report, evidence = evaluate_support_open_gate(
        support_gold,
        verifier_summary,
        census["sampling"],
        source_artifacts,
    )
    write_report(
        report,
        evidence,
        paths["support_result"],
        paths["support_report"],
        paths["support_evidence"],
        title="QuAC target-trained QA support-open gate (validation, v63)",
    )
    outcome = report["analysis"]["outcome"]
    record = {
        "experiment_id": EXPERIMENT_ID,
        "status": outcome["status"],
        "support_result_sha256": sha256(paths["support_result"]),
        "retrieval_scoring_open_authorized": outcome[
            "retrieval_scoring_open_authorized"
        ],
        "selector_adoption_authorized": False,
        "canary_or_default_authorized": False,
        "gate_2": "NO-GO/SHADOW",
    }
    if outcome["retrieval_scoring_open_authorized"]:
        record["schema_version"] = "frc-quac-v63-retrieval-open-v1"
        _write_json(paths["support_open"], record)
    else:
        record["schema_version"] = "frc-quac-v63-support-closure-v1"
        record["query_generation_started"] = False
        record["retrieval_scoring_started"] = False
        _write_json(paths["support_closure"], record)
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))


def queries(args: argparse.Namespace) -> None:
    paths = _paths(args)
    _validate_execution(args, paths)
    _validate_support_open(paths)
    if any(paths[name].exists() for name in ("queries", "scored", "coverage")):
        raise ValueError("QuAC v63 query cache order changed")
    prepared = list(read_jsonl(paths["prepared"]))
    rows = build_deterministic_queries(prepared)
    summary = validate_query_cache(prepared, rows)
    write_jsonl(paths["queries"], rows)
    print(json.dumps({**summary, "sha256": sha256(paths["queries"])}, sort_keys=True))


def score(args: argparse.Namespace) -> None:
    paths = _paths(args)
    _validate_execution(args, paths)
    _validate_support_open(paths)
    if paths["coverage"].exists():
        raise ValueError("QuAC v63 full gold joined before retrieval scoring")
    prepared = list(read_jsonl(paths["prepared"]))
    query_rows = list(read_jsonl(paths["queries"]))
    documents = list(read_jsonl(paths["document_store"]))
    validate_query_cache(prepared, query_rows)
    scorer = FrozenQuacV63Scorer(
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
    print(
        json.dumps(
            {"rows": len(rows), "sha256": sha256(paths["scored"])}, sort_keys=True
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    _validate_support_open(paths)
    if paths["final_result"].exists():
        raise FileExistsError("QuAC v63 final result is already frozen")
    prepared = list(read_jsonl(paths["prepared"]))
    maps = list(read_jsonl(paths["candidate_map"]))
    query_rows = list(read_jsonl(paths["queries"]))
    scored_rows = list(read_jsonl(paths["scored"]))
    qa_inputs = list(read_jsonl(paths["qa_input"]))
    support_rows = list(read_jsonl(paths["support"]))
    expected = [str(row["id"]) for row in prepared]
    for name, rows in (
        ("candidate map", maps),
        ("queries", query_rows),
        ("scores", scored_rows),
        ("QA inputs", qa_inputs),
        ("QA decisions", support_rows),
    ):
        if [str(row["id"]) for row in rows] != expected:
            raise ValueError(
                f"QuAC v63 {name} cache is incomplete before final gold join"
            )
    query_summary = validate_query_cache(prepared, query_rows)
    verifier_summary = validate_support_cache(
        qa_inputs,
        support_rows,
        cache_key=_qa_cache_key(args, execution),
    )
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    structural = census["structural_census"]
    source = read_validation_source(args.source_root)
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
        "support_result_sha256": sha256(paths["support_result"]),
        "retrieval_open_sha256": sha256(paths["support_open"]),
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
    report, evidence = evaluate_final_method(
        gold_rows,
        scored_rows,
        support_rows,
        query_summary,
        verifier_summary,
        source_artifacts,
    )
    write_report(
        report,
        evidence,
        paths["final_result"],
        paths["final_report"],
        paths["final_evidence"],
        title="QuAC target-trained QA support and selector evaluation (validation, v63)",
    )
    outcome = report["analysis"]["outcome"]
    _write_json(
        paths["final_closure"],
        {
            "schema_version": "frc-quac-v63-validation-closure-v1",
            "experiment_id": EXPERIMENT_ID,
            "status": outcome["status"],
            "validation_result_sha256": sha256(paths["final_result"]),
            "training_provenance_limited": True,
            "strict_independent_confirmation_claimed": False,
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
            "verify",
            "support-evaluate",
            "queries",
            "score",
            "evaluate",
            "runtime",
        ),
    )
    result.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/quac_target_trained_qa_support_protocol_v63.json"
        ),
    )
    result.add_argument(
        "--source-registration",
        type=Path,
        default=Path("docs/progressive_upgrade/quac_source_registration_v57.json"),
    )
    result.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/quac_target_trained_qa_support_implementation_v63.json"
        ),
    )
    result.add_argument(
        "--model-registration",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/quac_target_trained_qa_support_model_registration_v63.json"
        ),
    )
    result.add_argument(
        "--source-root", type=Path, default=Path(".cache/benchmarks/quac/v57_source")
    )
    result.add_argument(
        "--cache-root", type=Path, default=Path(".cache/benchmarks/quac/v63")
    )
    result.add_argument(
        "--execution-root", type=Path, default=Path("docs/progressive_upgrade")
    )
    result.add_argument(
        "--result-root", type=Path, default=Path("docs/progressive_upgrade")
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/rag_evaluation/quac_target_trained_qa_support"),
    )
    result.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    result.add_argument(
        "--qa-model",
        type=Path,
        default=Path(".cache/benchmarks/models/ixa_ehu_scibert_squad_quac_8d44c18"),
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--embedding-batch-size", type=int, default=64)
    result.add_argument("--reranker-batch-size", type=int, default=256)
    result.add_argument("--case-batch-size", type=int, default=2)
    result.add_argument("--qa-batch-size", type=int, default=16)
    result.add_argument("--no-fp16", action="store_true")
    result.add_argument("--registered-at", default="2026-08-03T22:00:00+08:00")
    return result


def main() -> int:
    args = parser().parse_args()
    for name in (
        "protocol",
        "source_registration",
        "implementation",
        "model_registration",
        "source_root",
        "cache_root",
        "execution_root",
        "result_root",
        "output_root",
        "hf_home",
        "qa_model",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    actions = {
        "model-manifest": model_manifest,
        "prepare": prepare,
        "register": register,
        "verify": verify,
        "support-evaluate": support_evaluate,
        "queries": queries,
        "score": score,
        "evaluate": evaluate,
        "runtime": runtime,
    }
    actions[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
