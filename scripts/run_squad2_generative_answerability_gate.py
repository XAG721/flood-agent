from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.rgb_cost_aware_frc import (  # noqa: E402
    load_frozen_tokenizer,
    read_jsonl,
    score_cases_resumable,
    sha256,
    write_jsonl,
)
from research.frc_rag.squad2_generative_answerability_gate import (  # noqa: E402
    BUDGETS,
    EXPERIMENT_ID,
    SOURCE_FILES,
    STAGES,
    TARGET_CASES,
    FrozenSquad2Scorer,
    LocalQwenSupportVerifier,
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
    validate_verifier_cache,
    write_report,
)


MODULE_PATH = REPO_ROOT / "research/frc_rag/squad2_generative_answerability_gate.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_squad2_generative_answerability_gate.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_squad2_generative_answerability_gate.py"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _model_hashes(model_path: Path) -> dict[str, str]:
    protocol = validate_protocol(DEFAULT_PROTOCOL)
    expected = protocol["blind_retrieval_and_models"][
        "support_verifier_model_file_sha256"
    ]
    actual: dict[str, str] = {}
    for name, expected_hash in expected.items():
        path = model_path / name
        if not path.is_file():
            raise ValueError(f"SQuAD2 v58 local verifier file is missing: {name}")
        actual[name] = sha256(path)
        if actual[name] != expected_hash:
            raise ValueError(f"SQuAD2 v58 local verifier file changed: {name}")
    return actual


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
        "dataset_version": "SQuAD 2.0",
        "license": "CC BY-SA 4.0 as stated on the official SQuAD homepage",
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
        "query_generator": "deterministic_templates",
        "token_budgets": list(BUDGETS),
        "learned_parameter": None,
        "support_verifier_batch_size": args.verifier_batch_size,
        "support_verifier_max_new_tokens": 4,
        "support_verifier_maximum_input_tokens": 3072,
        "support_verifier_do_sample": False,
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
        "verifier_input": cache / "verifier_input.jsonl",
        "census": cache / "blind_census.json",
        "queries": cache / "queries.jsonl",
        "scored": cache / "scored.jsonl",
        "support": cache / "support_decisions.jsonl",
        "coverage": cache / "coverage.json",
        "execution": args.execution_root
        / f"squad2_generative_answerability_gate_{args.stage}_execution_v58.json",
        "result": args.result_root
        / f"squad2_generative_answerability_gate_{args.stage}_result_v58.json",
        "closure": args.result_root
        / f"squad2_generative_answerability_gate_{args.stage}_closure_v58.json",
        "report": output / "report.md",
        "evidence": output / "cases.jsonl.gz",
    }


def _validate_confirmation_open(args: argparse.Namespace) -> None:
    if args.stage != "confirmation":
        return
    development = args.result_root / (
        "squad2_generative_answerability_gate_development_result_v58.json"
    )
    if not development.is_file():
        raise ValueError("SQuAD2 v58 confirmation cannot open before development")
    value = json.loads(development.read_text(encoding="utf-8"))
    outcome = value["analysis"]["outcome"]
    if (
        outcome["support_established"] is not True
        or outcome["confirmation_open_authorized"] is not True
    ):
        raise ValueError("SQuAD2 v58 development gate did not authorize confirmation")


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
        "verifier_input_sha256": sha256(paths["verifier_input"]),
        "blind_census_sha256": sha256(paths["census"]),
    }
    if value.get("hashes") != expected:
        raise ValueError("SQuAD2 v58 execution registration hash mismatch")
    if value.get("stage") != args.stage:
        raise ValueError("SQuAD2 v58 execution stage changed")
    if value.get("query_scoring_or_generation_started") is not False:
        raise ValueError("SQuAD2 v58 execution was registered too late")
    if value.get("gold_joined_for_coverage_or_metrics") is not False:
        raise ValueError("SQuAD2 v58 gold was joined before registration")
    return value


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _validate_confirmation_open(args)
    paths = _stage_paths(args)
    protected = [
        paths[name]
        for name in (
            "prepared",
            "candidate_map",
            "verifier_input",
            "census",
            "execution",
            "result",
            "closure",
        )
        if paths[name].exists()
    ]
    if protected:
        raise ValueError(
            "SQuAD2 v58 stage preparation is frozen: "
            + ", ".join(str(path) for path in protected)
        )
    source = read_stage_source(args.source_root, args.stage)
    selected, sampling = select_balanced_sample(source, stage=args.stage)
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared_rows, maps, verifier_inputs, structural = prepare_blind_cases(
        selected, tokenizer, stage=args.stage
    )
    write_jsonl(paths["prepared"], prepared_rows)
    write_jsonl(paths["candidate_map"], maps)
    write_jsonl(paths["verifier_input"], verifier_inputs)
    census = {
        "schema_version": "frc-squad2-v58-blind-census-v1",
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
        "verifier_input_sha256": sha256(paths["verifier_input"]),
        "is_impossible_answers_plausible_answers_or_raw_ids_exported_to_blind_cache": False,
        "gold_joined_for_coverage_or_metrics": False,
        "query_scoring_or_generation_started": False,
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
                    "verifier_input": sha256(paths["verifier_input"]),
                    "census": sha256(paths["census"]),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def register(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    _validate_confirmation_open(args)
    paths = _stage_paths(args)
    if paths["execution"].exists():
        raise ValueError("SQuAD2 v58 execution registration already exists")
    for name in ("prepared", "candidate_map", "verifier_input", "census"):
        if not paths[name].is_file():
            raise ValueError(f"SQuAD2 v58 preparation artifact missing: {name}")
    forbidden_existing = [
        paths[name]
        for name in ("queries", "scored", "support", "coverage", "result")
        if paths[name].exists()
    ]
    if forbidden_existing:
        raise ValueError("SQuAD2 v58 execution registration is too late")
    model_hashes = _model_hashes(args.verifier_model)
    value = {
        "schema_version": "frc-squad2-v58-execution-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "target_cases": TARGET_CASES,
        "hashes": {
            "implementation_registration_sha256": sha256(args.implementation),
            "stage_source_sha256": sha256(args.source_root / SOURCE_FILES[args.stage]),
            "prepared_blind_sha256": sha256(paths["prepared"]),
            "candidate_map_sha256": sha256(paths["candidate_map"]),
            "verifier_input_sha256": sha256(paths["verifier_input"]),
            "blind_census_sha256": sha256(paths["census"]),
        },
        "local_support_verifier": {
            "path": str(args.verifier_model),
            "file_sha256": model_hashes,
        },
        "runtime": _runtime_environment(args),
        "query_scoring_or_generation_started": False,
        "gold_joined_for_coverage_or_metrics": False,
        "other_stage_source_content_read": False,
    }
    _write_json(paths["execution"], value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def queries(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    _validate_execution(args, paths)
    if paths["queries"].exists():
        raise ValueError("SQuAD2 v58 query cache already exists")
    prepared = list(read_jsonl(paths["prepared"]))
    rows = build_deterministic_queries(prepared)
    write_jsonl(paths["queries"], rows)
    print(json.dumps(validate_query_cache(prepared, rows), sort_keys=True))


def score(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    _validate_execution(args, paths)
    prepared = list(read_jsonl(paths["prepared"]))
    query_rows = list(read_jsonl(paths["queries"]))
    validate_query_cache(prepared, query_rows)
    scorer = FrozenSquad2Scorer(
        query_rows=query_rows,
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        use_fp16=not args.no_fp16,
    )
    rows = score_cases_resumable(
        prepared, scorer, paths["scored"], case_batch_size=args.case_batch_size
    )
    print(
        json.dumps(
            {"rows": len(rows), "sha256": sha256(paths["scored"])},
            sort_keys=True,
        )
    )


def _verifier_cache_key(
    args: argparse.Namespace, paths: dict[str, Path], execution: dict[str, Any]
) -> str:
    value = {
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "verifier_input_sha256": sha256(paths["verifier_input"]),
        "model_file_sha256": execution["local_support_verifier"]["file_sha256"],
        "generation": {
            "batch_size": args.verifier_batch_size,
            "max_new_tokens": 4,
            "maximum_input_tokens": 3072,
            "do_sample": False,
        },
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    execution = _validate_execution(args, paths)
    verifier_inputs = list(read_jsonl(paths["verifier_input"]))
    cache_key = _verifier_cache_key(args, paths, execution)
    verifier = LocalQwenSupportVerifier(
        model_path=args.verifier_model,
        batch_size=args.verifier_batch_size,
        max_new_tokens=4,
        maximum_input_tokens=3072,
    )
    rows = verifier.verify(
        verifier_inputs, output_path=paths["support"], cache_key=cache_key
    )
    print(
        json.dumps(
            {
                **validate_verifier_cache(verifier_inputs, rows, cache_key=cache_key),
                "sha256": sha256(paths["support"]),
            },
            sort_keys=True,
        )
    )


def evaluate(args: argparse.Namespace) -> None:
    paths = _stage_paths(args)
    execution = _validate_execution(args, paths)
    if paths["result"].exists() or paths["report"].exists():
        raise ValueError("SQuAD2 v58 result is already frozen")
    prepared = list(read_jsonl(paths["prepared"]))
    query_rows = list(read_jsonl(paths["queries"]))
    scored_rows = list(read_jsonl(paths["scored"]))
    verifier_inputs = list(read_jsonl(paths["verifier_input"]))
    support_rows = list(read_jsonl(paths["support"]))
    if len(scored_rows) != len(prepared):
        raise ValueError("SQuAD2 v58 scoring is incomplete")
    query_summary = validate_query_cache(prepared, query_rows)
    cache_key = _verifier_cache_key(args, paths, execution)
    verifier_summary = validate_verifier_cache(
        verifier_inputs, support_rows, cache_key=cache_key
    )
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    source = read_stage_source(args.source_root, args.stage)
    candidate_maps = list(read_jsonl(paths["candidate_map"]))
    gold_rows = build_gold_rows(
        source,
        candidate_maps,
        scored_rows,
        structural_census=census["structural_census"],
    )
    coverage = build_candidate_coverage(gold_rows, census["sampling"])
    _write_json(paths["coverage"], coverage)
    source_artifacts = {
        "protocol_sha256": sha256(args.protocol),
        "source_registration_sha256": sha256(args.source_registration),
        "implementation_registration_sha256": sha256(args.implementation),
        "execution_registration_sha256": sha256(paths["execution"]),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "verifier_input_sha256": sha256(paths["verifier_input"]),
        "query_cache_sha256": sha256(paths["queries"]),
        "score_cache_sha256": sha256(paths["scored"]),
        "support_decision_cache_sha256": sha256(paths["support"]),
        "coverage_sha256": sha256(paths["coverage"]),
        "sampling": census["sampling"],
        "structural_census": census["structural_census"],
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
    write_report(
        report,
        evidence,
        paths["result"],
        paths["report"],
        paths["evidence"],
    )
    outcome = report["analysis"]["outcome"]
    if args.stage == "development" and not outcome["support_established"]:
        closure = {
            "schema_version": "frc-squad2-v58-development-closure-v1",
            "experiment_id": EXPERIMENT_ID,
            "status": outcome["status"],
            "development_result_sha256": sha256(paths["result"]),
            "confirmation_source_parsed_or_opened": False,
            "confirmation_execution_started": False,
            "selector_adoption_authorized": False,
            "canary_or_default_authorized": False,
            "gate_2": "NO-GO/SHADOW",
            "reason": "At least one prospectively registered development gate failed; confirmation remains sealed.",
        }
        _write_json(paths["closure"], closure)
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))


DEFAULT_PROTOCOL = (
    REPO_ROOT
    / "docs/progressive_upgrade/squad2_generative_answerability_gate_protocol_v58.json"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "command",
        choices=("prepare", "register", "queries", "score", "verify", "evaluate"),
    )
    result.add_argument("--stage", choices=STAGES, default="development")
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument(
        "--source-registration",
        type=Path,
        default=REPO_ROOT
        / "docs/progressive_upgrade/squad2_source_registration_v58.json",
    )
    result.add_argument(
        "--implementation",
        type=Path,
        default=REPO_ROOT
        / "docs/progressive_upgrade/squad2_generative_answerability_gate_implementation_v58.json",
    )
    result.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / ".cache/benchmarks/squad2_v2",
    )
    result.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT
        / ".cache/benchmarks/squad2_generative_answerability_gate_v58",
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "output/rag_evaluation/squad2_generative_answerability_gate",
    )
    result.add_argument(
        "--execution-root", type=Path, default=REPO_ROOT / "docs/progressive_upgrade"
    )
    result.add_argument(
        "--result-root", type=Path, default=REPO_ROOT / "docs/progressive_upgrade"
    )
    result.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
    result.add_argument(
        "--verifier-model",
        type=Path,
        default=Path(
            r"D:\RAG_test\.hf_cache\local_models\Qwen2.5-7B-Instruct-GPTQ-Int4"
        ),
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--embedding-batch-size", type=int, default=64)
    result.add_argument("--reranker-batch-size", type=int, default=256)
    result.add_argument("--case-batch-size", type=int, default=8)
    result.add_argument("--verifier-batch-size", type=int, default=16)
    result.add_argument("--no-fp16", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    commands = {
        "prepare": prepare,
        "register": register,
        "queries": queries,
        "score": score,
        "verify": verify,
        "evaluate": evaluate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
