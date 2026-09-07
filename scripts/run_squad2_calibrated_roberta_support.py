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
from research.frc_rag.squad2_calibrated_roberta_support import (  # noqa: E402
    BUDGETS,
    EXPERIMENT_ID,
    SOURCE_FILES,
    STAGES,
    TARGET_CASES,
    FrozenSquad2V64Scorer,
    LocalRobertaQASupportVerifier,
    build_candidate_coverage,
    build_deterministic_queries,
    build_gold_rows,
    build_support_gold_rows,
    evaluate_calibration_gate,
    evaluate_confirmation_support_gate,
    evaluate_final_method,
    load_prior_exclusion_union,
    prepare_blind_cases,
    read_stage_source,
    select_balanced_sample,
    threshold_support_rows,
    validate_implementation_registration,
    validate_model_registration,
    validate_protocol,
    validate_query_cache,
    validate_raw_qa_cache,
    validate_source_registration,
    write_report,
)


MODULE_PATH = REPO_ROOT / "research/frc_rag/squad2_calibrated_roberta_support.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_squad2_calibrated_roberta_support.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_squad2_calibrated_roberta_support.py"


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


def _exclusion_paths(args: argparse.Namespace) -> list[Path]:
    return [
        args.v58_exclusion_map,
        args.v59_exclusion_map,
        args.v60_exclusion_map,
        args.v61_exclusion_map,
    ]


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    cache = args.cache_root / args.stage
    output = args.output_root
    return {
        "prepared": cache / "prepared_blind.jsonl",
        "candidate_map": cache / "candidate_map.jsonl",
        "qa_input": cache / "qa_input.jsonl",
        "census": cache / "blind_census.json",
        "raw_support": cache / "raw_qa_decisions.jsonl",
        "thresholded_support": cache / "thresholded_support_decisions.jsonl",
        "queries": cache / "queries.jsonl",
        "scored": cache / "scored.jsonl",
        "coverage": cache / "coverage.json",
        "execution": args.execution_root
        / f"squad2_calibrated_roberta_support_{args.stage}_execution_v64.json",
        "calibration_result": args.result_root
        / "squad2_calibrated_roberta_support_calibration_result_v64.json",
        "calibration_open": args.result_root
        / "squad2_calibrated_roberta_support_confirmation_open_v64.json",
        "calibration_closure": args.result_root
        / "squad2_calibrated_roberta_support_calibration_closure_v64.json",
        "calibration_report": output / "calibration" / "report.md",
        "calibration_evidence": output / "calibration" / "cases.jsonl.gz",
        "support_result": args.result_root
        / "squad2_calibrated_roberta_support_confirmation_support_result_v64.json",
        "support_open": args.result_root
        / "squad2_calibrated_roberta_support_retrieval_open_v64.json",
        "support_closure": args.result_root
        / "squad2_calibrated_roberta_support_confirmation_support_closure_v64.json",
        "support_report": output / "confirmation_support" / "report.md",
        "support_evidence": output / "confirmation_support" / "cases.jsonl.gz",
        "final_result": args.result_root
        / "squad2_calibrated_roberta_support_confirmation_result_v64.json",
        "final_closure": args.result_root
        / "squad2_calibrated_roberta_support_confirmation_closure_v64.json",
        "final_report": output / "confirmation_final" / "report.md",
        "final_evidence": output / "confirmation_final" / "cases.jsonl.gz",
    }


def _runtime_environment(args: argparse.Namespace) -> dict[str, Any]:
    import numpy
    import torch
    import transformers

    gpu = None
    memory = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        memory = int(torch.cuda.get_device_properties(0).total_memory / 1024**2)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
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
            "qa_batch_size": args.qa_batch_size,
            "qa_max_length": 384,
            "qa_doc_stride": 128,
            "qa_maximum_answer_tokens": 30,
            "threshold_learned_parameter_count": 1,
            "token_budgets": list(BUDGETS),
        },
    }


def _validate_implementation(
    args: argparse.Namespace, *, allow_evaluation_erratum: bool = False
) -> dict[str, Any]:
    validate_protocol(args.protocol)
    validate_source_registration(
        args.source_registration,
        source_root=args.source_root,
    )
    validate_model_registration(
        args.model_registration,
        model_root=args.qa_model,
    )
    return validate_implementation_registration(
        args.implementation,
        protocol_path=args.protocol,
        source_registration_path=args.source_registration,
        model_registration_path=args.model_registration,
        module_path=MODULE_PATH,
        runner_path=RUNNER_PATH,
        test_path=TEST_PATH,
        successor_erratum_path=(
            args.evaluation_erratum if allow_evaluation_erratum else None
        ),
    )


def _validate_calibration_open(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args)
    if (
        not paths["calibration_result"].is_file()
        or not paths["calibration_open"].is_file()
    ):
        raise ValueError("SQuAD2 v64 confirmation dev is not authorized to open")
    result = json.loads(paths["calibration_result"].read_text(encoding="utf-8"))
    opened = json.loads(paths["calibration_open"].read_text(encoding="utf-8"))
    outcome = result["analysis"]["outcome"]
    if outcome["confirmation_dev_open_authorized"] is not True:
        raise ValueError("SQuAD2 v64 calibration gate failed")
    if opened["calibration_result_sha256"] != sha256(paths["calibration_result"]):
        raise ValueError("SQuAD2 v64 calibration-open registration changed")
    if round(float(opened["locked_threshold"]), 6) != float(
        outcome["locked_threshold"]
    ):
        raise ValueError("SQuAD2 v64 locked threshold changed")
    return result


def _locked_threshold(args: argparse.Namespace) -> float:
    _validate_calibration_open(args)
    paths = _paths(args)
    opened = json.loads(paths["calibration_open"].read_text(encoding="utf-8"))
    return float(opened["locked_threshold"])


def _validate_support_open(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args)
    if not paths["support_result"].is_file() or not paths["support_open"].is_file():
        raise ValueError("SQuAD2 v64 retrieval scoring is not authorized")
    result = json.loads(paths["support_result"].read_text(encoding="utf-8"))
    opened = json.loads(paths["support_open"].read_text(encoding="utf-8"))
    if result["analysis"]["outcome"]["retrieval_scoring_open_authorized"] is not True:
        raise ValueError("SQuAD2 v64 confirmation support gate failed")
    if opened["support_result_sha256"] != sha256(paths["support_result"]):
        raise ValueError("SQuAD2 v64 support-open registration changed")
    return result


def _qa_cache_key(args: argparse.Namespace, execution: dict[str, Any]) -> str:
    value = {
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "qa_input_sha256": execution["hashes"]["qa_input_sha256"],
        "model_registration_sha256": execution["hashes"]["model_registration_sha256"],
        "inference": {
            "max_length": 384,
            "doc_stride": 128,
            "maximum_answer_tokens": 30,
            "batch_size": args.qa_batch_size,
            "do_sample": False,
        },
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _expected_execution_hashes(
    args: argparse.Namespace, paths: dict[str, Path]
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "implementation_registration_sha256": sha256(args.implementation),
        "model_registration_sha256": sha256(args.model_registration),
        "stage_source_sha256": sha256(args.source_root / SOURCE_FILES[args.stage]),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "blind_census_sha256": sha256(paths["census"]),
    }
    if args.stage == "calibration":
        exclusions = load_prior_exclusion_union(_exclusion_paths(args))
        value.update(
            {
                "prior_exclusion_map_sha256": [
                    sha256(path) for path in _exclusion_paths(args)
                ],
                "prior_exclusion_union_sha256": hashlib.sha256(
                    "\n".join(sorted(exclusions)).encode()
                ).hexdigest(),
            }
        )
    else:
        value["calibration_open_sha256"] = sha256(paths["calibration_open"])
    return value


def _validate_execution(
    args: argparse.Namespace,
    paths: dict[str, Path],
    *,
    allow_evaluation_erratum: bool = False,
) -> dict[str, Any]:
    _validate_implementation(
        args, allow_evaluation_erratum=allow_evaluation_erratum
    )
    if args.stage == "confirmation":
        _validate_calibration_open(args)
    value = json.loads(paths["execution"].read_text(encoding="utf-8"))
    if value.get("stage") != args.stage:
        raise ValueError("SQuAD2 v64 execution stage changed")
    if value.get("hashes") != _expected_execution_hashes(args, paths):
        raise ValueError("SQuAD2 v64 execution registration hash mismatch")
    if value.get("qa_query_or_retrieval_started") is not False:
        raise ValueError("SQuAD2 v64 execution was registered too late")
    if value.get("gold_joined_for_metrics") is not False:
        raise ValueError("SQuAD2 v64 gold was joined before registration")
    return value


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    if args.stage == "confirmation":
        _validate_calibration_open(args)
    paths = _paths(args)
    if any(
        paths[name].exists()
        for name in (
            "prepared",
            "candidate_map",
            "qa_input",
            "census",
            "execution",
            "raw_support",
        )
    ):
        raise ValueError("SQuAD2 v64 stage preparation is already frozen")
    excluded = (
        load_prior_exclusion_union(_exclusion_paths(args))
        if args.stage == "calibration"
        else set()
    )
    source = read_stage_source(args.source_root, args.stage)
    selected, sampling = select_balanced_sample(
        source,
        stage=args.stage,
        excluded_commitments=excluded,
    )
    tokenizer = load_frozen_tokenizer(args.hf_home)
    prepared_rows, maps, qa_inputs, structural = prepare_blind_cases(
        selected,
        tokenizer,
        stage=args.stage,
    )
    write_jsonl(paths["prepared"], prepared_rows)
    write_jsonl(paths["candidate_map"], maps)
    write_jsonl(paths["qa_input"], qa_inputs)
    census = {
        "schema_version": "frc-squad2-v64-blind-census-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "source_manifest": {
            "opened_source_file": SOURCE_FILES[args.stage],
            "opened_source_sha256": sha256(args.source_root / SOURCE_FILES[args.stage]),
            "other_stage_source_content_read": False,
            "confirmation_opened_only_after_calibration_gate": args.stage
            == "confirmation",
        },
        "sampling": sampling,
        "structural_census": structural,
        "prior_exclusion": {
            "commitment_count": len(excluded),
            "selected_overlap": sampling["selected_prior_commitment_overlap"],
            "only_source_case_commitments_used": True,
        },
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "gold_joined_for_metrics": False,
        "qa_query_or_retrieval_started": False,
    }
    _write_json(paths["census"], census)
    print(json.dumps(census, ensure_ascii=False, sort_keys=True))


def register(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    if args.stage == "confirmation":
        _validate_calibration_open(args)
    paths = _paths(args)
    if paths["execution"].exists():
        raise FileExistsError("SQuAD2 v64 execution registration already exists")
    for name in ("prepared", "candidate_map", "qa_input", "census"):
        if not paths[name].is_file():
            raise ValueError(f"SQuAD2 v64 preparation artifact missing: {name}")
    if any(
        paths[name].exists()
        for name in ("raw_support", "thresholded_support", "queries", "scored")
    ):
        raise ValueError("SQuAD2 v64 execution registration is too late")
    runtime = _runtime_environment(args)
    if args.device == "cuda" and runtime["cuda_available"] is not True:
        raise ValueError("SQuAD2 v64 frozen CUDA runtime is unavailable")
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    value = {
        "schema_version": "frc-squad2-v64-execution-registration-v1",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "registered_at": args.registered_at,
        "target_cases": TARGET_CASES[args.stage],
        "hashes": _expected_execution_hashes(args, paths),
        "sampling": census["sampling"],
        "runtime": runtime,
        "qa_query_or_retrieval_started": False,
        "gold_joined_for_metrics": False,
        "other_stage_source_content_read": False,
    }
    _write_json(paths["execution"], value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def verify(args: argparse.Namespace) -> None:
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    if paths["raw_support"].exists():
        raise FileExistsError("SQuAD2 v64 raw QA cache already exists")
    if any(
        paths[name].exists()
        for name in ("thresholded_support", "queries", "scored", "coverage")
    ):
        raise ValueError("SQuAD2 v64 blind QA order changed")
    qa_inputs = list(read_jsonl(paths["qa_input"]))
    cache_key = _qa_cache_key(args, execution)
    verifier = LocalRobertaQASupportVerifier(
        args.qa_model,
        device=args.device,
        batch_size=args.qa_batch_size,
        fp16=not args.no_fp16,
    )
    rows = verifier.verify(qa_inputs, cache_key=cache_key)
    write_jsonl(paths["raw_support"], rows)
    summary = validate_raw_qa_cache(qa_inputs, rows, cache_key=cache_key)
    print(
        json.dumps({**summary, "sha256": sha256(paths["raw_support"])}, sort_keys=True)
    )


def calibrate(args: argparse.Namespace) -> None:
    if args.stage != "calibration":
        raise ValueError("SQuAD2 v64 calibration command requires --stage calibration")
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    if paths["calibration_result"].exists():
        raise FileExistsError("SQuAD2 v64 calibration result is already frozen")
    if any(paths[name].exists() for name in ("queries", "scored", "coverage")):
        raise ValueError("SQuAD2 v64 retrieval work started during calibration")
    maps = list(read_jsonl(paths["candidate_map"]))
    qa_inputs = list(read_jsonl(paths["qa_input"]))
    raw_support = list(read_jsonl(paths["raw_support"]))
    expected = [str(row["id"]) for row in maps]
    if (
        len(expected) != TARGET_CASES["calibration"]
        or [str(row["id"]) for row in qa_inputs] != expected
        or [str(row["id"]) for row in raw_support] != expected
    ):
        raise ValueError("SQuAD2 v64 calibration QA cache is incomplete")
    verifier_summary = validate_raw_qa_cache(
        qa_inputs,
        raw_support,
        cache_key=_qa_cache_key(args, execution),
    )
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    source = read_stage_source(args.source_root, "calibration")
    support_gold = build_support_gold_rows(source, maps, raw_support)
    source_artifacts = {
        "protocol_sha256": sha256(args.protocol),
        "source_registration_sha256": sha256(args.source_registration),
        "model_registration_sha256": sha256(args.model_registration),
        "implementation_registration_sha256": sha256(args.implementation),
        "execution_registration_sha256": sha256(paths["execution"]),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "raw_qa_cache_sha256": sha256(paths["raw_support"]),
        "sampling": census["sampling"],
        "gold_joined_after_complete_qa_cache": True,
        "confirmation_dev_content_read": False,
        "query_or_retrieval_scoring_started": False,
    }
    report, evidence = evaluate_calibration_gate(
        support_gold,
        verifier_summary,
        census["sampling"],
        source_artifacts,
    )
    write_report(
        report,
        evidence,
        paths["calibration_result"],
        paths["calibration_report"],
        paths["calibration_evidence"],
        title="SQuAD 2.0 calibrated RoBERTa support calibration (v64)",
    )
    outcome = report["analysis"]["outcome"]
    record = {
        "experiment_id": EXPERIMENT_ID,
        "status": outcome["status"],
        "calibration_result_sha256": sha256(paths["calibration_result"]),
        "confirmation_dev_open_authorized": outcome["confirmation_dev_open_authorized"],
        "locked_threshold": outcome["locked_threshold"],
        "selector_adoption_authorized": False,
        "canary_or_default_authorized": False,
        "gate_2": "NO-GO/SHADOW",
    }
    if outcome["confirmation_dev_open_authorized"]:
        record["schema_version"] = "frc-squad2-v64-confirmation-open-v1"
        _write_json(paths["calibration_open"], record)
    else:
        record["schema_version"] = "frc-squad2-v64-calibration-closure-v1"
        record["confirmation_dev_parsed_or_opened"] = False
        _write_json(paths["calibration_closure"], record)
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))


def support_evaluate(args: argparse.Namespace) -> None:
    if args.stage != "confirmation":
        raise ValueError("SQuAD2 v64 support-evaluate requires --stage confirmation")
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    if paths["support_result"].exists():
        raise FileExistsError("SQuAD2 v64 confirmation support result is frozen")
    if any(paths[name].exists() for name in ("queries", "scored", "coverage")):
        raise ValueError("SQuAD2 v64 retrieval work started before support gate")
    maps = list(read_jsonl(paths["candidate_map"]))
    qa_inputs = list(read_jsonl(paths["qa_input"]))
    raw_support = list(read_jsonl(paths["raw_support"]))
    expected = [str(row["id"]) for row in maps]
    if (
        len(expected) != TARGET_CASES["confirmation"]
        or [str(row["id"]) for row in qa_inputs] != expected
        or [str(row["id"]) for row in raw_support] != expected
    ):
        raise ValueError("SQuAD2 v64 confirmation QA cache is incomplete")
    verifier_summary = validate_raw_qa_cache(
        qa_inputs,
        raw_support,
        cache_key=_qa_cache_key(args, execution),
    )
    threshold = _locked_threshold(args)
    thresholded = threshold_support_rows(raw_support, threshold)
    write_jsonl(paths["thresholded_support"], thresholded)
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    source = read_stage_source(args.source_root, "confirmation")
    support_gold = build_support_gold_rows(source, maps, raw_support)
    source_artifacts = {
        "protocol_sha256": sha256(args.protocol),
        "source_registration_sha256": sha256(args.source_registration),
        "model_registration_sha256": sha256(args.model_registration),
        "implementation_registration_sha256": sha256(args.implementation),
        "calibration_result_sha256": sha256(paths["calibration_result"]),
        "calibration_open_sha256": sha256(paths["calibration_open"]),
        "execution_registration_sha256": sha256(paths["execution"]),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "raw_qa_cache_sha256": sha256(paths["raw_support"]),
        "thresholded_support_cache_sha256": sha256(paths["thresholded_support"]),
        "sampling": census["sampling"],
        "gold_joined_after_complete_qa_cache": True,
        "query_or_retrieval_scoring_started": False,
    }
    report, evidence = evaluate_confirmation_support_gate(
        support_gold,
        verifier_summary,
        census["sampling"],
        source_artifacts,
        locked_threshold=threshold,
    )
    write_report(
        report,
        evidence,
        paths["support_result"],
        paths["support_report"],
        paths["support_evidence"],
        title="SQuAD 2.0 calibrated RoBERTa confirmation support gate (v64)",
    )
    outcome = report["analysis"]["outcome"]
    record = {
        "experiment_id": EXPERIMENT_ID,
        "status": outcome["status"],
        "support_result_sha256": sha256(paths["support_result"]),
        "locked_threshold": threshold,
        "retrieval_scoring_open_authorized": outcome[
            "retrieval_scoring_open_authorized"
        ],
        "selector_adoption_authorized": False,
        "canary_or_default_authorized": False,
        "gate_2": "NO-GO/SHADOW",
    }
    if outcome["retrieval_scoring_open_authorized"]:
        record["schema_version"] = "frc-squad2-v64-retrieval-open-v1"
        _write_json(paths["support_open"], record)
    else:
        record["schema_version"] = "frc-squad2-v64-support-closure-v1"
        record["query_generation_started"] = False
        record["retrieval_scoring_started"] = False
        _write_json(paths["support_closure"], record)
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))


def queries(args: argparse.Namespace) -> None:
    if args.stage != "confirmation":
        raise ValueError("SQuAD2 v64 query generation is confirmation-only")
    paths = _paths(args)
    _validate_execution(args, paths)
    _validate_support_open(args)
    if paths["queries"].exists():
        raise FileExistsError("SQuAD2 v64 query cache already exists")
    prepared = list(read_jsonl(paths["prepared"]))
    rows = build_deterministic_queries(prepared)
    summary = validate_query_cache(prepared, rows)
    write_jsonl(paths["queries"], rows)
    print(json.dumps({**summary, "sha256": sha256(paths["queries"])}, sort_keys=True))


def score(args: argparse.Namespace) -> None:
    if args.stage != "confirmation":
        raise ValueError("SQuAD2 v64 retrieval scoring is confirmation-only")
    paths = _paths(args)
    _validate_execution(args, paths)
    _validate_support_open(args)
    if paths["coverage"].exists():
        raise ValueError("SQuAD2 v64 gold joined before retrieval scoring")
    prepared = list(read_jsonl(paths["prepared"]))
    query_rows = list(read_jsonl(paths["queries"]))
    validate_query_cache(prepared, query_rows)
    scorer = FrozenSquad2V64Scorer(
        query_rows=query_rows,
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
    if args.stage != "confirmation":
        raise ValueError("SQuAD2 v64 final evaluation is confirmation-only")
    paths = _paths(args)
    execution = _validate_execution(
        args, paths, allow_evaluation_erratum=True
    )
    support_result = _validate_support_open(args)
    if paths["final_result"].exists():
        raise FileExistsError("SQuAD2 v64 final result is already frozen")
    prepared = list(read_jsonl(paths["prepared"]))
    maps = list(read_jsonl(paths["candidate_map"]))
    qa_inputs = list(read_jsonl(paths["qa_input"]))
    raw_support = list(read_jsonl(paths["raw_support"]))
    thresholded = list(read_jsonl(paths["thresholded_support"]))
    query_rows = list(read_jsonl(paths["queries"]))
    scored_rows = list(read_jsonl(paths["scored"]))
    expected = [str(row["id"]) for row in prepared]
    for name, rows in (
        ("candidate map", maps),
        ("QA inputs", qa_inputs),
        ("raw QA", raw_support),
        ("thresholded support", thresholded),
        ("queries", query_rows),
        ("scores", scored_rows),
    ):
        if [str(row["id"]) for row in rows] != expected:
            raise ValueError(f"SQuAD2 v64 {name} cache is incomplete")
    query_summary = validate_query_cache(prepared, query_rows)
    verifier_summary = validate_raw_qa_cache(
        qa_inputs,
        raw_support,
        cache_key=_qa_cache_key(args, execution),
    )
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    evaluation_erratum = json.loads(
        args.evaluation_erratum.read_text(encoding="utf-8")
    )
    if evaluation_erratum.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("SQuAD2 v64 evaluation erratum experiment mismatch")
    expected_erratum_hashes = {
        "prior_implementation_erratum": sha256(args.implementation),
        "retrieval_open": sha256(paths["support_open"]),
        "score_cache": sha256(paths["scored"]),
    }
    for name, expected_hash in expected_erratum_hashes.items():
        if evaluation_erratum["hashes"].get(name) != expected_hash:
            raise ValueError(f"SQuAD2 v64 evaluation erratum {name} mismatch")
    correction = evaluation_erratum["correction"]
    if correction.get("sample_score_threshold_metric_or_gate_changed") is not False:
        raise ValueError("SQuAD2 v64 evaluation erratum changes frozen evidence")
    if correction.get("final_metrics_computed_before_erratum") is not False:
        raise ValueError("SQuAD2 v64 final metrics predate evaluation erratum")
    source = read_stage_source(args.source_root, "confirmation")
    gold_rows = build_gold_rows(
        source,
        maps,
        scored_rows,
        structural_census=census["structural_census"],
    )
    coverage = build_candidate_coverage(gold_rows, census["sampling"])
    coverage["gold_joined_after_complete_score_and_qa_caches"] = True
    _write_json(paths["coverage"], coverage)
    source_artifacts = {
        "protocol_sha256": sha256(args.protocol),
        "source_registration_sha256": sha256(args.source_registration),
        "model_registration_sha256": sha256(args.model_registration),
        "implementation_registration_sha256": sha256(args.implementation),
        "evaluation_implementation_erratum_sha256": sha256(
            args.evaluation_erratum
        ),
        "calibration_result_sha256": sha256(paths["calibration_result"]),
        "calibration_open_sha256": sha256(paths["calibration_open"]),
        "execution_registration_sha256": sha256(paths["execution"]),
        "support_result_sha256": sha256(paths["support_result"]),
        "retrieval_open_sha256": sha256(paths["support_open"]),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "raw_qa_cache_sha256": sha256(paths["raw_support"]),
        "thresholded_support_cache_sha256": sha256(paths["thresholded_support"]),
        "query_cache_sha256": sha256(paths["queries"]),
        "score_cache_sha256": sha256(paths["scored"]),
        "coverage_sha256": sha256(paths["coverage"]),
        "sampling": census["sampling"],
        "structural_census": census["structural_census"],
        "locked_threshold": _locked_threshold(args),
        "confirmation_support_gate_passed": support_result["analysis"]["outcome"][
            "support_gate_passed"
        ],
        "score_fallback_rate": 0.0,
    }
    report, evidence = evaluate_final_method(
        gold_rows,
        scored_rows,
        thresholded,
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
        title="SQuAD 2.0 calibrated RoBERTa support and selector confirmation (v64)",
    )
    outcome = report["analysis"]["outcome"]
    _write_json(
        paths["final_closure"],
        {
            "schema_version": "frc-squad2-v64-confirmation-closure-v1",
            "experiment_id": EXPERIMENT_ID,
            "status": outcome["status"],
            "confirmation_result_sha256": sha256(paths["final_result"]),
            "self_domain_held_out_component_feasibility_only": True,
            "independent_model_training_confirmation_claimed": False,
            "calibration_or_confirmation_reuse_for_additional_tuning_or_selection": False,
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
            "prepare",
            "register",
            "verify",
            "calibrate",
            "support-evaluate",
            "queries",
            "score",
            "evaluate",
            "runtime",
        ),
    )
    result.add_argument("--stage", choices=STAGES, default="calibration")
    result.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/squad2_calibrated_roberta_support_protocol_v64.json"
        ),
    )
    result.add_argument(
        "--source-registration",
        type=Path,
        default=Path("docs/progressive_upgrade/squad2_source_registration_v58.json"),
    )
    result.add_argument(
        "--model-registration",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/quac_roberta_qa_support_transfer_model_registration_v62.json"
        ),
    )
    result.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/squad2_calibrated_roberta_support_implementation_erratum_v64.json"
        ),
    )
    result.add_argument(
        "--evaluation-erratum",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/squad2_calibrated_roberta_support_evaluation_erratum_v64.json"
        ),
    )
    result.add_argument(
        "--source-root",
        type=Path,
        default=Path(".cache/benchmarks/squad2_v2"),
    )
    result.add_argument(
        "--v58-exclusion-map",
        type=Path,
        default=Path(
            ".cache/benchmarks/squad2_generative_answerability_gate_v58/development/candidate_map.jsonl"
        ),
    )
    result.add_argument(
        "--v59-exclusion-map",
        type=Path,
        default=Path(
            ".cache/benchmarks/squad2_extractive_support_gate_v59/development/candidate_map.jsonl"
        ),
    )
    result.add_argument(
        "--v60-exclusion-map",
        type=Path,
        default=Path(
            ".cache/benchmarks/squad2_structured_span_gate_v60/development/candidate_map.jsonl"
        ),
    )
    result.add_argument(
        "--v61-exclusion-map",
        type=Path,
        default=Path(
            ".cache/benchmarks/squad2_dual_support_union_v61/development/candidate_map.jsonl"
        ),
    )
    result.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".cache/benchmarks/squad2_calibrated_roberta_support_v64"),
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/rag_evaluation/squad2_calibrated_roberta_support"),
    )
    result.add_argument(
        "--execution-root", type=Path, default=Path("docs/progressive_upgrade")
    )
    result.add_argument(
        "--result-root", type=Path, default=Path("docs/progressive_upgrade")
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
    result.add_argument("--registered-at", default="2026-08-03T23:00:00+08:00")
    return result


def main() -> int:
    args = parser().parse_args()
    for name in (
        "protocol",
        "source_registration",
        "model_registration",
        "implementation",
        "source_root",
        "v58_exclusion_map",
        "v59_exclusion_map",
        "v60_exclusion_map",
        "v61_exclusion_map",
        "cache_root",
        "output_root",
        "execution_root",
        "result_root",
        "hf_home",
        "qa_model",
    ):
        setattr(args, name, _resolve(getattr(args, name)))
    commands = {
        "prepare": prepare,
        "register": register,
        "verify": verify,
        "calibrate": calibrate,
        "support-evaluate": support_evaluate,
        "queries": queries,
        "score": score,
        "evaluate": evaluate,
        "runtime": runtime,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
