from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag import squad2_calibrated_roberta_support as v64  # noqa: E402
from research.frc_rag.musique_full_roberta_transfer import (  # noqa: E402
    EXPERIMENT_ID,
    LOCKED_THRESHOLD,
    LocalRobertaQASupportVerifier,
    aggregate_paragraph_decisions,
    build_gold_evidence,
    evaluate_support_gate,
    extract_squad2_train_questions,
    load_v36_source_ids,
    prepare_blind_qa,
    select_balanced_sample,
    sha256,
    validate_protocol,
    validate_source_registration,
    write_report,
)


MODULE_PATH = REPO_ROOT / "research/frc_rag/musique_full_roberta_transfer.py"
RUNNER_PATH = REPO_ROOT / "scripts/run_musique_full_roberta_transfer.py"
TEST_PATH = REPO_ROOT / "tests/test_frc_musique_full_roberta_transfer.py"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "prepared": args.cache_root / "prepared_blind.jsonl",
        "candidate_map": args.cache_root / "candidate_map.jsonl",
        "qa_input": args.cache_root / "qa_input.jsonl",
        "census": args.cache_root / "blind_census.json",
        "raw_paragraph_qa": args.cache_root / "raw_paragraph_qa.jsonl",
        "case_support": args.cache_root / "case_support_decisions.jsonl",
        "execution": args.docs_root
        / "musique_full_roberta_transfer_corrected_execution_v65.json",
        "support_result": args.docs_root
        / "musique_full_roberta_transfer_support_result_v65.json",
        "closure": args.docs_root
        / "musique_full_roberta_transfer_support_closure_v65.json",
        "retrieval_open": args.docs_root
        / "musique_full_roberta_transfer_retrieval_open_v65.json",
        "report": args.output_root / "support_gate/report.md",
        "evidence": args.output_root / "support_gate/cases.jsonl.gz",
    }


def _runtime(args: argparse.Namespace) -> dict[str, Any]:
    import numpy
    import torch
    import transformers

    gpu = None
    gpu_memory_mib = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        gpu_memory_mib = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "gpu_memory_mib": gpu_memory_mib,
        "parameters": {
            "device": args.device,
            "fp16": not args.no_fp16,
            "qa_batch_size": args.qa_batch_size,
            "qa_chunk_rows": args.qa_chunk_rows,
            "qa_max_length": 384,
            "qa_doc_stride": 128,
            "qa_maximum_answer_tokens": 30,
            "locked_threshold": LOCKED_THRESHOLD,
            "offline_model_loading": True,
        },
    }


def _validate_implementation(args: argparse.Namespace) -> dict[str, Any]:
    validate_protocol(args.protocol)
    validate_source_registration(
        args.source_registration,
        target_path=args.target,
        squad2_train_path=args.squad2_train,
        v36_prepared_path=args.v36_prepared,
    )
    v64.validate_model_registration(
        args.model_registration,
        model_root=args.qa_model,
    )
    value = json.loads(args.implementation.read_text(encoding="utf-8"))
    expected = {
        "protocol": sha256(args.protocol),
        "source_registration": sha256(args.source_registration),
        "model_registration": sha256(args.model_registration),
        "module": sha256(MODULE_PATH),
        "runner": sha256(RUNNER_PATH),
        "tests": sha256(TEST_PATH),
        "original_implementation": sha256(args.original_implementation),
        "invalid_run_record": sha256(args.invalid_run_record),
        "invalid_candidate_map": sha256(args.invalid_candidate_map),
    }
    if value.get("hashes") != expected:
        raise ValueError("MuSiQue v65 implementation registration changed")
    if value.get("registered_after_invalid_run_before_corrected_target_parse") is not True:
        raise ValueError("MuSiQue v65 corrected implementation registration order changed")
    if value.get("gold_join_or_metric_computation_before_correction") is not False:
        raise ValueError("MuSiQue v65 correction followed metric access")
    return value


def _load_guard_inputs(args: argparse.Namespace) -> tuple[set[str], set[str]]:
    squad_source = json.loads(args.squad2_train.read_text(encoding="utf-8"))
    squad_questions = extract_squad2_train_questions(squad_source)
    v36_ids = load_v36_source_ids(read_jsonl(args.v36_prepared))
    return squad_questions, v36_ids


def _load_invalid_run_commitments(args: argparse.Namespace) -> set[str]:
    value = json.loads(args.invalid_run_record.read_text(encoding="utf-8"))
    if value["status"] != (
        "MUSIQUE_FULL_V65_PROCEDURAL_INVALID_DUPLICATE_SOURCE_ID_STOP_BEFORE_METRICS"
    ):
        raise ValueError("MuSiQue v65 invalid-run status changed")
    if value["information_revealed_before_invalidation"]["aggregate_or_per_stratum_metric"]:
        raise ValueError("MuSiQue v65 invalid run exposed metrics")
    commitments = {
        str(row["source_id_commitment"])
        for row in read_jsonl(args.invalid_candidate_map)
    }
    if len(commitments) != 300:
        raise ValueError("MuSiQue v65 invalid-run commitment count changed")
    return commitments


def _select(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    squad_questions, v36_ids = _load_guard_inputs(args)
    invalid_commitments = _load_invalid_run_commitments(args)
    return select_balanced_sample(
        read_jsonl(args.target),
        squad_questions=squad_questions,
        v36_source_ids=v36_ids,
        excluded_source_commitments=invalid_commitments,
    )


def _validate_prepared(paths: dict[str, Path]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    prepared = list(read_jsonl(paths["prepared"]))
    maps = list(read_jsonl(paths["candidate_map"]))
    qa_inputs = list(read_jsonl(paths["qa_input"]))
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    if len(prepared) != 600 or len(maps) != 600:
        raise ValueError("MuSiQue v65 prepared caches are incomplete")
    if [row["id"] for row in prepared] != [row["id"] for row in maps]:
        raise ValueError("MuSiQue v65 prepared case order changed")
    expected_qa_ids = [
        f"{row['id']}::{candidate['candidate_id']}"
        for row in prepared
        for candidate in row["candidates"]
    ]
    if [row["id"] for row in qa_inputs] != expected_qa_ids:
        raise ValueError("MuSiQue v65 QA input order changed")
    return prepared, maps, qa_inputs, census


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    paths = _paths(args)
    for path in (
        paths["prepared"],
        paths["candidate_map"],
        paths["qa_input"],
        paths["census"],
    ):
        if path.exists():
            raise FileExistsError(f"MuSiQue v65 preparation is frozen: {path}")
    selected, sampling = _select(args)
    prepared, maps, qa_inputs, structural = prepare_blind_qa(selected)
    _write_jsonl(paths["prepared"], prepared)
    _write_jsonl(paths["candidate_map"], maps)
    _write_jsonl(paths["qa_input"], qa_inputs)
    _write_json(
        paths["census"],
        {
            "schema_version": "frc-musique-v65-blind-census-v1",
            "experiment_id": EXPERIMENT_ID,
            "sampling": sampling,
            "structural_census": structural,
            "target_content_opened_after_implementation_registration": True,
            "gold_joined_for_metrics": False,
            "qa_or_retrieval_started": False,
        },
    )
    print(
        json.dumps(
            {
                "prepared_cases": len(prepared),
                "qa_input_rows": len(qa_inputs),
                "sampling": sampling,
                "structural_census": structural,
                "prepared_sha256": sha256(paths["prepared"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def register(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    paths = _paths(args)
    prepared, maps, qa_inputs, census = _validate_prepared(paths)
    if paths["execution"].exists():
        raise FileExistsError("MuSiQue v65 execution registration is frozen")
    _write_json(
        paths["execution"],
        {
            "schema_version": "frc-musique-v65-execution-registration-v1",
            "experiment_id": EXPERIMENT_ID,
            "registered_at": args.registered_at,
            "target_rows_parsed_only_after_implementation_registration": True,
            "gold_joined_for_metrics": False,
            "qa_or_retrieval_started": False,
            "query_or_neural_retrieval_started": False,
            "hashes": {
                "implementation_registration_sha256": sha256(args.implementation),
                "target_source_sha256": sha256(args.target),
                "squad2_guard_source_sha256": sha256(args.squad2_train),
                "v36_exclusion_source_sha256": sha256(args.v36_prepared),
                "prepared_blind_sha256": sha256(paths["prepared"]),
                "candidate_map_sha256": sha256(paths["candidate_map"]),
                "qa_input_sha256": sha256(paths["qa_input"]),
                "blind_census_sha256": sha256(paths["census"]),
            },
            "sampling": census["sampling"],
            "structural_census": census["structural_census"],
            "prepared_cases": len(prepared),
            "candidate_maps": len(maps),
            "qa_input_rows": len(qa_inputs),
            "runtime": _runtime(args),
        },
    )
    print(sha256(paths["execution"]))


def _validate_execution(
    args: argparse.Namespace, paths: dict[str, Path]
) -> dict[str, Any]:
    _validate_implementation(args)
    _, _, qa_inputs, census = _validate_prepared(paths)
    value = json.loads(paths["execution"].read_text(encoding="utf-8"))
    expected = {
        "implementation_registration_sha256": sha256(args.implementation),
        "target_source_sha256": sha256(args.target),
        "squad2_guard_source_sha256": sha256(args.squad2_train),
        "v36_exclusion_source_sha256": sha256(args.v36_prepared),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "blind_census_sha256": sha256(paths["census"]),
    }
    if value.get("hashes") != expected:
        raise ValueError("MuSiQue v65 execution registration changed")
    if value["qa_input_rows"] != len(qa_inputs):
        raise ValueError("MuSiQue v65 execution QA row count changed")
    if value["sampling"] != census["sampling"]:
        raise ValueError("MuSiQue v65 execution sampling changed")
    return value


def _qa_cache_key(args: argparse.Namespace, execution: dict[str, Any]) -> str:
    return json.dumps(
        {
            "experiment_id": EXPERIMENT_ID,
            "execution_sha256": sha256(_paths(args)["execution"]),
            "model_registration_sha256": sha256(args.model_registration),
            "qa_input_sha256": execution["hashes"]["qa_input_sha256"],
            "locked_threshold": LOCKED_THRESHOLD,
            "max_length": 384,
            "doc_stride": 128,
            "maximum_answer_tokens": 30,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def qa(args: argparse.Namespace) -> None:
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    _, _, qa_inputs, _ = _validate_prepared(paths)
    existing = list(read_jsonl(paths["raw_paragraph_qa"])) if paths["raw_paragraph_qa"].exists() else []
    if [row["id"] for row in existing] != [row["id"] for row in qa_inputs[: len(existing)]]:
        raise ValueError("MuSiQue v65 resumable QA prefix changed")
    cache_key = _qa_cache_key(args, execution)
    if any(row.get("cache_key") != cache_key for row in existing):
        raise ValueError("MuSiQue v65 resumable QA cache key changed")
    verifier = LocalRobertaQASupportVerifier(
        args.qa_model,
        device=args.device,
        batch_size=args.qa_batch_size,
        fp16=not args.no_fp16,
    )
    paths["raw_paragraph_qa"].parent.mkdir(parents=True, exist_ok=True)
    with paths["raw_paragraph_qa"].open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(len(existing), len(qa_inputs), args.qa_chunk_rows):
            chunk = qa_inputs[start : start + args.qa_chunk_rows]
            predictions = verifier.verify(chunk, cache_key=cache_key)
            for row in predictions:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
    decisions = list(read_jsonl(paths["raw_paragraph_qa"]))
    aggregated = aggregate_paragraph_decisions(
        qa_inputs,
        decisions,
        cache_key=cache_key,
    )
    _write_jsonl(paths["case_support"], aggregated)
    print(
        json.dumps(
            {
                "paragraph_rows": len(decisions),
                "cases": len(aggregated),
                "raw_qa_sha256": sha256(paths["raw_paragraph_qa"]),
                "case_support_sha256": sha256(paths["case_support"]),
            },
            sort_keys=True,
        )
    )


def support_evaluate(args: argparse.Namespace) -> None:
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    prepared, _, qa_inputs, census = _validate_prepared(paths)
    raw_qa = list(read_jsonl(paths["raw_paragraph_qa"]))
    case_support = list(read_jsonl(paths["case_support"]))
    if len(raw_qa) != len(qa_inputs) or len(case_support) != len(prepared):
        raise ValueError("MuSiQue v65 QA caches are incomplete")
    if paths["support_result"].exists():
        raise FileExistsError("MuSiQue v65 support result is frozen")
    selected, sampling = _select(args)
    if sampling != census["sampling"]:
        raise ValueError("MuSiQue v65 deterministic sample changed at gold join")
    evidence = build_gold_evidence(selected, prepared, case_support)
    source_artifacts = {
        "protocol_sha256": sha256(args.protocol),
        "source_registration_sha256": sha256(args.source_registration),
        "model_registration_sha256": sha256(args.model_registration),
        "implementation_registration_sha256": sha256(args.implementation),
        "execution_registration_sha256": sha256(paths["execution"]),
        "prepared_blind_sha256": sha256(paths["prepared"]),
        "candidate_map_sha256": sha256(paths["candidate_map"]),
        "qa_input_sha256": sha256(paths["qa_input"]),
        "raw_paragraph_qa_sha256": sha256(paths["raw_paragraph_qa"]),
        "case_support_sha256": sha256(paths["case_support"]),
        "gold_joined_after_complete_qa_cache": True,
        "query_or_neural_retrieval_started": False,
        "sampling": sampling,
        "structural_census": census["structural_census"],
        "runtime": execution["runtime"],
    }
    report = evaluate_support_gate(
        evidence,
        sampling=sampling,
        structural_census=census["structural_census"],
        source_artifacts=source_artifacts,
    )
    write_report(
        report,
        evidence,
        paths["support_result"],
        paths["report"],
        paths["evidence"],
    )
    outcome = report["analysis"]["outcome"]
    if outcome["retrieval_scoring_open_authorized"]:
        _write_json(
            paths["retrieval_open"],
            {
                "schema_version": "frc-musique-v65-retrieval-open-v1",
                "experiment_id": EXPERIMENT_ID,
                "status": outcome["status"],
                "support_result_sha256": sha256(paths["support_result"]),
                "retrieval_scoring_open_authorized": True,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        )
    else:
        _write_json(
            paths["closure"],
            {
                "schema_version": "frc-musique-v65-support-closure-v1",
                "experiment_id": EXPERIMENT_ID,
                "status": outcome["status"],
                "support_result_sha256": sha256(paths["support_result"]),
                "query_generation_started": False,
                "neural_retrieval_scoring_started": False,
                "retrieval_scoring_open_authorized": False,
                "reuse_v65_for_tuning_threshold_or_selection": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        )
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True))


def runtime(args: argparse.Namespace) -> None:
    print(json.dumps(_runtime(args), ensure_ascii=False, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "command",
        choices=("prepare", "register", "qa", "support-evaluate", "runtime"),
    )
    result.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/musique_full_roberta_transfer_protocol_v65.json"
        ),
    )
    result.add_argument(
        "--source-registration",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/musique_full_roberta_transfer_source_registration_v65.json"
        ),
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
            "docs/progressive_upgrade/musique_full_roberta_transfer_implementation_erratum_v65.json"
        ),
    )
    result.add_argument(
        "--original-implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/musique_full_roberta_transfer_implementation_v65.json"
        ),
    )
    result.add_argument(
        "--invalid-run-record",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/musique_full_roberta_transfer_invalid_run_v65.json"
        ),
    )
    result.add_argument(
        "--invalid-candidate-map",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique_full_roberta_transfer_v65/candidate_map.jsonl"
        ),
    )
    result.add_argument(
        "--target",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique/data/musique_full_v1.0_train.jsonl"
        ),
    )
    result.add_argument(
        "--squad2-train",
        type=Path,
        default=Path(".cache/benchmarks/squad2_v2/train-v2.0.json"),
    )
    result.add_argument(
        "--v36-prepared",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique/prepared_dual_resource_blind.jsonl"
        ),
    )
    result.add_argument(
        "--qa-model",
        type=Path,
        default=Path(
            ".cache/benchmarks/models/deepset_roberta_base_squad2_adc3b06_v64_registered"
        ),
    )
    result.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique_full_roberta_transfer_v65_corrected"
        ),
    )
    result.add_argument(
        "--docs-root", type=Path, default=Path("docs/progressive_upgrade")
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/rag_evaluation/musique_full_roberta_transfer"),
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--qa-batch-size", type=int, default=16)
    result.add_argument("--qa-chunk-rows", type=int, default=64)
    result.add_argument("--no-fp16", action="store_true")
    result.add_argument("--registered-at", default="2026-08-03T23:55:00+08:00")
    return result


def main() -> int:
    args = parser().parse_args()
    commands = {
        "prepare": prepare,
        "register": register,
        "qa": qa,
        "support-evaluate": support_evaluate,
        "runtime": runtime,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
