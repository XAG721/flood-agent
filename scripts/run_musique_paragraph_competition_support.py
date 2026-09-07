#!/usr/bin/env python3
"""Run the pre-registered MuSiQue paragraph-competition experiment (v70)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.frc_rag import (  # noqa: E402
    musique_paragraph_competition_support as experiment,
)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    experiment.write_json(path, payload)


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


def _with_stage(args: argparse.Namespace, stage: str) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(args), "stage": stage})


def _stage_target(args: argparse.Namespace) -> tuple[Path, str]:
    if args.stage == "confirmation":
        return args.dev_target, "dev"
    return args.train_target, "train"


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    cache = args.cache_root / args.stage
    output = args.output_root / args.stage
    docs = args.docs_root
    return {
        "prepared": cache / "prepared_blind.jsonl",
        "candidate_map": cache / "candidate_map.jsonl",
        "direct_input": cache / "direct_qa_input.jsonl",
        "direct_raw": cache / "direct_raw_qa.jsonl",
        "direct_decisions": cache / "direct_score_decisions.jsonl",
        "feature_decisions": cache / "feature_decisions.jsonl",
        "census": cache / "blind_census.json",
        "execution": docs
        / f"musique_paragraph_competition_support_{args.stage}_execution_v70.json",
        "result": docs
        / f"musique_paragraph_competition_support_{args.stage}_result_v70.json",
        "closure": docs
        / f"musique_paragraph_competition_support_{args.stage}_closure_v70.json",
        "confirmation_open": docs
        / "musique_paragraph_competition_support_confirmation_open_v70.json",
        "report": output / "report.md",
        "evidence": output / "cases.jsonl.gz",
    }


def _hop_paths(args: argparse.Namespace, hop: int) -> dict[str, Path]:
    cache = args.cache_root / args.stage
    return {
        "input": cache / f"hop_{hop}_qa_input.jsonl",
        "raw": cache / f"hop_{hop}_raw_qa.jsonl",
        "decisions": cache / f"hop_{hop}_feature_decisions.jsonl",
    }


def _protocol_and_source(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = experiment.validate_protocol(args.protocol)
    source = experiment.validate_source_registration(
        args.source_registration,
        protocol=protocol,
        train_path=args.train_target,
        dev_path=args.dev_target,
        squad2_path=args.squad2_train,
        prior_paths=args.prior_maps,
    )
    return protocol, source


def _validate_implementation(args: argparse.Namespace) -> dict[str, Any]:
    _protocol_and_source(args)
    value = json.loads(args.implementation.read_text(encoding="utf-8"))
    if value.get("experiment_id") != experiment.EXPERIMENT_ID:
        raise ValueError("MuSiQue v70 implementation experiment changed")
    erratum = json.loads(args.implementation_erratum.read_text(encoding="utf-8"))
    if (
        erratum.get("schema_version")
        != "frc-musique-paragraph-competition-support-implementation-erratum-v70"
        or erratum.get("experiment_id") != experiment.EXPERIMENT_ID
    ):
        raise ValueError("MuSiQue v70 implementation erratum changed")
    if erratum["original_registration_hashes"] != value["hashes"]:
        raise ValueError("MuSiQue v70 original implementation chain changed")
    expected = {
        "protocol": experiment.sha256(args.protocol),
        "source_registration": experiment.sha256(args.source_registration),
        "original_implementation": experiment.sha256(args.implementation),
        "model_registration": experiment.sha256(args.model_registration),
        "module": experiment.sha256(
            ROOT / "research/frc_rag/musique_paragraph_competition_support.py"
        ),
        "runner": experiment.sha256(
            ROOT / "scripts/run_musique_paragraph_competition_support.py"
        ),
        "tests": experiment.sha256(
            ROOT / "tests/test_frc_musique_paragraph_competition_support.py"
        ),
    }
    if erratum["corrected_hashes"] != expected:
        raise ValueError("MuSiQue v70 corrected implementation changed")
    if (
        int(value["total_learned_scalars_across_reported_methods"]) != 101
        or int(erratum["total_learned_scalars_across_reported_methods"]) != 101
    ):
        raise ValueError("MuSiQue v70 learned parameter boundary changed")
    return erratum


def _load_excluded(args: argparse.Namespace) -> set[str]:
    commitments: set[str] = set()
    for path in args.prior_maps:
        commitments.update(
            experiment.load_source_commitments(read_jsonl(path))
        )
    if len(commitments) != 7300:
        raise ValueError("MuSiQue v70 prior source commitment union changed")
    if args.stage in {"development", "confirmation"}:
        calibration_map = args.cache_root / "calibration" / "candidate_map.jsonl"
        commitments.update(
            experiment.load_source_commitments(read_jsonl(calibration_map))
        )
    if args.stage == "confirmation":
        development_map = args.cache_root / "development" / "candidate_map.jsonl"
        commitments.update(
            experiment.load_source_commitments(read_jsonl(development_map))
        )
    expected = {
        "calibration": 7300,
        "development": 8900,
        "confirmation": 9700,
    }[args.stage]
    if len(commitments) != expected:
        raise ValueError("MuSiQue v70 stage exclusion union changed")
    return commitments


def _select(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target, split = _stage_target(args)
    squad = json.loads(args.squad2_train.read_text(encoding="utf-8"))
    squad_questions = experiment.extract_squad2_questions(squad)
    return experiment.select_stage_sample(
        read_jsonl(target),
        stage=args.stage,
        source_split=split,
        squad_questions=squad_questions,
        excluded_source_commitments=_load_excluded(args),
    )


def prepare(args: argparse.Namespace) -> None:
    _validate_stage_authorization(args)
    _validate_implementation(args)
    paths = _paths(args)
    frozen = [
        paths["prepared"],
        paths["candidate_map"],
        paths["direct_input"],
        paths["census"],
    ]
    if any(path.exists() for path in frozen):
        raise FileExistsError("MuSiQue v70 prepared stage is frozen")
    selected, sampling = _select(args)
    prepared, maps, direct_inputs, structural = experiment.prepare_blind_stage(
        selected, stage=args.stage
    )
    _write_jsonl(paths["prepared"], prepared)
    _write_jsonl(paths["candidate_map"], maps)
    _write_jsonl(paths["direct_input"], direct_inputs)
    _write_json(
        paths["census"],
        {
            "schema_version": "frc-musique-v70-blind-census-v1",
            "experiment_id": experiment.EXPERIMENT_ID,
            "stage": args.stage,
            "sampling": sampling,
            "structural_census": structural,
        },
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "prepared_cases": len(prepared),
                "direct_qa_input_rows": len(direct_inputs),
                "prepared_sha256": experiment.sha256(paths["prepared"]),
                "sampling": sampling,
                "structural_census": structural,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _validate_prepared(
    args: argparse.Namespace, paths: dict[str, Path]
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    selected, sampling = _select(args)
    expected = experiment.prepare_blind_stage(selected, stage=args.stage)
    prepared = list(read_jsonl(paths["prepared"]))
    maps = list(read_jsonl(paths["candidate_map"]))
    direct_inputs = list(read_jsonl(paths["direct_input"]))
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    if (
        prepared != expected[0]
        or maps != expected[1]
        or direct_inputs != expected[2]
        or sampling != census["sampling"]
        or expected[3] != census["structural_census"]
    ):
        raise ValueError("MuSiQue v70 prepared stage changed")
    return prepared, maps, direct_inputs, census


def _validate_stage_authorization(args: argparse.Namespace) -> None:
    if args.stage == "calibration":
        return
    calibration_result = _paths(_with_stage(args, "calibration"))["result"]
    if not calibration_result.exists():
        raise FileNotFoundError("MuSiQue v70 calibration is not frozen")
    calibration = json.loads(calibration_result.read_text(encoding="utf-8"))
    if not calibration["outcome"]["development_open_authorized"]:
        raise ValueError("MuSiQue v70 development was not authorized")
    if args.stage == "confirmation":
        confirmation_open = _paths(args)["confirmation_open"]
        if not confirmation_open.exists():
            raise PermissionError("MuSiQue v70 confirmation is not open")
        authorization = json.loads(confirmation_open.read_text(encoding="utf-8"))
        if not authorization["confirmation_open_authorized"]:
            raise PermissionError("MuSiQue v70 confirmation authorization changed")


def _runtime(args: argparse.Namespace) -> dict[str, Any]:
    import numpy
    import torch
    import transformers

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    gpu_memory = (
        int(torch.cuda.get_device_properties(0).total_memory / 1024**2)
        if torch.cuda.is_available()
        else None
    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
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
            "qa_chunk_rows": args.qa_chunk_rows,
            "qa_max_length": 384,
            "qa_doc_stride": 128,
            "qa_maximum_answer_tokens": 30,
            "fp16": not args.no_fp16,
            "offline_model_loading": True,
            "intermediate_margin_threshold": None,
            "base_feature_count": len(experiment.BASE_FEATURE_NAMES),
            "additive_feature_count": len(experiment.ADDITIVE_FEATURE_NAMES),
            "interaction_feature_count": len(
                experiment.INTERACTION_FEATURE_NAMES
            ),
            "competition_feature_count": len(
                experiment.COMPETITION_FEATURE_NAMES
            ),
            "candidate_feature_count": len(experiment.CANDIDATE_FEATURE_NAMES),
            "optimizer_iterations": experiment.OPTIMIZER_ITERATIONS,
            "optimizer_learning_rate": experiment.OPTIMIZER_LEARNING_RATE,
            "optimizer_l2_weight": experiment.OPTIMIZER_L2_WEIGHT,
            "softmax_temperature": experiment.SOFTMAX_TEMPERATURE,
            "gap_clip_maximum": experiment.GAP_CLIP_MAXIMUM,
        },
    }


def register(args: argparse.Namespace) -> None:
    _validate_stage_authorization(args)
    _validate_implementation(args)
    paths = _paths(args)
    prepared, maps, direct_inputs, census = _validate_prepared(args, paths)
    if paths["execution"].exists():
        raise FileExistsError("MuSiQue v70 execution registration is frozen")
    calibration_result = _paths(_with_stage(args, "calibration"))["result"]
    calibration_hash = (
        experiment.sha256(calibration_result)
        if args.stage in {"development", "confirmation"}
        else None
    )
    target, _ = _stage_target(args)
    _write_json(
        paths["execution"],
        {
            "schema_version": "frc-musique-v70-execution-registration-v1",
            "experiment_id": experiment.EXPERIMENT_ID,
            "stage": args.stage,
            "registered_at": args.registered_at,
            "gold_joined_for_metrics": False,
            "qa_started": False,
            "feature_model_threshold_rule_or_gate_adjusted": False,
            "hashes": {
                "implementation_registration_sha256": experiment.sha256(
                    args.implementation
                ),
                "target_source_sha256": experiment.sha256(target),
                "squad2_guard_source_sha256": experiment.sha256(
                    args.squad2_train
                ),
                "prior_map_sha256": [
                    experiment.sha256(path) for path in args.prior_maps
                ],
                "calibration_result_sha256": calibration_hash,
                "prepared_blind_sha256": experiment.sha256(paths["prepared"]),
                "candidate_map_sha256": experiment.sha256(
                    paths["candidate_map"]
                ),
                "direct_qa_input_sha256": experiment.sha256(
                    paths["direct_input"]
                ),
                "blind_census_sha256": experiment.sha256(paths["census"]),
                "confirmation_open_sha256": experiment.sha256(
                    paths["confirmation_open"]
                )
                if args.stage == "confirmation"
                else None,
            },
            "sampling": census["sampling"],
            "structural_census": census["structural_census"],
            "prepared_cases": len(prepared),
            "candidate_maps": len(maps),
            "direct_qa_input_rows": len(direct_inputs),
            "runtime": _runtime(args),
        },
    )
    print(experiment.sha256(paths["execution"]))


def _validate_execution(
    args: argparse.Namespace, paths: dict[str, Path]
) -> dict[str, Any]:
    _validate_implementation(args)
    _, _, direct_inputs, census = _validate_prepared(args, paths)
    value = json.loads(paths["execution"].read_text(encoding="utf-8"))
    if value["stage"] != args.stage or value["gold_joined_for_metrics"]:
        raise ValueError("MuSiQue v70 execution boundary changed")
    if (
        value["sampling"] != census["sampling"]
        or value["direct_qa_input_rows"] != len(direct_inputs)
    ):
        raise ValueError("MuSiQue v70 execution census changed")
    return value


def _cache_key(
    args: argparse.Namespace,
    execution_path: Path,
    input_path: Path,
    *,
    phase: str,
) -> str:
    return json.dumps(
        {
            "experiment_id": experiment.EXPERIMENT_ID,
            "stage": args.stage,
            "phase": phase,
            "execution_sha256": experiment.sha256(execution_path),
            "model_registration_sha256": experiment.sha256(
                args.model_registration
            ),
            "input_sha256": experiment.sha256(input_path),
            "intermediate_margin_threshold": None,
            "max_length": 384,
            "doc_stride": 128,
            "maximum_answer_tokens": 30,
            "competition_softmax_temperature": experiment.SOFTMAX_TEMPERATURE,
            "competition_gap_clip_maximum": experiment.GAP_CLIP_MAXIMUM,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _run_qa_rows(
    verifier: experiment.LocalRobertaQASupportVerifier,
    rows: list[dict[str, Any]],
    *,
    raw_path: Path,
    cache_key: str,
    chunk_rows: int,
) -> list[dict[str, Any]]:
    existing = list(read_jsonl(raw_path)) if raw_path.exists() else []
    if [row["id"] for row in existing] != [
        row["id"] for row in rows[: len(existing)]
    ]:
        raise ValueError("MuSiQue v70 resumable QA prefix changed")
    if any(row.get("cache_key") != cache_key for row in existing):
        raise ValueError("MuSiQue v70 resumable QA cache key changed")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(len(existing), len(rows), chunk_rows):
            predictions = verifier.verify(
                rows[start : start + chunk_rows], cache_key=cache_key
            )
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
    result = list(read_jsonl(raw_path))
    if len(result) != len(rows):
        raise ValueError("MuSiQue v70 QA cache is incomplete")
    return result


def qa(args: argparse.Namespace) -> None:
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    prepared, _, direct_inputs, _ = _validate_prepared(args, paths)
    verifier = experiment.LocalRobertaQASupportVerifier(
        args.qa_model,
        device=args.device,
        batch_size=args.qa_batch_size,
        fp16=not args.no_fp16,
    )
    direct_key = _cache_key(
        args, paths["execution"], paths["direct_input"], phase="direct"
    )
    direct_raw = _run_qa_rows(
        verifier,
        direct_inputs,
        raw_path=paths["direct_raw"],
        cache_key=direct_key,
        chunk_rows=args.qa_chunk_rows,
    )
    direct_decisions = experiment.aggregate_qa_rows(
        direct_inputs, direct_raw, cache_key=direct_key
    )
    _write_jsonl(paths["direct_decisions"], direct_decisions)
    states = experiment.initial_chain_states(prepared)
    all_hop_decisions: list[dict[str, Any]] = []
    hop_artifacts: dict[str, Any] = {}
    for hop_index in range(1, 5):
        hop_paths = _hop_paths(args, hop_index)
        hop_inputs, blocked = experiment.build_hop_qa_inputs(
            prepared, states, hop_index=hop_index
        )
        if hop_paths["input"].exists():
            if list(read_jsonl(hop_paths["input"])) != hop_inputs:
                raise ValueError("MuSiQue v70 dynamic hop input changed")
        else:
            _write_jsonl(hop_paths["input"], hop_inputs)
        if hop_inputs:
            hop_key = _cache_key(
                args,
                paths["execution"],
                hop_paths["input"],
                phase=f"hop_{hop_index}",
            )
            raw = _run_qa_rows(
                verifier,
                hop_inputs,
                raw_path=hop_paths["raw"],
                cache_key=hop_key,
                chunk_rows=args.qa_chunk_rows,
            )
            aggregated = experiment.aggregate_qa_rows(
                hop_inputs, raw, cache_key=hop_key
            )
        else:
            raw = []
            aggregated = []
        decisions = experiment.apply_hop_results(
            prepared, states, aggregated, blocked, hop_index=hop_index
        )
        _write_jsonl(hop_paths["decisions"], decisions)
        all_hop_decisions.extend(decisions)
        hop_artifacts[str(hop_index)] = {
            "input_rows": len(hop_inputs),
            "blocked_cases": len(blocked),
            "raw_rows": len(raw),
            "decision_rows": len(decisions),
            "input_sha256": experiment.sha256(hop_paths["input"]),
            "raw_sha256": experiment.sha256(hop_paths["raw"])
            if hop_paths["raw"].exists()
            else None,
            "decisions_sha256": experiment.sha256(hop_paths["decisions"]),
        }
    feature_decisions = experiment.finalize_feature_decisions(
        prepared, direct_decisions, states, all_hop_decisions
    )
    _write_jsonl(paths["feature_decisions"], feature_decisions)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "cases": len(feature_decisions),
                "direct_paragraph_rows": len(direct_raw),
                "hop_artifacts": hop_artifacts,
                "direct_decisions_sha256": experiment.sha256(
                    paths["direct_decisions"]
                ),
                "feature_decisions_sha256": experiment.sha256(
                    paths["feature_decisions"]
                ),
                "execution_sha256": experiment.sha256(paths["execution"]),
                "runtime": execution["runtime"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _gold_evidence(
    args: argparse.Namespace, paths: dict[str, Path]
) -> tuple[
    list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    execution = _validate_execution(args, paths)
    prepared, _, direct_inputs, census = _validate_prepared(args, paths)
    direct_raw = list(read_jsonl(paths["direct_raw"]))
    direct_key = _cache_key(
        args, paths["execution"], paths["direct_input"], phase="direct"
    )
    direct_decisions = experiment.aggregate_qa_rows(
        direct_inputs, direct_raw, cache_key=direct_key
    )
    if direct_decisions != list(read_jsonl(paths["direct_decisions"])):
        raise ValueError("MuSiQue v70 direct decisions changed at gold join")
    feature_decisions = list(read_jsonl(paths["feature_decisions"]))
    selected, sampling = _select(args)
    evidence = experiment.build_gold_evidence(
        selected, prepared, feature_decisions
    )
    if sampling != census["sampling"]:
        raise ValueError("MuSiQue v70 deterministic sample changed at gold join")
    return evidence, sampling, census, execution


def _source_artifacts(
    args: argparse.Namespace,
    paths: dict[str, Path],
    execution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_sha256": experiment.sha256(args.protocol),
        "source_registration_sha256": experiment.sha256(
            args.source_registration
        ),
        "model_registration_sha256": experiment.sha256(args.model_registration),
        "implementation_registration_sha256": experiment.sha256(
            args.implementation
        ),
        "execution_registration_sha256": experiment.sha256(paths["execution"]),
        "prepared_blind_sha256": experiment.sha256(paths["prepared"]),
        "candidate_map_sha256": experiment.sha256(paths["candidate_map"]),
        "direct_qa_input_sha256": experiment.sha256(paths["direct_input"]),
        "direct_raw_qa_sha256": experiment.sha256(paths["direct_raw"]),
        "direct_decisions_sha256": experiment.sha256(
            paths["direct_decisions"]
        ),
        "hop_artifacts": {
            str(hop): {
                name: experiment.sha256(path)
                for name, path in _hop_paths(args, hop).items()
                if path.exists()
            }
            for hop in range(1, 5)
        },
        "feature_decisions_sha256": experiment.sha256(
            paths["feature_decisions"]
        ),
        "retrieval_scoring_started": False,
        "runtime": execution["runtime"],
    }


def calibrate(args: argparse.Namespace) -> None:
    if args.stage != "calibration":
        raise ValueError("MuSiQue v70 calibrate requires calibration stage")
    paths = _paths(args)
    if paths["result"].exists():
        raise FileExistsError("MuSiQue v70 calibration result is frozen")
    evidence, sampling, census, execution = _gold_evidence(args, paths)
    calibration = experiment.fit_calibration(evidence)
    experiment.write_calibration_report(
        calibration,
        evidence,
        result_path=paths["result"],
        report_path=paths["report"],
        evidence_path=paths["evidence"],
        metadata={
            "stage": "calibration",
            "cases": len(evidence),
            "answer_state_counts": {
                "answerable": sum(
                    row["answer_state"] == "answerable" for row in evidence
                ),
                "unanswerable": sum(
                    row["answer_state"] == "unanswerable" for row in evidence
                ),
            },
            "sampling": sampling,
            "structural_census": census["structural_census"],
            "source_artifacts": _source_artifacts(args, paths, execution),
            "gold_joined_after_complete_qa_and_feature_extraction": True,
            "total_learned_scalars_across_reported_methods": 101,
        },
    )
    print(json.dumps(calibration, ensure_ascii=False, sort_keys=True))


def _close_stage(
    args: argparse.Namespace,
    paths: dict[str, Path],
    report: dict[str, Any],
) -> None:
    outcome = report["analysis"]["outcome"]
    _write_json(
        paths["closure"],
        {
            "schema_version": "frc-musique-v70-stage-closure-v1",
            "experiment_id": experiment.EXPERIMENT_ID,
            "stage": args.stage,
            "stage_result_sha256": experiment.sha256(paths["result"]),
            "status": outcome["status"],
            "confirmation_open_authorized": outcome[
                "confirmation_open_authorized"
            ],
            "retrieval_scoring_started": False,
            "reuse_failed_stage_for_feature_model_threshold_rule_gate_or_selection": False,
            "selector_adoption_authorized": False,
            "gate_2": "NO-GO/SHADOW",
        },
    )
    if args.stage == "development" and outcome["confirmation_open_authorized"]:
        _write_json(
            paths["confirmation_open"],
            {
                "schema_version": "frc-musique-v70-confirmation-open-v1",
                "experiment_id": experiment.EXPERIMENT_ID,
                "confirmation_open_authorized": True,
                "development_result_sha256": experiment.sha256(paths["result"]),
                "development_closure_sha256": experiment.sha256(
                    paths["closure"]
                ),
                "calibration_result_sha256": experiment.sha256(
                    _paths(_with_stage(args, "calibration"))["result"]
                ),
                "no_feature_model_threshold_gate_or_method_change": True,
                "gate_2": "NO-GO/SHADOW",
            },
        )


def evaluate(args: argparse.Namespace) -> None:
    if args.stage == "calibration":
        raise ValueError("MuSiQue v70 evaluate requires development or confirmation")
    _validate_stage_authorization(args)
    paths = _paths(args)
    if paths["result"].exists() or paths["closure"].exists():
        raise FileExistsError("MuSiQue v70 stage result is frozen")
    evidence, sampling, census, execution = _gold_evidence(args, paths)
    calibration_path = _paths(_with_stage(args, "calibration"))["result"]
    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not calibration_payload["outcome"]["development_open_authorized"]:
        raise ValueError("MuSiQue v70 calibration result changed")
    report = experiment.evaluate_stage(
        evidence,
        stage=args.stage,
        calibration=calibration_payload["analysis"],
        sampling=sampling,
        structural_census=census["structural_census"],
        source_artifacts={
            **_source_artifacts(args, paths, execution),
            "calibration_result_sha256": experiment.sha256(calibration_path),
            "gold_joined_after_complete_qa_and_feature_extraction": True,
        },
    )
    experiment.write_stage_report(
        report,
        evidence,
        result_path=paths["result"],
        report_path=paths["report"],
        evidence_path=paths["evidence"],
    )
    _close_stage(args, paths, report)
    print(
        json.dumps(
            report["analysis"]["outcome"],
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def runtime(args: argparse.Namespace) -> None:
    print(json.dumps(_runtime(args), ensure_ascii=False, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("prepare", "register", "qa", "calibrate", "evaluate", "runtime"),
    )
    result.add_argument(
        "--stage",
        choices=("calibration", "development", "confirmation"),
        default="calibration",
    )
    result.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "musique_paragraph_competition_support_protocol_v70.json"
        ),
    )
    result.add_argument(
        "--source-registration",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "musique_paragraph_competition_support_source_registration_v70.json"
        ),
    )
    result.add_argument(
        "--model-registration",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "quac_roberta_qa_support_transfer_model_registration_v62.json"
        ),
    )
    result.add_argument(
        "--implementation",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "musique_paragraph_competition_support_implementation_v70.json"
        ),
    )
    result.add_argument(
        "--implementation-erratum",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/"
            "musique_paragraph_competition_support_implementation_erratum_v70.json"
        ),
    )
    result.add_argument(
        "--train-target",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique/data/musique_full_v1.0_train.jsonl"
        ),
    )
    result.add_argument(
        "--dev-target",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique/data/musique_full_v1.0_dev.jsonl"
        ),
    )
    result.add_argument(
        "--squad2-train",
        type=Path,
        default=Path(".cache/benchmarks/squad2_v2/train-v2.0.json"),
    )
    result.add_argument(
        "--prior-maps",
        type=Path,
        nargs=9,
        default=[
            Path(
                ".cache/benchmarks/"
                "musique_full_roberta_transfer_v65/candidate_map.jsonl"
            ),
            Path(
                ".cache/benchmarks/"
                "musique_full_roberta_transfer_v65_corrected/candidate_map.jsonl"
            ),
            Path(
                ".cache/benchmarks/"
                "musique_sequential_chain_support_v66/development/"
                "candidate_map.jsonl"
            ),
            Path(
                ".cache/benchmarks/"
                "musique_calibrated_chain_support_v67/calibration/"
                "candidate_map.jsonl"
            ),
            Path(
                ".cache/benchmarks/"
                "musique_calibrated_chain_support_v67/development/"
                "candidate_map.jsonl"
            ),
            Path(
                ".cache/benchmarks/"
                "musique_multisignal_chain_support_v68/calibration/"
                "candidate_map.jsonl"
            ),
            Path(
                ".cache/benchmarks/"
                "musique_multisignal_chain_support_v68/development/"
                "candidate_map.jsonl"
            ),
            Path(
                ".cache/benchmarks/"
                "musique_monotone_interaction_support_v69/calibration/"
                "candidate_map.jsonl"
            ),
            Path(
                ".cache/benchmarks/"
                "musique_monotone_interaction_support_v69/development/"
                "candidate_map.jsonl"
            ),
        ],
    )
    result.add_argument(
        "--qa-model",
        type=Path,
        default=Path(
            ".cache/benchmarks/models/"
            "deepset_roberta_base_squad2_adc3b06_v64_registered"
        ),
    )
    result.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique_paragraph_competition_support_v70"
        ),
    )
    result.add_argument(
        "--docs-root", type=Path, default=Path("docs/progressive_upgrade")
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "output/rag_evaluation/musique_paragraph_competition_support_v70"
        ),
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--qa-batch-size", type=int, default=16)
    result.add_argument("--qa-chunk-rows", type=int, default=64)
    result.add_argument("--no-fp16", action="store_true")
    result.add_argument("--registered-at", default="2026-08-04T08:10:00+08:00")
    return result


def main() -> int:
    args = parser().parse_args()
    commands = {
        "prepare": prepare,
        "register": register,
        "qa": qa,
        "calibrate": calibrate,
        "evaluate": evaluate,
        "runtime": runtime,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
