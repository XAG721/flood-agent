from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.frc_rag import musique_sequential_chain_support as experiment  # noqa: E402
from research.frc_rag import squad2_calibrated_roberta_support as v64  # noqa: E402


MODULE_PATH = ROOT / "research/frc_rag/musique_sequential_chain_support.py"
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = ROOT / "tests/test_frc_musique_sequential_chain_support.py"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
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
    stage_cache = args.cache_root / args.stage
    stage_output = args.output_root / args.stage
    return {
        "prepared": stage_cache / "prepared_blind.jsonl",
        "candidate_map": stage_cache / "candidate_map.jsonl",
        "direct_input": stage_cache / "direct_qa_input.jsonl",
        "census": stage_cache / "blind_census.json",
        "execution": args.docs_root
        / f"musique_sequential_chain_support_{args.stage}_execution_v66.json",
        "direct_raw": stage_cache / "direct_raw_qa.jsonl",
        "direct_decisions": stage_cache / "direct_case_decisions.jsonl",
        "chain_decisions": stage_cache / "sequential_chain_decisions.jsonl",
        "result": args.docs_root
        / f"musique_sequential_chain_support_{args.stage}_result_v66.json",
        "closure": args.docs_root
        / f"musique_sequential_chain_support_{args.stage}_closure_v66.json",
        "confirmation_open": args.docs_root
        / "musique_sequential_chain_support_confirmation_open_v66.json",
        "report": stage_output / "report.md",
        "evidence": stage_output / "cases.jsonl.gz",
    }


def _hop_paths(args: argparse.Namespace, hop_index: int) -> dict[str, Path]:
    root = args.cache_root / args.stage
    return {
        "input": root / f"hop_{hop_index}_qa_input.jsonl",
        "raw": root / f"hop_{hop_index}_raw_qa.jsonl",
        "decisions": root / f"hop_{hop_index}_decisions.jsonl",
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
            "locked_threshold": experiment.LOCKED_THRESHOLD,
            "offline_model_loading": True,
        },
    }


def _validate_implementation(args: argparse.Namespace) -> dict[str, Any]:
    experiment.validate_protocol(args.protocol)
    experiment.validate_source_registration(
        args.source_registration,
        target_path=args.target,
        squad2_train_path=args.squad2_train,
        invalid_map_path=args.v65_invalid_map,
        corrected_map_path=args.v65_corrected_map,
    )
    v64.validate_model_registration(args.model_registration, model_root=args.qa_model)
    value = json.loads(args.implementation.read_text(encoding="utf-8"))
    expected = {
        "protocol": experiment.sha256(args.protocol),
        "source_registration": experiment.sha256(args.source_registration),
        "model_registration": experiment.sha256(args.model_registration),
        "module": experiment.sha256(MODULE_PATH),
        "runner": experiment.sha256(RUNNER_PATH),
        "tests": experiment.sha256(TEST_PATH),
    }
    if value.get("hashes") != expected:
        raise ValueError("MuSiQue v66 implementation registration changed")
    if value.get("target_rows_read_for_v66_before_registration") != 0:
        raise ValueError("MuSiQue v66 target rows predate implementation freeze")
    if value.get("source_independent_synthetic_fixtures_only") is not True:
        raise ValueError("MuSiQue v66 implementation was not frozen synthetically")
    return value


def _guard_inputs(args: argparse.Namespace) -> tuple[set[str], set[str]]:
    squad_source = json.loads(args.squad2_train.read_text(encoding="utf-8"))
    squad_questions = experiment.extract_squad2_questions(squad_source)
    invalid = experiment.load_source_commitments(read_jsonl(args.v65_invalid_map))
    corrected = experiment.load_source_commitments(read_jsonl(args.v65_corrected_map))
    if len(invalid) != 300 or len(corrected) != 600 or invalid & corrected:
        raise ValueError("MuSiQue v66 v65 exclusion union changed")
    return squad_questions, invalid | corrected


def _development_commitments(args: argparse.Namespace) -> set[str]:
    path = _paths(argparse.Namespace(**{**vars(args), "stage": "development"}))[
        "candidate_map"
    ]
    if not path.exists():
        raise ValueError("MuSiQue v66 development commitments are unavailable")
    commitments = experiment.load_source_commitments(read_jsonl(path))
    if len(commitments) != experiment.TARGET_CASES:
        raise ValueError("MuSiQue v66 development commitment count changed")
    return commitments


def _validate_confirmation_open(args: argparse.Namespace) -> dict[str, Any]:
    dev_args = argparse.Namespace(**{**vars(args), "stage": "development"})
    dev_paths = _paths(dev_args)
    value = json.loads(dev_paths["confirmation_open"].read_text(encoding="utf-8"))
    if value["development_result_sha256"] != experiment.sha256(dev_paths["result"]):
        raise ValueError("MuSiQue v66 confirmation-open result hash changed")
    if value["confirmation_open_authorized"] is not True:
        raise ValueError("MuSiQue v66 confirmation was not authorized")
    return value


def _select(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    squad_questions, excluded = _guard_inputs(args)
    if args.stage == "confirmation":
        _validate_confirmation_open(args)
        excluded |= _development_commitments(args)
    return experiment.select_stage_sample(
        read_jsonl(args.target),
        stage=args.stage,
        squad_questions=squad_questions,
        excluded_source_commitments=excluded,
    )


def _validate_prepared(
    paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prepared = list(read_jsonl(paths["prepared"]))
    maps = list(read_jsonl(paths["candidate_map"]))
    direct_inputs = list(read_jsonl(paths["direct_input"]))
    census = json.loads(paths["census"].read_text(encoding="utf-8"))
    if len(prepared) != experiment.TARGET_CASES or len(maps) != len(prepared):
        raise ValueError("MuSiQue v66 prepared caches are incomplete")
    if [row["id"] for row in prepared] != [row["id"] for row in maps]:
        raise ValueError("MuSiQue v66 prepared case order changed")
    expected_ids = [
        f"{row['id']}::direct::p{context['paragraph_idx']}"
        for row in prepared
        for context in row["contexts"]
    ]
    if [row["id"] for row in direct_inputs] != expected_ids:
        raise ValueError("MuSiQue v66 direct QA input order changed")
    return prepared, maps, direct_inputs, census


def prepare(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    paths = _paths(args)
    for path in (
        paths["prepared"],
        paths["candidate_map"],
        paths["direct_input"],
        paths["census"],
    ):
        if path.exists():
            raise FileExistsError(f"MuSiQue v66 preparation is frozen: {path}")
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
            "schema_version": "frc-musique-v66-blind-census-v1",
            "experiment_id": experiment.EXPERIMENT_ID,
            "stage": args.stage,
            "sampling": sampling,
            "structural_census": structural,
            "gold_joined_for_metrics": False,
            "qa_started": False,
        },
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "prepared_cases": len(prepared),
                "direct_qa_input_rows": len(direct_inputs),
                "sampling": sampling,
                "structural_census": structural,
                "prepared_sha256": experiment.sha256(paths["prepared"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def register(args: argparse.Namespace) -> None:
    _validate_implementation(args)
    paths = _paths(args)
    prepared, maps, direct_inputs, census = _validate_prepared(paths)
    if paths["execution"].exists():
        raise FileExistsError("MuSiQue v66 execution registration is frozen")
    prior_open_sha = None
    if args.stage == "confirmation":
        prior_open_sha = experiment.sha256(
            _paths(args)["confirmation_open"]
        )
    _write_json(
        paths["execution"],
        {
            "schema_version": "frc-musique-v66-execution-registration-v1",
            "experiment_id": experiment.EXPERIMENT_ID,
            "stage": args.stage,
            "registered_at": args.registered_at,
            "gold_joined_for_metrics": False,
            "qa_started": False,
            "model_or_threshold_adjusted": False,
            "hashes": {
                "implementation_registration_sha256": experiment.sha256(
                    args.implementation
                ),
                "target_source_sha256": experiment.sha256(args.target),
                "squad2_guard_source_sha256": experiment.sha256(args.squad2_train),
                "v65_invalid_map_sha256": experiment.sha256(args.v65_invalid_map),
                "v65_corrected_map_sha256": experiment.sha256(
                    args.v65_corrected_map
                ),
                "prepared_blind_sha256": experiment.sha256(paths["prepared"]),
                "candidate_map_sha256": experiment.sha256(paths["candidate_map"]),
                "direct_qa_input_sha256": experiment.sha256(paths["direct_input"]),
                "blind_census_sha256": experiment.sha256(paths["census"]),
                "confirmation_open_sha256": prior_open_sha,
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
    _, _, direct_inputs, census = _validate_prepared(paths)
    value = json.loads(paths["execution"].read_text(encoding="utf-8"))
    if value["stage"] != args.stage or value["gold_joined_for_metrics"]:
        raise ValueError("MuSiQue v66 execution boundary changed")
    expected = {
        "implementation_registration_sha256": experiment.sha256(args.implementation),
        "target_source_sha256": experiment.sha256(args.target),
        "squad2_guard_source_sha256": experiment.sha256(args.squad2_train),
        "v65_invalid_map_sha256": experiment.sha256(args.v65_invalid_map),
        "v65_corrected_map_sha256": experiment.sha256(args.v65_corrected_map),
        "prepared_blind_sha256": experiment.sha256(paths["prepared"]),
        "candidate_map_sha256": experiment.sha256(paths["candidate_map"]),
        "direct_qa_input_sha256": experiment.sha256(paths["direct_input"]),
        "blind_census_sha256": experiment.sha256(paths["census"]),
        "confirmation_open_sha256": (
            experiment.sha256(paths["confirmation_open"])
            if args.stage == "confirmation"
            else None
        ),
    }
    if value["hashes"] != expected:
        raise ValueError("MuSiQue v66 execution registration changed")
    if value["sampling"] != census["sampling"]:
        raise ValueError("MuSiQue v66 execution sampling changed")
    if value["direct_qa_input_rows"] != len(direct_inputs):
        raise ValueError("MuSiQue v66 direct QA row count changed")
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
            "model_registration_sha256": experiment.sha256(args.model_registration),
            "input_sha256": experiment.sha256(input_path),
            "locked_threshold": experiment.LOCKED_THRESHOLD,
            "max_length": 384,
            "doc_stride": 128,
            "maximum_answer_tokens": 30,
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
    if [row["id"] for row in existing] != [row["id"] for row in rows[: len(existing)]]:
        raise ValueError("MuSiQue v66 resumable QA prefix changed")
    if any(row.get("cache_key") != cache_key for row in existing):
        raise ValueError("MuSiQue v66 resumable QA cache key changed")
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
        raise ValueError("MuSiQue v66 QA cache is incomplete")
    return result


def qa(args: argparse.Namespace) -> None:
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    prepared, _, direct_inputs, _ = _validate_prepared(paths)
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
                raise ValueError("MuSiQue v66 dynamic hop input changed")
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
            prepared,
            states,
            aggregated,
            blocked,
            hop_index=hop_index,
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
    chain_decisions = experiment.finalize_chain_decisions(
        prepared, states, all_hop_decisions
    )
    _write_jsonl(paths["chain_decisions"], chain_decisions)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "cases": len(chain_decisions),
                "direct_paragraph_rows": len(direct_raw),
                "hop_artifacts": hop_artifacts,
                "direct_decisions_sha256": experiment.sha256(
                    paths["direct_decisions"]
                ),
                "chain_decisions_sha256": experiment.sha256(
                    paths["chain_decisions"]
                ),
                "execution_sha256": experiment.sha256(paths["execution"]),
                "runtime": execution["runtime"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def support_evaluate(args: argparse.Namespace) -> None:
    paths = _paths(args)
    execution = _validate_execution(args, paths)
    prepared, _, direct_inputs, census = _validate_prepared(paths)
    direct_raw = list(read_jsonl(paths["direct_raw"]))
    direct_decisions = list(read_jsonl(paths["direct_decisions"]))
    chain_decisions = list(read_jsonl(paths["chain_decisions"]))
    if (
        len(direct_raw) != len(direct_inputs)
        or len(direct_decisions) != len(prepared)
        or len(chain_decisions) != len(prepared)
    ):
        raise ValueError("MuSiQue v66 execution caches are incomplete")
    if paths["result"].exists():
        raise FileExistsError("MuSiQue v66 stage result is frozen")
    selected, sampling = _select(args)
    if sampling != census["sampling"]:
        raise ValueError("MuSiQue v66 deterministic sample changed at gold join")
    evidence = experiment.build_gold_evidence(
        selected, prepared, direct_decisions, chain_decisions
    )
    hop_artifacts = {
        str(hop): {
            name: experiment.sha256(path)
            for name, path in _hop_paths(args, hop).items()
            if path.exists()
        }
        for hop in range(1, 5)
    }
    source_artifacts = {
        "protocol_sha256": experiment.sha256(args.protocol),
        "source_registration_sha256": experiment.sha256(args.source_registration),
        "model_registration_sha256": experiment.sha256(args.model_registration),
        "implementation_registration_sha256": experiment.sha256(args.implementation),
        "execution_registration_sha256": experiment.sha256(paths["execution"]),
        "prepared_blind_sha256": experiment.sha256(paths["prepared"]),
        "candidate_map_sha256": experiment.sha256(paths["candidate_map"]),
        "direct_qa_input_sha256": experiment.sha256(paths["direct_input"]),
        "direct_raw_qa_sha256": experiment.sha256(paths["direct_raw"]),
        "direct_decisions_sha256": experiment.sha256(paths["direct_decisions"]),
        "hop_artifacts": hop_artifacts,
        "chain_decisions_sha256": experiment.sha256(paths["chain_decisions"]),
        "gold_joined_after_complete_direct_and_sequential_qa": True,
        "retrieval_scoring_started": False,
        "sampling": sampling,
        "structural_census": census["structural_census"],
        "runtime": execution["runtime"],
    }
    report = experiment.evaluate_stage(
        evidence,
        stage=args.stage,
        sampling=sampling,
        structural_census=census["structural_census"],
        source_artifacts=source_artifacts,
    )
    experiment.write_stage_report(
        report,
        evidence,
        result_path=paths["result"],
        report_path=paths["report"],
        evidence_path=paths["evidence"],
    )
    outcome = report["analysis"]["outcome"]
    if args.stage == "development" and outcome["confirmation_open_authorized"]:
        _write_json(
            paths["confirmation_open"],
            {
                "schema_version": "frc-musique-v66-confirmation-open-v1",
                "experiment_id": experiment.EXPERIMENT_ID,
                "development_result_sha256": experiment.sha256(paths["result"]),
                "development_evidence_sha256": experiment.sha256(paths["evidence"]),
                "confirmation_open_authorized": True,
                "model_threshold_rule_or_gate_change_authorized": False,
                "selector_adoption_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        )
    else:
        _write_json(
            paths["closure"],
            {
                "schema_version": "frc-musique-v66-stage-closure-v1",
                "experiment_id": experiment.EXPERIMENT_ID,
                "stage": args.stage,
                "status": outcome["status"],
                "stage_result_sha256": experiment.sha256(paths["result"]),
                "confirmation_open_authorized": False,
                "retrieval_scoring_started": False,
                "reuse_failed_stage_for_tuning_threshold_rule_gate_or_selection": False,
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
        "command", choices=("prepare", "register", "qa", "support-evaluate", "runtime")
    )
    result.add_argument(
        "--stage", choices=("development", "confirmation"), default="development"
    )
    result.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/musique_sequential_chain_support_protocol_v66.json"
        ),
    )
    result.add_argument(
        "--source-registration",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/musique_sequential_chain_support_source_registration_v66.json"
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
            "docs/progressive_upgrade/musique_sequential_chain_support_implementation_erratum_v66.json"
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
        "--v65-invalid-map",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique_full_roberta_transfer_v65/candidate_map.jsonl"
        ),
    )
    result.add_argument(
        "--v65-corrected-map",
        type=Path,
        default=Path(
            ".cache/benchmarks/musique_full_roberta_transfer_v65_corrected/candidate_map.jsonl"
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
        default=Path(".cache/benchmarks/musique_sequential_chain_support_v66"),
    )
    result.add_argument(
        "--docs-root", type=Path, default=Path("docs/progressive_upgrade")
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "output/rag_evaluation/musique_sequential_chain_support_v66"
        ),
    )
    result.add_argument("--device", default="cuda")
    result.add_argument("--qa-batch-size", type=int, default=16)
    result.add_argument("--qa-chunk-rows", type=int, default=64)
    result.add_argument("--no-fp16", action="store_true")
    result.add_argument("--registered-at", default="2026-08-04T01:10:00+08:00")
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
