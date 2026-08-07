from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.hotpot_three_route_router import (
    load_router_artifact as load_history_router_artifact,
)
from research.frc_rag.musique_graph_router_transfer import load_source_commitments
from research.frc_rag.musique_mean_calibrated_three_route import load_calibration
from research.frc_rag.musique_target_three_route_router import (
    load_router_artifact as load_target_router_artifact,
)
from research.frc_rag.musique_target_three_route_transfer import (
    EXPERIMENT_ID,
    PRIOR_COMMITMENT_UNION,
    evaluate_stage,
    prepare_stage,
    validate_registered_protocol,
)
from research.frc_rag.twowiki_confirmation import (
    FrozenBgeScorer,
    score_cases_resumable,
)
from research.frc_rag.twowiki_question_router import (
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_support_path_closure import (
    read_json,
    read_jsonl,
    sha256,
)


DOCS_ROOT = REPOSITORY_ROOT / "docs/progressive_upgrade"
CACHE_ROOT = REPOSITORY_ROOT / ".cache/benchmarks/musique_target_three_route_v79"
OUTPUT_ROOT = REPOSITORY_ROOT / "output/rag_evaluation/musique_target_three_route_v79"
SOURCE_PATH = (
    REPOSITORY_ROOT / ".cache/benchmarks/musique/data/musique_full_v1.0_train.jsonl"
)
V77_GOLD_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/musique_graph_router_transfer_v77/development/sealed_gold.jsonl"
)
V78_GOLD_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/musique_mean_calibrated_three_route_v78/development/sealed_gold.jsonl"
)
V75_MODEL_PATH = (
    DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
)
V76_MODEL_PATH = DOCS_ROOT / "hotpot_graph_router_model_development_v76.json"
HISTORY_ROUTER_PATH = (
    DOCS_ROOT / "hotpot_three_route_router_model_development_v78.json"
)
CALIBRATION_PATH = DOCS_ROOT / "musique_three_route_mean_calibration_v78.json"
TARGET_ROUTER_PATH = (
    DOCS_ROOT / "musique_target_three_route_router_model_development_v79.json"
)
PROTOCOL_PATH = DOCS_ROOT / "musique_target_three_route_protocol_v79.json"
SOURCE_REGISTRATION_PATH = (
    DOCS_ROOT / "musique_target_three_route_source_registration_v79.json"
)
IMPLEMENTATION_LOCK_PATH = (
    DOCS_ROOT / "musique_target_three_route_implementation_v79.json"
)


def _paths(stage: str) -> dict[str, Path]:
    cache = CACHE_ROOT / stage
    output = OUTPUT_ROOT / stage
    return {
        "blind": cache / "blind_cases.jsonl",
        "gold": cache / "sealed_gold.jsonl",
        "scored": cache / "scored_blind.jsonl",
        "selection": cache / "selection_outputs.jsonl",
        "output": output,
        "execution": (
            DOCS_ROOT / f"musique_target_three_route_{stage}_execution_v79.json"
        ),
        "result": DOCS_ROOT / f"musique_target_three_route_{stage}_result_v79.json",
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_file(contract: dict[str, object]) -> Path:
    path = REPOSITORY_ROOT / str(contract["path"])
    if path.stat().st_size != int(contract["bytes"]) or sha256(path) != str(
        contract["sha256"]
    ):
        raise ValueError(f"v79 registered file changed: {contract['path']}")
    return path


def _verify_registration() -> tuple[
    dict[str, object], dict[str, object], list[Path]
]:
    protocol = read_json(PROTOCOL_PATH)
    source = read_json(SOURCE_REGISTRATION_PATH)
    lock = read_json(IMPLEMENTATION_LOCK_PATH)
    validate_registered_protocol(protocol)
    if (
        source.get("experiment_id") != EXPERIMENT_ID
        or lock.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("unexpected v79 registration experiment id")
    target = source["target"]
    if SOURCE_PATH.stat().st_size != int(target["bytes"]) or sha256(
        SOURCE_PATH
    ) != str(target["sha256"]):
        raise ValueError("v79 registered MuSiQue source changed")
    registry_path = _verify_file(source["prior_exclusion_registry"])
    v77_registry = read_json(registry_path)
    exclusion_paths: list[Path] = []
    for contract in v77_registry["prior_exclusion_sources"].values():
        if not isinstance(contract, dict) or "path" not in contract:
            continue
        path = _verify_file(contract)
        local = load_source_commitments([path])
        if len(local) != int(contract["expected_unique_source_commitments"]):
            raise ValueError(f"v79 prior exclusion count changed: {contract['path']}")
        exclusion_paths.append(path)
    if len(load_source_commitments(exclusion_paths)) != PRIOR_COMMITMENT_UNION:
        raise ValueError("v79 prior exclusion union changed")
    for key, expected_path, expected_rows in (
        ("v77_target_domain_calibration_ids", V77_GOLD_PATH, 800),
        ("v78_target_domain_training_ids", V78_GOLD_PATH, 600),
    ):
        registered_path = _verify_file(source[key])
        if registered_path != expected_path:
            raise ValueError(f"v79 registered exclusion path changed: {key}")
        rows = read_jsonl(registered_path)
        if len(rows) != expected_rows or len({str(row["id"]) for row in rows}) != len(
            rows
        ):
            raise ValueError(f"v79 registered exclusion ids changed: {key}")
    dependencies = source["frozen_dependencies"]
    for prefix, expected_path in (
        ("target_router", TARGET_ROUTER_PATH),
        ("history_router", HISTORY_ROUTER_PATH),
        ("mean_calibration", CALIBRATION_PATH),
        ("v76_router", V76_MODEL_PATH),
        ("v75_model", V75_MODEL_PATH),
    ):
        _verify_file(
            {
                "path": expected_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "bytes": dependencies[f"{prefix}_bytes"],
                "sha256": dependencies[f"{prefix}_sha256"],
            }
        )
    for prefix in ("target_training_scored", "target_training_cases"):
        _verify_file(
            {
                "path": dependencies[f"{prefix}_path"],
                "bytes": dependencies[f"{prefix}_bytes"],
                "sha256": dependencies[f"{prefix}_sha256"],
            }
        )
    target_router = load_target_router_artifact(TARGET_ROUTER_PATH)
    history_router = load_history_router_artifact(HISTORY_ROUTER_PATH)
    load_calibration(CALIBRATION_PATH, history_router)
    v75_artifact, _ = load_v75_model_artifact(V75_MODEL_PATH)
    if target_router["v75_router_model_payload_sha256"] != v75_artifact[
        "model_sha256"
    ]:
        raise ValueError("v79 target router and v75 model are inconsistent")
    for contract in lock.get("files", {}).values():
        _verify_file(contract)
    return protocol, source, exclusion_paths


def prepare(stage: str) -> dict[str, object]:
    protocol, source, exclusion_paths = _verify_registration()
    if stage == "confirmation":
        open_path = DOCS_ROOT / "musique_target_three_route_confirmation_open_v79.json"
        if (
            not open_path.is_file()
            or read_json(open_path).get("confirmation_open_authorized") is not True
        ):
            raise ValueError("v79 confirmation has not been authorized")
    paths = _paths(stage)
    preparation = prepare_stage(
        SOURCE_PATH,
        exclusion_paths,
        V77_GOLD_PATH,
        V78_GOLD_PATH,
        stage=stage,
        blind_path=paths["blind"],
        gold_path=paths["gold"],
    )
    execution: dict[str, object] = {
        "schema_version": "frc-musique-target-three-route-execution-v79",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "status": "PREPARED_BLIND_NOT_SCORED",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_registration_sha256": sha256(SOURCE_REGISTRATION_PATH),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK_PATH),
        "target_router_sha256": sha256(TARGET_ROUTER_PATH),
        "source_sha256": sha256(SOURCE_PATH),
        "preparation": preparation,
        "leakage_boundary": {
            "blind_rows_exclude_answer_decomposition_support_flags_and_gold_ids": True,
            "hop_count_is_id_derived_and_not_used_by_runtime_router": True,
            "v78_used_only_for_frozen_target_domain_model_selection": True,
            "gold_metrics_computed": False,
            "selection_outputs_created": False,
        },
        "frozen_resources": protocol["common_resources"],
        "registered_claims": source["claims"],
    }
    _write_json(paths["execution"], execution)
    return execution


def score(stage: str, *, hf_home: Path, device: str) -> dict[str, object]:
    _verify_registration()
    paths = _paths(stage)
    execution = read_json(paths["execution"])
    if execution.get("status") != "PREPARED_BLIND_NOT_SCORED":
        raise ValueError("v79 stage is not at the prepared boundary")
    started = time.perf_counter()

    def progress(index: int, total: int, case_id: str) -> None:
        if index == 1 or index % 10 == 0 or index == total:
            print(f"scored {index}/{total}: {case_id}", flush=True)

    scorer = FrozenBgeScorer(
        hf_home=hf_home,
        device=device,
        embedding_batch_size=32,
        rerank_batch_size=8,
    )
    count = score_cases_resumable(
        paths["blind"], paths["scored"], scorer=scorer, progress=progress
    )
    execution.update(
        {
            "status": "SCORED_BLIND_GOLD_NOT_JOINED",
            "scored_cases": count,
            "scored_blind_sha256": sha256(paths["scored"]),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    )
    _write_json(paths["execution"], execution)
    return execution


def evaluate(stage: str) -> dict[str, object]:
    protocol, _, _ = _verify_registration()
    paths = _paths(stage)
    execution = read_json(paths["execution"])
    if execution.get("status") != "SCORED_BLIND_GOLD_NOT_JOINED":
        raise ValueError("v79 stage is not at the scored pre-gold boundary")
    stage_overlap = 0
    if stage == "confirmation":
        development_ids = {
            str(row["id"]) for row in read_jsonl(_paths("development")["gold"])
        }
        confirmation_ids = {str(row["id"]) for row in read_jsonl(paths["gold"])}
        stage_overlap = len(development_ids & confirmation_ids)
    preparation = execution["preparation"]
    report = evaluate_stage(
        paths["scored"],
        paths["gold"],
        paths["selection"],
        TARGET_ROUTER_PATH,
        HISTORY_ROUTER_PATH,
        V76_MODEL_PATH,
        CALIBRATION_PATH,
        V75_MODEL_PATH,
        stage=stage,
        seed=int(protocol["metrics"][f"{stage}_seed"]),
        output_dir=paths["output"],
        prior_overlap=int(preparation["prior_source_overlap"]),
        v77_overlap=int(preparation["v77_calibration_overlap"]),
        v78_overlap=int(preparation["v78_training_overlap"]),
        stage_overlap=stage_overlap,
    )
    _write_json(paths["result"], report)
    outcome = report["analysis"]["outcome"]
    execution.update(
        {
            "status": "EVALUATED_GOLD_JOINED",
            "result_sha256": sha256(paths["result"]),
            "selection_output_sha256": sha256(paths["selection"]),
            "stage_gate_passed": bool(outcome["stage_gate_passed"]),
            "confirmation_open_authorized": bool(
                outcome["confirmation_open_authorized"]
            ),
        }
    )
    _write_json(paths["execution"], execution)
    if stage == "development":
        if outcome["confirmation_open_authorized"]:
            _write_json(
                DOCS_ROOT / "musique_target_three_route_confirmation_open_v79.json",
                {
                    "schema_version": (
                        "frc-musique-target-three-route-confirmation-open-v79"
                    ),
                    "experiment_id": EXPERIMENT_ID,
                    "confirmation_open_authorized": True,
                    "development_result_sha256": sha256(paths["result"]),
                    "all_strict_development_gates_passed": True,
                    "candidate_model_threshold_or_gate_changed": False,
                },
            )
        else:
            _write_json(
                DOCS_ROOT / "musique_target_three_route_development_closure_v79.json",
                {
                    "schema_version": (
                        "frc-musique-target-three-route-development-closure-v79"
                    ),
                    "experiment_id": EXPERIMENT_ID,
                    "status": outcome["status"],
                    "development_result_sha256": sha256(paths["result"]),
                    "strict_gate_passed": False,
                    "noninferiority_supported": bool(
                        outcome["noninferiority_envelope_supported"]
                    ),
                    "confirmation_open_authorized": False,
                    "candidate_model_threshold_or_gate_changed": False,
                    "gate_2": "NO-GO/SHADOW",
                },
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the v79 target-trained three-route experiment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--stage", choices=("development", "confirmation"), required=True
        )
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument(
        "--stage", choices=("development", "confirmation"), required=True
    )
    score_parser.add_argument("--hf-home", type=Path, required=True)
    score_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.stage)
    elif args.command == "score":
        result = score(args.stage, hf_home=args.hf_home, device=args.device)
    else:
        result = evaluate(args.stage)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
