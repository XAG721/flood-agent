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

from research.frc_rag.hotpot_graph_router import load_router_artifact
from research.frc_rag.musique_graph_router_transfer import (
    EXPERIMENT_ID,
    PRIOR_COMMITMENT_UNION,
    evaluate_stage,
    load_source_commitments,
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
CACHE_ROOT = REPOSITORY_ROOT / ".cache/benchmarks/musique_graph_router_transfer_v77"
OUTPUT_ROOT = REPOSITORY_ROOT / "output/rag_evaluation/musique_graph_router_transfer_v77"
SOURCE_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/musique/data/musique_full_v1.0_train.jsonl"
)
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
ROUTER_MODEL_PATH = DOCS_ROOT / "hotpot_graph_router_model_development_v76.json"
PROTOCOL_PATH = DOCS_ROOT / "musique_graph_router_transfer_protocol_v77.json"
SOURCE_REGISTRATION_PATH = (
    DOCS_ROOT / "musique_graph_router_transfer_source_registration_v77.json"
)
IMPLEMENTATION_LOCK_PATH = (
    DOCS_ROOT / "musique_graph_router_transfer_implementation_v77.json"
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
        "execution": DOCS_ROOT / f"musique_graph_router_transfer_{stage}_execution_v77.json",
        "result": DOCS_ROOT / f"musique_graph_router_transfer_{stage}_result_v77.json",
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_registration() -> tuple[
    dict[str, object], dict[str, object], list[Path]
]:
    protocol = read_json(PROTOCOL_PATH)
    source = read_json(SOURCE_REGISTRATION_PATH)
    lock = read_json(IMPLEMENTATION_LOCK_PATH)
    validate_registered_protocol(protocol)
    if source.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v77 source registration experiment id")
    if lock.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v77 implementation lock experiment id")
    if sha256(SOURCE_PATH) != source["target"]["sha256"]:
        raise ValueError("v77 registered MuSiQue source changed")
    if sha256(ROUTER_MODEL_PATH) != source["frozen_dependencies"]["v76_router_sha256"]:
        raise ValueError("v77 registered v76 router artifact changed")
    if sha256(V75_MODEL_PATH) != source["frozen_dependencies"]["v75_model_sha256"]:
        raise ValueError("v77 registered v75 feature model changed")
    router = load_router_artifact(ROUTER_MODEL_PATH)
    v75_artifact, _ = load_v75_model_artifact(V75_MODEL_PATH)
    if router["v75_router_model_payload_sha256"] != v75_artifact["model_sha256"]:
        raise ValueError("v77 v76 router and v75 feature model are inconsistent")
    exclusion_paths: list[Path] = []
    for contract in source["prior_exclusion_sources"].values():
        if not isinstance(contract, dict) or "path" not in contract:
            continue
        path = REPOSITORY_ROOT / str(contract["path"])
        if path.stat().st_size != int(contract["bytes"]) or sha256(path) != str(
            contract["sha256"]
        ):
            raise ValueError(f"v77 prior exclusion source changed: {contract['path']}")
        local = load_source_commitments([path])
        if len(local) != int(contract["expected_unique_source_commitments"]):
            raise ValueError(f"v77 prior exclusion count changed: {contract['path']}")
        exclusion_paths.append(path)
    if len(load_source_commitments(exclusion_paths)) != PRIOR_COMMITMENT_UNION:
        raise ValueError("v77 prior exclusion union changed")
    for contract in lock.get("files", {}).values():
        path = REPOSITORY_ROOT / str(contract["path"])
        if path.stat().st_size != int(contract["bytes"]) or sha256(path) != str(
            contract["sha256"]
        ):
            raise ValueError(f"v77 frozen file changed: {contract['path']}")
    return protocol, source, exclusion_paths


def prepare(stage: str) -> dict[str, object]:
    protocol, source, exclusion_paths = _verify_registration()
    if stage == "confirmation":
        open_path = DOCS_ROOT / "musique_graph_router_transfer_confirmation_open_v77.json"
        if (
            not open_path.is_file()
            or read_json(open_path).get("confirmation_open_authorized") is not True
        ):
            raise ValueError("v77 confirmation has not been authorized")
    paths = _paths(stage)
    preparation = prepare_stage(
        SOURCE_PATH,
        exclusion_paths,
        stage=stage,
        blind_path=paths["blind"],
        gold_path=paths["gold"],
    )
    execution: dict[str, object] = {
        "schema_version": "frc-musique-graph-router-transfer-execution-v77",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "status": "PREPARED_BLIND_NOT_SCORED",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_registration_sha256": sha256(SOURCE_REGISTRATION_PATH),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK_PATH),
        "router_model_sha256": sha256(ROUTER_MODEL_PATH),
        "source_sha256": sha256(SOURCE_PATH),
        "preparation": preparation,
        "leakage_boundary": {
            "blind_rows_exclude_answer_decomposition_support_flags_and_gold_ids": True,
            "hop_count_is_id_derived_and_not_used_by_runtime_router": True,
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
        raise ValueError("v77 stage is not at the prepared boundary")
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
        raise ValueError("v77 stage is not at the scored pre-gold boundary")
    stage_overlap = 0
    if stage == "confirmation":
        development_ids = {
            str(row["id"]) for row in read_jsonl(_paths("development")["gold"])
        }
        confirmation_ids = {str(row["id"]) for row in read_jsonl(paths["gold"])}
        stage_overlap = len(development_ids & confirmation_ids)
    report = evaluate_stage(
        paths["scored"],
        paths["gold"],
        paths["selection"],
        ROUTER_MODEL_PATH,
        V75_MODEL_PATH,
        stage=stage,
        seed=int(protocol["metrics"][f"{stage}_seed"]),
        output_dir=paths["output"],
        prior_overlap=int(execution["preparation"]["prior_source_overlap"]),
        stage_overlap=stage_overlap,
    )
    _write_json(paths["result"], report)
    execution.update(
        {
            "status": "EVALUATED",
            "selection_output_sha256": sha256(paths["selection"]),
            "result_sha256": sha256(paths["output"] / "result.json"),
            "report_sha256": sha256(paths["output"] / "report.md"),
            "cases_sha256": sha256(paths["output"] / "cases.jsonl.gz"),
            "leakage_boundary": {
                "blind_rows_exclude_answer_decomposition_support_flags_and_gold_ids": True,
                "hop_count_is_id_derived_and_not_used_by_runtime_router": True,
                "gold_metrics_computed": True,
                "selection_outputs_created_before_gold_join": True,
            },
        }
    )
    _write_json(paths["execution"], execution)
    outcome = report["analysis"]["outcome"]
    if stage == "development":
        target = (
            DOCS_ROOT / "musique_graph_router_transfer_confirmation_open_v77.json"
            if outcome["confirmation_open_authorized"]
            else DOCS_ROOT / "musique_graph_router_transfer_development_closure_v77.json"
        )
        _write_json(
            target,
            {
                "schema_version": "frc-musique-graph-router-transfer-development-decision-v77",
                "experiment_id": EXPERIMENT_ID,
                "development_result_sha256": sha256(paths["result"]),
                "status": outcome["status"],
                "strict_transfer_gate_passed": outcome["stage_gate_passed"],
                "noninferiority_envelope_supported": outcome[
                    "noninferiority_envelope_supported"
                ],
                "confirmation_open_authorized": outcome[
                    "confirmation_open_authorized"
                ],
                "feature_model_threshold_controls_gates_or_stage_size_changed": False,
                "reuse_development_for_selection_or_tuning": False,
                "selector_adoption_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        )
    else:
        _write_json(
            DOCS_ROOT / "musique_graph_router_transfer_confirmation_closure_v77.json",
            {
                "schema_version": "frc-musique-graph-router-transfer-confirmation-closure-v77",
                "experiment_id": EXPERIMENT_ID,
                "confirmation_result_sha256": sha256(paths["result"]),
                "status": outcome["status"],
                "confirmation_gate_passed": outcome["stage_gate_passed"],
                "reuse_target_stages_for_selection_or_tuning": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen v76 graph router on untouched MuSiQue cases"
    )
    parser.add_argument("command", choices=("prepare", "score", "evaluate", "run"))
    parser.add_argument(
        "--stage", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--hf-home", type=Path, default=Path("D:/RAG_test/.hf_cache"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command in {"prepare", "run"}:
        print(json.dumps(prepare(args.stage), ensure_ascii=False, sort_keys=True))
    if args.command in {"score", "run"}:
        print(
            json.dumps(
                score(args.stage, hf_home=args.hf_home, device=args.device),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    if args.command in {"evaluate", "run"}:
        print(json.dumps(evaluate(args.stage), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
