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

from research.frc_rag.hotpot_graph_router import (
    EXPERIMENT_ID,
    develop_model,
    evaluate_stage,
    load_router_artifact,
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
CACHE_ROOT = REPOSITORY_ROOT / ".cache/benchmarks/hotpot_graph_router_v76"
OUTPUT_ROOT = REPOSITORY_ROOT / "output/rag_evaluation/hotpot_graph_router_v76"
SOURCE_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/hotpot_hf/distractor/validation-00000-of-00001.parquet"
)
HISTORY_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/frc_public_reference/role_scores_hotpotqa.jsonl"
)
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
MODEL_PATH = DOCS_ROOT / "hotpot_graph_router_model_development_v76.json"
PROTOCOL_PATH = DOCS_ROOT / "hotpot_graph_router_protocol_v76.json"
SOURCE_REGISTRATION_PATH = (
    DOCS_ROOT / "hotpot_graph_router_source_registration_v76.json"
)
IMPLEMENTATION_LOCK_PATH = DOCS_ROOT / "hotpot_graph_router_implementation_v76.json"


def _paths(stage: str) -> dict[str, Path]:
    cache = CACHE_ROOT / stage
    output = OUTPUT_ROOT / stage
    return {
        "blind": cache / "blind_cases.jsonl",
        "gold": cache / "sealed_gold.jsonl",
        "scored": cache / "scored_blind.jsonl",
        "selection": cache / "selection_outputs.jsonl",
        "output": output,
        "execution": DOCS_ROOT / f"hotpot_graph_router_{stage}_execution_v76.json",
        "result": DOCS_ROOT / f"hotpot_graph_router_{stage}_result_v76.json",
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def develop() -> dict[str, object]:
    artifact = develop_model(HISTORY_PATH, V75_MODEL_PATH)
    artifact["history_artifacts"] = {
        "path": ".cache/benchmarks/frc_public_reference/role_scores_hotpotqa.jsonl",
        "bytes": HISTORY_PATH.stat().st_size,
        "sha256": sha256(HISTORY_PATH),
        "v75_model_path": "docs/progressive_upgrade/twowiki_question_router_model_development_v75.json",
        "v75_model_sha256": sha256(V75_MODEL_PATH),
    }
    _write_json(MODEL_PATH, artifact)
    return artifact


def _verify_registration() -> tuple[dict[str, object], dict[str, object]]:
    protocol = read_json(PROTOCOL_PATH)
    source = read_json(SOURCE_REGISTRATION_PATH)
    implementation_lock = read_json(IMPLEMENTATION_LOCK_PATH)
    router_artifact = load_router_artifact(MODEL_PATH)
    v75_artifact, _ = load_v75_model_artifact(V75_MODEL_PATH)
    validate_registered_protocol(protocol)
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v76 protocol experiment id")
    if source.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v76 source registration experiment id")
    if sha256(SOURCE_PATH) != source["source"]["sha256"]:
        raise ValueError("v76 registered source hash changed")
    if sha256(HISTORY_PATH) != source["history_exclusion"]["sha256"]:
        raise ValueError("v76 registered history hash changed")
    if sha256(V75_MODEL_PATH) != source["frozen_dependencies"]["v75_model_sha256"]:
        raise ValueError("v76 registered v75 model file changed")
    if (
        router_artifact["v75_router_model_payload_sha256"]
        != v75_artifact["model_sha256"]
    ):
        raise ValueError("v76 registered v75 model payload changed")
    if sha256(MODEL_PATH) != protocol["candidate"]["model_artifact_sha256"]:
        raise ValueError("v76 router model artifact changed")
    if implementation_lock.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v76 implementation lock experiment id")
    for contract in implementation_lock.get("files", {}).values():
        path = REPOSITORY_ROOT / str(contract["path"])
        if sha256(path) != contract["sha256"]:
            raise ValueError(f"v76 frozen file changed: {contract['path']}")
    return protocol, source


def prepare(stage: str) -> dict[str, object]:
    protocol, source = _verify_registration()
    if stage == "confirmation":
        open_path = DOCS_ROOT / "hotpot_graph_router_confirmation_open_v76.json"
        if (
            not open_path.is_file()
            or read_json(open_path).get("confirmation_open_authorized") is not True
        ):
            raise ValueError("v76 confirmation has not been authorized")
    paths = _paths(stage)
    preparation = prepare_stage(
        SOURCE_PATH,
        HISTORY_PATH,
        stage=stage,
        blind_path=paths["blind"],
        gold_path=paths["gold"],
    )
    execution: dict[str, object] = {
        "schema_version": "frc-hotpot-graph-router-execution-v76",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "status": "PREPARED_BLIND_NOT_SCORED",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_registration_sha256": sha256(SOURCE_REGISTRATION_PATH),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK_PATH),
        "model_artifact_sha256": sha256(MODEL_PATH),
        "source_sha256": sha256(SOURCE_PATH),
        "preparation": preparation,
        "leakage_boundary": {
            "blind_rows_exclude_answer_gold_ids_gold_flags_and_gold_roles": True,
            "official_type_answer_or_gold_used_by_runtime_router": False,
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
        raise ValueError("v76 stage is not at the prepared boundary")
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
        paths["blind"],
        paths["scored"],
        scorer=scorer,
        progress=progress,
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
    protocol, _ = _verify_registration()
    paths = _paths(stage)
    execution = read_json(paths["execution"])
    if execution.get("status") != "SCORED_BLIND_GOLD_NOT_JOINED":
        raise ValueError("v76 stage is not at the scored pre-gold boundary")
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
        MODEL_PATH,
        V75_MODEL_PATH,
        stage=stage,
        seed=int(protocol["metrics"][f"{stage}_seed"]),
        output_dir=paths["output"],
        history_overlap=int(execution["preparation"]["history_overlap"]),
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
                "blind_rows_exclude_answer_gold_ids_gold_flags_and_gold_roles": True,
                "official_type_answer_or_gold_used_by_runtime_router": False,
                "gold_metrics_computed": True,
                "selection_outputs_created_before_gold_join": True,
            },
        }
    )
    _write_json(paths["execution"], execution)
    outcome = report["analysis"]["outcome"]
    if stage == "development":
        target = (
            DOCS_ROOT / "hotpot_graph_router_confirmation_open_v76.json"
            if outcome["confirmation_open_authorized"]
            else DOCS_ROOT / "hotpot_graph_router_development_closure_v76.json"
        )
        _write_json(
            target,
            {
                "schema_version": "frc-hotpot-graph-router-development-decision-v76",
                "experiment_id": EXPERIMENT_ID,
                "development_result_sha256": sha256(paths["result"]),
                "status": outcome["status"],
                "confirmation_open_authorized": outcome["confirmation_open_authorized"],
                "feature_model_threshold_controls_or_gates_changed": False,
                "reuse_development_for_feature_model_threshold_gate_or_selection": False,
                "selector_adoption_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        )
    else:
        _write_json(
            DOCS_ROOT / "hotpot_graph_router_confirmation_closure_v76.json",
            {
                "schema_version": "frc-hotpot-graph-router-confirmation-closure-v76",
                "experiment_id": EXPERIMENT_ID,
                "confirmation_result_sha256": sha256(paths["result"]),
                "status": outcome["status"],
                "confirmation_gate_passed": outcome["stage_gate_passed"],
                "reuse_target_stages_for_feature_model_threshold_gate_or_selection": False,
                "selector_adoption_authorized": False,
                "canary_or_default_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen HotpotQA v76 router")
    parser.add_argument(
        "command", choices=("develop", "prepare", "score", "evaluate", "run")
    )
    parser.add_argument(
        "--stage", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--hf-home", type=Path, default=Path("D:/RAG_test/.hf_cache"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "develop":
        result = develop()
        print(
            json.dumps(
                {
                    "experiment_id": result["experiment_id"],
                    "history_cases": result["history"]["cases"],
                    "crossfit": result["crossfit"],
                    "model_sha256": result["model_sha256"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if args.command in {"prepare", "run"}:
        print(
            json.dumps(prepare(args.stage), ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    if args.command in {"score", "run"}:
        print(
            json.dumps(
                score(args.stage, hf_home=args.hf_home, device=args.device),
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    if args.command in {"evaluate", "run"}:
        print(
            json.dumps(evaluate(args.stage), ensure_ascii=False, sort_keys=True),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
