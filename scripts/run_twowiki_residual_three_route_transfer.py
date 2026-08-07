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
    load_router_artifact as load_v76_router_artifact,
)
from research.frc_rag.hotpot_three_route_router import (
    load_router_artifact as load_v78_router_artifact,
)
from research.frc_rag.musique_anchor_default_router import (
    load_router_artifact as load_v80_router_artifact,
)
from research.frc_rag.musique_target_three_route_router import (
    load_router_artifact as load_v79_router_artifact,
)
from research.frc_rag.twowiki_confirmation import FrozenBgeScorer, score_cases_resumable
from research.frc_rag.twowiki_question_router import (
    load_model_artifact as load_v75_model_artifact,
)
from research.frc_rag.twowiki_residual_three_route_router import (
    load_router_artifact as load_v81_router_artifact,
)
from research.frc_rag.twowiki_residual_three_route_transfer import (
    EXPERIMENT_ID,
    PRIOR_EXCLUDED_CASES,
    evaluate_stage,
    load_prior_excluded_ids,
    prepare_stage,
    validate_registered_protocol,
)
from research.frc_rag.twowiki_support_path_closure import read_json, read_jsonl, sha256


DOCS_ROOT = REPOSITORY_ROOT / "docs/progressive_upgrade"
CACHE_ROOT = REPOSITORY_ROOT / ".cache/benchmarks/twowiki_residual_router_v81"
OUTPUT_ROOT = REPOSITORY_ROOT / "output/rag_evaluation/twowiki_residual_router_v81"
SOURCE_PATH = REPOSITORY_ROOT / ".cache/benchmarks/2wikimultihopqa/dev.parquet"
HISTORY_POOL_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/frc_public_reference/candidate_pool_2wikimultihopqa.jsonl"
)
V74_GOLD_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/twowiki_support_path_closure_v74/development/sealed_gold.jsonl"
)
V75_DEVELOPMENT_GOLD_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/twowiki_question_router_v75/development/sealed_gold.jsonl"
)
V75_CONFIRMATION_GOLD_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/twowiki_question_router_v75/confirmation/sealed_gold.jsonl"
)
V75_MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
V76_MODEL_PATH = DOCS_ROOT / "hotpot_graph_router_model_development_v76.json"
V78_MODEL_PATH = DOCS_ROOT / "hotpot_three_route_router_model_development_v78.json"
V79_MODEL_PATH = DOCS_ROOT / "musique_target_three_route_router_model_development_v79.json"
V80_MODEL_PATH = DOCS_ROOT / "musique_anchor_default_router_model_development_v80.json"
V81_MODEL_PATH = (
    DOCS_ROOT / "twowiki_residual_three_route_router_model_development_v81.json"
)
PROTOCOL_PATH = DOCS_ROOT / "twowiki_residual_three_route_protocol_v81.json"
SOURCE_REGISTRATION_PATH = (
    DOCS_ROOT / "twowiki_residual_three_route_source_registration_v81.json"
)
IMPLEMENTATION_LOCK_PATH = (
    DOCS_ROOT / "twowiki_residual_three_route_implementation_v81.json"
)
CONFIRMATION_OPEN_PATH = (
    DOCS_ROOT / "twowiki_residual_three_route_confirmation_open_v81.json"
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
        "execution": DOCS_ROOT
        / f"twowiki_residual_three_route_{stage}_execution_v81.json",
        "result": DOCS_ROOT / f"twowiki_residual_three_route_{stage}_result_v81.json",
        "closure": DOCS_ROOT
        / f"twowiki_residual_three_route_{stage}_closure_v81.json",
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
        raise ValueError(f"v81 registered file changed: {contract['path']}")
    return path


def _verify_registration() -> tuple[dict[str, object], dict[str, object]]:
    protocol = read_json(PROTOCOL_PATH)
    source = read_json(SOURCE_REGISTRATION_PATH)
    lock = read_json(IMPLEMENTATION_LOCK_PATH)
    validate_registered_protocol(protocol)
    if (
        source.get("experiment_id") != EXPERIMENT_ID
        or lock.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("unexpected v81 registration experiment id")
    if _verify_file(source["target_source"]) != SOURCE_PATH:
        raise ValueError("v81 target source path changed")
    exclusion_paths = source["prior_exclusions"]
    registered_exclusions = (
        ("history", HISTORY_POOL_PATH),
        ("v74", V74_GOLD_PATH),
        ("v75_development", V75_DEVELOPMENT_GOLD_PATH),
        ("v75_confirmation", V75_CONFIRMATION_GOLD_PATH),
    )
    for name, expected_path in registered_exclusions:
        if _verify_file(exclusion_paths[name]) != expected_path:
            raise ValueError(f"v81 {name} exclusion path changed")
    excluded, _ = load_prior_excluded_ids(
        HISTORY_POOL_PATH,
        V74_GOLD_PATH,
        V75_DEVELOPMENT_GOLD_PATH,
        V75_CONFIRMATION_GOLD_PATH,
    )
    if len(excluded) != PRIOR_EXCLUDED_CASES:
        raise ValueError("v81 prior exclusion count changed")
    dependencies = source["frozen_dependencies"]
    dependency_paths = (
        ("v75_model", V75_MODEL_PATH),
        ("v76_router", V76_MODEL_PATH),
        ("v78_router", V78_MODEL_PATH),
        ("v79_router", V79_MODEL_PATH),
        ("v80_router", V80_MODEL_PATH),
        ("v81_router", V81_MODEL_PATH),
    )
    for name, expected_path in dependency_paths:
        if _verify_file(dependencies[name]) != expected_path:
            raise ValueError(f"v81 frozen dependency path changed: {name}")
    v75_artifact, _ = load_v75_model_artifact(V75_MODEL_PATH)
    load_v76_router_artifact(V76_MODEL_PATH)
    load_v78_router_artifact(V78_MODEL_PATH)
    load_v79_router_artifact(V79_MODEL_PATH)
    load_v80_router_artifact(V80_MODEL_PATH)
    v81_router = load_v81_router_artifact(V81_MODEL_PATH)
    if (
        v81_router["v75_router_model_payload_sha256"]
        != v75_artifact["model_sha256"]
    ):
        raise ValueError("v81 router is inconsistent with frozen v75")
    diagnostic = v81_router["crossfit_model_selection_diagnostic"]
    if float(diagnostic["candidate_minus_v75"]["ci_low"]) <= 0.0:
        raise ValueError("v81 frozen-history OOF diagnostic lost positive support")
    for contract in lock.get("files", {}).values():
        _verify_file(contract)
    return protocol, source


def prepare(stage: str) -> dict[str, object]:
    protocol, source = _verify_registration()
    if stage == "confirmation":
        if (
            not CONFIRMATION_OPEN_PATH.is_file()
            or read_json(CONFIRMATION_OPEN_PATH).get("confirmation_open_authorized")
            is not True
        ):
            raise ValueError("v81 confirmation has not been authorized")
    paths = _paths(stage)
    summary = prepare_stage(
        SOURCE_PATH,
        HISTORY_POOL_PATH,
        V74_GOLD_PATH,
        V75_DEVELOPMENT_GOLD_PATH,
        V75_CONFIRMATION_GOLD_PATH,
        stage=stage,
        blind_path=paths["blind"],
        gold_path=paths["gold"],
    )
    execution: dict[str, object] = {
        "schema_version": "frc-twowiki-residual-three-route-execution-v81",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "status": "PREPARED_BLIND_NOT_SCORED",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_registration_sha256": sha256(SOURCE_REGISTRATION_PATH),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK_PATH),
        "model_artifact_sha256": sha256(V81_MODEL_PATH),
        "source_sha256": sha256(SOURCE_PATH),
        "preparation": summary,
        "leakage_boundary": {
            "blind_rows_exclude_answer_gold_ids_gold_flags_and_gold_roles": True,
            "official_type_used_only_for_pre_registered_balanced_sampling": True,
            "official_type_used_by_runtime_router": False,
            "gold_metrics_computed": False,
            "selection_outputs_created": False,
        },
        "frozen_resources": protocol["common_resources"],
        "registered_claim_limits": source["claim_limits"],
    }
    _write_json(paths["execution"], execution)
    return execution


def score(stage: str, *, hf_home: Path, device: str) -> dict[str, object]:
    _verify_registration()
    paths = _paths(stage)
    execution = read_json(paths["execution"])
    if execution.get("status") != "PREPARED_BLIND_NOT_SCORED":
        raise ValueError("v81 stage is not at the prepared boundary")
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
    _verify_registration()
    paths = _paths(stage)
    execution = read_json(paths["execution"])
    if execution.get("status") != "SCORED_BLIND_GOLD_NOT_JOINED":
        raise ValueError("v81 stage is not at the scored pre-gold boundary")
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
        V81_MODEL_PATH,
        V80_MODEL_PATH,
        V79_MODEL_PATH,
        V78_MODEL_PATH,
        V76_MODEL_PATH,
        V75_MODEL_PATH,
        stage=stage,
        output_dir=paths["output"],
        prior_overlap=int(execution["preparation"]["prior_overlap"]),
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
                "official_type_used_by_runtime_router": False,
                "gold_metrics_computed": True,
                "selection_outputs_created_before_gold_join": True,
            },
        }
    )
    _write_json(paths["execution"], execution)
    outcome = report["analysis"]["outcome"]
    if stage == "development" and outcome["confirmation_open_authorized"]:
        _write_json(
            CONFIRMATION_OPEN_PATH,
            {
                "schema_version": "frc-twowiki-residual-three-route-confirmation-open-v81",
                "experiment_id": EXPERIMENT_ID,
                "development_result_sha256": sha256(paths["result"]),
                "status": outcome["status"],
                "confirmation_open_authorized": True,
                "model_feature_threshold_or_gates_changed": False,
                "gate_2": "NO-GO/SHADOW",
            },
        )
    else:
        _write_json(
            paths["closure"],
            {
                "schema_version": "frc-twowiki-residual-three-route-closure-v81",
                "experiment_id": EXPERIMENT_ID,
                "stage": stage,
                "result_sha256": sha256(paths["result"]),
                "status": outcome["status"],
                "confirmation_open_authorized": False,
                "further_same_source_confirmation_authorized": False,
                "reuse_stage_for_model_feature_threshold_gate_or_selection": False,
                "selector_adoption_authorized": False,
                "gate_2": "NO-GO/SHADOW",
            },
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen prospective 2Wiki v81 residual-router experiment."
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
