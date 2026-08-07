from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.twowiki_confirmation import FrozenBgeScorer, score_cases_resumable
from research.frc_rag.twowiki_question_router import (
    EXPERIMENT_ID,
    develop_router,
    evaluate_stage,
    load_model_artifact,
    prepare_stage,
    validate_registered_protocol,
)
from research.frc_rag.twowiki_support_path_closure import read_json, read_jsonl, sha256


DOCS_ROOT = REPOSITORY_ROOT / "docs/progressive_upgrade"
CACHE_ROOT = REPOSITORY_ROOT / ".cache/benchmarks/twowiki_question_router_v75"
OUTPUT_ROOT = REPOSITORY_ROOT / "output/rag_evaluation/twowiki_question_router_v75"
SOURCE_PATH = REPOSITORY_ROOT / ".cache/benchmarks/2wikimultihopqa/dev.parquet"
HISTORY_POOL_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/frc_public_reference/candidate_pool_2wikimultihopqa.jsonl"
)
V74_BLIND_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/twowiki_support_path_closure_v74/development/blind_cases.jsonl"
)
V74_SCORED_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/twowiki_support_path_closure_v74/development/scored_blind.jsonl"
)
V74_CASES_PATH = (
    REPOSITORY_ROOT
    / "output/rag_evaluation/twowiki_support_path_closure_v74/development/cases.jsonl.gz"
)
MODEL_PATH = DOCS_ROOT / "twowiki_question_router_model_development_v75.json"
PROTOCOL_PATH = DOCS_ROOT / "twowiki_question_router_protocol_v75.json"
SOURCE_REGISTRATION_PATH = (
    DOCS_ROOT / "twowiki_question_router_source_registration_v75.json"
)
IMPLEMENTATION_LOCK_PATH = DOCS_ROOT / "twowiki_question_router_implementation_v75.json"


def _paths(stage: str) -> dict[str, Path]:
    cache = CACHE_ROOT / stage
    output = OUTPUT_ROOT / stage
    return {
        "blind": cache / "blind_cases.jsonl",
        "gold": cache / "sealed_gold.jsonl",
        "scored": cache / "scored_blind.jsonl",
        "selection": cache / "selection_outputs.jsonl",
        "output": output,
        "execution": DOCS_ROOT / f"twowiki_question_router_{stage}_execution_v75.json",
        "result": DOCS_ROOT / f"twowiki_question_router_{stage}_result_v75.json",
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def develop() -> dict[str, object]:
    scored = read_jsonl(V74_SCORED_PATH)
    with gzip.open(V74_CASES_PATH, "rt", encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    artifact = develop_router(scored, cases)
    artifact["history_artifacts"] = {
        "scored_path": ".cache/benchmarks/twowiki_support_path_closure_v74/development/scored_blind.jsonl",
        "scored_sha256": sha256(V74_SCORED_PATH),
        "cases_path": "output/rag_evaluation/twowiki_support_path_closure_v74/development/cases.jsonl.gz",
        "cases_sha256": sha256(V74_CASES_PATH),
    }
    _write_json(MODEL_PATH, artifact)
    return artifact


def _verify_registration() -> tuple[dict[str, object], dict[str, object]]:
    protocol = read_json(PROTOCOL_PATH)
    source = read_json(SOURCE_REGISTRATION_PATH)
    implementation_lock = read_json(IMPLEMENTATION_LOCK_PATH)
    artifact, _ = load_model_artifact(MODEL_PATH)
    validate_registered_protocol(protocol)
    if (
        protocol.get("experiment_id") != EXPERIMENT_ID
        or source.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("unexpected v75 registration experiment id")
    if sha256(SOURCE_PATH) != source["source"]["sha256"]:
        raise ValueError("v75 registered source hash changed")
    if sha256(HISTORY_POOL_PATH) != source["prior_exclusions"]["history_pool_sha256"]:
        raise ValueError("v75 history pool hash changed")
    if sha256(V74_BLIND_PATH) != source["prior_exclusions"]["v74_blind_sha256"]:
        raise ValueError("v75 v74 exclusion source hash changed")
    if sha256(MODEL_PATH) != protocol["router"]["model_artifact_sha256"]:
        raise ValueError("v75 router model artifact hash changed")
    if artifact["selected_crossfit"]["crossfit_router_balanced_accuracy"] < 0.9:
        raise ValueError("v75 history router failed its frozen development diagnostic")
    if implementation_lock.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v75 implementation lock experiment id")
    for file_contract in implementation_lock.get("files", {}).values():
        path = REPOSITORY_ROOT / str(file_contract["path"])
        if sha256(path) != file_contract["sha256"]:
            raise ValueError(f"v75 frozen file changed: {file_contract['path']}")
    return protocol, source


def prepare(stage: str) -> dict[str, object]:
    protocol, source = _verify_registration()
    if stage == "confirmation":
        open_path = DOCS_ROOT / "twowiki_question_router_confirmation_open_v75.json"
        if (
            not open_path.is_file()
            or read_json(open_path).get("confirmation_open_authorized") is not True
        ):
            raise ValueError("v75 confirmation has not been authorized")
    paths = _paths(stage)
    summary = prepare_stage(
        SOURCE_PATH,
        HISTORY_POOL_PATH,
        V74_BLIND_PATH,
        stage=stage,
        blind_path=paths["blind"],
        gold_path=paths["gold"],
    )
    execution: dict[str, object] = {
        "schema_version": "frc-twowiki-question-router-execution-v75",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "status": "PREPARED_BLIND_NOT_SCORED",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_registration_sha256": sha256(SOURCE_REGISTRATION_PATH),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK_PATH),
        "model_artifact_sha256": sha256(MODEL_PATH),
        "source_sha256": sha256(SOURCE_PATH),
        "preparation": summary,
        "leakage_boundary": {
            "blind_rows_exclude_answer_gold_ids_gold_flags_and_gold_roles": True,
            "official_type_used_by_router": False,
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
        raise ValueError("v75 stage is not at the prepared boundary")
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
        raise ValueError("v75 stage is not at the scored pre-gold boundary")
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
        stage=stage,
        seed=int(protocol["metrics"][f"{stage}_seed"]),
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
        }
    )
    execution["leakage_boundary"] = {
        "blind_rows_exclude_answer_gold_ids_gold_flags_and_gold_roles": True,
        "official_type_used_by_router": False,
        "gold_metrics_computed": True,
        "selection_outputs_created_before_gold_join": True,
    }
    _write_json(paths["execution"], execution)
    outcome = report["analysis"]["outcome"]
    if stage == "development":
        if outcome["confirmation_open_authorized"]:
            _write_json(
                DOCS_ROOT / "twowiki_question_router_confirmation_open_v75.json",
                {
                    "schema_version": "frc-twowiki-question-router-confirmation-open-v75",
                    "experiment_id": EXPERIMENT_ID,
                    "development_result_sha256": sha256(paths["result"]),
                    "status": outcome["status"],
                    "confirmation_open_authorized": True,
                    "router_model_threshold_or_gates_changed": False,
                    "gate_2": "NO-GO/SHADOW",
                },
            )
        else:
            _write_json(
                DOCS_ROOT / "twowiki_question_router_development_closure_v75.json",
                {
                    "schema_version": "frc-twowiki-question-router-development-closure-v75",
                    "experiment_id": EXPERIMENT_ID,
                    "development_result_sha256": sha256(paths["result"]),
                    "status": outcome["status"],
                    "confirmation_open_authorized": False,
                    "reuse_development_for_router_feature_model_threshold_gate_or_selection": False,
                    "selector_adoption_authorized": False,
                    "gate_2": "NO-GO/SHADOW",
                },
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen 2Wiki v75 router experiment"
    )
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
        print(json.dumps(develop(), ensure_ascii=False, sort_keys=True), flush=True)
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
