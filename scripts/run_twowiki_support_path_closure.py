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

from research.frc_rag.twowiki_confirmation import score_cases_resumable
from research.frc_rag.twowiki_support_path_closure import (
    EXPERIMENT_ID,
    FrozenBgeScorer,
    evaluate_stage,
    prepare_stage,
    read_json,
    sha256,
    validate_registered_protocol,
)


DOCS_ROOT = REPOSITORY_ROOT / "docs/progressive_upgrade"
CACHE_ROOT = REPOSITORY_ROOT / ".cache/benchmarks/twowiki_support_path_closure_v74"
OUTPUT_ROOT = REPOSITORY_ROOT / "output/rag_evaluation/twowiki_support_path_closure_v74"
PROTOCOL_PATH = DOCS_ROOT / "twowiki_support_path_closure_protocol_v74.json"
SOURCE_REGISTRATION_PATH = (
    DOCS_ROOT / "twowiki_support_path_closure_source_registration_v74.json"
)
HISTORY_PATH = (
    REPOSITORY_ROOT
    / ".cache/benchmarks/frc_public_reference/candidate_pool_2wikimultihopqa.jsonl"
)
SOURCE_PATH = REPOSITORY_ROOT / ".cache/benchmarks/2wikimultihopqa/dev.parquet"


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
        / f"twowiki_support_path_closure_{stage}_execution_v74.json",
        "result": DOCS_ROOT
        / f"twowiki_support_path_closure_{stage}_result_v74.json",
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_and_verify_registration() -> tuple[dict[str, object], dict[str, object]]:
    protocol = read_json(PROTOCOL_PATH)
    source = read_json(SOURCE_REGISTRATION_PATH)
    validate_registered_protocol(protocol)
    if source.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v74 source registration experiment id")
    expected_source = source["source"]
    if SOURCE_PATH.stat().st_size != int(expected_source["bytes"]):
        raise ValueError("registered 2Wiki source byte count changed")
    if sha256(SOURCE_PATH) != expected_source["sha256"]:
        raise ValueError("registered 2Wiki source hash changed")
    history = source["prior_history_exclusion"]
    if HISTORY_PATH.stat().st_size != int(history["candidate_pool_bytes"]):
        raise ValueError("registered 2Wiki history byte count changed")
    if sha256(HISTORY_PATH) != history["candidate_pool_sha256"]:
        raise ValueError("registered 2Wiki history hash changed")
    return protocol, source


def prepare(stage: str) -> dict[str, object]:
    protocol, source = _load_and_verify_registration()
    if stage == "confirmation":
        open_path = DOCS_ROOT / "twowiki_support_path_closure_confirmation_open_v74.json"
        if not open_path.is_file():
            raise ValueError("v74 confirmation has not been opened")
        opened = read_json(open_path)
        if opened.get("confirmation_open_authorized") is not True:
            raise ValueError("v74 confirmation open artifact is not authorized")
    paths = _paths(stage)
    summary = prepare_stage(
        SOURCE_PATH,
        HISTORY_PATH,
        stage=stage,
        blind_path=paths["blind"],
        gold_path=paths["gold"],
    )
    execution: dict[str, object] = {
        "schema_version": "frc-twowiki-support-path-closure-execution-v74",
        "experiment_id": EXPERIMENT_ID,
        "stage": stage,
        "status": "PREPARED_BLIND_NOT_SCORED",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_registration_sha256": sha256(SOURCE_REGISTRATION_PATH),
        "source_sha256": sha256(SOURCE_PATH),
        "history_sha256": sha256(HISTORY_PATH),
        "preparation": summary,
        "leakage_boundary": {
            "blind_rows_exclude_answer_gold_ids_gold_flags_and_gold_roles": True,
            "gold_metrics_computed": False,
            "selection_outputs_created": False,
        },
        "frozen_resources": protocol["common_resources"],
        "registered_source_claims": source["claims"],
    }
    _write_json(paths["execution"], execution)
    return execution


def score(stage: str, *, hf_home: Path, device: str) -> dict[str, object]:
    _load_and_verify_registration()
    paths = _paths(stage)
    if not paths["blind"].is_file() or not paths["execution"].is_file():
        raise ValueError(f"v74 {stage} must be prepared before scoring")
    execution = read_json(paths["execution"])
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
    cases = score_cases_resumable(
        paths["blind"],
        paths["scored"],
        scorer=scorer,
        progress=progress,
    )
    execution.update(
        {
            "status": "SCORED_BLIND_GOLD_NOT_JOINED",
            "scored_cases": cases,
            "scored_blind_sha256": sha256(paths["scored"]),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
    )
    execution["leakage_boundary"] = {
        "blind_rows_exclude_answer_gold_ids_gold_flags_and_gold_roles": True,
        "gold_metrics_computed": False,
        "selection_outputs_created": False,
    }
    _write_json(paths["execution"], execution)
    return execution


def evaluate(stage: str) -> dict[str, object]:
    _load_and_verify_registration()
    paths = _paths(stage)
    if not paths["scored"].is_file() or not paths["gold"].is_file():
        raise ValueError(f"v74 {stage} must be scored before evaluation")
    execution = read_json(paths["execution"])
    if execution.get("status") != "SCORED_BLIND_GOLD_NOT_JOINED":
        raise ValueError("v74 execution is not at the pre-gold-join boundary")
    stage_overlap = 0
    if stage == "confirmation":
        development_gold = _paths("development")["gold"]
        development_ids = {
            str(row["id"])
            for row in (
                json.loads(line)
                for line in development_gold.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        confirmation_ids = {
            str(row["id"])
            for row in (
                json.loads(line)
                for line in paths["gold"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        stage_overlap = len(development_ids & confirmation_ids)
    seed = 20261001 if stage == "development" else 20261002
    report = evaluate_stage(
        paths["scored"],
        paths["gold"],
        paths["selection"],
        stage=stage,
        seed=seed,
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
        }
    )
    execution["leakage_boundary"] = {
        "blind_rows_exclude_answer_gold_ids_gold_flags_and_gold_roles": True,
        "gold_metrics_computed": True,
        "selection_outputs_created_before_gold_join": True,
    }
    _write_json(paths["execution"], execution)
    outcome = report["analysis"]["outcome"]
    if stage == "development":
        if outcome["confirmation_open_authorized"]:
            _write_json(
                DOCS_ROOT / "twowiki_support_path_closure_confirmation_open_v74.json",
                {
                    "schema_version": "frc-twowiki-support-path-closure-confirmation-open-v74",
                    "experiment_id": EXPERIMENT_ID,
                    "development_result_sha256": sha256(paths["result"]),
                    "status": outcome["status"],
                    "confirmation_open_authorized": True,
                    "selector_parameters_or_gates_changed": False,
                    "gate_2": "NO-GO/SHADOW",
                },
            )
        else:
            _write_json(
                DOCS_ROOT / "twowiki_support_path_closure_development_closure_v74.json",
                {
                    "schema_version": "frc-twowiki-support-path-closure-development-closure-v74",
                    "experiment_id": EXPERIMENT_ID,
                    "development_result_sha256": sha256(paths["result"]),
                    "status": outcome["status"],
                    "confirmation_open_authorized": False,
                    "reuse_development_for_method_weight_gate_or_selection": False,
                    "selector_adoption_authorized": False,
                    "gate_2": "NO-GO/SHADOW",
                },
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen 2Wiki v74 experiment")
    parser.add_argument("command", choices=("prepare", "score", "evaluate", "run"))
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--hf-home", type=Path, default=Path("D:/RAG_test/.hf_cache"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command in {"prepare", "run"}:
        result = prepare(args.stage)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if args.command in {"score", "run"}:
        result = score(args.stage, hf_home=args.hf_home, device=args.device)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if args.command in {"evaluate", "run"}:
        result = evaluate(args.stage)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
