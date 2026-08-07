from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.frc_rag.iirc_cardinality_transfer import validate_sources  # noqa: E402
from research.frc_rag.iirc_low_candidate_robustness_transfer import (  # noqa: E402
    EXPERIMENT_ID,
    evaluate_stage,
    prepare_stage,
    sha256,
    write_report,
    write_selection_outputs,
)
from research.frc_rag.rgb_cost_aware_frc import load_frozen_tokenizer  # noqa: E402
from research.frc_rag.twowiki_confirmation import (  # noqa: E402
    FrozenBgeScorer,
    score_cases_resumable,
)


DOCS = REPO_ROOT / "docs/progressive_upgrade"
CACHE = REPO_ROOT / ".cache/benchmarks/iirc"
OUTPUT = REPO_ROOT / "output/rag_evaluation/iirc_low_candidate_robustness_v89"
TRAIN_PATH = CACHE / "iirc_train_dev/train.json"
DEV_PATH = CACHE / "iirc_train_dev/dev.json"
TRAIN_DEV_ARCHIVE = CACHE / "iirc_train_dev.tgz"
CONTEXT_ARCHIVE = CACHE / "context_articles.tar.gz"
CONTEXT_JSON = CACHE / "context_articles.json"
CONTEXT_INDEX = CACHE / "context_articles.offsets.sqlite"
V86_MODEL_PATH = DOCS / "iirc_cardinality_transfer_router_model_development_v86.json"
V87_MODEL_PATH = DOCS / "iirc_score_gap_router_model_development_v87.json"
V88_MODEL_PATH = DOCS / "iirc_pooled_score_gap_router_model_development_v88.json"
PROTOCOL_PATH = DOCS / "iirc_low_candidate_robustness_protocol_v89.json"
SOURCE_REGISTRATION_PATH = DOCS / "iirc_low_candidate_robustness_source_registration_v89.json"
IMPLEMENTATION_LOCK_PATH = DOCS / "iirc_low_candidate_robustness_implementation_v89.json"
CONFIRMATION_OPEN_PATH = DOCS / "iirc_low_candidate_robustness_confirmation_open_v89.json"


def _paths(stage: str) -> dict[str, Path]:
    root = CACHE / "v89" / stage
    return {
        "blind": root / "blind_cases.jsonl",
        "gold": root / "sealed_gold.jsonl",
        "scored": root / "scored_blind.jsonl",
        "selection": root / "selection_outputs.jsonl",
        "execution": DOCS / f"iirc_low_candidate_robustness_{stage}_execution_v89.json",
        "result": DOCS / f"iirc_low_candidate_robustness_{stage}_result_v89.json",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_registration() -> None:
    lock = _read_json(IMPLEMENTATION_LOCK_PATH)
    for name, contract in lock.get("files", {}).items():
        path = REPO_ROOT / str(contract["path"])
        if not path.is_file() or sha256(path) != str(contract["sha256"]):
            raise ValueError(f"v89 locked file changed: {name}")
    validate_sources(
        train_path=TRAIN_PATH,
        dev_path=DEV_PATH,
        train_dev_archive=TRAIN_DEV_ARCHIVE,
        context_archive=CONTEXT_ARCHIVE,
        context_json=CONTEXT_JSON,
    )


def prepare(args: argparse.Namespace) -> None:
    _verify_registration()
    if args.stage == "confirmation":
        opened = _read_json(CONFIRMATION_OPEN_PATH)
        if opened.get("confirmation_open_authorized") is not True:
            raise ValueError("v89 confirmation is not open")
    if not CONTEXT_INDEX.is_file():
        raise FileNotFoundError("build the shared IIRC context offset index first")
    tokenizer = load_frozen_tokenizer(args.hf_home)
    paths = _paths(args.stage)
    summary = prepare_stage(
        TRAIN_PATH,
        CONTEXT_JSON,
        CONTEXT_INDEX,
        tokenizer,
        stage=args.stage,
        blind_path=paths["blind"],
        gold_path=paths["gold"],
    )
    execution = {
        "schema_version": "frc-iirc-low-candidate-robustness-execution-v89",
        "experiment_id": EXPERIMENT_ID,
        "stage": args.stage,
        "status": "PREPARED_BLIND_NOT_SCORED",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_registration_sha256": sha256(SOURCE_REGISTRATION_PATH),
        "implementation_lock_sha256": sha256(IMPLEMENTATION_LOCK_PATH),
        "v86_model_sha256": sha256(V86_MODEL_PATH),
        "v87_model_sha256": sha256(V87_MODEL_PATH),
        "v88_model_sha256": sha256(V88_MODEL_PATH),
        "preparation": summary,
        "leakage_boundary": {
            "blind_cache_excludes_answer_answer_type_context_question_links_and_gold": True,
            "sealed_gold_created_but_not_read_by_scorer_or_selector": True,
            "selection_metrics_computed": False,
        },
    }
    _write_json(paths["execution"], execution)
    print(json.dumps(execution, ensure_ascii=False, sort_keys=True))


def score(args: argparse.Namespace) -> None:
    _verify_registration()
    paths = _paths(args.stage)
    execution = _read_json(paths["execution"])
    if execution.get("status") != "PREPARED_BLIND_NOT_SCORED":
        raise ValueError("v89 stage is not at the prepared boundary")
    scorer = FrozenBgeScorer(
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=args.embedding_batch_size,
        rerank_batch_size=args.rerank_batch_size,
    )
    started = time.perf_counter()

    def progress(index: int, total: int, case_id: str) -> None:
        if index == 1 or index % 20 == 0 or index == total:
            print(f"scored {index}/{total}: {case_id}", flush=True)

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
    print(json.dumps(execution, ensure_ascii=False, sort_keys=True))


def select(args: argparse.Namespace) -> None:
    _verify_registration()
    paths = _paths(args.stage)
    execution = _read_json(paths["execution"])
    if execution.get("status") != "SCORED_BLIND_GOLD_NOT_JOINED":
        raise ValueError("v89 stage is not at the scored boundary")
    outputs = write_selection_outputs(
        paths["scored"],
        V86_MODEL_PATH,
        V87_MODEL_PATH,
        V88_MODEL_PATH,
        paths["selection"],
    )
    execution.update(
        {
            "status": "SELECTED_BLIND_GOLD_NOT_JOINED",
            "selection_rows": len(outputs),
            "selection_sha256": sha256(paths["selection"]),
        }
    )
    _write_json(paths["execution"], execution)
    print(json.dumps(execution, ensure_ascii=False, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    _verify_registration()
    paths = _paths(args.stage)
    execution = _read_json(paths["execution"])
    if execution.get("status") != "SELECTED_BLIND_GOLD_NOT_JOINED":
        raise ValueError("v89 stage is not at the selected boundary")
    report, cases = evaluate_stage(paths["selection"], paths["gold"], stage=args.stage)
    published = write_report(report, cases, OUTPUT)
    report["artifacts"] = {
        "selection_sha256": sha256(paths["selection"]),
        "sealed_gold_sha256": sha256(paths["gold"]),
        **{name: sha256(path) for name, path in published.items()},
    }
    _write_json(paths["result"], report)
    execution.update(
        {
            "status": "EVALUATED_GOLD_JOINED",
            "result_sha256": sha256(paths["result"]),
            "outcome": report["status"],
        }
    )
    _write_json(paths["execution"], execution)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def open_confirmation(_: argparse.Namespace) -> None:
    _verify_registration()
    result_path = _paths("development")["result"]
    result = _read_json(result_path)
    if result.get("all_strict_gates_pass") is not True:
        raise ValueError("v89 development did not pass every strict gate")
    record = {
        "schema_version": "frc-iirc-low-candidate-robustness-confirmation-open-v89",
        "experiment_id": EXPERIMENT_ID,
        "development_result_sha256": sha256(result_path),
        "development_status": result["status"],
        "confirmation_open_authorized": True,
        "confirmation_content_accessed_before_open": False,
        "algorithm_model_adapter_baselines_metrics_or_gates_changed": False,
        "gate_2": "NO-GO/SHADOW",
    }
    _write_json(CONFIRMATION_OPEN_PATH, record)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the IIRC v89 experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, handler in (
        ("prepare", prepare),
        ("score", score),
        ("select", select),
        ("evaluate", evaluate),
    ):
        local = subparsers.add_parser(command)
        local.add_argument("--stage", choices=("development", "confirmation"), required=True)
        local.add_argument("--hf-home", type=Path, default=Path(r"D:\RAG_test\.hf_cache"))
        local.add_argument("--device", default="cuda")
        local.add_argument("--embedding-batch-size", type=int, default=64)
        local.add_argument("--rerank-batch-size", type=int, default=128)
        local.set_defaults(handler=handler)
    open_parser = subparsers.add_parser("open-confirmation")
    open_parser.set_defaults(handler=open_confirmation)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
