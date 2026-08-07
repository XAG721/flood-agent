from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.qasc_confirmation import (
    QascGoldBlindScorer,
    read_jsonl,
    sha256,
)
from research.frc_rag.qasc_scoring_optimization import (
    SCHEMA_VERSION,
    GlobalBatchedFrozenBgeScorer,
    QascGoldBlindBatchScorer,
    compare_scored_cases,
    select_benchmark_cases,
    write_json,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    import torch

    return round(torch.cuda.max_memory_allocated() / 1024**2, 3)


def _timed(call: Any, *, device: str) -> tuple[Any, float, float | None]:
    _reset_peak_memory(device)
    _synchronize(device)
    started = time.perf_counter()
    result = call()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return result, round(elapsed, 6), _peak_memory_mb(device)


def _render_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# QASC Global-batch Scoring Optimization Benchmark",
        "",
        f"- Decision: `{decision['status']}`",
        f"- Selected rerank batch size: `{decision['selected_rerank_batch_size']}`",
        f"- Frozen QASC confirmation changed: `{decision['frozen_confirmation_changed']}`",
        f"- Benchmark cases: {report['metadata']['benchmark_case_count']}",
        "",
        "| Pipeline | Rerank batch | Seconds | Speedup | Exactness | Peak GPU MiB |",
        "|---|---:|---:|---:|---|---:|",
    ]
    baseline = report["baseline"]
    lines.append(
        f"| per-case baseline | {baseline['rerank_batch_size']} | "
        f"{baseline['elapsed_seconds']:.6f} | 1.000000 | "
        f"`{baseline['equivalence']['passed']}` | "
        f"{baseline['peak_gpu_memory_mb']} |"
    )
    for candidate in report["candidates"]:
        lines.append(
            f"| global batch | {candidate['rerank_batch_size']} | "
            f"{candidate['elapsed_seconds']:.6f} | "
            f"{candidate['speedup_vs_per_case']:.6f} | "
            f"`{candidate['equivalence']['passed']}` | "
            f"{candidate['peak_gpu_memory_mb']} |"
        )
    lines.extend(
        [
            "",
            "The optimization is adopted only as a future scoring entrypoint when "
            "all score/rank/FRC-selection checks pass and measured speedup reaches "
            "the frozen threshold. It cannot alter the existing QASC scores, "
            "`NOT_CONFIRMED` result, Gate 2, or MuSiQue boundary.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact global batching against the frozen QASC per-case scorer."
        )
    )
    parser.add_argument(
        "--prepared-source",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/candidate_pool_qasc.jsonl"
        ),
    )
    parser.add_argument(
        "--reference-scores",
        type=Path,
        default=Path(
            ".cache/benchmarks/frc_public_reference/role_scores_qasc.jsonl"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/qasc_scoring_optimization_protocol.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/qasc_scoring_optimization"),
    )
    parser.add_argument("--hf-home", type=Path, default=Path("D:/RAG_test/.hf_cache"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for path in (args.prepared_source, args.reference_scores, args.protocol):
        if not path.is_file():
            parser.error(f"required benchmark input does not exist: {path}")
    protocol = _load_json(args.protocol)
    if protocol.get("schema_version") != "frc-qasc-scoring-optimization-protocol-v1":
        parser.error("unsupported QASC scoring optimization protocol")
    frozen_inputs = protocol["frozen_inputs"]
    if sha256(args.prepared_source) != frozen_inputs["prepared_source_sha256"]:
        parser.error("prepared QASC source hash does not match protocol")
    if sha256(args.reference_scores) != frozen_inputs["reference_scores_sha256"]:
        parser.error("reference QASC score hash does not match protocol")
    implementation = protocol["implementation"]
    for kind in ("module", "runner"):
        path = REPOSITORY_ROOT / implementation[f"{kind}_path"]
        if sha256(path) != implementation[f"{kind}_sha256"]:
            parser.error(f"QASC optimization {kind} hash does not match protocol")

    prepared = read_jsonl(args.prepared_source)
    references = read_jsonl(args.reference_scores)
    benchmark = select_benchmark_cases(
        prepared,
        count=int(protocol["benchmark"]["case_count"]),
        seed=str(protocol["benchmark"]["selection_seed"]),
    )
    reference_by_id = {str(case["id"]): case for case in references}
    expected = [reference_by_id[str(case["id"])] for case in benchmark]
    benchmark_ids = [str(case["id"]) for case in benchmark]
    if protocol["benchmark"]["selected_case_ids_sha256"] != (
        hashlib.sha256(
            json.dumps(
                benchmark_ids,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    ):
        parser.error("QASC benchmark case selection does not match protocol")

    import sentence_transformers
    import torch

    scorer = GlobalBatchedFrozenBgeScorer(
        hf_home=args.hf_home,
        device=args.device,
        embedding_batch_size=int(protocol["benchmark"]["embedding_batch_size"]),
        rerank_batch_size=int(protocol["benchmark"]["baseline_rerank_batch_size"]),
    )
    benchmark_ids_set = set(benchmark_ids)
    warmup = next(case for case in prepared if str(case["id"]) not in benchmark_ids_set)
    QascGoldBlindScorer(scorer).score_case(warmup)

    tolerance = float(protocol["equivalence"]["absolute_tolerance"])
    sequential, sequential_seconds, sequential_peak = _timed(
        lambda: [QascGoldBlindScorer(scorer).score_case(case) for case in benchmark],
        device=args.device,
    )
    baseline_equivalence = compare_scored_cases(
        expected, sequential, tolerance=tolerance
    )
    candidates = []
    for batch_size in protocol["benchmark"]["candidate_rerank_batch_sizes"]:
        scorer.rerank_batch_size = int(batch_size)
        scored, elapsed, peak = _timed(
            lambda: QascGoldBlindBatchScorer(scorer).score_cases(benchmark),
            device=args.device,
        )
        equivalence = compare_scored_cases(expected, scored, tolerance=tolerance)
        candidates.append(
            {
                "rerank_batch_size": int(batch_size),
                "elapsed_seconds": elapsed,
                "seconds_per_case": round(elapsed / len(benchmark), 6),
                "speedup_vs_per_case": round(sequential_seconds / elapsed, 6),
                "peak_gpu_memory_mb": peak,
                "equivalence": equivalence,
            }
        )
    minimum_speedup = float(protocol["adoption_rule"]["minimum_speedup"])
    eligible = [
        candidate
        for candidate in candidates
        if candidate["equivalence"]["passed"]
        and candidate["speedup_vs_per_case"] >= minimum_speedup
    ]
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["elapsed_seconds"],
            candidate["rerank_batch_size"],
        ),
        default=None,
    )
    report = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "status": "RUN_REAL_MODEL_QASC_GLOBAL_BATCH_SCORING_BENCHMARK",
            "benchmark_case_count": len(benchmark),
            "benchmark_candidate_count": sum(
                len(case["candidates"]) for case in benchmark
            ),
            "selected_case_ids_sha256": protocol["benchmark"][
                "selected_case_ids_sha256"
            ],
            "device": args.device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        },
        "provenance": {
            "protocol_path_label": args.protocol.name,
            "protocol_sha256": sha256(args.protocol),
            "prepared_source_path_label": args.prepared_source.name,
            "prepared_source_sha256": sha256(args.prepared_source),
            "reference_scores_path_label": args.reference_scores.name,
            "reference_scores_sha256": sha256(args.reference_scores),
            "reference_case_count": len(references),
        },
        "baseline": {
            "pipeline": "frozen_per_case",
            "rerank_batch_size": int(
                protocol["benchmark"]["baseline_rerank_batch_size"]
            ),
            "elapsed_seconds": sequential_seconds,
            "seconds_per_case": round(sequential_seconds / len(benchmark), 6),
            "peak_gpu_memory_mb": sequential_peak,
            "equivalence": baseline_equivalence,
        },
        "candidates": candidates,
        "decision": {
            "status": (
                "ADOPT_GLOBAL_BATCH_SCORING_ENTRYPOINT"
                if selected is not None and baseline_equivalence["passed"]
                else "KEEP_FROZEN_PER_CASE_SCORING_ENTRYPOINT"
            ),
            "selected_rerank_batch_size": (
                selected["rerank_batch_size"] if selected is not None else None
            ),
            "minimum_speedup": minimum_speedup,
            "all_equivalence_checks_required": True,
            "frozen_confirmation_changed": False,
            "existing_reference_scores_rewritten": False,
            "gate_2": "NO-GO/SHADOW",
            "musique_downloaded_or_inspected": False,
        },
    }
    json_path = write_json(
        args.output_dir / "qasc_scoring_optimization.json", report
    )
    markdown_path = args.output_dir / "qasc_scoring_optimization.md"
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
