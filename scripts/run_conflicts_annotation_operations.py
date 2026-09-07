from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conflicts_annotation_operations import (
    REVIEWER_SLOTS,
    build_annotation_operations,
    merge_batch_submissions,
)
from research.frc_rag.conflicts_expected_behavior import (
    canonical_json_sha256,
    file_sha256,
    read_jsonl,
)


PROTOCOL_SCHEMA_VERSION = "frc-conflicts-annotation-operations-protocol-v1"
MANIFEST_SCHEMA_VERSION = "frc-conflicts-annotation-operations-manifest-v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def _verify_protocol(protocol_path: Path, protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported CONFLICTS annotation operations protocol")
    if protocol.get("protocol_path_label") != protocol_path.name:
        raise ValueError("annotation operations protocol path label differs")
    for label in ("module", "runner"):
        relative = protocol["implementation"][f"{label}_path"]
        expected = protocol["implementation"][f"{label}_sha256"]
        if file_sha256(REPOSITORY_ROOT / relative) != expected:
            raise ValueError(f"annotation operations {label} hash does not match protocol")


def _protocol_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CONFLICTS 独立评审批次作业说明",
            "",
            f"- 状态：`{manifest['status']}`",
            f"- 作业 ID：`{manifest['operations_id']}`",
            f"- 原盲评包 ID：`{manifest['package_id']}`",
            f"- 角色：{len(manifest['reviewer_slots'])}；每角色批次：{manifest['batch_count']}",
            f"- 每角色主任务：{manifest['primary_task_count_per_slot']}；隐藏复测：{manifest['repeat_task_count_per_slot']}",
            f"- 私有路由承诺：`{manifest['routing_sha256']}`",
            "",
            "## 执行要求",
            "",
            "1. `reviewer_1`、`reviewer_2` 和 `adjudicator` 必须由三名不同人员承担，不得互看决定。",
            "2. 每人依次完成自己目录下 8 个 `submission-template`，8 份文件使用同一个私有 annotator_id。",
            "3. 不修改 task、batch、package 或 operations ID；每项四个评分、理由和偏好都必须完成。",
            "4. 部分题目会跨批重复，用于组内一致性质控；重复身份不在公开批次中标识。",
            "5. 使用 `merge` 合并同一角色的 8 份表单。合并器验证私有路由哈希、恢复原 458 例顺序并输出组内一致性。",
            "6. 组内任一评分或偏好精确一致率低于 0.80 时状态为 `REVIEW_REQUIRED`，不得进入跨人员比较。",
            "7. 三个合并提交均通过后，再使用冻结 expected-behavior 工作流的 `compare` 和 `finalize`。",
            "",
            "> 批次文件、空模板和质控工具不是人工结果；它们不改变 Gate 2，也不会发布方法映射。",
            "",
        ]
    )


def prepare(args: argparse.Namespace) -> None:
    protocol = _load_json(args.protocol)
    _verify_protocol(args.protocol, protocol)
    source_manifest = _load_json(args.source_manifest)
    package_items = read_jsonl(args.package)
    frozen = protocol["frozen_inputs"]
    if file_sha256(args.package) != frozen["package_file_sha256"]:
        raise ValueError("CONFLICTS behavior package hash differs from protocol")
    if file_sha256(args.source_manifest) != frozen["source_manifest_file_sha256"]:
        raise ValueError("CONFLICTS behavior manifest hash differs from protocol")
    if source_manifest["package_id"] != frozen["package_id"]:
        raise ValueError("CONFLICTS behavior package id differs from protocol")
    if len(package_items) != frozen["case_count"]:
        raise ValueError("CONFLICTS behavior case count differs from protocol")
    if canonical_json_sha256(package_items) != frozen["package_items_sha256"]:
        raise ValueError("CONFLICTS behavior package items differ from protocol")

    batching = protocol["batching"]
    operations = build_annotation_operations(
        package_items,
        package_id=source_manifest["package_id"],
        seed=batching["seed"],
        batch_count=int(batching["batch_count_per_slot"]),
        repeat_count=int(batching["repeat_task_count_per_slot"]),
        minimum_repeat_batch_distance=int(
            batching["minimum_repeat_batch_distance"]
        ),
        reviewer_slots=tuple(batching["reviewer_slots"]),
    )
    if operations["operations_id"] != protocol["expected_operations_id"]:
        raise ValueError("annotation operations id differs from protocol")

    routing_sha256 = canonical_json_sha256(operations["routing"])
    if routing_sha256 != protocol["expected_routing_sha256"]:
        raise ValueError("annotation routing commitment differs from protocol")
    routing_path = _write_json(args.routing, operations["routing"])
    files: list[dict[str, Any]] = []
    for batch, template in zip(
        operations["batches"], operations["templates"], strict=True
    ):
        slot_dir = batch["reviewer_slot"].replace("_", "-")
        batch_stem = f"batch-{int(batch['batch_index']):02d}"
        batch_path = _write_json(args.output_dir / slot_dir / f"{batch_stem}.json", batch)
        template_path = _write_json(
            args.output_dir / slot_dir / f"{batch_stem}-submission-template.json",
            template,
        )
        for kind, path in (("batch", batch_path), ("submission_template", template_path)):
            files.append(
                {
                    "kind": kind,
                    "reviewer_slot": batch["reviewer_slot"],
                    "batch_id": batch["batch_id"],
                    "path": path.relative_to(args.output_dir).as_posix(),
                    "sha256": file_sha256(path),
                }
            )

    item_by_id = {str(item["item_id"]): item for item in package_items}
    stratum_by_item = {
        item_id: (
            f"{item['conflict_type']}|"
            + (
                "answer_available"
                if item.get("correct_answer") is not None
                else "no_answer"
            )
        )
        for item_id, item in item_by_id.items()
    }
    primary_strata_by_batch: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    batch_task_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in operations["routing"]["rows"]:
        slot = str(row["reviewer_slot"])
        batch_index = int(row["batch_index"])
        batch_task_counts[slot][batch_index] += 1
        if row["occurrence"] == "primary":
            primary_strata_by_batch[slot][batch_index][
                stratum_by_item[str(row["item_id"])]
            ] += 1
    stratum_balance = {}
    for slot in batching["reviewer_slots"]:
        stratum_balance[slot] = {}
        for stratum in operations["primary_counts_by_stratum"]:
            values = [
                primary_strata_by_batch[slot][batch_index].get(stratum, 0)
                for batch_index in range(1, int(batching["batch_count_per_slot"]) + 1)
            ]
            stratum_balance[slot][stratum] = {
                "minimum": min(values),
                "maximum": max(values),
                "range": max(values) - min(values),
            }

    public_payload = json.dumps(
        {"batches": operations["batches"], "templates": operations["templates"]},
        ensure_ascii=False,
        sort_keys=True,
    )
    if any(value in public_payload for value in ("coverage_greedy_proxy", "frc_select")):
        raise ValueError("public annotation operations leak method identity")
    if '"item_id"' in public_payload or '"occurrence"' in public_payload:
        raise ValueError("public annotation operations leak private routing fields")

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "PREPARED_AWAITING_INDEPENDENT_HUMAN_REVIEW",
        "operations_id": operations["operations_id"],
        "package_id": source_manifest["package_id"],
        "protocol_path_label": args.protocol.name,
        "protocol_sha256": file_sha256(args.protocol),
        "source_manifest_file_sha256": frozen["source_manifest_file_sha256"],
        "package_file_sha256": frozen["package_file_sha256"],
        "package_items_sha256": frozen["package_items_sha256"],
        "seed": str(batching["seed"]),
        "reviewer_slots": list(batching["reviewer_slots"]),
        "batch_count": int(batching["batch_count_per_slot"]),
        "primary_task_count_per_slot": len(package_items),
        "repeat_task_count_per_slot": len(operations["repeat_item_ids"]),
        "total_task_count_per_slot": len(package_items)
        + len(operations["repeat_item_ids"]),
        "total_public_batch_count": len(operations["batches"]),
        "total_submission_template_count": len(operations["templates"]),
        "repeat_counts_by_conflict_type": operations[
            "repeat_counts_by_conflict_type"
        ],
        "batch_task_counts": {
            slot: [
                batch_task_counts[slot][batch_index]
                for batch_index in range(1, int(batching["batch_count_per_slot"]) + 1)
            ]
            for slot in batching["reviewer_slots"]
        },
        "primary_stratum_balance": stratum_balance,
        "public_files": files,
        "routing_sha256": routing_sha256,
        "routing_path_label": routing_path.name,
        "routing_published": False,
        "method_identity_present_in_public_files": False,
        "original_item_id_present_in_public_files": False,
        "human_evidence_complete": False,
        "gate_2": "NO-GO/SHADOW",
        "quality_control": protocol["quality_control"],
        "limitations": [
            "Prepared batches and empty templates are not human evaluation results.",
            "The routing map is required for merge but remains in ignored local cache.",
            "Within-reviewer repeats measure consistency and do not create gold labels.",
            "This operational preparation cannot replace independent people or change Gate 2.",
        ],
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    protocol_markdown = _protocol_markdown(manifest)
    with (args.output_dir / "PROTOCOL.md").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(protocol_markdown)
    print(args.output_dir / "manifest.json")
    print(f"prepared {len(operations['batches'])} method-blind review batches")
    print(manifest["status"])


def merge(args: argparse.Namespace) -> None:
    protocol = _load_json(args.protocol)
    _verify_protocol(args.protocol, protocol)
    operations_manifest = _load_json(args.operations_manifest)
    package_items = read_jsonl(args.package)
    routing = _load_json(args.routing)
    submissions = [_load_json(path) for path in args.submissions]
    merged, qc = merge_batch_submissions(
        package_items,
        operations_manifest=operations_manifest,
        routing=routing,
        reviewer_slot=args.reviewer_slot,
        submissions=submissions,
        minimum_exact_agreement=float(
            protocol["quality_control"]["minimum_exact_agreement_each_dimension"]
        ),
    )
    _write_json(args.output, merged)
    _write_json(args.qc_output, qc)
    print(args.output)
    print(args.qc_output)
    print(qc["status"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare and merge method-blind CONFLICTS human-review batches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/conflicts_annotation_operations_protocol.json"
        ),
    )
    prepare_parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("output/rag_evaluation/conflicts_expected_behavior/manifest.json"),
    )
    prepare_parser.add_argument(
        "--package",
        type=Path,
        default=Path(
            "output/rag_evaluation/conflicts_expected_behavior/package.jsonl.gz"
        ),
    )
    prepare_parser.add_argument(
        "--routing",
        type=Path,
        default=Path(
            ".cache/benchmarks/rag_conflicts/expected_behavior_workflow/annotation_operations_routing.json"
        ),
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conflicts_annotation_operations"),
    )
    prepare_parser.set_defaults(func=prepare)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--protocol", type=Path, required=True)
    merge_parser.add_argument("--operations-manifest", type=Path, required=True)
    merge_parser.add_argument("--package", type=Path, required=True)
    merge_parser.add_argument("--routing", type=Path, required=True)
    merge_parser.add_argument(
        "--reviewer-slot", choices=REVIEWER_SLOTS, required=True
    )
    merge_parser.add_argument("--submissions", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--qc-output", type=Path, required=True)
    merge_parser.set_defaults(func=merge)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
