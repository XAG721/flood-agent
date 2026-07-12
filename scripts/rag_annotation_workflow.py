from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from flood_system.rag_annotation import (
    AdjudicationSubmission,
    AnnotationPackage,
    AnnotationSubmission,
    blank_submission,
    compare_submissions,
    finalize_adjudication,
    prepare_annotation_package,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, compare and adjudicate independent district RAG annotations.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", type=Path, default=Path("flood_system/rag_benchmarks/district_policy_benchmark.json"))
    prepare.add_argument("--output-dir", type=Path, default=Path("output/rag_annotation/round-1"))
    prepare.add_argument("--seed", type=int, default=20260712)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--package", type=Path, required=True)
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--package", type=Path, required=True)
    finalize.add_argument("--first", type=Path, required=True)
    finalize.add_argument("--second", type=Path, required=True)
    finalize.add_argument("--adjudication", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "prepare":
        package = prepare_annotation_package(args.source, seed=args.seed)
        write_json(args.output_dir / "annotation_package.json", package)
        write_json(args.output_dir / "annotator-a.json", blank_submission(package, "REPLACE_WITH_ANNOTATOR_A_ID"))
        write_json(args.output_dir / "annotator-b.json", blank_submission(package, "REPLACE_WITH_ANNOTATOR_B_ID"))
        adjudication_template = {
            "package_id": package.package_id,
            "adjudicator_id": "REPLACE_WITH_INDEPENDENT_ADJUDICATOR_ID",
            "resolutions": [
                {
                    "item_id": item.item_id,
                    "relevant_doc_ids": [],
                    "role_labels": {},
                    "reference_answer": "",
                    "rationale": "REQUIRED: explain the final evidence decision",
                }
                for item in package.items
            ],
            "submitted_at": datetime.now(UTC).isoformat(),
        }
        write_json(args.output_dir / "adjudication-template.json", adjudication_template)
        (args.output_dir / "PROTOCOL.md").write_text(
            "# 区县 RAG 独立标注协议\n\n"
            "1. 两名标注员分别复制各自表单，替换匿名标识并独立完成，不交换结果。\n"
            "2. 每题只选择回答必需且有直接依据的文档，并为已选文档标注六类证据角色。\n"
            "3. 标注完成后运行 `compare`，查看 Cohen kappa、Jaccard、角色和答案一致率。\n"
            "4. 第三名独立裁决员逐题填写最终证据、角色、答案和理由。\n"
            "5. 运行 `finalize`；工具拒绝同一人员兼任标注员和裁决员，并保存哈希与一致性证据。\n",
            encoding="utf-8",
        )
        print(f"Prepared blinded annotation package: {args.output_dir}")
        return

    package = AnnotationPackage.model_validate_json(args.package.read_text(encoding="utf-8"))
    first = AnnotationSubmission.model_validate_json(args.first.read_text(encoding="utf-8"))
    second = AnnotationSubmission.model_validate_json(args.second.read_text(encoding="utf-8"))
    if args.command == "compare":
        try:
            comparison = compare_submissions(package, first, second)
        except ValueError as exc:
            parser.error(str(exc))
        write_json(args.output, comparison)
        print(f"Wrote annotation comparison: {args.output}")
        return

    adjudication = AdjudicationSubmission.model_validate_json(args.adjudication.read_text(encoding="utf-8"))
    try:
        benchmark = finalize_adjudication(package, first, second, adjudication)
    except ValueError as exc:
        parser.error(str(exc))
    write_json(args.output, benchmark)
    print(f"Wrote adjudicated benchmark: {args.output}")


if __name__ == "__main__":
    main()
