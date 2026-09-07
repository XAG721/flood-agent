from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conflicts_expected_behavior import (
    COMPARISON_SCHEMA_VERSION,
    METHODS,
    SUBMISSION_SCHEMA_VERSION,
    LocalQwenAnswerGenerator,
    blank_submission_rows,
    build_blinded_package,
    canonical_json_sha256,
    compare_submissions,
    file_sha256,
    finalize_adjudication,
    generate_answers,
    load_selected_methods,
    read_jsonl,
    write_gzip_jsonl,
)


PROTOCOL_SCHEMA_VERSION = "frc-conflicts-expected-behavior-protocol-v1"
MANIFEST_SCHEMA_VERSION = "frc-conflicts-expected-behavior-manifest-v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _verify_protocol(protocol_path: Path, protocol: dict[str, Any]) -> None:
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported CONFLICTS expected-behavior protocol")
    for label in ("module", "runner"):
        relative = protocol["implementation"][f"{label}_path"]
        expected = protocol["implementation"][f"{label}_sha256"]
        if file_sha256(REPOSITORY_ROOT / relative) != expected:
            raise ValueError(f"CONFLICTS behavior {label} hash does not match protocol")
    if protocol["protocol_path_label"] != protocol_path.name:
        raise ValueError("CONFLICTS behavior protocol path label differs")


def _verify_model(protocol: dict[str, Any], model_path: Path) -> None:
    for filename, expected in protocol["model"]["file_sha256"].items():
        path = model_path / filename
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"local generator artifact differs: {filename}")


def _protocol_markdown(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# CONFLICTS expected-behavior 独立盲评说明",
            "",
            f"- 状态：`{manifest['status']}`",
            f"- 包 ID：`{manifest['package_id']}`",
            f"- 用例：{manifest['case_count']}；盲化回答：{manifest['response_count']}",
            f"- 方法映射承诺：`{manifest['blind_mapping_sha256']}`",
            "- 当前工件只是待评审包，不是人工评测结果。",
            "",
            "## 评审步骤",
            "",
            "1. 两名评审分别复制 `annotator-template.json`，替换匿名 ID，独立完成全部条目。",
            "2. 每个回答分别评定 expected behavior、事实依据、引用正确性和答案正确性；每项必须写理由。",
            "3. `correct_answer=null` 时，答案正确性可填 `NOT_APPLICABLE`；其他维度仍须评定。",
            "4. 完成后运行 `compare`，只查看一致性和分歧，不解盲方法身份。",
            "5. 第三名独立裁决员逐条给出最终评分，再运行 `finalize`；工具验证映射 SHA-256 后才解盲汇总。",
            "6. 任何空白、占位 ID、同一人员兼任或缺少理由都会被拒绝。",
            "",
            "## 判定口径",
            "",
            "- `PASS`：回答满足该维度；`FAIL`：明确不满足；`UNCERTAIN`：证据不足以判断。",
            "- 偏好为 `A`、`B`、`TIE`、`NEITHER` 或 `UNCERTAIN`。",
            "- A/B 每例独立随机化；包中不包含方法名、分数、角色分或选择器解释。",
            "- 该跨领域评审不能替代真实防汛领域双专家验证，也不能单独改变 Gate 2。",
            "",
        ]
    )


def _render_final(report: dict[str, Any]) -> str:
    lines = [
        "# CONFLICTS expected-behavior 独立裁决结果",
        "",
        f"- 状态：`{report['status']}`",
        f"- 用例：{report['case_count']}",
        f"- Gate 2：`{report['decision']['gate_2']}`",
        "",
        "| 方法 | Expected behavior | 事实依据 | 引用正确 | 答案正确 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        metrics = report["method_metrics"][method]
        lines.append(
            f"| `{method}` | {metrics['expected_behavior_adherence']['rate']:.6f} | "
            f"{metrics['factual_grounding']['rate']:.6f} | "
            f"{metrics['citation_correctness']['rate']:.6f} | "
            f"{metrics['answer_correctness']['rate']:.6f} |"
        )
    lines.extend(["", report["decision"]["reason"], ""])
    return "\n".join(lines)


def prepare(args: argparse.Namespace) -> None:
    protocol = _load_json(args.protocol)
    _verify_protocol(args.protocol, protocol)
    frozen = protocol["frozen_inputs"]
    if file_sha256(args.dataset) != frozen["dataset_sha256"]:
        raise ValueError("CONFLICTS dataset hash does not match protocol")
    if file_sha256(args.selected_evidence) != frozen["selected_evidence_sha256"]:
        raise ValueError("CONFLICTS selected-evidence hash does not match protocol")
    _verify_model(protocol, args.model_path)
    selected = load_selected_methods(args.selected_evidence)
    case_count = len({str(row["case_id"]) for row in selected})
    if case_count != frozen["case_count"] or len(selected) != frozen["comparison_row_count"]:
        raise ValueError("CONFLICTS comparison row counts do not match protocol")
    generation = protocol["generation"]
    cache_key = canonical_json_sha256(
        {
            "protocol_sha256": file_sha256(args.protocol),
            "dataset_sha256": frozen["dataset_sha256"],
            "selected_evidence_sha256": frozen["selected_evidence_sha256"],
            "methods": list(METHODS),
            "model": protocol["model"],
            "generation": generation,
        }
    )
    generator = LocalQwenAnswerGenerator(
        model_path=args.model_path,
        batch_size=int(generation["model_batch_size"]),
        max_input_tokens=int(generation["max_input_tokens"]),
        max_new_tokens=int(generation["max_new_tokens"]),
    )
    generations = generate_answers(
        selected,
        generator=generator,
        output_path=args.cache_dir / "generations.jsonl",
        cache_key=cache_key,
        prompt_batch_size=int(generation["checkpoint_batch_size"]),
    )
    package_id, items, mapping = build_blinded_package(
        selected, generations, blind_seed=protocol["blinding"]["seed"]
    )
    package_path = write_gzip_jsonl(args.output_dir / "package.jsonl.gz", items)
    mapping_path = _write_json(args.cache_dir / "blind_mapping.json", mapping)
    template = {
        "schema_version": SUBMISSION_SCHEMA_VERSION,
        "package_id": package_id,
        "annotator_id": "REPLACE_WITH_INDEPENDENT_ANNOTATOR_ID",
        "decisions": blank_submission_rows(items),
    }
    _write_json(args.output_dir / "annotator-template.json", template)
    _write_json(
        args.output_dir / "adjudicator-template.json",
        {
            **template,
            "annotator_id": "REPLACE_WITH_INDEPENDENT_ADJUDICATOR_ID",
        },
    )
    label_counts: dict[str, int] = {}
    for item in items:
        label_counts[item["conflict_type"]] = label_counts.get(item["conflict_type"], 0) + 1
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "GENERATED_AWAITING_INDEPENDENT_HUMAN_ANNOTATION",
        "package_id": package_id,
        "case_count": len(items),
        "response_count": len(items) * len(METHODS),
        "methods_hidden_from_reviewers": True,
        "methods": list(METHODS),
        "conflict_type_counts": label_counts,
        "protocol_path_label": args.protocol.name,
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_sha256": frozen["dataset_sha256"],
        "selected_evidence_sha256": frozen["selected_evidence_sha256"],
        "generation_cache_sha256": file_sha256(args.cache_dir / "generations.jsonl"),
        "package_file_sha256": file_sha256(package_path),
        "package_items_sha256": canonical_json_sha256(items),
        "blind_mapping_sha256": canonical_json_sha256(mapping),
        "blind_mapping_path_label": mapping_path.name,
        "human_evidence_complete": False,
        "gate_2": "NO-GO/SHADOW",
        "limitations": [
            "No adherence, grounding, citation, correctness, or preference result exists until two independent submissions and independent adjudication are completed.",
            "The local Qwen generator is shared across methods and is not an evaluator or expert judge.",
            "CONFLICTS is cross-domain and cannot replace a flood-domain expert benchmark.",
            "coverage_greedy_proxy is not a SetR reproduction.",
        ],
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "PROTOCOL.md").write_text(
        _protocol_markdown(manifest), encoding="utf-8"
    )
    print(args.output_dir / "manifest.json")
    print(package_path)
    print("GENERATED_AWAITING_INDEPENDENT_HUMAN_ANNOTATION")


def compare(args: argparse.Namespace) -> None:
    manifest = _load_json(args.manifest)
    items = read_jsonl(args.package)
    first = _load_json(args.first)
    second = _load_json(args.second)
    if file_sha256(args.package) != manifest["package_file_sha256"]:
        raise ValueError("CONFLICTS behavior package differs from manifest")
    comparison = compare_submissions(items, first, second)
    if comparison["schema_version"] != COMPARISON_SCHEMA_VERSION:
        raise AssertionError("unexpected comparison schema")
    _write_json(args.output, comparison)
    print(args.output)


def finalize(args: argparse.Namespace) -> None:
    manifest = _load_json(args.manifest)
    items = read_jsonl(args.package)
    mapping = _load_json(args.mapping)
    first = _load_json(args.first)
    second = _load_json(args.second)
    adjudication = _load_json(args.adjudication)
    if file_sha256(args.package) != manifest["package_file_sha256"]:
        raise ValueError("CONFLICTS behavior package differs from manifest")
    report = finalize_adjudication(
        items, manifest, mapping, first, second, adjudication
    )
    _write_json(args.output, report)
    markdown = args.output.with_suffix(".md")
    markdown.write_text(_render_final(report), encoding="utf-8")
    print(args.output)
    print(markdown)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate, compare, and adjudicate a blinded CONFLICTS behavior audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "docs/progressive_upgrade/conflicts_expected_behavior_protocol.json"
        ),
    )
    prepare_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(".cache/benchmarks/rag_conflicts/conflicts.jsonl"),
    )
    prepare_parser.add_argument(
        "--selected-evidence",
        type=Path,
        default=Path(
            ".cache/benchmarks/rag_conflicts/evaluation_cache_full/selected_evidence.jsonl"
        ),
    )
    prepare_parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "D:/RAG_test/.hf_cache/local_models/Qwen2.5-7B-Instruct-GPTQ-Int4"
        ),
    )
    prepare_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            ".cache/benchmarks/rag_conflicts/expected_behavior_workflow"
        ),
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rag_evaluation/conflicts_expected_behavior"),
    )
    prepare_parser.set_defaults(func=prepare)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--manifest", type=Path, required=True)
    compare_parser.add_argument("--package", type=Path, required=True)
    compare_parser.add_argument("--first", type=Path, required=True)
    compare_parser.add_argument("--second", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(func=compare)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--manifest", type=Path, required=True)
    finalize_parser.add_argument("--package", type=Path, required=True)
    finalize_parser.add_argument("--mapping", type=Path, required=True)
    finalize_parser.add_argument("--first", type=Path, required=True)
    finalize_parser.add_argument("--second", type=Path, required=True)
    finalize_parser.add_argument("--adjudication", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    finalize_parser.set_defaults(func=finalize)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
