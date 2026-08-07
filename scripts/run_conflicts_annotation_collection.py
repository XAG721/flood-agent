from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conflicts_annotation_collection import CollectionManager


DEFAULT_PUBLIC_ROOT = Path("output/rag_evaluation/conflicts_annotation_operations")
DEFAULT_DRAFT_ROOT = Path(
    ".cache/benchmarks/rag_conflicts/expected_behavior_workflow/annotation_submissions"
)
DEFAULT_PACKAGE = Path(
    "output/rag_evaluation/conflicts_expected_behavior/package.jsonl.gz"
)
DEFAULT_ROUTING = Path(
    ".cache/benchmarks/rag_conflicts/expected_behavior_workflow/"
    "annotation_operations_routing.json"
)
DEFAULT_MERGE_ROOT = Path(
    ".cache/benchmarks/rag_conflicts/expected_behavior_workflow/private_merge"
)


def _inside_repository(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _manager(args: argparse.Namespace) -> CollectionManager:
    return CollectionManager(
        repository_root=REPOSITORY_ROOT,
        public_root=_inside_repository(args.public_root),
        draft_root=_inside_repository(args.draft_root),
    )


def status(args: argparse.Namespace) -> None:
    manager = _manager(args)
    report = (
        manager.write_status(_inside_repository(args.output))
        if args.output is not None
        else manager.audit()
    )
    summary = {
        "status": report["status"],
        "present_batch_count": report["present_batch_count"],
        "finalized_batch_count": report["finalized_batch_count"],
        "expected_batch_count": report["expected_batch_count"],
        "independent_role_identity_count": report[
            "independent_role_identity_count"
        ],
        "human_evidence_complete": report["human_evidence_complete"],
        "gate_2": report["gate_2"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output is not None:
        print(_inside_repository(args.output))


def merge(args: argparse.Namespace) -> None:
    manager = _manager(args)
    manifest = manager.merge_private_collection(
        package_path=_inside_repository(args.package),
        routing_path=_inside_repository(args.routing),
        output_root=_inside_repository(args.output_root),
        minimum_exact_agreement=args.minimum_exact_agreement,
    )
    print(_inside_repository(args.output_root) / "manifest.json")
    print(manifest["status"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and privately merge finalized CONFLICTS review batches without "
            "opening the blind mapping."
        )
    )
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output", type=Path)
    status_parser.set_defaults(func=status)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    merge_parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    merge_parser.add_argument("--output-root", type=Path, default=DEFAULT_MERGE_ROOT)
    merge_parser.add_argument(
        "--minimum-exact-agreement", type=float, default=0.8
    )
    merge_parser.set_defaults(func=merge)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
