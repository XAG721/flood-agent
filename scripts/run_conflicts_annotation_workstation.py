from __future__ import annotations

# ruff: noqa: E402 -- research modules intentionally stay outside the runtime wheel.

import argparse
import sys
import webbrowser
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.frc_rag.conflicts_annotation_operations import REVIEWER_SLOTS
from research.frc_rag.conflicts_annotation_workstation import (
    WorkstationSession,
    create_workstation_server,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local-only CONFLICTS independent-review workstation."
    )
    parser.add_argument("--reviewer-slot", choices=REVIEWER_SLOTS, required=True)
    parser.add_argument("--batch-index", type=int, choices=range(1, 9), required=True)
    parser.add_argument(
        "--public-root",
        type=Path,
        default=Path("output/rag_evaluation/conflicts_annotation_operations"),
    )
    parser.add_argument(
        "--draft-root",
        type=Path,
        default=Path(
            ".cache/benchmarks/rag_conflicts/expected_behavior_workflow/annotation_submissions"
        ),
    )
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    session = WorkstationSession(
        repository_root=REPOSITORY_ROOT,
        public_root=(REPOSITORY_ROOT / args.public_root),
        draft_root=(REPOSITORY_ROOT / args.draft_root),
        reviewer_slot=args.reviewer_slot,
        batch_index=args.batch_index,
    )
    server, launch_url = create_workstation_server(
        session, host=args.host, port=args.port
    )
    print("CONFLICTS 独立盲评工作台已启动：")
    print(launch_url)
    print(f"私有草稿：{session.draft_path}")
    print("仅监听本机；按 Ctrl+C 停止。")
    if args.open_browser:
        webbrowser.open(launch_url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
