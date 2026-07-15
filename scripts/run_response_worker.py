from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path

from flood_system.response_workflow.models import OperatorRole, OutboxProcessRequest, TaskActionRequest
from flood_system.system import FloodWarningSystem


def run_cycle(system: FloodWarningSystem) -> dict[str, int]:
    workflow = system.response_workflow
    processed = workflow.process_outbox(
        OutboxProcessRequest(
            max_messages=100,
            operator_id="response-worker",
            operator_role=OperatorRole.ADMIN,
            terminal_id="response-worker",
        )
    )
    escalated = 0
    for event in workflow.list_events():
        if event.status.value == "closed":
            continue
        escalated += len(
            workflow.run_deadline_sweep(
                event.event_id,
                TaskActionRequest(
                    operator_id="response-worker",
                    operator_role=OperatorRole.DUTY_OFFICER,
                    terminal_id="response-worker",
                    note="scheduled deadline sweep",
                ),
            )
        )
    return {"outbox_processed": len(processed), "tasks_escalated": escalated}


def write_heartbeat(path: Path, result: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps({"updated_at_epoch": time.time(), "last_cycle": result}, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Outbox and deadline worker cycles.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=float(os.getenv("FLOOD_RESPONSE_WORKER_INTERVAL", "10")))
    parser.add_argument(
        "--health-file",
        type=Path,
        default=Path(os.getenv("FLOOD_RESPONSE_WORKER_HEALTH_FILE", "tmp/response-worker-health.json")),
    )
    args = parser.parse_args()
    if args.interval < 1:
        raise SystemExit("worker interval must be at least one second")
    db_path = Path(os.environ["FLOOD_DB_PATH"]).expanduser().resolve()
    system = FloodWarningSystem(db_path)
    if args.once:
        result = run_cycle(system)
        write_heartbeat(args.health_file, result)
        print(result)
        return 0

    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    while not stopped.is_set():
        try:
            result = run_cycle(system)
            write_heartbeat(args.health_file, result)
            print(result, flush=True)
        except Exception as exc:  # keep the worker alive; individual operations fail closed
            print({"worker_error": str(exc)}, flush=True)
        stopped.wait(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
