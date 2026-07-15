from __future__ import annotations

import json

from scripts.run_response_worker import write_heartbeat


def test_worker_heartbeat_is_atomic_and_records_last_successful_cycle(tmp_path) -> None:
    target = tmp_path / "health" / "response-worker.json"

    write_heartbeat(target, {"outbox_processed": 2, "tasks_escalated": 1})

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["updated_at_epoch"] > 0
    assert payload["last_cycle"] == {"outbox_processed": 2, "tasks_escalated": 1}
    assert not target.with_suffix(".json.tmp").exists()
