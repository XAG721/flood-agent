from __future__ import annotations

import copy
import hashlib
import http.client
import json
import threading
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from research.frc_rag.conflicts_annotation_operations import (
    BATCH_SCHEMA_VERSION,
    BATCH_SUBMISSION_SCHEMA_VERSION,
)
from research.frc_rag.conflicts_annotation_workstation import (
    ASSET_ROOT,
    MANIFEST_SCHEMA_VERSION,
    WorkstationSession,
    WorkstationValidationError,
    create_workstation_server,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(citation: str) -> dict:
    return {
        "citation": citation,
        "title": "Public evidence",
        "url": "https://example.test/evidence",
        "date": "2026-01-01",
        "text": "Evidence supplied to both blinded reviewers.",
    }


def _task(task_id: str, *, correct_answer: str | None) -> dict:
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "review_task_id": task_id,
        "question": f"Question for {task_id}?",
        "conflict_type": "No conflict",
        "expected_behavior": "Use the supplied evidence and preserve uncertainty.",
        "correct_answer": correct_answer,
        "responses": [
            {
                "response_id": alias,
                "response": f"Blinded response {alias} cites the evidence [1].",
                "sources": [_source("[1]")],
            }
            for alias in ("A", "B")
        ],
    }


def _blank_decision(task: dict) -> dict:
    answer = "REQUIRED" if task["correct_answer"] is not None else "NOT_APPLICABLE"
    return {
        "review_task_id": task["review_task_id"],
        "ratings": {
            alias: {
                "expected_behavior_adherence": "REQUIRED",
                "factual_grounding": "REQUIRED",
                "citation_correctness": "REQUIRED",
                "answer_correctness": answer,
                "rationale": "REQUIRED: explain all four ratings",
            }
            for alias in ("A", "B")
        },
        "preference": "REQUIRED",
        "notes": "",
    }


def _public_fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    public_root = tmp_path / "public"
    tasks = [
        _task("TASK-ONE", correct_answer="gold"),
        _task("TASK-TWO", correct_answer=None),
    ]
    batch = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "operations_id": "CONFLICTS-ANNOTATION-OPS-TEST",
        "package_id": "CONFLICTS-BEHAVIOR-TEST",
        "reviewer_slot": "reviewer_1",
        "batch_id": "REVIEWER_1-B01",
        "batch_index": 1,
        "batch_count": 1,
        "task_count": len(tasks),
        "instructions": ["Review independently."],
        "tasks": tasks,
    }
    template = {
        "schema_version": BATCH_SUBMISSION_SCHEMA_VERSION,
        "operations_id": batch["operations_id"],
        "package_id": batch["package_id"],
        "reviewer_slot": batch["reviewer_slot"],
        "batch_id": batch["batch_id"],
        "annotator_id": "REPLACE_WITH_PRIVATE_REVIEWER_1_ID",
        "decisions": [_blank_decision(task) for task in tasks],
    }
    batch_path = public_root / "reviewer-1" / "batch-01.json"
    template_path = public_root / "reviewer-1" / "batch-01-submission-template.json"
    _write_json(batch_path, batch)
    _write_json(template_path, template)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "PREPARED_AWAITING_INDEPENDENT_HUMAN_REVIEW",
        "operations_id": batch["operations_id"],
        "package_id": batch["package_id"],
        "reviewer_slots": ["reviewer_1"],
        "batch_count": 1,
        "public_files": [
            {
                "kind": "batch",
                "reviewer_slot": "reviewer_1",
                "batch_id": batch["batch_id"],
                "path": batch_path.relative_to(public_root).as_posix(),
                "sha256": _sha256(batch_path),
            },
            {
                "kind": "submission_template",
                "reviewer_slot": "reviewer_1",
                "batch_id": batch["batch_id"],
                "path": template_path.relative_to(public_root).as_posix(),
                "sha256": _sha256(template_path),
            },
        ],
    }
    _write_json(public_root / "manifest.json", manifest)
    return public_root, tmp_path / ".cache" / "annotation-drafts", batch, manifest


def _session(tmp_path: Path) -> WorkstationSession:
    public_root, draft_root, _, _ = _public_fixture(tmp_path)
    return WorkstationSession(
        repository_root=tmp_path,
        public_root=public_root,
        draft_root=draft_root,
        reviewer_slot="reviewer_1",
        batch_index=1,
        expected_manifest_sha256=_sha256(public_root / "manifest.json"),
    )


def _completed(draft: dict) -> dict:
    completed = copy.deepcopy(draft)
    completed["annotator_id"] = "private-reviewer-alpha"
    for decision in completed["decisions"]:
        for rating in decision["ratings"].values():
            rating["expected_behavior_adherence"] = "PASS"
            rating["factual_grounding"] = "PASS"
            rating["citation_correctness"] = "PASS"
            if rating["answer_correctness"] == "REQUIRED":
                rating["answer_correctness"] = "PASS"
            rating["rationale"] = "The response is supported by the visible evidence."
        decision["preference"] = "TIE"
    return completed


def _request(
    port: int,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_workstation_persists_only_ignored_private_drafts_and_resumes(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    state = session.state()

    assert session.draft_path.is_relative_to(tmp_path / ".cache")
    assert state["status"] == "IN_PROGRESS"
    assert state["progress"]["completed_task_count"] == 0
    assert state["privacy"] == {
        "local_only": True,
        "draft_git_ignored": True,
        "routing_loaded": False,
        "method_identity_loaded": False,
    }

    partial = copy.deepcopy(state["draft"])
    partial["annotator_id"] = "private-reviewer-alpha"
    partial["decisions"][0]["notes"] = "Resume marker"
    session.save(partial)
    resumed = WorkstationSession(
        repository_root=tmp_path,
        public_root=session.public_root,
        draft_root=session.draft_root,
        reviewer_slot="reviewer_1",
        batch_index=1,
        expected_manifest_sha256=_sha256(session.public_root / "manifest.json"),
    )
    assert resumed.draft["decisions"][0]["notes"] == "Resume marker"

    with pytest.raises(ValueError, match="ignored .cache"):
        WorkstationSession(
            repository_root=tmp_path,
            public_root=session.public_root,
            draft_root=tmp_path / "tracked-drafts",
            reviewer_slot="reviewer_1",
            batch_index=1,
            expected_manifest_sha256=_sha256(session.public_root / "manifest.json"),
        )


def test_workstation_finalizes_only_complete_batches_and_can_reopen(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    with pytest.raises(WorkstationValidationError, match="评审员标识"):
        session.finalize(session.draft)

    incomplete = copy.deepcopy(session.draft)
    incomplete["annotator_id"] = "private-reviewer-alpha"
    with pytest.raises(WorkstationValidationError) as error:
        session.finalize(incomplete)
    assert error.value.details["code"] == "INCOMPLETE_BATCH"
    assert error.value.details["first_review_task_id"] == "TASK-ONE"

    completed = _completed(session.draft)
    state = session.finalize(completed)
    assert state["status"] == "FINALIZED"
    assert state["receipt"]["human_evidence_complete"] is False
    assert state["receipt"]["gate_2"] == "NO-GO/SHADOW"
    assert state["progress"]["batch_complete"] is True

    edited = copy.deepcopy(completed)
    edited["decisions"][0]["notes"] = "A post-finalization correction."
    assert session.save(edited)["status"] == "IN_PROGRESS"
    assert not session.receipt_path.exists()
    session.finalize(edited)
    assert session.reopen()["status"] == "IN_PROGRESS"


def test_workstation_rejects_hash_tampering_and_public_routing_leaks(
    tmp_path: Path,
) -> None:
    public_root, draft_root, batch, manifest = _public_fixture(tmp_path)
    frozen_manifest_sha256 = _sha256(public_root / "manifest.json")
    batch_path = public_root / manifest["public_files"][0]["path"]
    batch["tasks"][0]["question"] = "Tampered question"
    _write_json(batch_path, batch)
    with pytest.raises(ValueError, match="hash differs"):
        WorkstationSession(
            repository_root=tmp_path,
            public_root=public_root,
            draft_root=draft_root,
            reviewer_slot="reviewer_1",
            batch_index=1,
            expected_manifest_sha256=_sha256(public_root / "manifest.json"),
        )

    batch["tasks"][0]["item_id"] = "PRIVATE-CASE-ID"
    _write_json(batch_path, batch)
    manifest["public_files"][0]["sha256"] = _sha256(batch_path)
    _write_json(public_root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest hash differs"):
        WorkstationSession(
            repository_root=tmp_path,
            public_root=public_root,
            draft_root=draft_root,
            reviewer_slot="reviewer_1",
            batch_index=1,
            expected_manifest_sha256=frozen_manifest_sha256,
        )
    with pytest.raises(ValueError, match="private routing"):
        WorkstationSession(
            repository_root=tmp_path,
            public_root=public_root,
            draft_root=draft_root,
            reviewer_slot="reviewer_1",
            batch_index=1,
            expected_manifest_sha256=_sha256(public_root / "manifest.json"),
        )


def test_loopback_server_requires_token_origin_and_secure_headers(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    server, launch_url = create_workstation_server(session, token="fixed-test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = urlsplit(launch_url).port
    assert port is not None
    try:
        status, headers, _ = _request(port, "GET", "/")
        assert status == 403
        assert headers["Cache-Control"] == "no-store"

        status, headers, html = _request(
            port, "GET", "/?token=fixed-test-token"
        )
        assert status == 200
        assert b"CONFLICTS" in html
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert headers["X-Frame-Options"] == "DENY"
        page_cookie = headers["Set-Cookie"]
        assert "HttpOnly" in page_cookie
        assert "SameSite=Strict" in page_cookie
        status, _, _ = _request(
            port,
            "GET",
            "/",
            headers={"Cookie": page_cookie.split(";", maxsplit=1)[0]},
        )
        assert status == 200

        status, _, _ = _request(port, "GET", "/api/session")
        assert status == 403
        token_header = {"X-Review-Token": "fixed-test-token"}
        status, _, payload = _request(
            port, "GET", "/api/session", headers=token_header
        )
        assert status == 200
        state = json.loads(payload)
        assert state["privacy"]["routing_loaded"] is False

        status, _, _ = _request(
            port,
            "GET",
            "/api/session",
            headers={
                **token_header,
                "Origin": "https://attacker.example",
            },
        )
        assert status == 403
        status, _, _ = _request(
            port, "GET", "/api/unknown", headers=token_header
        )
        assert status == 404

        state["draft"]["annotator_id"] = "private-reviewer-alpha"
        status, _, saved = _request(
            port,
            "PUT",
            "/api/draft",
            body=json.dumps(state["draft"]).encode(),
            headers={
                **token_header,
                "Content-Type": "application/json",
            },
        )
        assert status == 200
        assert json.loads(saved)["progress"]["identity_ready"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_static_workstation_is_external_free_accessible_and_method_blind() -> None:
    html = (ASSET_ROOT / "index.html").read_text(encoding="utf-8")
    css = (ASSET_ROOT / "styles.css").read_text(encoding="utf-8")
    javascript = (ASSET_ROOT / "app.js").read_text(encoding="utf-8")
    combined = f"{html}\n{css}\n{javascript}".lower()

    assert "coverage_greedy_proxy" not in combined
    assert "frc_select" not in combined
    assert "innerhtml" not in javascript.lower()
    assert "http://" not in html and "https://" not in html
    assert "focus-visible" in css
    assert "prefers-reduced-motion" in css
    assert "aria-live" in html
    assert "matchmedia" in javascript.lower()
    assert "sessionstorage" in javascript.lower()
