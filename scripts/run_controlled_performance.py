from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from flood_system.http.response_router import create_response_router
from flood_system.identity import TrustedIdentityVerifier
from flood_system.performance_acceptance import evaluate_budget, summarize_samples
from flood_system.response_workflow.models import OperatorRole
from flood_system.system import FloodWarningSystem


IDENTITY_SECRET = "controlled-performance-identity-secret-at-least-32-bytes"


def _headers(method: str, path: str) -> dict[str, str]:
    nonce = f"perf-{uuid4().hex}"
    headers = TrustedIdentityVerifier.build_headers(
        secret=IDENTITY_SECRET,
        operator_id="performance-commander",
        operator_role=OperatorRole.COMMANDER,
        assurance_level="aal2",
        terminal_id="controlled-performance-runner",
        nonce=nonce,
        method=method,
        path=path,
    )
    headers["X-Correlation-ID"] = nonce
    return headers


def run_controlled_performance(db_path: Path, budget: dict) -> dict:
    db_path = db_path.expanduser().resolve()
    if db_path.exists():
        raise FileExistsError(f"performance database already exists: {db_path}")
    system = FloodWarningSystem(db_path)
    system.response_identity = TrustedIdentityVerifier(system.repository, IDENTITY_SECRET)
    dashboard = system.response_workflow.bootstrap_demo()
    event_id = dashboard.event.event_id
    app = FastAPI()
    app.include_router(create_response_router(lambda: system))
    endpoints = {
        endpoint_id: {
            "method": spec["method"],
            "path": spec["path_template"].format(event_id=event_id),
        }
        for endpoint_id, spec in budget["endpoints"].items()
    }
    sample_count = int(budget["sample_count"])
    warmup_count = int(budget["warmup_count"])
    results: dict[str, dict] = {}
    with TestClient(app) as client:
        for endpoint_id, endpoint in endpoints.items():
            for _ in range(warmup_count):
                response = client.request(
                    endpoint["method"], endpoint["path"], headers=_headers(endpoint["method"], endpoint["path"])
                )
                if not 200 <= response.status_code < 300:
                    raise RuntimeError(f"warmup failed for {endpoint_id}: HTTP {response.status_code}")
            durations: list[float] = []
            statuses: list[int] = []
            for _ in range(sample_count):
                started = time.perf_counter()
                response = client.request(
                    endpoint["method"], endpoint["path"], headers=_headers(endpoint["method"], endpoint["path"])
                )
                durations.append((time.perf_counter() - started) * 1000)
                statuses.append(response.status_code)
            results[endpoint_id] = {
                "method": endpoint["method"],
                "path": endpoint["path"],
                **summarize_samples(durations, statuses),
            }
    decision = evaluate_budget(results, budget)
    return {
        "budget_name": budget["name"],
        "budget_version": budget["version"],
        "scope": budget["scope"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "results": results,
        "decision": decision,
        "limitations": budget.get("limitations", []),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# 受控 API 性能预算报告",
        "",
        f"- 预算：{report['budget_name']}（{report['budget_version']}）",
        f"- 判定：{report['decision']['status']}",
        f"- 范围：{report['scope']}",
        "",
        "| 端点 | 样本 | P50 ms | P95 ms | Max ms | 错误率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for endpoint_id, result in report["results"].items():
        lines.append(
            f"| {endpoint_id} | {result['sample_count']} | {result['p50_ms']:.3f} | "
            f"{result['p95_ms']:.3f} | {result['max_ms']:.3f} | {result['error_rate']:.6f} |"
        )
    lines.extend(["", "## 限制", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the controlled response API performance regression budget.")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument(
        "--budget",
        type=Path,
        default=Path("benchmarks/controlled_performance_budget.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/performance"))
    args = parser.parse_args()
    budget = json.loads(args.budget.read_text(encoding="utf-8"))
    report = run_controlled_performance(args.db, budget)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "controlled_performance_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "controlled_performance_report.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["decision"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
