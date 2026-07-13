from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "legacy-baseline-snapshot-v1"
DEFAULT_COMMIT = "2db016a"
DEFAULT_FREEZE_DATE = "2026-07-14"
TREE_ENTRY = re.compile(
    r"^(?P<mode>\d+)\s+(?P<type>\w+)\s+(?P<object>[0-9a-f]+)\s+(?P<size>\d+|-)\t(?P<path>.+)$"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def _tree_entries(repo_root: Path, commit: str) -> list[dict[str, Any]]:
    raw = _git(repo_root, "ls-tree", "-r", "-l", "-z", commit, text=False)
    entries: list[dict[str, Any]] = []
    for record in bytes(raw).split(b"\0"):
        if not record:
            continue
        match = TREE_ENTRY.match(record.decode("utf-8"))
        if match is None:
            raise ValueError(f"unexpected git tree entry: {record!r}")
        size = match.group("size")
        entries.append(
            {
                "path": match.group("path"),
                "mode": match.group("mode"),
                "object": match.group("object"),
                "bytes": None if size == "-" else int(size),
            }
        )
    return sorted(entries, key=lambda item: item["path"])


def _category(
    entries: Iterable[dict[str, Any]], prefixes: tuple[str, ...]
) -> dict[str, Any]:
    selected = [
        entry
        for entry in entries
        if any(
            entry["path"] == prefix.rstrip("/") or entry["path"].startswith(prefix)
            for prefix in prefixes
        )
    ]
    canonical = json.dumps(
        selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "prefixes": list(prefixes),
        "file_count": len(selected),
        "blob_bytes": sum(int(item["bytes"] or 0) for item in selected),
        "listing_sha256": sha256_bytes(canonical),
        "entries": selected,
    }


def _safe_extract_tar(payload: bytes, target: Path) -> None:
    target_resolved = target.resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            destination = (target / member.name).resolve()
            if (
                destination != target_resolved
                and target_resolved not in destination.parents
            ):
                raise ValueError(f"unsafe git archive member: {member.name}")
        archive.extractall(target, filter="data")


def _canonical_sql_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float):
        return {"float": format(value, ".17g")}
    return value


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def fingerprint_sqlite(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        table_rows = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        tables: list[dict[str, Any]] = []
        for table_name, create_sql in table_rows:
            columns = connection.execute(
                f"PRAGMA table_info({_quote_identifier(table_name)})"
            ).fetchall()
            column_names = [str(column[1]) for column in columns]
            primary_key = [
                str(column[1])
                for column in sorted(columns, key=lambda column: int(column[5]) or 9999)
                if int(column[5]) > 0
            ]
            order_columns = primary_key or column_names
            order_sql = ", ".join(_quote_identifier(name) for name in order_columns)
            query = f"SELECT * FROM {_quote_identifier(table_name)}"
            if order_sql:
                query += f" ORDER BY {order_sql}"
            digest = hashlib.sha256()
            row_count = 0
            for row in connection.execute(query):
                canonical = json.dumps(
                    [_canonical_sql_value(value) for value in row],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                digest.update(canonical + b"\n")
                row_count += 1
            tables.append(
                {
                    "name": table_name,
                    "columns": column_names,
                    "primary_key": primary_key,
                    "row_count": row_count,
                    "schema_sha256": sha256_bytes((create_sql or "").encode("utf-8")),
                    "rows_sha256": digest.hexdigest(),
                }
            )
        indexes = [
            {
                "name": name,
                "table": table_name,
                "schema_sha256": sha256_bytes((sql or "").encode("utf-8")),
            }
            for name, table_name, sql in connection.execute(
                "SELECT name, tbl_name, sql FROM sqlite_master "
                "WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        ]
        return {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "table_count": len(tables),
            "index_count": len(indexes),
            "tables": tables,
            "indexes": indexes,
        }
    finally:
        connection.close()


def normalize_legacy_bootstrap_clock(path: Path, frozen_timestamp: str) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE v2_area_resource_status SET updated_at = ?", (frozen_timestamp,)
        )
        rows = connection.execute(
            "SELECT component_key, payload FROM v2_supervisor_health_state"
        ).fetchall()
        for component_key, raw_payload in rows:
            payload = json.loads(raw_payload)
            payload["updated_at"] = frozen_timestamp
            connection.execute(
                "UPDATE v2_supervisor_health_state "
                "SET updated_at = ?, payload = ? WHERE component_key = ?",
                (
                    frozen_timestamp,
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    component_key,
                ),
            )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()


def build_legacy_baseline_manifest(
    repo_root: Path,
    *,
    commit: str,
    freeze_date: str,
    local_snapshot_root: Path,
) -> dict[str, Any]:
    full_commit = str(_git(repo_root, "rev-parse", f"{commit}^{{commit}}")).strip()
    commit_lines = str(
        _git(repo_root, "show", "-s", "--format=%H%n%T%n%cI%n%s", full_commit)
    ).splitlines()
    if len(commit_lines) < 4:
        raise ValueError("legacy commit metadata is incomplete")
    entries = _tree_entries(repo_root, full_commit)
    archive = bytes(_git(repo_root, "archive", "--format=tar", full_commit, text=False))
    snapshot_dir = local_snapshot_root / full_commit
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="legacy-baseline-build-", dir=local_snapshot_root
    ) as temporary:
        source_root = Path(temporary) / "source"
        source_root.mkdir()
        _safe_extract_tar(archive, source_root)
        database_path = source_root / "data" / "flood_warning_system_v2.db"
        database_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root)
        environment["FLOOD_SUPERVISOR_LOOP_ENABLED"] = "0"
        baseline_index = database_path.parent / "rag_documents.baseline.json"
        bootstrap = (
            "import json; from pathlib import Path; "
            "from flood_system.system import FloodWarningSystem; "
            f"system=FloodWarningSystem(r'{database_path}'); "
            f"Path(r'{baseline_index}').write_text(json.dumps("
            "[item.model_dump(mode='json') for item in system.rag_service.list_documents()], "
            "ensure_ascii=False, indent=2), encoding='utf-8')"
        )
        subprocess.run(
            [sys.executable, "-c", bootstrap],
            cwd=source_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        if not database_path.is_file() or not baseline_index.is_file():
            raise ValueError(
                "legacy bootstrap did not create the database and RAG snapshot"
            )
        normalize_legacy_bootstrap_clock(database_path, commit_lines[2])
        frozen_database = snapshot_dir / database_path.name
        frozen_index = snapshot_dir / baseline_index.name
        shutil.copy2(database_path, frozen_database)
        shutil.copy2(baseline_index, frozen_index)

    categories = {
        "application": _category(
            entries, ("flood_system/", "frontend/", "3D_visual/", "scripts/")
        ),
        "database_rebuild_inputs": _category(
            entries,
            (
                "flood_system/bootstrap_data/",
                "flood_system/sample_data.py",
                "flood_system/v2/bootstrap.py",
                "data_sources/beilin/normalized/",
            ),
        ),
        "rag_index_rebuild_inputs": _category(
            entries,
            (
                "flood_system/rag_raw/",
                "flood_system/rag.py",
                "flood_system/rag_runtime.py",
                "data_sources/beilin/normalized/rag_documents.beilin.json",
            ),
        ),
        "models_prompts_and_contracts": _category(
            entries,
            (
                "flood_system/v2/prompt_profiles.json",
                "docs/current_baseline/",
                "AgentTwin-Flood-Requirements.md",
            ),
        ),
    }
    database = fingerprint_sqlite(frozen_database)
    runtime_payload = json.loads(frozen_index.read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "freeze_date": freeze_date,
            "scope": "controlled pre-upgrade legacy application baseline",
            "source_commit": full_commit,
            "source_tree": commit_lines[1],
            "source_commit_date": commit_lines[2],
            "source_subject": commit_lines[3],
            "git_archive_sha256": sha256_bytes(archive),
            "bootstrap_clock_normalization": (
                "runtime-only resource/supervisor timestamps are normalized to the source "
                "commit timestamp before VACUUM so repeated offline rebuilds are byte-identical"
            ),
            "local_snapshot_policy": (
                "binary database and runtime index are retained under ignored .cache; "
                "the tracked manifest contains only hashes, schema, counts, and Git blob identities"
            ),
        },
        "tracked_tree": {
            "file_count": len(entries),
            "blob_bytes": sum(int(item["bytes"] or 0) for item in entries),
            "categories": categories,
        },
        "database_snapshot": {
            "logical_path": "data/flood_warning_system_v2.db",
            "local_cache_path": str(frozen_database.relative_to(repo_root)).replace(
                "\\", "/"
            ),
            "bootstrap_entrypoint": "flood_system.system.FloodWarningSystem",
            **database,
        },
        "rag_index_snapshot": {
            "logical_path": "data/rag_documents.baseline.json",
            "local_cache_path": str(frozen_index.relative_to(repo_root)).replace(
                "\\", "/"
            ),
            "bytes": frozen_index.stat().st_size,
            "sha256": sha256_file(frozen_index),
            "document_count": len(runtime_payload),
        },
        "reproduction": {
            "command": (
                "python scripts/freeze_legacy_baseline.py --commit 2db016a "
                "--freeze-date 2026-07-14"
            ),
            "network_required": False,
            "writes_formal_response_state": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    database = report["database_snapshot"]
    rag = report["rag_index_snapshot"]
    lines = [
        "# 原系统可复现基线冻结报告",
        "",
        f"- 冻结日期：`{metadata['freeze_date']}`",
        f"- Git 提交：`{metadata['source_commit']}`",
        f"- Git Tree：`{metadata['source_tree']}`",
        f"- 原提交时间：`{metadata['source_commit_date']}`",
        f"- 原提交说明：{metadata['source_subject']}",
        f"- Git archive SHA-256：`{metadata['git_archive_sha256']}`",
        "",
        "## 数据库与索引",
        "",
        f"- 从冻结提交离线启动旧系统生成 SQLite 基线：{database['table_count']} 张表、{database['index_count']} 个显式索引、{database['bytes']} bytes。",
        f"- 数据库 SHA-256：`{database['sha256']}`。",
        f"- RAG 运行索引：{rag['document_count']} 篇文档、{rag['bytes']} bytes，SHA-256=`{rag['sha256']}`。",
        "- 二进制数据库和索引仅保留在忽略的 `.cache/legacy_baseline/`，避免把运行数据或密钥提交到 Git；仓库提交完整表结构、逐表行数/内容哈希和所有重建输入 Git blob 身份。",
        "",
        "## 重建输入",
        "",
        "| 类别 | 文件数 | Blob bytes | 清单 SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for name, category in report["tracked_tree"]["categories"].items():
        lines.append(
            f"| {name} | {category['file_count']} | {category['blob_bytes']} | `{category['listing_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "该报告证明受控旧系统代码、数据库重建输入、RAG 原文/索引输入、Prompt 与合同可以从指定 Git 对象离线重建。它不代表旧生产流量已经归零，也不替代生产数据库归档、账号撤权或真实恢复演练。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze an auditable, offline-reproducible legacy code/database/RAG baseline."
    )
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--freeze-date", default=DEFAULT_FREEZE_DATE)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/acceptance/legacy_baseline_manifest.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("output/acceptance/legacy_baseline_manifest.md"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    report = build_legacy_baseline_manifest(
        repo_root,
        commit=args.commit,
        freeze_date=args.freeze_date,
        local_snapshot_root=repo_root / ".cache" / "legacy_baseline",
    )
    json_target = (repo_root / args.output_json).resolve()
    markdown_target = (repo_root / args.output_markdown).resolve()
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_target.write_text(render_markdown(report), encoding="utf-8")
    print(json_target)
    print(markdown_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
