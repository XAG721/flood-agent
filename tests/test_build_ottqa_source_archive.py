from __future__ import annotations

import io
import tarfile
from pathlib import Path

import scripts.build_ottqa_source_archive as builder


def _add(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 123456
    archive.addfile(info, io.BytesIO(payload))


def test_builder_streams_only_registered_members_without_extracting(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(builder, "EXPECTED_TABLE_FILES", 1)
    monkeypatch.setattr(builder, "EXPECTED_PASSAGE_FILES", 1)
    source = tmp_path / "codeload.tar.gz"
    output = tmp_path / "registered.tar"
    prefix = "OTT-QA-pinned"
    expected = {
        "LICENSE": b"MIT\n",
        "preprocessed_data/dev_linked.json": b"[]",
        'data/traindev_tables_tok/unsafe:"table*0.json': b'{"table": 1}',
        'data/traindev_request_tok/unsafe:"table*0.json': b'{"passage": 1}',
    }
    with tarfile.open(source, mode="w:gz") as archive:
        for name, payload in expected.items():
            _add(archive, f"{prefix}/{name}", payload)
        _add(archive, f"{prefix}/README.md", b"excluded")
    report = builder.build(source, output)
    assert report["json_content_parsed"] is False
    assert report["members_extracted_to_filesystem"] is False
    with tarfile.open(output, mode="r:") as archive:
        assert {member.name for member in archive if member.isfile()} == set(expected)
        for name, payload in expected.items():
            handle = archive.extractfile(name)
            assert handle is not None
            assert handle.read() == payload
