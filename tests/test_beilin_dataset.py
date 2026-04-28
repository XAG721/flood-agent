from __future__ import annotations

import json
import time
from pathlib import Path

import flood_system.data_pipeline.beilin_dataset as dataset_module
from flood_system.data_pipeline.service import BeilinDatasetService
from flood_system.data_pipeline.beilin_dataset import (
    BEILIN_AREA_ID,
    build_dataset,
    fetch_beilin_sources,
    normalize_beilin_sources,
)
from flood_system.sample_data import build_area_profiles
from flood_system.v2.bootstrap import build_entity_profiles


def test_build_beilin_dataset_generates_importable_files(tmp_path: Path):
    result = build_dataset(tmp_path, sync_db=False)

    bootstrap_dir = tmp_path / "flood_system" / "bootstrap_data"
    assert (bootstrap_dir / "area_profiles.csv").exists()
    assert (bootstrap_dir / "roads.csv").exists()
    assert (bootstrap_dir / "shelters.csv").exists()
    assert (bootstrap_dir / "resource_status.csv").exists()
    assert (bootstrap_dir / "observations_beilin_mild.csv").exists()
    assert (bootstrap_dir / "observations_beilin_warning.csv").exists()
    assert (bootstrap_dir / "observations_beilin_extreme.csv").exists()
    assert (tmp_path / "data" / "rag_documents.runtime.json").exists()
    assert result["validation"]["entity_profile_count"] >= 8
    assert result["validation"]["rag_query_hit_count"] >= 3


def test_build_entity_profiles_prefers_external_seed(tmp_path: Path, monkeypatch):
    build_dataset(tmp_path, sync_db=False)
    bootstrap_dir = tmp_path / "flood_system" / "bootstrap_data"
    monkeypatch.setenv("FLOOD_BOOTSTRAP_DATA_DIR", str(bootstrap_dir))

    try:
        area_profile = build_area_profiles(bootstrap_dir)[BEILIN_AREA_ID]
        profiles = build_entity_profiles(area_profile)
    finally:
        monkeypatch.delenv("FLOOD_BOOTSTRAP_DATA_DIR", raising=False)

    assert "school_wyl_primary" in profiles
    school_profile = profiles["school_wyl_primary"]
    assert school_profile.name in {"文艺路小学", "Wenyi Road Primary School"}
    assert school_profile.area_id == BEILIN_AREA_ID
    assert school_profile.location_hint
    assert school_profile.custom_attributes["provenance"] == "real_poi"


def test_fetch_and_normalize_can_parse_cached_shelter_html(tmp_path: Path, monkeypatch):
    html_source = tmp_path / "shelters.html"
    html_source.write_text(
        """
        <table>
          <tr data-shelter-id="beilin_shelter_html_1">
            <td>Nanmen Plaza Shelter</td>
            <td>Nanyuanmen Street</td>
            <td>1200</td>
            <td>850</td>
            <td>108.9555</td>
            <td>34.2431</td>
          </tr>
        </table>
        """,
        encoding="utf-8",
    )
    registry = [
        {
            "source_id": "xa_emergency_shelters_2025",
            "title": "Test shelter source",
            "category": "shelters",
            "source_type": "official",
            "source_ref": html_source.resolve().as_uri(),
            "download_url": "",
            "last_verified_at": "2026-04-03T00:00:00+08:00",
            "notes": "local test source",
        }
    ]
    monkeypatch.setattr(dataset_module, "SOURCE_REGISTRY", registry)

    fetch_beilin_sources(tmp_path, download=True)
    result = normalize_beilin_sources(tmp_path)
    shelters = dataset_module._load_json(
        tmp_path / "data_sources" / "beilin" / "normalized" / "shelters.beilin.json",
    )

    assert result["parsed_shelter_count"] == 1
    assert shelters[0]["shelter_id"] == "beilin_shelter_html_1"
    assert shelters[0]["name"] == "Nanmen Plaza Shelter"
    assert (tmp_path / "data_sources" / "beilin" / "raw" / "xa_emergency_shelters_2025" / "artifacts.json").exists()
    assert (tmp_path / "data_sources" / "beilin" / "raw" / "xa_emergency_shelters_2025" / "versions.json").exists()


def test_fetch_and_normalize_can_parse_cached_shelter_pdf(tmp_path: Path, monkeypatch):
    pdf_source = tmp_path / "shelters.pdf"
    pdf_source.write_bytes(b"%PDF-1.4 fake pdf payload")
    registry = [
        {
            "source_id": "xa_emergency_shelters_2025",
            "title": "Test shelter pdf source",
            "category": "shelters",
            "source_type": "official",
            "source_ref": pdf_source.resolve().as_uri(),
            "download_url": "",
            "last_verified_at": "2026-04-03T00:00:00+08:00",
            "notes": "local pdf source",
            "parser_kind": "pdf_html_table",
        }
    ]
    monkeypatch.setattr(dataset_module, "SOURCE_REGISTRY", registry)
    monkeypatch.setattr(
        dataset_module,
        "_load_pdf_text",
        lambda _path: "Nanmen Plaza Shelter|Nanyuanmen Street|1200|850|108.9555|34.2431",
    )

    fetch_beilin_sources(tmp_path, download=True)
    result = normalize_beilin_sources(tmp_path)
    shelters = dataset_module._load_json(
        tmp_path / "data_sources" / "beilin" / "normalized" / "shelters.beilin.json",
    )

    assert result["parsed_shelter_count"] == 1
    assert shelters[0]["name"] == "Nanmen Plaza Shelter"


def test_fetch_and_normalize_can_parse_cached_osm_xml(tmp_path: Path, monkeypatch):
    osm_source = tmp_path / "beilin.osm"
    osm_source.write_text(
        """
        <osm version="0.6">
          <node id="1" lat="34.2431" lon="108.9555" />
          <node id="2" lat="34.2440" lon="108.9570" />
          <node id="3" lat="34.2450" lon="108.9580">
            <tag k="amenity" v="school" />
            <tag k="name" v="Beilin Test School" />
          </node>
          <way id="10">
            <nd ref="1" />
            <nd ref="2" />
            <tag k="highway" v="primary" />
            <tag k="name" v="Test Road" />
          </way>
        </osm>
        """,
        encoding="utf-8",
    )
    registry = [
        {
            "source_id": "geofabrik_shaanxi_osm",
            "title": "Test OSM source",
            "category": "roads_poi",
            "source_type": "open_map",
            "source_ref": osm_source.resolve().as_uri(),
            "download_url": "",
            "last_verified_at": "2026-04-03T00:00:00+08:00",
            "notes": "local osm source",
            "parser_kind": "osm_bundle",
        }
    ]
    monkeypatch.setattr(dataset_module, "SOURCE_REGISTRY", registry)

    fetch_beilin_sources(tmp_path, download=True)
    result = normalize_beilin_sources(tmp_path)
    roads = dataset_module._load_json(
        tmp_path / "data_sources" / "beilin" / "normalized" / "roads.beilin.json",
    )
    osm_extract = dataset_module._load_json(
        tmp_path / "data_sources" / "beilin" / "normalized" / "osm_extract.beilin.json",
    )

    assert result["parsed_osm_road_count"] >= 1
    assert roads[0]["name"] == "Test Road"
    assert osm_extract["poi_features"][0]["name"] == "Beilin Test School"
    assert osm_extract["artifacts_used"]


def test_dataset_service_records_history_and_manual_retry(tmp_path: Path, monkeypatch):
    service = BeilinDatasetService(repo_root=tmp_path, db_path=tmp_path / "demo.db")
    attempts = {"count": 0}

    def flaky_validate(update, check_cancel):
        attempts["count"] += 1
        update(progress_percent=30, current_step="validation", message=f"attempt {attempts['count']}")
        if attempts["count"] == 1:
            raise RuntimeError("synthetic validation failure")
        check_cancel()
        return {"summary": "Validated synthetic dataset."}

    monkeypatch.setattr(service, "_run_validate", flaky_validate)

    job = service.start_validate()
    deadline = time.time() + 5
    while time.time() < deadline:
        history = service.list_jobs()
        if history and history[0].job_id == job.job_id and history[0].status == "completed":
            break
        time.sleep(0.05)

    history = service.list_jobs()
    assert history[0].job_id == job.job_id
    assert history[0].status == "completed"
    assert history[0].retry_count == 1
    assert history[0].attempt_count == 2

    def always_fail_validate(update, check_cancel):
        update(progress_percent=30, current_step="validation", message="always failing")
        raise RuntimeError("synthetic terminal failure")

    monkeypatch.setattr(service, "_run_validate", always_fail_validate)

    failed_job = service.start_validate()
    deadline = time.time() + 5
    while time.time() < deadline:
        history = service.list_jobs()
        if history and history[0].job_id == failed_job.job_id and history[0].status == "failed":
            break
        time.sleep(0.05)

    retry_job = service.retry_job(failed_job.job_id)
    assert retry_job.retry_of_job_id == failed_job.job_id


def test_dataset_service_can_cancel_running_job(tmp_path: Path, monkeypatch):
    service = BeilinDatasetService(repo_root=tmp_path, db_path=tmp_path / "demo.db")

    def slow_validate(update, check_cancel):
        update(progress_percent=25, current_step="validation", message="working")
        for _ in range(50):
            time.sleep(0.01)
            check_cancel()
        return {"summary": "Validated synthetic dataset."}

    monkeypatch.setattr(service, "_run_validate", slow_validate)

    job = service.start_validate()
    canceled = service.cancel_job(job.job_id)
    assert canceled.status == "cancel_requested"

    deadline = time.time() + 5
    while time.time() < deadline:
        history = service.list_jobs()
        if history and history[0].job_id == job.job_id and history[0].status == "canceled":
            break
        time.sleep(0.05)

    history = service.list_jobs()
    assert history[0].job_id == job.job_id
    assert history[0].status == "canceled"


def test_dataset_status_reports_manifest_only_when_registry_exists_without_download(tmp_path: Path):
    fetch_beilin_sources(tmp_path, download=False)

    status = dataset_module.inspect_dataset_status(tmp_path)

    assert status["raw_ready"] is False
    assert status["sources"]
    assert all(source["completeness_status"] == "manifest_only" for source in status["sources"])
    assert status["missing_required_sources"]


def test_fetch_writes_artifact_and_version_manifests(tmp_path: Path, monkeypatch):
    html_source = tmp_path / "source.html"
    html_source.write_text(
        """
        <table>
          <tr data-shelter-id="beilin_shelter_html_2">
            <td>South Gate Shelter</td>
            <td>Nanyuanmen Street</td>
            <td>900</td>
            <td>620</td>
            <td>108.9555</td>
            <td>34.2431</td>
          </tr>
        </table>
        """,
        encoding="utf-8",
    )
    registry = [
        {
            "source_id": "xa_emergency_shelters_2025",
            "title": "Test shelter source",
            "category": "shelters",
            "source_type": "official",
            "source_ref": html_source.resolve().as_uri(),
            "download_url": "",
            "last_verified_at": "2026-04-03T00:00:00+08:00",
            "notes": "local test source",
            "parser_kind": "pdf_html_table",
        }
    ]
    monkeypatch.setattr(dataset_module, "SOURCE_REGISTRY", registry)

    fetch_beilin_sources(tmp_path, download=True)

    source_dir = tmp_path / "data_sources" / "beilin" / "raw" / "xa_emergency_shelters_2025"
    artifacts = json.loads((source_dir / "artifacts.json").read_text(encoding="utf-8"))
    versions = json.loads((source_dir / "versions.json").read_text(encoding="utf-8"))
    metadata = json.loads((source_dir / "cache_metadata.json").read_text(encoding="utf-8"))

    assert artifacts
    assert any(item["artifact_name"] == "source_ref" for item in artifacts)
    assert versions
    assert metadata["cache_status"] in {"cached", "parsed"}
    assert metadata["artifacts_manifest_path"].endswith("artifacts.json")
