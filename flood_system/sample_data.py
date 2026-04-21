from __future__ import annotations

import csv
import json
import re
import subprocess
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

from .models import (
    AreaProfile,
    CorpusType,
    Observation,
    RAGDocument,
    ResourceStatus,
    RoadStatus,
    Shelter,
)


CSV_ROOT = Path(__file__).resolve().parent / "bootstrap_data"
RAG_RAW_ROOT = Path(__file__).resolve().parent / "rag_raw"
def build_area_profiles(data_dir: str | Path | None = None) -> dict[str, AreaProfile]:
    root = Path(data_dir) if data_dir else CSV_ROOT
    area_rows = _read_csv(root / "area_profiles.csv")
    shelter_rows = _read_csv(root / "shelters.csv")
    road_rows = _read_csv(root / "roads.csv")

    shelters_by_area: dict[str, list[Shelter]] = {}
    for row in shelter_rows:
        shelters_by_area.setdefault(row["area_id"], []).append(
            Shelter(
                shelter_id=row["shelter_id"],
                name=row["name"],
                village=row["village"],
                capacity=_as_int(row.get("capacity")),
                available_capacity=_as_int(row.get("available_capacity")),
                accessible=_as_bool(row.get("accessible")),
            )
        )

    roads_by_area: dict[str, list[RoadStatus]] = {}
    for row in road_rows:
        roads_by_area.setdefault(row["area_id"], []).append(
            RoadStatus(
                road_id=row["road_id"],
                name=row["name"],
                from_village=row["from_village"],
                to_location=row["to_location"],
                accessible=_as_bool(row.get("accessible")),
                risk_note=row.get("risk_note", ""),
            )
        )

    profiles: dict[str, AreaProfile] = {}
    for row in area_rows:
        area_id = row["area_id"]
        profiles[area_id] = AreaProfile(
            area_id=area_id,
            region=row["region"],
            villages=_split_list(row.get("villages")),
            population=_as_int(row.get("population")),
            household_count=_as_int(row.get("household_count")),
            vulnerable_population=_as_int(row.get("vulnerable_population")),
            elderly_population=_as_int(row.get("elderly_population")),
            children_population=_as_int(row.get("children_population")),
            disabled_population=_as_int(row.get("disabled_population")),
            historical_risk_level=row.get("historical_risk_level", "unknown"),
            key_assets=_split_list(row.get("key_assets")),
            medical_facilities=_split_list(row.get("medical_facilities")),
            schools=_split_list(row.get("schools")),
            monitoring_points=_split_list(row.get("monitoring_points")),
            flood_prone_spots=_split_list(row.get("flood_prone_spots")),
            shelters=shelters_by_area.get(area_id, []),
            roads=roads_by_area.get(area_id, []),
        )
    return profiles


def build_resource_status(data_dir: str | Path | None = None) -> dict[str, ResourceStatus]:
    root = Path(data_dir) if data_dir else CSV_ROOT
    rows = _read_csv(root / "resource_status.csv")
    resources: dict[str, ResourceStatus] = {}
    for row in rows:
        resource = ResourceStatus(
            area_id=row["area_id"],
            vehicle_count=_as_int(row.get("vehicle_count")),
            staff_count=_as_int(row.get("staff_count")),
            supply_kits=_as_int(row.get("supply_kits")),
            rescue_boats=_as_int(row.get("rescue_boats")),
            ambulance_count=_as_int(row.get("ambulance_count")),
            drone_count=_as_int(row.get("drone_count")),
            portable_pumps=_as_int(row.get("portable_pumps")),
            power_generators=_as_int(row.get("power_generators")),
            medical_staff_count=_as_int(row.get("medical_staff_count")),
            volunteer_count=_as_int(row.get("volunteer_count")),
            satellite_phones=_as_int(row.get("satellite_phones")),
            notes=row.get("notes", ""),
        )
        resources[resource.area_id] = resource
    return resources


def load_observations_from_csv(csv_path: str | Path) -> list[Observation]:
    rows = _read_csv(Path(csv_path))
    observations: list[Observation] = []
    for row in rows:
        observations.append(
            Observation(
                observed_at=_as_datetime(row["observed_at"]),
                rainfall_mm=_as_float(row.get("rainfall_mm")),
                water_level_m=_as_float(row.get("water_level_m")),
                road_blocked=_as_bool(row.get("road_blocked")),
                citizen_reports=_as_int(row.get("citizen_reports")),
                notes=row.get("notes", ""),
            )
        )
    return observations


def build_rag_documents(data_dir: str | Path | None = None) -> list[RAGDocument]:
    builtin = _build_builtin_rag_documents()
    root = Path(data_dir) if data_dir else RAG_RAW_ROOT
    if not root.exists():
        return builtin

    raw_documents = load_rag_documents_from_directory(root)
    if not raw_documents:
        return builtin

    loaded_ids = {document.doc_id for document in raw_documents}
    fallback_documents = [document for document in builtin if document.doc_id not in loaded_ids]
    return raw_documents + fallback_documents


def load_rag_documents_from_directory(root: str | Path) -> list[RAGDocument]:
    root_path = Path(root)
    corpus_dirs = {
        CorpusType.POLICY: root_path / "policy",
        CorpusType.CASE: root_path / "case",
        CorpusType.PROFILE: root_path / "profile",
    }
    documents: list[RAGDocument] = []
    for corpus, corpus_dir in corpus_dirs.items():
        if not corpus_dir.exists():
            continue
        for path in sorted(corpus_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".pdf", ".docx"}:
                continue
            try:
                content = _extract_raw_document_text(path)
            except Exception as exc:
                warnings.warn(f"Skipped RAG source {path.name}: {exc}", stacklevel=2)
                continue
            if not content.strip():
                continue
            metadata = _load_document_metadata(path)
            title = str(metadata.pop("title", path.stem.replace("_", " ")))
            doc_id = str(metadata.pop("doc_id", _make_doc_id(corpus, path)))
            documents.append(
                RAGDocument(
                    doc_id=doc_id,
                    corpus=corpus,
                    title=title,
                    content=content,
                    metadata=metadata,
                )
            )
    return documents


def _build_builtin_rag_documents() -> list[RAGDocument]:
    return [
        RAGDocument(
            doc_id="policy_warning_public",
            corpus=CorpusType.POLICY,
            title="碑林区强降雨公众预警模板",
            content=(
                "在预警阶段，系统应明确受影响片区、下穿通道和低洼路段，"
                "向公众提供简明避险指引，并明确可用安置点信息。"
            ),
            metadata={
                "stage": "Warning",
                "risk_level": "Yellow",
                "audience": "public",
                "region": "西安市碑林区",
                "section_number": "4.2",
            },
        ),
        RAGDocument(
            doc_id="policy_response_approval",
            corpus=CorpusType.POLICY,
            title="碑林区疏散与应急响应审批规则",
            content=(
                "当预警等级达到橙色或红色时，人员转移、道路封控和应急资源调度"
                "应在执行前完成审批。"
            ),
            metadata={
                "stage": "Warning",
                "risk_level": "Orange",
                "audience": "government",
                "region": "西安市碑林区",
                "section_number": "5.1",
            },
        ),
        RAGDocument(
            doc_id="policy_compensation_review",
            corpus=CorpusType.POLICY,
            title="碑林区灾后补偿审核指引",
            content=(
                "进入恢复阶段后，应汇总住户受损申报、核验证明材料并形成补偿审核清单。"
                "对老年人和残障成员家庭、首层和地下空间进水住户以及关键公共资产损失，"
                "应优先进入人工复核。"
            ),
            metadata={
                "stage": "Compensation",
                "audience": "government",
                "region": "西安市碑林区",
                "section_number": "7.3",
            },
        ),
        RAGDocument(
            doc_id="case_road_block",
            corpus=CorpusType.CASE,
            title="碑林区下穿通道积水处置案例",
            content=(
                "在类似碑林区暴雨内涝事件中，提前封控南稍门下穿通道并引导车辆绕行，"
                "有助于降低救援压力并提升排水处置效率。"
            ),
            metadata={
                "stage": "Response",
                "scenario_type": "road_blocked",
                "outcome_label": "successful",
                "region": "西安市碑林区",
            },
        ),
        RAGDocument(
            doc_id="case_early_transfer",
            corpus=CorpusType.CASE,
            title="碑林区脆弱人群提前转移案例",
            content=(
                "对低洼居民楼中的学生、老年人和残障居民进行提前转移，"
                "可减少重复求助并释放救护资源。"
            ),
            metadata={
                "stage": "Warning",
                "scenario_type": "vulnerable_population",
                "outcome_label": "successful",
                "region": "西安市碑林区",
            },
        ),
        RAGDocument(
            doc_id="case_compensation_household",
            corpus=CorpusType.CASE,
            title="碑林区灾后住户补偿筛查案例",
            content=(
                "在类似城区内涝事件中，将住户受损、地铁出入口通行受阻和医疗点服务中断"
                "联合纳入筛查，可提升补偿审核效率，并及时识别需人工复核的争议事项。"
            ),
            metadata={
                "stage": "Compensation",
                "scenario_type": "household_damage",
                "outcome_label": "review_required",
                "region": "西安市碑林区",
            },
        ),
        RAGDocument(
            doc_id="profile_shelter_school",
            corpus=CorpusType.PROFILE,
            title="碑林体育中心安置点画像",
            content=(
                "碑林体育中心是该模拟区域的一级安置点，可承接南稍门与东大街片区的大规模转移人群。"
            ),
            metadata={"object_type": "shelter", "region": "西安市碑林区", "village": "南稍门"},
        ),
        RAGDocument(
            doc_id="profile_bridge_asset",
            corpus=CorpusType.PROFILE,
            title="南稍门地铁枢纽画像",
            content=(
                "南稍门地铁枢纽是重要交通节点，短时强降雨叠加客流高峰时，"
                "需优先关注下穿通道积水与出入口通行安全。"
            ),
            metadata={"object_type": "asset", "region": "西安市碑林区", "village": "南稍门"},
        ),
        RAGDocument(
            doc_id="profile_vulnerable",
            corpus=CorpusType.PROFILE,
            title="碑林区脆弱人群画像",
            content=(
                "该区域老年住户、学生群体和地下空间居住人群较为集中，"
                "在南稍门、文艺路和李家村低洼片区需优先预警和协助转移。"
            ),
            metadata={"object_type": "population", "region": "西安市碑林区", "village": "文艺路"},
        ),
        RAGDocument(
            doc_id="profile_monitoring_points",
            corpus=CorpusType.PROFILE,
            title="碑林区监测点与易涝点画像",
            content=(
                "南稍门下穿通道水尺、文艺路雨量站、东大街泵站传感器和城墙排水监控"
                "是核心监测点。南稍门下穿通道、和平路低洼路段和文艺路排水瓶颈"
                "是持续性易涝点。"
            ),
            metadata={"object_type": "monitoring", "region": "西安市碑林区", "village": "文艺路"},
        ),
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing bootstrap CSV file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split("|") if item.strip()]


def _as_int(raw: str | None) -> int:
    if raw in (None, ""):
        return 0
    return int(raw)


def _as_float(raw: str | None) -> float:
    if raw in (None, ""):
        return 0.0
    return float(raw)


def _as_bool(raw: str | None) -> bool:
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _as_datetime(raw: str) -> datetime:
    normalized = raw.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _extract_raw_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _extract_docx_text(path)
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    raise ValueError(f"Unsupported document type: {path.suffix}")


def _extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        with archive.open("word/document.xml") as handle:
            xml_bytes = handle.read()
    root = ElementTree.fromstring(xml_bytes)
    text_nodes = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            text_nodes.append(node.text)
        elif node.tag.endswith("}p"):
            text_nodes.append("\n")
    return _normalize_text(" ".join(text_nodes))


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return _normalize_text("\n".join(page.extract_text() or "" for page in reader.pages))
    except ImportError:
        pass

    pdftotext = _find_command("pdftotext")
    if pdftotext:
        result = subprocess.run(
            [pdftotext, str(path), "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return _normalize_text(result.stdout)

    raise RuntimeError("PDF extraction requires pypdf or pdftotext to be installed.")


def _load_document_metadata(path: Path) -> dict[str, object]:
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _make_doc_id(corpus: CorpusType, path: Path) -> str:
    relative_name = path.stem.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", relative_name).strip("_")
    return f"{corpus.value}_{normalized or 'document'}"


def _normalize_text(text: str) -> str:
    collapsed = re.sub(r"[ \t]+", " ", text)
    collapsed = re.sub(r"\n\s*\n+", "\n", collapsed)
    return collapsed.strip()


def _find_command(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["where.exe", name],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.splitlines()[0].strip()
    return first or None
