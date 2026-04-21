from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AppModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class Stage(StrEnum):
    MONITORING = "Monitoring"
    ATTENTION = "Attention"
    WARNING = "Warning"
    RESPONSE = "Response"
    RESCUE_COORDINATION = "Rescue Coordination"
    RECOVERY = "Recovery"
    COMPENSATION = "Compensation"
    POSTMORTEM = "Postmortem"


class RiskLevel(StrEnum):
    NONE = "None"
    BLUE = "Blue"
    YELLOW = "Yellow"
    ORANGE = "Orange"
    RED = "Red"


class CorpusType(StrEnum):
    POLICY = "policy"
    CASE = "case"
    PROFILE = "profile"
    MEMORY = "memory"


class Observation(AppModel):
    observed_at: datetime
    rainfall_mm: float = 0.0
    water_level_m: float = 0.0
    road_blocked: bool = False
    citizen_reports: int = 0
    notes: str = ""


class Shelter(AppModel):
    shelter_id: str
    name: str
    village: str
    capacity: int
    available_capacity: int
    accessible: bool = True


class RoadStatus(AppModel):
    road_id: str
    name: str
    from_village: str
    to_location: str
    accessible: bool = True
    risk_note: str = ""


class ResourceStatus(AppModel):
    area_id: str
    vehicle_count: int
    staff_count: int
    supply_kits: int
    rescue_boats: int = 0
    ambulance_count: int = 0
    drone_count: int = 0
    portable_pumps: int = 0
    power_generators: int = 0
    medical_staff_count: int = 0
    volunteer_count: int = 0
    satellite_phones: int = 0
    notes: str = ""


class AreaProfile(AppModel):
    area_id: str
    region: str
    villages: list[str]
    population: int
    household_count: int = 0
    vulnerable_population: int
    elderly_population: int = 0
    children_population: int = 0
    disabled_population: int = 0
    historical_risk_level: str = "unknown"
    key_assets: list[str]
    medical_facilities: list[str] = Field(default_factory=list)
    schools: list[str] = Field(default_factory=list)
    monitoring_points: list[str] = Field(default_factory=list)
    flood_prone_spots: list[str] = Field(default_factory=list)
    shelters: list[Shelter]
    roads: list[RoadStatus]


class RAGDocument(AppModel):
    doc_id: str
    corpus: CorpusType
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRef(AppModel):
    doc_id: str
    title: str
    excerpt: str
    retrieval_explain: dict[str, Any] = Field(default_factory=dict)
