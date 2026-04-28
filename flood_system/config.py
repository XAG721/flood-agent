from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "flood_warning_system_v2.db"


@dataclass(frozen=True)
class AppSettings:
    db_path: Path
    title: str = "Flood Warning System"
    version: str = "3.0.0"


def load_settings() -> AppSettings:
    db_path = Path(os.getenv("FLOOD_DB_PATH", str(DEFAULT_DB_PATH))).expanduser().resolve()
    return AppSettings(db_path=db_path)
