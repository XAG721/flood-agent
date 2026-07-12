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
    require_https: bool = False
    trust_proxy_headers: bool = False


def load_settings() -> AppSettings:
    db_path = Path(os.getenv("FLOOD_DB_PATH", str(DEFAULT_DB_PATH))).expanduser().resolve()
    return AppSettings(
        db_path=db_path,
        require_https=_env_flag("FLOOD_REQUIRE_HTTPS", False),
        trust_proxy_headers=_env_flag("FLOOD_TRUST_PROXY_HEADERS", False),
    )


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
