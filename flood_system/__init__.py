from __future__ import annotations

from typing import Any

__all__ = ["app", "create_app", "FloodWarningSystem", "create_default_system"]


def __getattr__(name: str) -> Any:
    if name in {"app", "create_app"}:
        from .api import app, create_app

        return {"app": app, "create_app": create_app}[name]
    if name in {"FloodWarningSystem", "create_default_system"}:
        from .system import FloodWarningSystem, create_default_system

        return {
            "FloodWarningSystem": FloodWarningSystem,
            "create_default_system": create_default_system,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
