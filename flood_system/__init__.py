from __future__ import annotations

from typing import Any

__all__ = ["app", "FloodWarningSystem", "create_default_system"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from .api import app

        return app
    if name in {"FloodWarningSystem", "create_default_system"}:
        from .system import FloodWarningSystem, create_default_system

        return {
            "FloodWarningSystem": FloodWarningSystem,
            "create_default_system": create_default_system,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
