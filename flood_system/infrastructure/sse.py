from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from typing import Any


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def encode_sse(value: Any) -> str:
    return f"data: {json.dumps(to_jsonable(value), ensure_ascii=False)}\n\n"


async def repeated_snapshot_stream(
    snapshot_factory: Callable[[], Any],
    *,
    version_getter: Callable[[Any], str],
    interval_seconds: float = 1.0,
):
    last_version: str | None = None
    while True:
        snapshot = snapshot_factory()
        version = version_getter(snapshot)
        if version != last_version:
            last_version = version
            yield encode_sse(snapshot)
        await asyncio.sleep(interval_seconds)


async def typed_event_stream(
    event_factory: Callable[[], Iterable[Any]],
    *,
    event_type_getter: Callable[[Any], str],
    version_getter: Callable[[Any], str],
    interval_seconds: float = 1.0,
):
    last_versions: dict[str, str] = {}
    while True:
        for event in event_factory():
            event_type = event_type_getter(event)
            version = version_getter(event)
            if last_versions.get(event_type) == version:
                continue
            last_versions[event_type] = version
            yield encode_sse(event)
        await asyncio.sleep(interval_seconds)
