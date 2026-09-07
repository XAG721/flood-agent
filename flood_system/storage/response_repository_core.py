from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def _load(model_type: type[T], payload: str) -> T:
    return model_type.model_validate(json.loads(payload))


class ResponseRepositoryCoreMixin:
    def _secure_dump(self, model: BaseModel) -> str:
        return self.data_protector.encrypt_payload(model.model_dump_json())

    def _secure_load(self, model_type: type[T], payload: str) -> T:
        return _load(model_type, self.data_protector.decrypt_payload(payload))
