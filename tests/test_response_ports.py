from __future__ import annotations

from typing import get_type_hints

from flood_system.response_workflow.ports import ResponseWorkflowRepositoryPort
from flood_system.response_workflow.service_core import ResponseWorkflowCoreMixin


def test_response_workflow_constructor_exposes_repository_port_contract():
    hints = get_type_hints(ResponseWorkflowCoreMixin.__init__)

    assert hints["repository"] is ResponseWorkflowRepositoryPort
