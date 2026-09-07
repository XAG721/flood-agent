from __future__ import annotations

from .document_governance import PARSER_VERSION
from .service_core import ResponseWorkflowCoreMixin
from .service_dispatch import DispatchWorkflowMixin
from .service_documents import DocumentWorkflowMixin
from .service_evidence import EvidenceWorkflowMixin
from .service_operations import WorkflowOperationsMixin
from .service_risk_objects import RiskObjectWorkflowMixin
from .service_tasks import TaskWorkflowMixin


class ResponseWorkflowService(
    ResponseWorkflowCoreMixin,
    RiskObjectWorkflowMixin,
    EvidenceWorkflowMixin,
    TaskWorkflowMixin,
    DocumentWorkflowMixin,
    DispatchWorkflowMixin,
    WorkflowOperationsMixin,
):
    """Deterministic district flood-response workflow; no model call is required."""

    GENERATION_VERSION = "response-rag-task-v1"

    WORKFLOW_ENGINE_VERSION = "response-workflow-v2"

    RULE_SET_VERSION = "response-rules-v2"

    TASK_SCHEMA_VERSION = "response-task-schema-v2"

    DOCUMENT_PARSER_VERSION = PARSER_VERSION
