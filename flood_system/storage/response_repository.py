from __future__ import annotations

from .response_repository_core import ResponseRepositoryCoreMixin
from .response_repository_dispatch import ResponseDispatchRepositoryMixin
from .response_repository_documents import ResponseDocumentRepositoryMixin
from .response_repository_events import ResponseEventRepositoryMixin
from .response_repository_metrics import ResponseMetricsRepositoryMixin
from .response_repository_resilience import ResponseResilienceRepositoryMixin
from .response_repository_tasks import ResponseTaskRepositoryMixin


class ResponseRepositoryMixin(
    ResponseRepositoryCoreMixin,
    ResponseEventRepositoryMixin,
    ResponseMetricsRepositoryMixin,
    ResponseTaskRepositoryMixin,
    ResponseDispatchRepositoryMixin,
    ResponseDocumentRepositoryMixin,
    ResponseResilienceRepositoryMixin,
):
    """Compatibility facade composed from response-domain repository mixins."""
