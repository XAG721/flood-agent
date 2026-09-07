from __future__ import annotations

from .models import OperatorRole


EDIT_ROLES = {
    OperatorRole.DUTY_OFFICER,
    OperatorRole.REVIEWER,
    OperatorRole.COMMANDER,
    OperatorRole.ADMIN,
}
APPROVAL_ROLES = {OperatorRole.REVIEWER, OperatorRole.COMMANDER}
EXECUTION_ROLES = {
    OperatorRole.LIAISON,
    OperatorRole.FIELD_OPERATOR,
    OperatorRole.DUTY_OFFICER,
    OperatorRole.REVIEWER,
    OperatorRole.COMMANDER,
}
