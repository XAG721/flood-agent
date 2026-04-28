from __future__ import annotations

from ..v2.models import ActionProposal, Advisory, ProposalStatus


class ProposalRepositoryMixin:
    def save_v2_advisory(self, advisory: Advisory) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_advisories (advisory_id, event_id, generated_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(advisory_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    generated_at = excluded.generated_at,
                    payload = excluded.payload
                """,
                (advisory.advisory_id, advisory.event_id, advisory.generated_at.isoformat(), advisory.model_dump_json()),
            )

    def save_v2_action_proposal(self, proposal: ActionProposal) -> None:
        resolved_at = proposal.resolved_at.isoformat() if proposal.resolved_at else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_action_proposals (
                    proposal_id, event_id, entity_id, created_at, resolved_at, status, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET
                    event_id = excluded.event_id,
                    entity_id = excluded.entity_id,
                    created_at = excluded.created_at,
                    resolved_at = excluded.resolved_at,
                    status = excluded.status,
                    payload = excluded.payload
                """,
                (
                    proposal.proposal_id,
                    proposal.event_id,
                    proposal.entity_id,
                    proposal.created_at.isoformat(),
                    resolved_at,
                    proposal.status.value,
                    proposal.model_dump_json(),
                ),
            )

    def get_v2_action_proposal(self, proposal_id: str) -> ActionProposal | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM v2_action_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        return ActionProposal.model_validate_json(row["payload"]) if row else None

    def list_v2_action_proposals(
        self,
        event_id: str | None = None,
        *,
        proposal_scope: str | None = None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[ActionProposal]:
        query = "SELECT payload FROM v2_action_proposals WHERE 1 = 1"
        parameters: list[object] = []
        if event_id is not None:
            query += " AND event_id = ?"
            parameters.append(event_id)
        query += " ORDER BY created_at DESC, proposal_id DESC"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, parameters).fetchall()
        proposals = [ActionProposal.model_validate_json(row["payload"]) for row in rows]
        if proposal_scope is not None:
            proposals = [item for item in proposals if item.proposal_scope == proposal_scope]
        if statuses is not None:
            allowed = set(statuses)
            proposals = [item for item in proposals if item.status.value in allowed]
        return proposals

    def list_v2_pending_regional_proposals(self) -> list[ActionProposal]:
        return self.list_v2_action_proposals(
            proposal_scope="regional",
            statuses=[ProposalStatus.PENDING.value],
        )
