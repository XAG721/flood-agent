from __future__ import annotations

import json

from flood_system.security import OutboundPrivacyFilter
from flood_system.system import FloodWarningSystem


def test_outbound_privacy_filter_redacts_sensitive_fields_and_inline_identifiers():
    payload = {
        "event_id": "FLOOD-001",
        "emergency_contacts": [{"name": "张某", "phone": "13800000001"}],
        "resident": {
            "resident_name": "李某",
            "message": "身份证 610101199001011234，邮箱 resident@example.com，电话 13900000002",
        },
        "public_facility": "长安北路下穿通道",
    }

    report = OutboundPrivacyFilter.sanitize(payload)
    serialized = json.dumps(report.sanitized_payload, ensure_ascii=False)

    assert report.redaction_count >= 4
    assert report.source_digest != report.sanitized_digest
    assert "张某" not in serialized
    assert "李某" not in serialized
    assert "13800000001" not in serialized
    assert "610101199001011234" not in serialized
    assert "resident@example.com" not in serialized
    assert "长安北路下穿通道" in serialized


def test_external_llm_gateway_only_posts_sanitized_payload_and_persists_audit(tmp_path):
    system = FloodWarningSystem(tmp_path / "privacy.db")
    gateway = system.production_platform.llm_gateway
    captured: dict = {}
    gateway._load_api_key = lambda: "test-key"

    def fake_post(_api_key: str, body: dict):
        captured.update(body)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "已生成不含个人信息的建议。",
                                "impact_summary": [],
                                "recommended_actions": [],
                                "confidence": 0.8,
                                "confidence_explanation": "证据充分",
                                "missing_data": [],
                                "grounding_summary": "已脱敏",
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

    gateway._post = fake_post
    gateway.generate_object_advisory(
        {
            "object_id": "resident-001",
            "resident_name": "王某",
            "phone": "13700000003",
            "special_population_notes": "独居老人，需要搀扶转移",
            "operational_context": "橙色暴雨预警",
        }
    )

    outbound = json.dumps(captured, ensure_ascii=False)
    assert "王某" not in outbound
    assert "13700000003" not in outbound
    assert "独居老人" not in outbound
    assert "[REDACTED:resident_name]" in outbound
    assert "personal_data_redacted" in outbound
    assert "true" in outbound
    audits = system.repository.list_audit_records(source_type="model_egress_privacy")
    assert len(audits) == 1
    assert audits[0].details["redaction_count"] == 3
    assert "source_digest" in audits[0].details
    assert "sanitized_digest" in audits[0].details
