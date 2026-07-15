from unittest.mock import Mock

from flood_system.v2.llm_gateway import (
    MockLLMGateway,
    RegionalAnalysisPackageOutput,
    ResponsesLLMGateway,
)


def test_mock_gateway_owns_deterministic_regional_analysis_fallback():
    result = MockLLMGateway().generate_regional_analysis_package(
        {
            "hazard_state": {"overall_risk_level": "Red"},
            "exposure_summary": {
                "top_risks": ["下穿通道积水风险上升"],
                "affected_entities": [{"entity": {"name": "北门下穿通道"}}],
            },
            "knowledge_evidence": [{"title": "区级防汛预案"}],
            "pending_proposals": [{"title": "封控下穿通道"}],
        }
    )

    assert "Red" in result.risk_assessment
    assert "北门下穿通道" in result.rescue_plan


def test_responses_gateway_routes_regional_analysis_through_prompt_profile():
    expected = RegionalAnalysisPackageOutput(
        analysis_message="analysis",
        risk_assessment="risk",
        rescue_plan="rescue",
        resource_dispatch_plan="resources",
    )
    gateway = ResponsesLLMGateway()
    gateway._generate_output = Mock(return_value=expected)

    result = gateway.generate_regional_analysis_package({"event_id": "EVENT-1"})

    assert result == expected
    gateway._generate_output.assert_called_once_with(
        prompt_profile="regional_analysis_package",
        payload={"event_id": "EVENT-1"},
        response_model=RegionalAnalysisPackageOutput,
    )
