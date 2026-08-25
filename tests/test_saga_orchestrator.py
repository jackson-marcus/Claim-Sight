"""Unit tests for Saga Orchestration and Compensating Actions in ClaimSight."""

from __future__ import annotations

import pytest

from claimsight.saga import (
    ClaimsSagaOrchestrator,
    SagaStatus,
)


@pytest.fixture
def valid_auto_claim() -> dict:
    return {
        "vehicle_age": 4,
        "injuries": 0,
        "police_report": 1,
        "report_delay_days": 2,
        "policy_tenure_days": 450,
        "prior_claims_3y": 0,
        "claimed_amount": 4200.0,
        "claim_type": "auto_collision",
    }


@pytest.fixture
def lapsed_policy_claim(valid_auto_claim) -> dict:
    return {**valid_auto_claim, "policy_tenure_days": 0}


@pytest.fixture
def over_limit_claim(valid_auto_claim) -> dict:
    return {**valid_auto_claim, "claimed_amount": 250000.0}


@pytest.fixture
def fraudulent_claim(valid_auto_claim) -> dict:
    return {
        **valid_auto_claim,
        "report_delay_days": 45,  # Late report flag
        "claimed_amount": 10000.0,  # Round 10k flag
        "policy_tenure_days": 10,  # Inception flag
    }


def test_saga_successful_end_to_end(valid_auto_claim):
    orchestrator = ClaimsSagaOrchestrator()
    ctx, trace = orchestrator.execute_saga("CLM-1001", valid_auto_claim)

    assert trace.final_status == SagaStatus.COMPLETED
    assert ctx.policy_active is True
    assert ctx.policy_hold_id is not None
    assert ctx.reserve_amount_usd == pytest.approx(4200.0 * 1.15)
    assert ctx.reserve_hold_id is not None
    assert ctx.assigned_adjuster_id is not None
    assert ctx.settlement_token is not None
    assert len(trace.compensated_steps) == 0
    assert "status" in trace.as_dict()


def test_saga_failure_at_step_1_policy_coverage(lapsed_policy_claim):
    orchestrator = ClaimsSagaOrchestrator()
    ctx, trace = orchestrator.execute_saga("CLM-1002", lapsed_policy_claim)

    assert trace.final_status == SagaStatus.FAILED_AND_COMPENSATED
    assert ctx.policy_active is False
    assert "Policy lapsed" in (trace.error_message or "")
    assert len(trace.executed_steps) == 0
    assert len(trace.compensated_steps) == 0


def test_saga_compensation_on_financial_limit_breach(over_limit_claim):
    orchestrator = ClaimsSagaOrchestrator()
    ctx, trace = orchestrator.execute_saga("CLM-1003", over_limit_claim)

    assert trace.final_status == SagaStatus.FAILED_AND_COMPENSATED
    assert "exceeds policy authority limit" in (trace.error_message or "")
    # Step 1 succeeded before Step 2 failed -> Step 1 must have been compensated
    assert "validate_policy_coverage" in trace.compensated_steps
    assert ctx.policy_active is False
    assert ctx.policy_hold_id is None
    assert ctx.reserve_hold_id is None


def test_saga_compensation_on_fraud_hard_block(fraudulent_claim):
    orchestrator = ClaimsSagaOrchestrator()
    ctx, trace = orchestrator.execute_saga("CLM-1004", fraudulent_claim)

    assert trace.final_status == SagaStatus.FAILED_AND_COMPENSATED
    assert "fraud indicators detected" in (trace.error_message or "")
    # Steps 1 and 2 were executed, then Step 3 failed -> Steps 2 and 1 compensated in reverse
    assert trace.executed_steps == ("validate_policy_coverage", "financial_reserve_allocation")
    assert trace.compensated_steps == ("financial_reserve_allocation", "validate_policy_coverage")

    # Verify both financial reserve and policy hold were cleanly rolled back
    assert ctx.reserve_hold_id is None
    assert ctx.reserve_amount_usd == 0.0
    assert ctx.policy_active is False
    assert ctx.policy_hold_id is None
