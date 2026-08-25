"""Saga Orchestration - Concrete Saga Steps and Compensating Actions.

Implements discrete forward steps and compensating rollback actions for
claims intake, reserve holds, SIU audit, and adjuster routing.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from claimsight.models.triage import fraud_flags
from claimsight.saga.types import SagaContext


class SagaStep(ABC):
    """Abstract base for a single transactional step in a Saga."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the saga step."""

    @abstractmethod
    def execute(self, ctx: SagaContext) -> bool:
        """Executes forward step. Returns True on success, False on failure."""

    @abstractmethod
    def compensate(self, ctx: SagaContext) -> None:
        """Executes compensating rollback action if a downstream step fails."""


class ValidatePolicyCoverageStep(SagaStep):
    """Verifies that the policy is active and establishes a coverage hold."""

    name = "validate_policy_coverage"

    def execute(self, ctx: SagaContext) -> bool:
        policy_tenure = ctx.claim_payload.get("policy_tenure_days", 0)
        # Policy is invalid if tenure is 0 or negative
        if policy_tenure <= 0:
            ctx.failure_reason = "Policy lapsed or inactive"
            return False

        ctx.policy_active = True
        ctx.policy_hold_id = f"hold_pol_{uuid.uuid4().hex[:8]}"
        return True

    def compensate(self, ctx: SagaContext) -> None:
        if ctx.policy_hold_id:
            ctx.policy_hold_id = None
            ctx.policy_active = False


class FinancialReserveAllocationStep(SagaStep):
    """Calculates and locks financial reserve from company balance sheet."""

    name = "financial_reserve_allocation"

    def __init__(self, max_single_claim_limit: float = 100000.0) -> None:
        self.max_limit = max_single_claim_limit

    def execute(self, ctx: SagaContext) -> bool:
        claimed = float(ctx.claim_payload.get("claimed_amount", 0.0))
        if claimed > self.max_limit:
            ctx.failure_reason = f"Claimed amount (${claimed:,.2f}) exceeds policy authority limit (${self.max_limit:,.2f})"
            return False

        # Allocate 115% reserve buffer
        ctx.reserve_amount_usd = round(claimed * 1.15, 2)
        ctx.reserve_hold_id = f"res_hold_{uuid.uuid4().hex[:8]}"
        return True

    def compensate(self, ctx: SagaContext) -> None:
        if ctx.reserve_hold_id:
            ctx.reserve_hold_id = None
            ctx.reserve_amount_usd = 0.0


class FraudRiskScreeningStep(SagaStep):
    """Screens claim against rule heuristics and triggers SIU if high-risk."""

    name = "fraud_risk_screening"

    def __init__(self, max_allowed_flags: int = 2) -> None:
        self.max_allowed_flags = max_allowed_flags

    def execute(self, ctx: SagaContext) -> bool:
        flags = fraud_flags(ctx.claim_payload)
        ctx.fraud_flags = flags

        if len(flags) > self.max_allowed_flags:
            ctx.siu_referral_id = f"siu_case_{uuid.uuid4().hex[:8]}"
            ctx.failure_reason = f"Hard block: {len(flags)} critical fraud indicators detected"
            return False

        return True

    def compensate(self, ctx: SagaContext) -> None:
        if ctx.siu_referral_id:
            ctx.siu_referral_id = None


class AdjusterRoutingStep(SagaStep):
    """Assigns an adjuster to the claim."""

    name = "adjuster_routing"

    def execute(self, ctx: SagaContext) -> bool:
        claim_type = ctx.claim_payload.get("claim_type", "auto_collision")
        # Route based on type
        if "liability" in claim_type:
            ctx.assigned_adjuster_id = "adj_senior_liability_09"
        else:
            ctx.assigned_adjuster_id = "adj_standard_auto_14"
        return True

    def compensate(self, ctx: SagaContext) -> None:
        if ctx.assigned_adjuster_id:
            ctx.assigned_adjuster_id = None


class SettlementAuthorizationStep(SagaStep):
    """Mints settlement authorization token."""

    name = "settlement_authorization"

    def execute(self, ctx: SagaContext) -> bool:
        # Final authorization step
        ctx.settlement_token = f"settle_tok_{uuid.uuid4().hex[:12]}"
        return True

    def compensate(self, ctx: SagaContext) -> None:
        if ctx.settlement_token:
            ctx.settlement_token = None
