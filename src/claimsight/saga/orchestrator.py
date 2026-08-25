"""Saga Orchestrator for Insurance Claims.

Drives multi-step forward execution and manages automatic compensating rollbacks
in reverse chronological order upon any step failure.
"""

from __future__ import annotations

from typing import Any

from claimsight.saga.steps import (
    AdjusterRoutingStep,
    FinancialReserveAllocationStep,
    FraudRiskScreeningStep,
    SagaStep,
    SettlementAuthorizationStep,
    ValidatePolicyCoverageStep,
)
from claimsight.saga.types import (
    SagaContext,
    SagaExecutionTrace,
    SagaStatus,
    SagaStepRecord,
)


class ClaimsSagaOrchestrator:
    """Coordinates forward execution and backward compensation for a claim."""

    def __init__(self, steps: list[SagaStep] | None = None) -> None:
        self.steps: list[SagaStep] = steps or self.default_claim_pipeline()

    @classmethod
    def default_claim_pipeline(cls) -> list[SagaStep]:
        """Standard 5-stage claims adjudication saga."""
        return [
            ValidatePolicyCoverageStep(),
            FinancialReserveAllocationStep(),
            FraudRiskScreeningStep(),
            AdjusterRoutingStep(),
            SettlementAuthorizationStep(),
        ]

    def execute_saga(
        self, claim_id: str, claim_payload: dict[str, Any]
    ) -> tuple[SagaContext, SagaExecutionTrace]:
        """Executes the forward saga; if any step fails, compensates in reverse."""
        ctx = SagaContext(claim_id=claim_id, claim_payload=claim_payload)
        executed_steps: list[SagaStep] = []
        history: list[SagaStepRecord] = []
        failure_encountered = False

        # Forward execution phase
        for step in self.steps:
            try:
                success = step.execute(ctx)
                if success:
                    executed_steps.append(step)
                    history.append(
                        SagaStepRecord(
                            step_name=step.name,
                            action="EXECUTE",
                            success=True,
                            details="Step completed successfully",
                        )
                    )
                else:
                    failure_encountered = True
                    history.append(
                        SagaStepRecord(
                            step_name=step.name,
                            action="EXECUTE",
                            success=False,
                            details=ctx.failure_reason or "Step returned failure",
                        )
                    )
                    break
            except Exception as e:
                failure_encountered = True
                ctx.failure_reason = f"Exception in {step.name}: {e}"
                history.append(
                    SagaStepRecord(
                        step_name=step.name,
                        action="EXECUTE",
                        success=False,
                        details=str(e),
                    )
                )
                break

        # Compensation phase if failure occurred
        compensated_names: list[str] = []
        if failure_encountered:
            for step in reversed(executed_steps):
                try:
                    step.compensate(ctx)
                    compensated_names.append(step.name)
                    history.append(
                        SagaStepRecord(
                            step_name=step.name,
                            action="COMPENSATE",
                            success=True,
                            details="Compensating rollback applied",
                        )
                    )
                except Exception as e:
                    history.append(
                        SagaStepRecord(
                            step_name=step.name,
                            action="COMPENSATE",
                            success=False,
                            details=f"Compensation failed: {e}",
                        )
                    )

            trace = SagaExecutionTrace(
                claim_id=claim_id,
                final_status=SagaStatus.FAILED_AND_COMPENSATED,
                executed_steps=tuple(s.name for s in executed_steps),
                compensated_steps=tuple(compensated_names),
                history=tuple(history),
                error_message=ctx.failure_reason,
            )
            return ctx, trace

        # Success outcome
        trace = SagaExecutionTrace(
            claim_id=claim_id,
            final_status=SagaStatus.COMPLETED,
            executed_steps=tuple(s.name for s in executed_steps),
            compensated_steps=(),
            history=tuple(history),
            error_message=None,
        )
        return ctx, trace
