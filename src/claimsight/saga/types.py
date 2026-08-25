"""Saga Orchestration - Domain Types and Context.

Defines execution context, step statuses, and compensation audit traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SagaStatus(StrEnum):
    PENDING = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    COMPLETED = "COMPLETED"
    FAILED_AND_COMPENSATED = "FAILED_AND_COMPENSATED"
    CRITICAL_ERROR = "CRITICAL_ERROR"


@dataclass
class SagaContext:
    """Shared state passed between saga steps and compensations."""

    claim_id: str
    claim_payload: dict[str, Any]
    policy_active: bool = False
    policy_hold_id: str | None = None
    reserve_amount_usd: float = 0.0
    reserve_hold_id: str | None = None
    fraud_flags: list[dict[str, Any]] = field(default_factory=list)
    siu_referral_id: str | None = None
    assigned_adjuster_id: str | None = None
    settlement_token: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class SagaStepRecord:
    """Execution outcome of a single forward or backward step."""

    step_name: str
    action: str  # "EXECUTE" | "COMPENSATE"
    success: bool
    details: str


@dataclass(frozen=True)
class SagaExecutionTrace:
    """Audit log of complete saga execution."""

    claim_id: str
    final_status: SagaStatus
    executed_steps: tuple[str, ...]
    compensated_steps: tuple[str, ...]
    history: tuple[SagaStepRecord, ...]
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "status": str(self.final_status),
            "executed_steps": list(self.executed_steps),
            "compensated_steps": list(self.compensated_steps),
            "error": self.error_message,
        }
