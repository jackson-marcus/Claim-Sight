"""API routes: /triage, /queue-stats, /models, /health."""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from claimsight.model.predictor import ReservingPredictor
from claimsight.model.registry import ModelRegistry
from claimsight.settings import get_config, resolve_path

logger = logging.getLogger(__name__)
router = APIRouter()

_CACHED: dict[str, object] = {}


class Claim(BaseModel):
    claim_type: str = Field(
        pattern="^(auto_collision|auto_theft|property_water|property_fire|liability)$"
    )
    vehicle_age: int = Field(ge=0, le=40, default=5)
    injuries: int = Field(ge=0, le=20, default=0)
    police_report: int = Field(ge=0, le=1, default=1)
    report_delay_days: int = Field(ge=0, le=365)
    policy_tenure_days: int = Field(ge=0, le=20000)
    prior_claims_3y: int = Field(ge=0, le=30, default=0)
    claimed_amount: float = Field(gt=0)
    reserve_confidence: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Share of claims the booked reserve should cover. Must be a "
        "calibrated level; /models lists them.",
    )


def _registry() -> ModelRegistry:
    return ModelRegistry(resolve_path(get_config()["data"]["artifacts_dir"]) / "registry")


def _predictor() -> ReservingPredictor:
    """Serve the Production version, reloading when the registry index changes.

    Keyed on the index mtime so a promotion (or a rollback) takes effect
    without bouncing the process.
    """
    registry = _registry()
    stamp = (str(registry.root), registry.index_stamp())
    if _CACHED.get("stamp") != stamp:
        try:
            predictor = ReservingPredictor.from_registry(registry)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No model in Production; run scripts/make_claims.py then "
                    "python -m claimsight.models.triage"
                ),
            ) from exc
        _CACHED["stamp"], _CACHED["predictor"] = stamp, predictor
        logger.info("serving %s v%d", predictor.version.name, predictor.version.version)
    return _CACHED["predictor"]  # type: ignore[return-value]


def _reset_cache() -> None:
    _CACHED.clear()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/models")
def models() -> dict:
    """Registered versions, their stage, and the adequacy metrics behind them."""
    registry = _registry()
    versions = registry.versions()
    if not versions:
        raise HTTPException(status_code=503, detail="Registry is empty; train a model first")
    production = next((v for v in versions if v.stage == "Production"), None)
    return {
        "versions": [v.as_dict() for v in versions],
        "production_version": production.version if production else None,
        "reserve_levels": list(_predictor().confidence_levels) if production else [],
    }


@router.post("/triage")
def triage(claim: Claim) -> dict:
    predictor = _predictor()
    payload = claim.model_dump()
    confidence = payload.pop("reserve_confidence")
    try:
        return predictor.triage(payload, confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/queue-stats")
def queue_stats(reserve_confidence: float | None = None) -> dict:
    """Route and reserve a sampled book; queue depth vs capacity, capital booked."""
    predictor = _predictor()
    df = pd.read_parquet(resolve_path(get_config()["data"]["processed_dir"]) / "claims.parquet")
    sample = df.sample(min(2000, len(df)), random_state=0)
    try:
        scored = predictor.triage_book(sample, reserve_confidence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    counts = scored["route_to"].value_counts().to_dict()
    capacity = {a["name"]: a["capacity"] for a in get_config()["routing"]["adjusters"]}
    level = (
        predictor.calibration.primary_level if reserve_confidence is None else reserve_confidence
    )
    return {
        "sampled_claims": len(sample),
        "queue_depth": counts,
        "weekly_capacity": capacity,
        "siu_share": round(counts.get("special_investigations", 0) / len(sample), 4),
        "reserve_confidence": round(float(level), 4),
        "measured_coverage": round(predictor.calibration.coverage_for(reserve_confidence), 4),
        "reserves_booked_usd": round(float(scored["suggested_reserve_usd"].sum()), 2),
        "claimed_total_usd": round(float(sample["claimed_amount"].sum()), 2),
        "model_version": predictor.version.version,
    }
