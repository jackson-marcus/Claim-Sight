"""API routes: /triage, /queue-stats, /health."""

from __future__ import annotations

import functools
import logging
import pickle

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from claimsight.models.triage import fraud_flags, predict_claim, prepare, route
from claimsight.settings import get_config, resolve_path

logger = logging.getLogger(__name__)
router = APIRouter()


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


@functools.lru_cache(maxsize=1)
def _bundle():
    path = resolve_path(get_config()["data"]["artifacts_dir"]) / "severity.pkl"
    if not path.exists():
        raise FileNotFoundError("No model; run make_claims.py + models.triage")
    with open(path, "rb") as f:
        return pickle.load(f)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/triage")
def triage(claim: Claim) -> dict:
    try:
        bundle = _bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return predict_claim(bundle, claim.model_dump())


@router.get("/queue-stats")
def queue_stats() -> dict:
    """Route the whole book of claims; report queue depths vs capacity."""
    try:
        bundle = _bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    df = pd.read_parquet(resolve_path(get_config()["data"]["processed_dir"]) / "claims.parquet")
    sample = df.sample(min(2000, len(df)), random_state=0)
    frame = prepare(sample)
    import numpy as np

    pred = np.expm1(bundle["model"].predict(frame[bundle["features"]]))
    routes_assigned = []
    for (_, row), severity in zip(sample.iterrows(), pred, strict=True):
        flags = fraud_flags(row.to_dict())
        routes_assigned.append(route(float(severity), len(flags)))
    counts = pd.Series(routes_assigned).value_counts().to_dict()
    capacity = {a["name"]: a["capacity"] for a in get_config()["routing"]["adjusters"]}
    return {
        "sampled_claims": len(sample),
        "queue_depth": counts,
        "weekly_capacity": capacity,
        "siu_share": round(counts.get("special_investigations", 0) / len(sample), 4),
    }
