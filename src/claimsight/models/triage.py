"""Claims triage: severity model, explainable fraud flags, routing.

Severity: LGBM on log-severity. The reserve on top of that point estimate is
calibrated on its own slice and validated on a third, disjoint one — see
``claimsight.features.extractor``. Fraud flags are auditable rules (not a black
box); each flag names its evidence. Routing combines predicted severity, flags,
and adjuster capacity.

Training registers a new model version and asks the registry to promote it;
the reserve-adequacy gate can refuse, in which case Production keeps serving
whatever it was already serving.

Usage (train):
    python -m claimsight.models.triage
"""

from __future__ import annotations

import logging

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from claimsight.features.extractor import calibrate_reserve, extract_log_residuals
from claimsight.model.registry import (
    ModelRegistry,
    PromotionRefusedError,
    ReserveAdequacyGate,
)
from claimsight.settings import get_config, get_settings, resolve_path

logger = logging.getLogger(__name__)

FEATURES = [
    "vehicle_age",
    "injuries",
    "police_report",
    "report_delay_days",
    "policy_tenure_days",
    "prior_claims_3y",
    "claimed_amount",
]
TYPE_DUMMIES = [
    f"type_{t}"
    for t in ("auto_collision", "auto_theft", "property_water", "property_fire", "liability")
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for t in TYPE_DUMMIES:
        out[t] = (out["claim_type"] == t.removeprefix("type_")).astype(int)
    return out


def fraud_flags(claim: dict) -> list[dict]:
    cfg = get_config()["fraud_flags"]
    flags = []
    if claim["report_delay_days"] >= cfg["late_report_days"]:
        flags.append(
            {
                "flag": "late_report",
                "evidence": f"reported {claim['report_delay_days']} days after loss",
            }
        )
    if (
        abs(claim["claimed_amount"] - round(claim["claimed_amount"], -3))
        <= cfg["round_amount_tolerance"]
        and claim["claimed_amount"] >= 1000
    ):
        flags.append(
            {"flag": "round_amount", "evidence": f"claimed exactly {claim['claimed_amount']:.0f}"}
        )
    if claim["policy_tenure_days"] <= cfg["claim_soon_after_policy_days"]:
        flags.append(
            {
                "flag": "new_policy",
                "evidence": f"policy only {claim['policy_tenure_days']} days old",
            }
        )
    if claim["prior_claims_3y"] >= 3:
        flags.append(
            {"flag": "frequent_claimant", "evidence": f"{claim['prior_claims_3y']} claims in 3y"}
        )
    if not claim.get("police_report", 1) and claim["claim_type"] in ("auto_theft", "property_fire"):
        flags.append(
            {"flag": "no_police_report", "evidence": f"{claim['claim_type']} without police report"}
        )
    return flags


def route(severity_usd: float, n_flags: int) -> str:
    cfg = get_config()["routing"]
    if n_flags >= cfg["fraud_unit_threshold"]:
        return "special_investigations"
    for adjuster in cfg["adjusters"]:
        if severity_usd <= adjuster["max_severity_usd"]:
            return adjuster["name"]
    return cfg["adjusters"][-1]["name"]


def _three_way_split(
    frame: pd.DataFrame, cfg: dict, seed: int = 42
) -> tuple[pd.Index, pd.Index, pd.Index]:
    """train / calibration / test — the uplift must never see the test slice."""
    test_frac = float(cfg["severity"]["test_frac"])
    calibration_frac = float(cfg["severity"]["calibration_frac"])
    held_out = test_frac + calibration_frac
    if not 0 < held_out < 1:
        raise ValueError("test_frac + calibration_frac must be strictly between 0 and 1")
    train_idx, rest = train_test_split(frame.index, test_size=held_out, random_state=seed)
    calibration_idx, test_idx = train_test_split(
        rest, test_size=test_frac / held_out, random_state=seed
    )
    return train_idx, calibration_idx, test_idx


def train() -> dict:
    """Fit the severity model, calibrate the reserve, register, try to promote."""
    cfg = get_config()
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["eval"]["experiment_name"])

    raw = pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "claims.parquet")
    frame = prepare(raw)
    cols = FEATURES + TYPE_DUMMIES
    y_log = np.log1p(raw["final_severity"])
    train_idx, calibration_idx, test_idx = _three_way_split(frame, cfg)

    model = LGBMRegressor(**cfg["severity"]["lgbm"], random_state=42)
    model.fit(frame.loc[train_idx, cols], y_log.loc[train_idx])

    _, calibration_residuals = extract_log_residuals(
        model, frame.loc[calibration_idx], y_log.loc[calibration_idx].to_numpy(), cols
    )
    test_predicted, test_residuals = extract_log_residuals(
        model, frame.loc[test_idx], y_log.loc[test_idx].to_numpy(), cols
    )
    calibration = calibrate_reserve(
        calibration_residuals=calibration_residuals,
        test_predicted_log=test_predicted,
        test_residuals=test_residuals,
        test_incurred_usd=raw.loc[test_idx, "final_severity"].to_numpy(),
        test_segments=raw.loc[test_idx, "claim_type"].tolist(),
        levels=cfg["severity"]["reserve_levels"],
        primary_level=cfg["severity"]["reserve_primary_level"],
    )

    actual = np.expm1(y_log.loc[test_idx].to_numpy())
    predicted = np.expm1(test_predicted)
    mape = float(np.median(np.abs(actual - predicted) / np.maximum(actual, 1)))
    primary = calibration.primary_level
    metrics = {
        "median_ape": mape,
        "reserve_uplift_log": calibration.uplift_for(primary),
        "reserve_coverage": calibration.coverage_for(primary),
        "reserve_to_incurred": calibration.capital_ratio_for(primary),
    }

    registry = ModelRegistry(resolve_path(cfg["data"]["artifacts_dir"]) / "registry")
    version = registry.register(
        {"model": model, "features": cols, "calibration": calibration}, metrics
    )
    gate = ReserveAdequacyGate(
        nominal=primary, tolerance=float(cfg["severity"]["adequacy_tolerance"])
    )
    try:
        registry.promote(version.version, gate=gate)
        promoted, refusal = True, None
    except PromotionRefusedError as exc:
        promoted, refusal = False, str(exc)
        logger.warning("promotion refused, Production unchanged: %s", refusal)

    with mlflow.start_run(run_name="severity-lgbm"):
        mlflow.log_params(
            {"n_claims": len(raw), "model_version": version.version, "nominal_level": primary}
        )
        mlflow.log_metrics({**metrics, "promoted": float(promoted)})
    worst_level, worst_dev = calibration.worst_deviation()
    logger.info(
        "v%d median APE %.1f%% | reserve coverage %.3f at nominal %.2f (worst level %.2f, %+.3f)"
        " | reserves are %.2fx incurred | promoted=%s",
        version.version,
        mape * 100,
        metrics["reserve_coverage"],
        primary,
        worst_level,
        worst_dev,
        metrics["reserve_to_incurred"],
        promoted,
    )
    for name, seg in sorted(calibration.segment_coverage.items()):
        logger.info("  segment %-16s n=%-5d coverage %.3f", name, seg.n_test, seg.coverage)
    return {**metrics, "model_version": version.version, "promoted": promoted, "refusal": refusal}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
