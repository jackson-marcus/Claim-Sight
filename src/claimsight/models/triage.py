"""Claims triage: severity model, explainable fraud flags, routing.

Severity: LGBM on log-severity with a reserve suggestion (P75-ish via
residual quantile). Fraud flags are auditable rules (not a black box) —
each flag names its evidence. Routing combines predicted severity, flags,
and adjuster capacity.

Usage (train):
    python -m claimsight.models.triage
"""

from __future__ import annotations

import logging
import pickle

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

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


def train() -> dict:
    cfg = get_config()
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["eval"]["experiment_name"])

    df = prepare(pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "claims.parquet"))
    cols = FEATURES + TYPE_DUMMIES
    x_train, x_test, y_train, y_test = train_test_split(
        df[cols],
        np.log1p(df["final_severity"]),
        test_size=cfg["severity"]["test_frac"],
        random_state=42,
    )
    model = LGBMRegressor(**cfg["severity"]["lgbm"], random_state=42)
    model.fit(x_train, y_train)
    pred_log = model.predict(x_test)
    residual_q75 = float(np.quantile(y_test - pred_log, 0.75))
    actual = np.expm1(y_test)
    pred = np.expm1(pred_log)
    mape = float(np.median(np.abs(actual - pred) / np.maximum(actual, 1)))

    with mlflow.start_run(run_name="severity-lgbm"):
        mlflow.log_params({"n_claims": len(df)})
        mlflow.log_metrics({"median_ape": mape, "reserve_uplift_log": residual_q75})
    logger.info("severity median APE %.1f%% | reserve log-uplift %.3f", mape * 100, residual_q75)

    artifacts = resolve_path(cfg["data"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "severity.pkl", "wb") as f:
        pickle.dump({"model": model, "features": cols, "reserve_uplift_log": residual_q75}, f)
    return {"median_ape": mape}


def predict_claim(bundle: dict, claim: dict) -> dict:
    frame = prepare(pd.DataFrame([claim]))
    pred_log = float(bundle["model"].predict(frame[bundle["features"]])[0])
    severity = float(np.expm1(pred_log))
    reserve = float(np.expm1(pred_log + bundle["reserve_uplift_log"]))
    flags = fraud_flags(claim)
    return {
        "predicted_severity_usd": round(severity, 2),
        "suggested_reserve_usd": round(reserve, 2),
        "fraud_flags": flags,
        "route_to": route(severity, len(flags)),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
