"""Synthetic auto/property claims with latent severity drivers + planted fraud.

Usage:
    uv run python scripts/make_claims.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claimsight.settings import get_config, resolve_path

CLAIM_TYPES = ["auto_collision", "auto_theft", "property_water", "property_fire", "liability"]
TYPE_BASE = {
    "auto_collision": 3500,
    "auto_theft": 9000,
    "property_water": 6000,
    "property_fire": 24000,
    "liability": 15000,
}


def generate(n_claims: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ctype = rng.choice(CLAIM_TYPES, n_claims, p=[0.42, 0.08, 0.28, 0.07, 0.15])
    base = np.array([TYPE_BASE[t] for t in ctype])
    vehicle_age = rng.integers(0, 20, n_claims)
    injuries = rng.poisson(0.25, n_claims)
    police_report = rng.random(n_claims) < 0.55
    report_delay_days = rng.exponential(5, n_claims).astype(int)
    policy_tenure_days = rng.integers(5, 3000, n_claims)
    prior_claims_3y = rng.poisson(0.5, n_claims)

    severity = base * np.exp(
        0.35 * injuries
        + 0.015 * vehicle_age * (np.char.startswith(ctype.astype(str), "auto"))
        + rng.normal(0, 0.55, n_claims)
    )

    is_fraud = rng.random(n_claims) < 0.06
    # Fraudulent claims: inflated, rounder amounts, later reports, young policies.
    severity = np.where(is_fraud, severity * rng.uniform(1.4, 2.4, n_claims), severity)
    # First-notice claimed amounts are rough estimates (±30-40%), so the
    # severity model must blend the anchor with loss drivers, not pass it through.
    claimed = np.where(
        is_fraud & (rng.random(n_claims) < 0.6),
        np.round(severity, -3),  # suspiciously round
        np.round(severity * rng.uniform(0.65, 1.45, n_claims), 2),
    )
    report_delay_days = np.where(
        is_fraud & (rng.random(n_claims) < 0.5),
        report_delay_days + rng.integers(20, 60, n_claims),
        report_delay_days,
    )
    policy_tenure_days = np.where(
        is_fraud & (rng.random(n_claims) < 0.4),
        rng.integers(1, 30, n_claims),
        policy_tenure_days,
    )
    prior_claims_3y = np.where(
        is_fraud, prior_claims_3y + rng.poisson(1.0, n_claims), prior_claims_3y
    )

    return pd.DataFrame(
        {
            "claim_id": np.arange(1, n_claims + 1),
            "claim_type": ctype,
            "vehicle_age": vehicle_age,
            "injuries": injuries,
            "police_report": police_report.astype(int),
            "report_delay_days": report_delay_days,
            "policy_tenure_days": policy_tenure_days,
            "prior_claims_3y": prior_claims_3y,
            "claimed_amount": np.round(claimed, 2),
            "final_severity": np.round(severity, 2),
            "is_fraud": is_fraud.astype(int),
        }
    )


def main() -> None:
    cfg = get_config()["data"]
    df = generate(cfg["n_claims"], cfg["seed"])
    out = resolve_path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "claims.parquet", index=False)
    print(f"Wrote {len(df):,} claims (fraud rate {df['is_fraud'].mean():.1%}) -> {out}")


if __name__ == "__main__":
    main()
