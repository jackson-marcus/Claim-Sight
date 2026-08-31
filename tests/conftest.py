"""Shared fixtures: one synthetic book, fitted once, split the way training does."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_claims import generate

from claimsight.features.extractor import calibrate_reserve, extract_log_residuals
from claimsight.models.triage import FEATURES, TYPE_DUMMIES, prepare

COLS = FEATURES + TYPE_DUMMIES


@pytest.fixture(scope="session")
def claims():
    return generate(n_claims=9000, seed=4)


@pytest.fixture(scope="session")
def fitted(claims):
    """Model plus residuals for each of the three disjoint slices."""
    frame = prepare(claims)
    y_log = np.log1p(claims["final_severity"])
    train_idx, rest = train_test_split(claims.index, test_size=0.4, random_state=4)
    cal_idx, test_idx = train_test_split(rest, test_size=0.5, random_state=4)
    model = LGBMRegressor(n_estimators=200, verbose=-1, random_state=0).fit(
        frame.loc[train_idx, COLS], y_log.loc[train_idx]
    )

    def residuals(idx):
        return extract_log_residuals(model, frame.loc[idx], y_log.loc[idx].to_numpy(), COLS)

    _, train_res = residuals(train_idx)
    _, cal_res = residuals(cal_idx)
    test_pred, test_res = residuals(test_idx)
    return {
        "model": model,
        "features": COLS,
        "train_residuals": train_res,
        "calibration_residuals": cal_res,
        "test_predicted_log": test_pred,
        "test_residuals": test_res,
        "test_index": test_idx,
    }


@pytest.fixture(scope="session")
def calibration(claims, fitted):
    return calibrate_reserve(
        calibration_residuals=fitted["calibration_residuals"],
        test_predicted_log=fitted["test_predicted_log"],
        test_residuals=fitted["test_residuals"],
        test_incurred_usd=claims.loc[fitted["test_index"], "final_severity"].to_numpy(),
        test_segments=claims.loc[fitted["test_index"], "claim_type"].tolist(),
    )


@pytest.fixture(scope="session")
def bundle(fitted, calibration):
    return {"model": fitted["model"], "features": COLS, "calibration": calibration}
