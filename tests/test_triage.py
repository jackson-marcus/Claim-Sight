"""Severity model, flags-vs-planted-fraud, routing, API."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_claims import generate

from claimsight.api.main import create_app
from claimsight.models.triage import fraud_flags, route


@pytest.fixture(scope="session")
def claims():
    return generate(n_claims=6000, seed=4)


def test_flags_catch_planted_fraud_more_often(claims):
    fraud = claims[claims["is_fraud"] == 1]
    clean = claims[claims["is_fraud"] == 0]
    fraud_rate = fraud.apply(lambda r: len(fraud_flags(r.to_dict())) >= 2, axis=1).mean()
    clean_rate = clean.apply(lambda r: len(fraud_flags(r.to_dict())) >= 2, axis=1).mean()
    assert fraud_rate > clean_rate * 2, f"fraud {fraud_rate:.2f} vs clean {clean_rate:.2f}"


def test_flags_are_explained(claims):
    row = claims.iloc[0].to_dict()
    row.update(report_delay_days=45, policy_tenure_days=10, prior_claims_3y=4)
    flags = fraud_flags(row)
    assert len(flags) >= 3
    assert all("evidence" in f and f["evidence"] for f in flags)


def test_routing_by_severity_and_flags():
    assert route(3000, 0) == "junior_pool"
    assert route(20000, 0) == "senior_pool"
    assert route(100000, 0) == "complex_unit"
    assert route(3000, 2) == "special_investigations"


def test_severity_model_learns(claims, tmp_path, monkeypatch):
    import numpy as np
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import train_test_split

    from claimsight.models.triage import FEATURES, TYPE_DUMMIES, prepare

    frame = prepare(claims)
    cols = FEATURES + TYPE_DUMMIES
    x_train, x_test, y_train, y_test = train_test_split(
        frame[cols], np.log1p(claims["final_severity"]), test_size=0.3, random_state=0
    )
    model = LGBMRegressor(n_estimators=150, verbose=-1, random_state=0).fit(x_train, y_train)
    pred = np.expm1(model.predict(x_test))
    actual = np.expm1(y_test)
    median_ape = float(np.median(np.abs(actual - pred) / np.maximum(actual, 1)))
    assert median_ape < 0.5, f"median APE {median_ape:.2f}"
    assert median_ape > 0.05, "noise floor should prevent near-perfect severity"


def test_api_triage(claims, tmp_path):
    import pickle

    import numpy as np
    from lightgbm import LGBMRegressor

    import claimsight.api.routes as routes
    from claimsight.models.triage import FEATURES, TYPE_DUMMIES, prepare
    from claimsight.settings import get_config

    frame = prepare(claims)
    cols = FEATURES + TYPE_DUMMIES
    model = LGBMRegressor(n_estimators=80, verbose=-1, random_state=0)
    model.fit(frame[cols], np.log1p(claims["final_severity"]))

    cfg = get_config()
    orig_art, orig_proc = cfg["data"]["artifacts_dir"], cfg["data"]["processed_dir"]
    art, proc = tmp_path / "art", tmp_path / "proc"
    art.mkdir()
    proc.mkdir()
    claims.to_parquet(proc / "claims.parquet", index=False)
    cfg["data"]["artifacts_dir"], cfg["data"]["processed_dir"] = str(art), str(proc)
    with open(art / "severity.pkl", "wb") as f:
        pickle.dump({"model": model, "features": cols, "reserve_uplift_log": 0.4}, f)
    routes._bundle.cache_clear()
    try:
        client = TestClient(create_app())
        r = client.post(
            "/triage",
            json={
                "claim_type": "property_fire",
                "report_delay_days": 40,
                "policy_tenure_days": 12,
                "claimed_amount": 50000,
                "police_report": 0,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["suggested_reserve_usd"] > body["predicted_severity_usd"]
        assert body["route_to"] == "special_investigations"
        stats = client.get("/queue-stats").json()
        assert stats["sampled_claims"] > 0
    finally:
        cfg["data"]["artifacts_dir"], cfg["data"]["processed_dir"] = orig_art, orig_proc
        routes._bundle.cache_clear()
