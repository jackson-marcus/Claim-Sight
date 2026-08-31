"""Fraud rules, routing, the severity model, and the HTTP contract."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from claimsight.api import routes
from claimsight.api.main import create_app
from claimsight.model.registry import ModelRegistry, ReserveAdequacyGate
from claimsight.models.triage import fraud_flags, route

GATE = ReserveAdequacyGate(nominal=0.75, tolerance=0.03)

SUSPECT_CLAIM = {
    "claim_type": "property_fire",
    "report_delay_days": 40,
    "policy_tenure_days": 12,
    "claimed_amount": 50000,
    "police_report": 0,
}


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


def test_severity_model_learns(claims, fitted):
    actual = np.expm1(np.log1p(claims.loc[fitted["test_index"], "final_severity"]).to_numpy())
    predicted = np.expm1(fitted["test_predicted_log"])
    median_ape = float(np.median(np.abs(actual - predicted) / np.maximum(actual, 1)))
    assert median_ape < 0.5, f"median APE {median_ape:.2f}"
    assert median_ape > 0.05, "noise floor should prevent near-perfect severity"


@pytest.fixture
def served(tmp_path, claims, bundle, monkeypatch):
    """A registry with one promoted version, wired into a live TestClient."""
    from claimsight.settings import get_config

    cfg = get_config()
    artifacts, processed = tmp_path / "art", tmp_path / "proc"
    artifacts.mkdir()
    processed.mkdir()
    claims.to_parquet(processed / "claims.parquet", index=False)
    monkeypatch.setitem(cfg["data"], "artifacts_dir", str(artifacts))
    monkeypatch.setitem(cfg["data"], "processed_dir", str(processed))
    routes._reset_cache()

    registry = ModelRegistry(artifacts / "registry")
    registry.register(bundle, {"reserve_coverage": bundle["calibration"].coverage_for(0.75)})
    registry.promote(1, gate=GATE)
    try:
        yield TestClient(create_app()), registry
    finally:
        routes._reset_cache()


def test_triage_reserves_above_severity_and_names_its_coverage(served):
    client, _ = served
    body = client.post("/triage", json=SUSPECT_CLAIM).json()
    assert body["suggested_reserve_usd"] > body["predicted_severity_usd"]
    assert body["route_to"] == "special_investigations"
    assert body["reserve_confidence"] == 0.75
    assert abs(body["measured_coverage"] - 0.75) < 0.05
    assert body["segment_coverage"]["segment"] == "property_fire"
    assert body["model_version"] == 1


def test_reserve_grows_with_requested_confidence(served):
    client, _ = served
    reserves = []
    for level in (0.5, 0.75, 0.9, 0.99):
        body = client.post("/triage", json={**SUSPECT_CLAIM, "reserve_confidence": level}).json()
        assert body["reserve_confidence"] == level
        reserves.append(body["suggested_reserve_usd"])
    assert reserves == sorted(reserves)
    assert reserves[0] < reserves[-1]


def test_uncalibrated_confidence_is_a_400_listing_the_offered_levels(served):
    client, _ = served
    response = client.post("/triage", json={**SUSPECT_CLAIM, "reserve_confidence": 0.8})
    assert response.status_code == 400
    assert "0.75" in response.json()["detail"]


def test_queue_stats_books_more_capital_at_higher_confidence(served):
    client, _ = served
    low = client.get("/queue-stats", params={"reserve_confidence": 0.5}).json()
    high = client.get("/queue-stats", params={"reserve_confidence": 0.95}).json()
    assert low["sampled_claims"] == high["sampled_claims"] > 0
    assert high["reserves_booked_usd"] > low["reserves_booked_usd"]
    assert low["queue_depth"] == high["queue_depth"], "routing must not move with reserve policy"


def test_promoting_a_new_version_swaps_what_is_served_without_a_restart(served, bundle):
    client, registry = served
    assert client.post("/triage", json=SUSPECT_CLAIM).json()["model_version"] == 1
    registry.register(bundle, {"reserve_coverage": 0.75})
    registry.promote(2, gate=GATE)
    assert client.post("/triage", json=SUSPECT_CLAIM).json()["model_version"] == 2
    assert client.get("/models").json()["production_version"] == 2


def test_service_is_503_until_something_reaches_production(tmp_path, claims, monkeypatch):
    from claimsight.settings import get_config

    cfg = get_config()
    monkeypatch.setitem(cfg["data"], "artifacts_dir", str(tmp_path))
    routes._reset_cache()
    try:
        client = TestClient(create_app())
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/triage", json=SUSPECT_CLAIM).status_code == 503
    finally:
        routes._reset_cache()
