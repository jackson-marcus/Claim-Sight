"""Promotion is gated on reserve adequacy, not on error alone."""

from __future__ import annotations

import pytest

from claimsight.model.predictor import ReservingPredictor
from claimsight.model.registry import (
    ARCHIVED,
    PRODUCTION,
    STAGING,
    ModelRegistry,
    PromotionRefusedError,
    ReserveAdequacyGate,
)

GATE = ReserveAdequacyGate(nominal=0.75, tolerance=0.03)


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(tmp_path / "registry")


def test_versions_number_upwards_and_start_in_staging(registry, bundle):
    first = registry.register(bundle, {"reserve_coverage": 0.75})
    second = registry.register(bundle, {"reserve_coverage": 0.75})
    assert (first.version, second.version) == (1, 2)
    assert [v.stage for v in registry.versions()] == [STAGING, STAGING]
    with pytest.raises(KeyError):
        registry.production()


def test_promotion_archives_the_incumbent(registry, bundle):
    registry.register(bundle, {"reserve_coverage": 0.752})
    registry.register(bundle, {"reserve_coverage": 0.748})
    registry.promote(1, gate=GATE)
    assert registry.production().version == 1

    registry.promote(2, gate=GATE)
    stages = {v.version: v.stage for v in registry.versions()}
    assert stages == {1: ARCHIVED, 2: PRODUCTION}
    assert registry.production().version == 2


def test_gate_refuses_an_under_reserving_version_and_production_does_not_move(registry, bundle):
    registry.register(bundle, {"reserve_coverage": 0.751, "median_ape": 0.20})
    registry.promote(1, gate=GATE)
    # Lower median error, but it only covers 71% of claims: adverse development.
    registry.register(bundle, {"reserve_coverage": 0.712, "median_ape": 0.15})

    with pytest.raises(PromotionRefusedError, match="under-reserves"):
        registry.promote(2, gate=GATE)

    assert registry.production().version == 1
    assert registry.get(2).stage == STAGING


def test_gate_refuses_an_over_reserving_version(registry, bundle):
    registry.register(bundle, {"reserve_coverage": 0.83})
    with pytest.raises(PromotionRefusedError, match="over-reserves"):
        registry.promote(1, gate=GATE)


def test_gate_refuses_a_version_that_never_measured_coverage(registry, bundle):
    registry.register(bundle, {"median_ape": 0.10})
    with pytest.raises(PromotionRefusedError, match="cannot verify adequacy"):
        registry.promote(1, gate=GATE)


def test_promotion_without_a_gate_is_allowed_for_manual_rollback(registry, bundle):
    registry.register(bundle, {"reserve_coverage": 0.60})
    registry.promote(1)
    assert registry.production().version == 1


def test_production_bundle_round_trips_into_a_working_predictor(registry, bundle):
    registry.register(bundle, {"reserve_coverage": 0.75})
    registry.promote(1, gate=GATE)
    predictor = ReservingPredictor.from_registry(registry)
    assert predictor.version.version == 1
    assert predictor.confidence_levels == bundle["calibration"].levels


def test_predictor_rejects_a_bundle_with_no_calibration(registry, bundle):
    registry.register({"model": bundle["model"], "features": bundle["features"]}, {})
    version = registry.get(1)
    with pytest.raises(ValueError, match="no reserve calibration"):
        ReservingPredictor(version, registry.load(version))
