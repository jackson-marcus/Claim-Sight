"""The reserve uplift has to earn its nominal level on data it never saw."""

from __future__ import annotations

import numpy as np
import pytest

from claimsight.features.extractor import DEFAULT_LEVELS, calibrate_reserve

NOMINAL = 0.75


def test_uplift_rises_with_requested_confidence(calibration):
    uplifts = [calibration.uplift_for(level) for level in DEFAULT_LEVELS]
    assert uplifts == sorted(uplifts)
    assert uplifts[0] < uplifts[-1], "0.99 must cost more log-uplift than 0.50"


def test_measured_coverage_tracks_nominal_on_the_untouched_slice(calibration):
    for level in DEFAULT_LEVELS:
        realized = calibration.coverage_for(level)
        assert abs(realized - level) < 0.05, f"nominal {level}: realized {realized:.4f}"


def test_higher_confidence_books_strictly_more_capital(calibration):
    ratios = [calibration.capital_ratio_for(level) for level in DEFAULT_LEVELS]
    assert ratios == sorted(ratios)
    # Reserving at the median leaves the book short of incurred losses.
    assert calibration.capital_ratio_for(0.50) < 1.0 < calibration.capital_ratio_for(0.90)


def test_calibrating_on_the_training_fold_silently_under_reserves(claims, fitted, calibration):
    """The shortcut a two-way split invites: the model already fits train, so
    its residuals are too tight and the uplift comes out too small."""
    in_sample_uplift = float(np.quantile(fitted["train_residuals"], NOMINAL))
    honest_uplift = calibration.uplift_for(NOMINAL)
    assert in_sample_uplift < honest_uplift

    in_sample_coverage = float((fitted["test_residuals"] <= in_sample_uplift).mean())
    assert in_sample_coverage < calibration.coverage_for(NOMINAL)
    assert in_sample_coverage < NOMINAL - 0.015, (
        f"training-fold calibration only covered {in_sample_coverage:.4f}"
    )


def test_segment_coverage_partitions_the_test_slice(calibration):
    segments = calibration.segment_coverage.values()
    assert sum(s.n_test for s in segments) == calibration.n_test
    weighted = sum(s.coverage * s.n_test for s in segments) / calibration.n_test
    assert weighted == pytest.approx(calibration.coverage_for(NOMINAL), abs=1e-9)


def test_uncalibrated_confidence_is_refused_by_name(calibration):
    with pytest.raises(ValueError, match=r"0\.8 not calibrated"):
        calibration.uplift_for(0.8)


def test_empty_calibration_slice_is_rejected(fitted):
    with pytest.raises(ValueError, match="non-empty"):
        calibrate_reserve(
            calibration_residuals=np.array([]),
            test_predicted_log=fitted["test_predicted_log"],
            test_residuals=fitted["test_residuals"],
            test_incurred_usd=np.ones_like(fitted["test_residuals"]),
            test_segments=["auto_collision"] * len(fitted["test_residuals"]),
        )


def test_segments_must_align_with_the_test_slice(fitted):
    with pytest.raises(ValueError, match="align"):
        calibrate_reserve(
            calibration_residuals=fitted["calibration_residuals"],
            test_predicted_log=fitted["test_predicted_log"],
            test_residuals=fitted["test_residuals"],
            test_incurred_usd=np.ones_like(fitted["test_residuals"]),
            test_segments=["auto_collision"],
        )
