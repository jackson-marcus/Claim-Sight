"""Reserve-adequacy calibration from held-out log-residuals.

A point severity estimate is not a reserve. What an insurer actually books is
an amount that should *cover* the ultimate loss some agreed fraction of the
time, and the only honest way to know that fraction is to measure it on data
the uplift was not fitted on.

So the book is cut three ways: the model is fitted on ``train``, the log-space
uplift is fitted on ``calibration``, and the coverage the uplift really
achieves is measured on a disjoint ``test`` slice. Fitting the uplift and
reporting its coverage on the same rows gives back the nominal level by
construction and tells you nothing.

Per-claim-type coverage is carried alongside as *monitoring* only. Giving each
claim type its own uplift was tried and measured to be worse — see
``scripts/reserve_calibration_report.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_LEVELS: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99)
DEFAULT_PRIMARY_LEVEL = 0.75


def _key(level: float) -> float:
    return round(float(level), 4)


@dataclass(frozen=True)
class SegmentCoverage:
    """How the *global* uplift performs on one claim type, for monitoring."""

    segment: str
    n_test: int
    coverage: float


@dataclass(frozen=True)
class ReserveCalibration:
    """Log-space uplifts and the coverage they actually achieved out of sample."""

    uplift_log: dict[float, float]
    measured_coverage: dict[float, float]
    reserve_to_incurred: dict[float, float]
    segment_coverage: dict[str, SegmentCoverage]
    primary_level: float
    n_calibration: int
    n_test: int

    @property
    def levels(self) -> tuple[float, ...]:
        return tuple(sorted(self.uplift_log))

    def _lookup(self, table: dict[float, float], level: float | None) -> float:
        chosen = _key(self.primary_level if level is None else level)
        if chosen not in table:
            offered = ", ".join(f"{lv:g}" for lv in self.levels)
            raise ValueError(f"reserve confidence {chosen:g} not calibrated; offered: {offered}")
        return table[chosen]

    def uplift_for(self, level: float | None = None) -> float:
        return self._lookup(self.uplift_log, level)

    def coverage_for(self, level: float | None = None) -> float:
        return self._lookup(self.measured_coverage, level)

    def capital_ratio_for(self, level: float | None = None) -> float:
        """Booked reserve / incurred loss over the test slice at this level."""
        return self._lookup(self.reserve_to_incurred, level)

    def worst_deviation(self) -> tuple[float, float]:
        """The level whose measured coverage strays furthest from nominal."""
        level = max(self.levels, key=lambda lv: abs(self.measured_coverage[lv] - lv))
        return level, self.measured_coverage[level] - level

    def as_dict(self) -> dict:
        return {
            "levels": list(self.levels),
            "uplift_log": {f"{lv:g}": round(v, 4) for lv, v in sorted(self.uplift_log.items())},
            "measured_coverage": {
                f"{lv:g}": round(v, 4) for lv, v in sorted(self.measured_coverage.items())
            },
            "reserve_to_incurred": {
                f"{lv:g}": round(v, 4) for lv, v in sorted(self.reserve_to_incurred.items())
            },
            "segment_coverage": {
                name: {"n_test": s.n_test, "coverage": round(s.coverage, 4)}
                for name, s in sorted(self.segment_coverage.items())
            },
            "primary_level": self.primary_level,
            "n_calibration": self.n_calibration,
            "n_test": self.n_test,
        }


def extract_log_residuals(
    model, frame: pd.DataFrame, y_log: np.ndarray, features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predicted_log, residual_log)`` for a slice the model didn't see."""
    predicted = np.asarray(model.predict(frame[list(features)]), dtype=float)
    return predicted, np.asarray(y_log, dtype=float) - predicted


def calibrate_reserve(
    calibration_residuals: np.ndarray,
    test_predicted_log: np.ndarray,
    test_residuals: np.ndarray,
    test_incurred_usd: np.ndarray,
    test_segments: Sequence[str],
    levels: Sequence[float] = DEFAULT_LEVELS,
    primary_level: float = DEFAULT_PRIMARY_LEVEL,
) -> ReserveCalibration:
    """Fit uplifts on the calibration slice, measure their coverage on the test slice.

    ``coverage`` is the share of test claims whose booked reserve was at least
    the incurred loss. Because the uplift never saw these rows, the number can
    (and does) land off nominal.
    """
    calibration_residuals = np.asarray(calibration_residuals, dtype=float)
    test_residuals = np.asarray(test_residuals, dtype=float)
    test_predicted_log = np.asarray(test_predicted_log, dtype=float)
    test_incurred_usd = np.asarray(test_incurred_usd, dtype=float)
    if calibration_residuals.size == 0 or test_residuals.size == 0:
        raise ValueError("calibration and test slices must both be non-empty")

    incurred_total = float(test_incurred_usd.sum())
    uplifts: dict[float, float] = {}
    coverage: dict[float, float] = {}
    capital: dict[float, float] = {}
    for level in levels:
        key = _key(level)
        uplift = float(np.quantile(calibration_residuals, key))
        uplifts[key] = uplift
        coverage[key] = float((test_residuals <= uplift).mean())
        booked = float(np.expm1(test_predicted_log + uplift).sum())
        capital[key] = booked / incurred_total if incurred_total > 0 else float("nan")

    primary = _key(primary_level)
    if primary not in uplifts:
        raise ValueError(f"primary level {primary:g} must be one of the calibrated levels")

    covered = test_residuals <= uplifts[primary]
    segments = pd.Series(list(test_segments), dtype="object")
    if len(segments) != len(test_residuals):
        raise ValueError("test_segments must align with test_residuals")
    per_segment: dict[str, SegmentCoverage] = {}
    for name, idx in segments.groupby(segments).groups.items():
        mask = segments.index.isin(idx)
        per_segment[str(name)] = SegmentCoverage(
            segment=str(name), n_test=int(mask.sum()), coverage=float(covered[mask].mean())
        )

    return ReserveCalibration(
        uplift_log=uplifts,
        measured_coverage=coverage,
        reserve_to_incurred=capital,
        segment_coverage=per_segment,
        primary_level=primary,
        n_calibration=int(calibration_residuals.size),
        n_test=int(test_residuals.size),
    )
