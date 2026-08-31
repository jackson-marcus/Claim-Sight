"""Serve the Production severity model as a reserve, at a chosen confidence.

The reserve an adjuster books is a policy choice, not a model output: booking
at 0.50 means half the claims blow through their reserve, booking at 0.99 ties
up capital that could be writing business. This predictor exposes that dial and
answers it with the coverage each setting *actually* achieved on held-out
claims, plus how the global uplift has been performing on this claim's own line
of business.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from claimsight.features.extractor import ReserveCalibration
from claimsight.model.registry import DEFAULT_MODEL, ModelRegistry, ModelVersion
from claimsight.models.triage import fraud_flags, prepare, route


class ReservingPredictor:
    """Wraps one registered model version and its reserve calibration."""

    def __init__(self, version: ModelVersion, bundle: dict[str, Any]) -> None:
        self.version = version
        self.model = bundle["model"]
        self.features: list[str] = list(bundle["features"])
        calibration = bundle.get("calibration")
        if not isinstance(calibration, ReserveCalibration):
            raise ValueError(
                f"{version.name} v{version.version} has no reserve calibration; "
                "retrain with claimsight.models.triage"
            )
        self.calibration: ReserveCalibration = calibration

    @classmethod
    def from_registry(
        cls, registry: ModelRegistry, name: str = DEFAULT_MODEL
    ) -> ReservingPredictor:
        version = registry.production(name)
        return cls(version, registry.load(version))

    @property
    def confidence_levels(self) -> tuple[float, ...]:
        return self.calibration.levels

    def severity_usd(self, claims: pd.DataFrame) -> np.ndarray:
        """Point severity estimate in dollars for a frame of raw claims."""
        frame = prepare(claims)
        return np.expm1(np.asarray(self.model.predict(frame[self.features]), dtype=float))

    def _predicted_log(self, claims: pd.DataFrame) -> np.ndarray:
        frame = prepare(claims)
        return np.asarray(self.model.predict(frame[self.features]), dtype=float)

    def triage(self, claim: dict[str, Any], confidence: float | None = None) -> dict[str, Any]:
        """Severity, a reserve at the requested confidence, flags and a route."""
        predicted_log = float(self._predicted_log(pd.DataFrame([claim]))[0])
        uplift = self.calibration.uplift_for(confidence)
        level = self.calibration.primary_level if confidence is None else round(confidence, 4)
        severity = float(np.expm1(predicted_log))
        reserve = float(np.expm1(predicted_log + uplift))
        flags = fraud_flags(claim)
        segment = str(claim.get("claim_type", ""))
        observed = self.calibration.segment_coverage.get(segment)
        return {
            "predicted_severity_usd": round(severity, 2),
            "suggested_reserve_usd": round(reserve, 2),
            "reserve_confidence": level,
            "reserve_uplift_log": round(uplift, 4),
            "measured_coverage": round(self.calibration.coverage_for(confidence), 4),
            "segment_coverage": (
                None
                if observed is None
                else {
                    "segment": observed.segment,
                    "n_test": observed.n_test,
                    "coverage": round(observed.coverage, 4),
                }
            ),
            "fraud_flags": flags,
            "route_to": route(severity, len(flags)),
            "model_version": self.version.version,
        }

    def triage_book(self, claims: pd.DataFrame, confidence: float | None = None) -> pd.DataFrame:
        """Route and reserve a whole book at once, for capacity planning."""
        predicted_log = self._predicted_log(claims)
        uplift = self.calibration.uplift_for(confidence)
        severity = np.expm1(predicted_log)
        flag_counts = [len(fraud_flags(row)) for row in claims.to_dict("records")]
        return pd.DataFrame(
            {
                "predicted_severity_usd": severity,
                "suggested_reserve_usd": np.expm1(predicted_log + uplift),
                "n_fraud_flags": flag_counts,
                "route_to": [
                    route(float(s), n) for s, n in zip(severity, flag_counts, strict=True)
                ],
            },
            index=claims.index,
        )
