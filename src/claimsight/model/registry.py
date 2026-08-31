"""On-disk registry of severity models, with a reserve-adequacy promotion gate.

Every trained bundle lands here as a numbered version in ``Staging``. A version
only reaches ``Production`` — the stage ``/triage`` serves — if its measured
out-of-sample reserve coverage sits inside a tolerance band around nominal.
That gate exists because a model can improve its median error while quietly
getting worse at reserving; the two are not the same objective, and only the
second one shows up on the balance sheet.

The index is a small JSON file next to the pickles, so the API can notice a
newly promoted version by its mtime instead of needing a restart.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGING = "Staging"
PRODUCTION = "Production"
ARCHIVED = "Archived"
DEFAULT_MODEL = "claims_severity"


class PromotionRefusedError(RuntimeError):
    """Raised when a version fails the promotion gate. Production is unchanged."""


@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: int
    stage: str
    created_at: str
    metrics: dict[str, float]
    filename: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "stage": self.stage,
            "created_at": self.created_at,
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
        }


@dataclass(frozen=True)
class ReserveAdequacyGate:
    """Refuse promotion when measured coverage strays too far from nominal.

    Both directions matter: under-covering means adverse development, and
    over-covering means capital locked up for nothing.
    """

    nominal: float = 0.75
    tolerance: float = 0.03
    metric: str = "reserve_coverage"

    def refusal_reason(self, version: ModelVersion) -> str | None:
        if self.metric not in version.metrics:
            return f"v{version.version} has no {self.metric} metric; cannot verify adequacy"
        measured = float(version.metrics[self.metric])
        deviation = measured - self.nominal
        if abs(deviation) > self.tolerance:
            direction = "under" if deviation < 0 else "over"
            return (
                f"v{version.version} {direction}-reserves: measured coverage "
                f"{measured:.4f} vs nominal {self.nominal:.2f} "
                f"(deviation {deviation:+.4f}, tolerance +/-{self.tolerance:.2f})"
            )
        return None


class ModelRegistry:
    """Versioned model store backed by a directory of pickles plus an index."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    def index_stamp(self) -> tuple[float, int]:
        """Cheap change detector for the API's served-version cache."""
        if not self.index_path.exists():
            return (0.0, 0)
        stat = self.index_path.stat()
        return (stat.st_mtime, stat.st_size)

    def _read(self) -> dict[str, list[dict[str, Any]]]:
        if not self.index_path.exists():
            return {}
        with open(self.index_path, encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, index: dict[str, list[dict[str, Any]]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
        tmp.replace(self.index_path)

    @staticmethod
    def _row(entry: dict[str, Any], name: str) -> ModelVersion:
        return ModelVersion(
            name=name,
            version=int(entry["version"]),
            stage=str(entry["stage"]),
            created_at=str(entry["created_at"]),
            metrics={k: float(v) for k, v in entry.get("metrics", {}).items()},
            filename=str(entry["filename"]),
        )

    def versions(self, name: str = DEFAULT_MODEL) -> list[ModelVersion]:
        return [self._row(entry, name) for entry in self._read().get(name, [])]

    def get(self, version: int, name: str = DEFAULT_MODEL) -> ModelVersion:
        for candidate in self.versions(name):
            if candidate.version == version:
                return candidate
        raise KeyError(f"{name} has no version {version}")

    def latest(self, name: str = DEFAULT_MODEL, stage: str | None = None) -> ModelVersion:
        history = self.versions(name)
        if stage is not None:
            history = [item for item in history if item.stage == stage]
        if not history:
            raise KeyError(f"no versions for {name!r} (stage={stage!r})")
        return history[-1]

    def production(self, name: str = DEFAULT_MODEL) -> ModelVersion:
        return self.latest(name, stage=PRODUCTION)

    def register(
        self,
        bundle: dict[str, Any],
        metrics: dict[str, float],
        name: str = DEFAULT_MODEL,
        stage: str = STAGING,
    ) -> ModelVersion:
        index = self._read()
        history = index.setdefault(name, [])
        number = max((int(e["version"]) for e in history), default=0) + 1
        filename = f"{name}.v{number}.pkl"
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / filename, "wb") as handle:
            pickle.dump(bundle, handle)
        history.append(
            {
                "version": number,
                "stage": stage,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "metrics": {k: float(v) for k, v in metrics.items()},
                "filename": filename,
            }
        )
        self._write(index)
        return self._row(history[-1], name)

    def load(self, version: ModelVersion) -> dict[str, Any]:
        with open(self.root / version.filename, "rb") as handle:
            return pickle.load(handle)

    def promote(
        self,
        version: int,
        name: str = DEFAULT_MODEL,
        gate: ReserveAdequacyGate | None = None,
    ) -> ModelVersion:
        """Move a version to Production, archiving whatever held it.

        Raises ``PromotionRefusedError`` — leaving the index untouched — when the
        gate rejects the candidate.
        """
        candidate = self.get(version, name)
        if gate is not None:
            reason = gate.refusal_reason(candidate)
            if reason is not None:
                raise PromotionRefusedError(reason)

        index = self._read()
        rows = index[name]
        for entry in rows:
            if entry["stage"] == PRODUCTION and int(entry["version"]) != version:
                entry["stage"] = ARCHIVED
            if int(entry["version"]) == version:
                entry["stage"] = PRODUCTION
        self._write(index)
        return replace(candidate, stage=PRODUCTION)
