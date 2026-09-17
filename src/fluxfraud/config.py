"""Validated, serialisable experiment configuration; time is measured in hours."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    seed: int = 42
    customers: int = 160
    days: int = 45
    base_rate: float = 0.10
    branching_ratio: float = 0.35
    decay: float = 1.5
    ou_kappa: float = 0.06
    ou_sigma: float = 0.12
    temperature: float = 0.8
    network_memory: float = 1.0
    compromise_hazard: float = 0.0007
    recovery_hazard: float = 0.10
    drift_fraction: float = 0.75
    drift_strength: float = 1.0
    label_delay_hours: float = 24.0
    alert_budget: int = 20
    max_events: int = 1_000_000

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be numeric, not {type(value).__name__}")
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        for key in ("seed", "customers", "days", "alert_budget", "max_events"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer")
        if self.seed < 0 or not 10 <= self.customers <= 2000 or not 10 <= self.days <= 365:
            raise ValueError("Require seed >= 0, 10 <= customers <= 2000, 10 <= days <= 365")
        if not 0 <= self.branching_ratio < 0.95:
            raise ValueError("branching_ratio must be in [0, 0.95) for stable, tractable runs")
        for key in ("base_rate", "decay", "ou_kappa", "temperature", "recovery_hazard"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in (
            "ou_sigma",
            "network_memory",
            "compromise_hazard",
            "drift_strength",
            "label_delay_hours",
        ):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not 0.70 <= self.drift_fraction <= 0.90:
            raise ValueError("drift_fraction must be in [0.70, 0.90]")
        if self.label_delay_hours >= 0.15 * self.days * 24:
            raise ValueError("Label delay must be less than 15% of the simulation horizon")
        if self.alert_budget < 1 or self.max_events < 1:
            raise ValueError("alert_budget and max_events must be positive")

    @property
    def horizon(self):
        return self.days * 24

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
