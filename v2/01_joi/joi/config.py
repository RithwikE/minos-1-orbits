from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt
from pathlib import Path
import tomllib


@dataclass(slots=True)
class LegBounds:
    minimum_days: float
    maximum_days: float


@dataclass(slots=True)
class SequenceConfig:
    name: str
    label: str
    bodies: list[str]
    departure_start: str
    departure_end: str
    vinf_kms_min: float
    max_c3_kms2: float
    max_total_tof_days: float
    tof_days: list[LegBounds]
    objective: str = "total_dsm"
    tof_encoding: str = "direct"
    add_vinf_dep: bool = False
    add_vinf_arr: bool = False
    multi_objective: bool = False
    top_n: int = 50
    dense_output_samples_per_leg: int = 128
    notes: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def vinf_kms_max(self) -> float:
        return sqrt(self.max_c3_kms2)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["vinf_kms_max"] = self.vinf_kms_max
        return data


@dataclass(slots=True)
class ComputeProfile:
    level: int
    phase1_islands: int
    phase1_pop_size: int
    phase1_generations: int
    phase1_rounds: int
    phase2_seed_count: int
    phase2_pop_size: int
    phase2_generations: int
    phase3_candidate_count: int
    phase3_compass_fevals: int
    archive_top_n: int
    log_frequency: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


def load_sequence_config(path: str | Path) -> SequenceConfig:
    cfg_path = Path(path)
    raw = tomllib.loads(cfg_path.read_text())
    legs = [
        LegBounds(
            minimum_days=float(leg["min"]),
            maximum_days=float(leg["max"]),
        )
        for leg in raw["tof_days"]
    ]
    config = SequenceConfig(
        name=raw["name"],
        label=raw["label"],
        bodies=list(raw["bodies"]),
        departure_start=raw["departure_start"],
        departure_end=raw["departure_end"],
        vinf_kms_min=float(raw["vinf_kms_min"]),
        max_c3_kms2=float(raw["max_c3_kms2"]),
        max_total_tof_days=float(raw["max_total_tof_days"]),
        tof_days=legs,
        objective=raw.get("objective", "total_dsm"),
        tof_encoding=raw.get("tof_encoding", "direct"),
        add_vinf_dep=bool(raw.get("add_vinf_dep", False)),
        add_vinf_arr=bool(raw.get("add_vinf_arr", False)),
        multi_objective=bool(raw.get("multi_objective", False)),
        top_n=int(raw.get("top_n", 50)),
        dense_output_samples_per_leg=int(raw.get("dense_output_samples_per_leg", 128)),
        notes=raw.get("notes", ""),
        metadata=dict(raw.get("metadata", {})),
    )
    validate_sequence_config(config)
    return config


def validate_sequence_config(config: SequenceConfig) -> None:
    if len(config.bodies) < 2:
        raise ValueError("A sequence needs at least a departure and arrival body.")
    if len(config.tof_days) != len(config.bodies) - 1:
        raise ValueError("`tof_days` must have one entry per leg.")
    if config.max_c3_kms2 <= 0.0:
        raise ValueError("`max_c3_kms2` must be positive.")
    if config.vinf_kms_min < 0.0:
        raise ValueError("`vinf_kms_min` must be non-negative.")
    if config.vinf_kms_min >= config.vinf_kms_max:
        raise ValueError("`vinf_kms_min` must be below the implied `vinf_kms_max`.")
    if config.max_total_tof_days <= 0.0:
        raise ValueError("`max_total_tof_days` must be positive.")
    total_leg_min = sum(leg.minimum_days for leg in config.tof_days)
    total_leg_max = sum(leg.maximum_days for leg in config.tof_days)
    if total_leg_max < 1.0:
        raise ValueError("Total leg TOF maxima must be positive.")
    if total_leg_min > config.max_total_tof_days:
        raise ValueError(
            "`max_total_tof_days` is smaller than the sum of the minimum leg TOFs."
        )
    if config.tof_encoding == "direct" and total_leg_max > config.max_total_tof_days:
        raise ValueError(
            "For `direct` encoding, the sum of leg TOF maxima cannot exceed "
            "`max_total_tof_days`, otherwise the search domain includes invalid "
            "total mission durations."
        )


def build_compute_profile(level: int) -> ComputeProfile:
    if not 1 <= level <= 10:
        raise ValueError("Compute level must be between 1 and 10.")

    return ComputeProfile(
        level=level,
        phase1_islands=4 * level,
        phase1_pop_size=8 + 4 * level,
        phase1_generations=50 * level,
        phase1_rounds=1 + level,
        phase2_seed_count=max(4, 6 * level),
        phase2_pop_size=8 + 4 * level,
        phase2_generations=100 * level,
        phase3_candidate_count=max(1, min(2 * level, 12)),
        phase3_compass_fevals=5_000 * level * level,
        archive_top_n=max(25, 25 * level),
        log_frequency=1,
    )
