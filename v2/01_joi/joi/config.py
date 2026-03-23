from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import sqrt
from pathlib import Path
import tomllib

from shared.search import ComputeProfile, build_compute_profile, override_compute_profile


@dataclass(slots=True)
class LegBounds:
    minimum_days: float
    maximum_days: float


@dataclass(slots=True)
class SearchOverrides:
    phase1_islands: int | None = None
    phase1_pop_size: int | None = None
    phase1_generations: int | None = None
    phase1_rounds: int | None = None
    phase2_seed_count: int | None = None
    phase2_pop_size: int | None = None
    phase2_generations: int | None = None
    phase3_candidate_count: int | None = None
    phase3_compass_fevals: int | None = None
    archive_top_n: int | None = None
    log_frequency: int | None = None

    def to_dict(self) -> dict[str, int]:
        return {
            key: int(value)
            for key, value in asdict(self).items()
            if value is not None
        }


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
    tof_days: list[LegBounds] = field(default_factory=list)
    total_tof_days: LegBounds | None = None
    objective: str = "total_dsm"
    tof_encoding: str = "direct"
    add_vinf_dep: bool = False
    add_vinf_arr: bool = False
    multi_objective: bool = False
    top_n: int = 50
    dense_output_samples_per_leg: int = 128
    notes: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    search_overrides: SearchOverrides | None = None
    seed_candidate_paths: list[str] = field(default_factory=list)

    @property
    def vinf_kms_max(self) -> float:
        return sqrt(self.max_c3_kms2)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["vinf_kms_max"] = self.vinf_kms_max
        return data


def load_sequence_config(path: str | Path) -> SequenceConfig:
    cfg_path = Path(path)
    raw = tomllib.loads(cfg_path.read_text())
    legs = [
        LegBounds(
            minimum_days=float(leg["min"]),
            maximum_days=float(leg["max"]),
        )
        for leg in raw.get("tof_days", [])
    ]
    total_tof = None
    if "tof_total_days" in raw:
        total_tof = LegBounds(
            minimum_days=float(raw["tof_total_days"]["min"]),
            maximum_days=float(raw["tof_total_days"]["max"]),
        )
    search_overrides = None
    if "search" in raw:
        search_raw = raw["search"]
        search_overrides = SearchOverrides(
            phase1_islands=int(search_raw["phase1_islands"]) if "phase1_islands" in search_raw else None,
            phase1_pop_size=int(search_raw["phase1_pop_size"]) if "phase1_pop_size" in search_raw else None,
            phase1_generations=int(search_raw["phase1_generations"]) if "phase1_generations" in search_raw else None,
            phase1_rounds=int(search_raw["phase1_rounds"]) if "phase1_rounds" in search_raw else None,
            phase2_seed_count=int(search_raw["phase2_seed_count"]) if "phase2_seed_count" in search_raw else None,
            phase2_pop_size=int(search_raw["phase2_pop_size"]) if "phase2_pop_size" in search_raw else None,
            phase2_generations=int(search_raw["phase2_generations"]) if "phase2_generations" in search_raw else None,
            phase3_candidate_count=int(search_raw["phase3_candidate_count"]) if "phase3_candidate_count" in search_raw else None,
            phase3_compass_fevals=int(search_raw["phase3_compass_fevals"]) if "phase3_compass_fevals" in search_raw else None,
            archive_top_n=int(search_raw["archive_top_n"]) if "archive_top_n" in search_raw else None,
            log_frequency=int(search_raw["log_frequency"]) if "log_frequency" in search_raw else None,
        )
    seed_candidate_paths = [
        str(resolve_support_path(cfg_path, raw_path))
        for raw_path in raw.get("seed_candidate_paths", [])
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
        total_tof_days=total_tof,
        objective=raw.get("objective", "total_dsm"),
        tof_encoding=raw.get("tof_encoding", "direct"),
        add_vinf_dep=bool(raw.get("add_vinf_dep", False)),
        add_vinf_arr=bool(raw.get("add_vinf_arr", False)),
        multi_objective=bool(raw.get("multi_objective", False)),
        top_n=int(raw.get("top_n", 50)),
        dense_output_samples_per_leg=int(raw.get("dense_output_samples_per_leg", 128)),
        notes=raw.get("notes", ""),
        metadata=dict(raw.get("metadata", {})),
        search_overrides=search_overrides,
        seed_candidate_paths=seed_candidate_paths,
    )
    validate_sequence_config(config)
    return config


def resolve_support_path(cfg_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    relative_to_config = (cfg_path.parent / candidate).resolve()
    if relative_to_config.exists():
        return relative_to_config

    project_root = Path(__file__).resolve().parents[1]
    relative_to_project_configs = (project_root / "configs" / candidate).resolve()
    if relative_to_project_configs.exists():
        return relative_to_project_configs

    return relative_to_config


def build_search_profile(config: SequenceConfig, level: int) -> ComputeProfile:
    profile = build_compute_profile(level)
    if config.search_overrides is None:
        return profile
    return override_compute_profile(profile, **config.search_overrides.to_dict())


def validate_sequence_config(config: SequenceConfig) -> None:
    if len(config.bodies) < 2:
        raise ValueError("A sequence needs at least a departure and arrival body.")
    if config.max_c3_kms2 <= 0.0:
        raise ValueError("`max_c3_kms2` must be positive.")
    if config.vinf_kms_min < 0.0:
        raise ValueError("`vinf_kms_min` must be non-negative.")
    if config.vinf_kms_min >= config.vinf_kms_max:
        raise ValueError("`vinf_kms_min` must be below the implied `vinf_kms_max`.")
    if config.max_total_tof_days <= 0.0:
        raise ValueError("`max_total_tof_days` must be positive.")
    if config.tof_encoding not in {"direct", "alpha", "eta"}:
        raise ValueError("`tof_encoding` must be one of `direct`, `alpha`, or `eta`.")
    if config.search_overrides is not None:
        for key, value in config.search_overrides.to_dict().items():
            if value <= 0:
                raise ValueError(f"`search.{key}` must be positive when provided.")
    for seed_candidate_path in config.seed_candidate_paths:
        if not Path(seed_candidate_path).is_file():
            raise ValueError(f"Seed candidate path does not exist: {seed_candidate_path}")

    if config.tof_encoding == "direct":
        if len(config.tof_days) != len(config.bodies) - 1:
            raise ValueError("`tof_days` must have one entry per leg for `direct` encoding.")
        total_leg_min = sum(leg.minimum_days for leg in config.tof_days)
        total_leg_max = sum(leg.maximum_days for leg in config.tof_days)
        if total_leg_max < 1.0:
            raise ValueError("Total leg TOF maxima must be positive.")
        if total_leg_min > config.max_total_tof_days:
            raise ValueError(
                "`max_total_tof_days` is smaller than the sum of the minimum leg TOFs."
            )
        if total_leg_max > config.max_total_tof_days:
            raise ValueError(
                "For `direct` encoding, the sum of leg TOF maxima cannot exceed "
                "`max_total_tof_days`, otherwise the search domain includes invalid "
                "total mission durations."
            )
        return

    if config.tof_encoding == "alpha":
        if config.total_tof_days is None:
            raise ValueError("`tof_total_days` is required for `alpha` encoding.")
        if config.total_tof_days.maximum_days > config.max_total_tof_days:
            raise ValueError(
                "The `alpha` total TOF upper bound cannot exceed `max_total_tof_days`."
            )
        if config.total_tof_days.minimum_days <= 0.0:
            raise ValueError("`alpha` total TOF minimum must be positive.")
        if config.total_tof_days.minimum_days >= config.total_tof_days.maximum_days:
            raise ValueError("`tof_total_days.min` must be below `tof_total_days.max`.")
        return

    if config.total_tof_days is not None:
        if config.total_tof_days.maximum_days > config.max_total_tof_days:
            raise ValueError(
                "The optional eta total TOF upper bound cannot exceed `max_total_tof_days`."
            )
        if config.total_tof_days.minimum_days > 0.0:
            raise ValueError(
                "`eta` encoding only supports an upper total TOF bound in pykep. "
                "Do not provide a positive `tof_total_days.min`."
            )
