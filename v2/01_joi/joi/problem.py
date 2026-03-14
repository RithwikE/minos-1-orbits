from __future__ import annotations

from pathlib import Path

import pykep as pk
import pygmo as pg

from .config import SequenceConfig


def build_sequence(config: SequenceConfig) -> list:
    return [pk.planet.jpl_lp(name) for name in config.bodies]


def build_udp(config: SequenceConfig):
    sequence = build_sequence(config)
    # pykep's mga_1dsm constructor expects TOFs in days and the vinf bounds in km/s.
    if config.tof_encoding == "direct":
        tof = [
            [leg.minimum_days, leg.maximum_days]
            for leg in config.tof_days
        ]
    elif config.tof_encoding == "alpha":
        if config.total_tof_days is None:
            raise ValueError("`alpha` encoding requires `total_tof_days` bounds.")
        tof = [
            config.total_tof_days.minimum_days,
            config.total_tof_days.maximum_days,
        ]
    else:
        tof = (
            config.total_tof_days.maximum_days
            if config.total_tof_days is not None
            else config.max_total_tof_days
        )
    udp = pk.trajopt.mga_1dsm(
        seq=sequence,
        t0=[
            pk.epoch_from_string(config.departure_start),
            pk.epoch_from_string(config.departure_end),
        ],
        tof=tof,
        vinf=[config.vinf_kms_min, config.vinf_kms_max],
        add_vinf_dep=config.add_vinf_dep,
        add_vinf_arr=config.add_vinf_arr,
        tof_encoding=config.tof_encoding,
        multi_objective=config.multi_objective,
    )
    return udp, sequence


def build_problem(config: SequenceConfig) -> tuple[pg.problem, object, list]:
    udp, sequence = build_udp(config)
    problem = pg.problem(udp)
    return problem, udp, sequence


def ensure_results_dir(path: str | Path) -> Path:
    result_path = Path(path)
    result_path.mkdir(parents=True, exist_ok=True)
    return result_path
