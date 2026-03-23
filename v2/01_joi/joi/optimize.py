from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .config import ComputeProfile, SequenceConfig
from .postprocess import parse_decision_vector, reconstruct_candidate
from shared.search import (
    collect_population_candidates,
    run_single_objective_search as run_shared_search,
    save_search_results as save_shared_search_results,
)


def run_single_objective_search(
    problem,
    udp,
    sequence: list,
    config: SequenceConfig,
    profile: ComputeProfile,
    base_seed: int,
    progress_callback=None,
) -> dict:
    _ = sequence
    initial_candidates = load_seed_candidates(problem, config.seed_candidate_paths)
    return run_shared_search(
        problem=problem,
        profile=profile,
        base_seed=base_seed,
        multi_objective=config.multi_objective,
        top_n=config.top_n,
        mission_filter=lambda candidates: apply_mission_filters(candidates, udp=udp, config=config),
        initial_candidates=initial_candidates,
        progress_callback=progress_callback,
    )


def apply_mission_filters(candidates: list[dict], udp, config: SequenceConfig) -> list[dict]:
    n_legs = len(config.bodies) - 1
    filtered = []
    for candidate in candidates:
        parsed = parse_decision_vector(udp, np.array(candidate["x"], dtype=float), n_legs)
        total_tof_days = float(sum(parsed["tofs_days"]))
        candidate["total_tof_days"] = total_tof_days
        candidate["c3_kms2"] = (parsed["vinf_dep_m_s"] / 1000.0) ** 2
        candidate["mission_feasible"] = (
            total_tof_days <= config.max_total_tof_days + 1e-9
            and candidate["c3_kms2"] <= config.max_c3_kms2 + 1e-9
        )
        filtered.append(candidate)
    return filtered


def load_seed_candidates(problem, seed_candidate_paths: list[str]) -> list[dict]:
    candidates: list[dict] = []
    for raw_path in seed_candidate_paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        decision_vector = payload.get("decision_vector") or payload.get("x")
        if decision_vector is None:
            raise ValueError(f"Seed candidate file {path} did not contain `decision_vector` or `x`.")
        pop = collect_population_candidates(
            problem=problem,
            population=build_seed_population(problem, np.array(decision_vector, dtype=float)),
            stage=f"seed_file:{path.stem}",
        )
        if not pop:
            continue
        pop[0]["seed_source"] = str(path)
        candidates.append(pop[0])
    return candidates


def build_seed_population(problem, x: np.ndarray):
    import pygmo as pg

    pop = pg.population(problem, size=1, seed=0)
    pop.set_x(0, x)
    return pop


def save_search_results(
    run_dir: Path,
    config: SequenceConfig,
    profile: ComputeProfile,
    search_result: dict,
    udp,
    sequence: list,
    source_root: Path | None = None,
    git_metadata: dict[str, str | bool | None] | None = None,
) -> None:
    def reconstruct(candidate: dict) -> dict:
        return reconstruct_candidate(
            udp=udp,
            sequence=sequence,
            config=config,
            x=np.array(candidate["x"], dtype=float),
        )

    save_shared_search_results(
        run_dir,
        config_payload=config.to_dict(),
        profile=profile,
        search_result=search_result,
        reconstruct_candidate=reconstruct,
        source_root=source_root,
        git_metadata=git_metadata,
    )
