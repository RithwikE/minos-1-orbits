from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import time

import numpy as np
import pygmo as pg

from .config import ComputeProfile, SequenceConfig
from .postprocess import parse_decision_vector, reconstruct_candidate


def run_single_objective_search(
    problem: pg.problem,
    udp,
    sequence: list,
    config: SequenceConfig,
    profile: ComputeProfile,
    base_seed: int,
) -> dict:
    if config.multi_objective:
        raise NotImplementedError(
            "The first v2 runner only supports single-objective mode. "
            "Use archived candidate comparison for secondary metrics for now."
        )

    started_at_utc = datetime.now(UTC).isoformat()
    run_started = time.perf_counter()
    archive: list[dict] = []

    archipelago = pg.archipelago()
    for island_idx in range(profile.phase1_islands):
        algo1 = pg.algorithm(pg.sade(gen=profile.phase1_generations))
        algo1.set_verbosity(0)
        algo1.set_seed(base_seed + island_idx)
        archipelago.push_back(
            algo=algo1,
            prob=problem,
            size=profile.phase1_pop_size,
            seed=base_seed + 10_000 + island_idx,
        )

    for round_idx in range(profile.phase1_rounds):
        archipelago.evolve()
        archipelago.wait()
        archive.extend(
            apply_mission_filters(
                collect_archipelago_candidates(
                    problem=problem,
                    archipelago=archipelago,
                    stage=f"phase1_round_{round_idx + 1}",
                ),
                config=config,
            )
        )

    ranked = rank_single_objective_candidates(archive)
    if not ranked:
        raise RuntimeError("No feasible candidates were found in phase 1.")

    seed_candidates = ranked[: profile.phase2_seed_count]
    for seed_idx, candidate in enumerate(seed_candidates):
        algo2 = pg.algorithm(pg.sade(gen=profile.phase2_generations))
        algo2.set_verbosity(0)
        algo2.set_seed(base_seed + 20_000 + seed_idx)
        pop = pg.population(problem, size=profile.phase2_pop_size, seed=base_seed + 30_000 + seed_idx)
        pop.set_x(0, np.array(candidate["x"], dtype=float))
        pop = algo2.evolve(pop)
        archive.extend(
            apply_mission_filters(
                collect_population_candidates(
                    problem=problem,
                    population=pop,
                    stage=f"phase2_seed_{seed_idx + 1}",
                ),
                config=config,
            )
        )

    ranked = rank_single_objective_candidates(archive)
    polished_inputs = ranked[: profile.phase3_candidate_count]
    for polish_idx, candidate in enumerate(polished_inputs):
        algo3 = pg.algorithm(
            pg.compass_search(
                max_fevals=profile.phase3_compass_fevals,
                start_range=0.01,
                stop_range=1e-8,
            )
        )
        algo3.set_verbosity(0)
        pop = pg.population(problem, size=1, seed=base_seed + 50_000 + polish_idx)
        pop.set_x(0, np.array(candidate["x"], dtype=float))
        pop = algo3.evolve(pop)
        archive.extend(
            apply_mission_filters(
                collect_population_candidates(
                    problem=problem,
                    population=pop,
                    stage=f"phase3_polish_{polish_idx + 1}",
                ),
                config=config,
            )
        )

    deduped = deduplicate_candidates(archive)
    ranked = rank_single_objective_candidates(deduped)
    if not ranked:
        raise RuntimeError("No candidates satisfied the configured mission filters.")
    top_candidates = ranked[: min(config.top_n, profile.archive_top_n)]
    run_elapsed = time.perf_counter() - run_started

    return {
        "started_at_utc": started_at_utc,
        "runtime_seconds": run_elapsed,
        "base_seed": base_seed,
        "archive": deduped,
        "top_candidates": top_candidates,
        "best_candidate": top_candidates[0],
    }


def collect_archipelago_candidates(problem: pg.problem, archipelago: pg.archipelago, stage: str) -> list[dict]:
    candidates = []
    for island_idx, island in enumerate(archipelago):
        population = island.get_population()
        candidates.extend(
            collect_population_candidates(
                problem=problem,
                population=population,
                stage=stage,
                island_index=island_idx,
            )
        )
    return candidates


def collect_population_candidates(
    problem: pg.problem,
    population: pg.population,
    stage: str,
    island_index: int | None = None,
) -> list[dict]:
    xs = population.get_x()
    fs = population.get_f()
    candidates = []
    for idx, (x_vec, f_vec) in enumerate(zip(xs, fs, strict=True)):
        x_arr = np.array(x_vec, dtype=float)
        f_arr = np.array(f_vec, dtype=float)
        candidates.append(
            {
                "stage": stage,
                "island_index": island_index,
                "population_index": idx,
                "feasible": bool(problem.feasibility_x(x_arr)),
                "mission_feasible": True,
                "x": [float(val) for val in x_arr.tolist()],
                "f": [float(val) for val in f_arr.tolist()],
                "objective": float(f_arr[0]),
            }
        )
    return candidates


def deduplicate_candidates(candidates: list[dict], ndigits: int = 9) -> list[dict]:
    best_by_key: dict[tuple, dict] = {}
    for candidate in candidates:
        key = tuple(round(val, ndigits) for val in candidate["x"])
        incumbent = best_by_key.get(key)
        if incumbent is None or candidate["objective"] < incumbent["objective"]:
            best_by_key[key] = candidate
    return list(best_by_key.values())


def rank_single_objective_candidates(candidates: list[dict]) -> list[dict]:
    feasible = [
        candidate
        for candidate in candidates
        if candidate["feasible"] and candidate.get("mission_feasible", True)
    ]
    return sorted(feasible, key=lambda item: item["objective"])


def apply_mission_filters(candidates: list[dict], config: SequenceConfig) -> list[dict]:
    n_legs = len(config.bodies) - 1
    filtered = []
    for candidate in candidates:
        parsed = parse_decision_vector(np.array(candidate["x"], dtype=float), n_legs)
        total_tof_days = float(sum(parsed["tofs_days"]))
        candidate["total_tof_days"] = total_tof_days
        candidate["c3_kms2"] = (parsed["vinf_dep_m_s"] / 1000.0) ** 2
        candidate["mission_feasible"] = (
            total_tof_days <= config.max_total_tof_days + 1e-9
            and candidate["c3_kms2"] <= config.max_c3_kms2 + 1e-9
        )
        filtered.append(candidate)
    return filtered


def save_search_results(
    run_dir: Path,
    config: SequenceConfig,
    profile: ComputeProfile,
    search_result: dict,
    udp,
    sequence: list,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    details_dir = run_dir / "candidate_details"
    details_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "compute_profile.json", profile.to_dict())
    write_json(
        run_dir / "run_summary.json",
        {
            "started_at_utc": search_result["started_at_utc"],
            "runtime_seconds": search_result["runtime_seconds"],
            "base_seed": search_result["base_seed"],
            "archive_size": len(search_result["archive"]),
            "top_candidate_count": len(search_result["top_candidates"]),
            "best_objective_m_s": search_result["best_candidate"]["objective"],
        },
    )
    write_jsonl(run_dir / "all_candidates.jsonl", search_result["archive"])
    write_jsonl(run_dir / "top_candidates.jsonl", search_result["top_candidates"])

    top_summaries = []
    for rank_idx, candidate in enumerate(search_result["top_candidates"], start=1):
        detail = reconstruct_candidate(
            udp=udp,
            sequence=sequence,
            config=config,
            x=np.array(candidate["x"], dtype=float),
        )
        detail["rank"] = rank_idx
        detail["objective_m_s"] = candidate["objective"]
        detail_path = details_dir / f"rank_{rank_idx:03d}.json"
        write_json(detail_path, detail)
        top_summaries.append(
            {
                "rank": rank_idx,
                "objective_m_s": candidate["objective"],
                "launch_epoch": detail["summary"]["launch_epoch"],
                "arrival_epoch": detail["summary"]["arrival_epoch"],
                "total_tof_years": detail["summary"]["total_tof_years"],
                "c3_kms2": detail["summary"]["c3_kms2"],
                "total_dsm_kms": detail["summary"]["total_dsm_kms"],
                "arrival_vinf_kms": detail["summary"]["arrival_vinf_kms"],
                "detail_file": str(detail_path.name),
            }
        )
    write_json(run_dir / "top_candidate_summaries.json", top_summaries)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(to_jsonable(row)))
            stream.write("\n")


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
