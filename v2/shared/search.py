from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pygmo as pg

from .runtime import get_git_metadata


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

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


ProgressCallback = Callable[[dict[str, Any]], None]


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


def override_compute_profile(profile: ComputeProfile, **overrides: int | None) -> ComputeProfile:
    payload = profile.to_dict()
    payload["level"] = profile.level
    for key, value in overrides.items():
        if value is not None:
            payload[key] = int(value)
    return ComputeProfile(**payload)


def run_single_objective_search(
    *,
    problem: pg.problem,
    profile: ComputeProfile,
    base_seed: int,
    multi_objective: bool,
    top_n: int,
    mission_filter: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    initial_candidates: list[dict[str, Any]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if multi_objective:
        raise NotImplementedError(
            "The first v2 runner only supports single-objective mode. "
            "Use archived candidate comparison for secondary metrics for now."
        )

    started_at_utc = datetime.now(UTC).isoformat()
    run_started = time.perf_counter()
    archive: list[dict[str, Any]] = []
    phase_fevals = {
        "phase1": 0,
        "phase2": 0,
        "phase3": 0,
    }
    progress_events: list[dict[str, Any]] = []

    emit_progress(
        progress_events,
        progress_callback,
        started_at_perf=run_started,
        archive=archive,
        phase_fevals=phase_fevals,
        event="search_started",
        profile=profile,
    )

    if initial_candidates:
        archive.extend(mission_filter(initial_candidates))
        emit_progress(
            progress_events,
            progress_callback,
            started_at_perf=run_started,
            archive=archive,
            phase_fevals=phase_fevals,
            event="seed_candidates_loaded",
            profile=profile,
            completed=len(initial_candidates),
            total=len(initial_candidates),
        )

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
            mission_filter(
                collect_archipelago_candidates(
                    problem=problem,
                    archipelago=archipelago,
                    stage=f"phase1_round_{round_idx + 1}",
                )
            )
        )
        phase_fevals["phase1"] = sum(island.get_population().problem.get_fevals() for island in archipelago)
        emit_progress(
            progress_events,
            progress_callback,
            started_at_perf=run_started,
            archive=archive,
            phase_fevals=phase_fevals,
            event="phase1_round_complete",
            profile=profile,
            completed=round_idx + 1,
            total=profile.phase1_rounds,
        )
    phase_fevals["phase1"] = sum(island.get_population().problem.get_fevals() for island in archipelago)

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
        phase_fevals["phase2"] += pop.problem.get_fevals()
        archive.extend(
            mission_filter(
                collect_population_candidates(
                    problem=problem,
                    population=pop,
                    stage=f"phase2_seed_{seed_idx + 1}",
                )
            )
        )
        if should_emit_progress(step_index=seed_idx, total=profile.phase2_seed_count, frequency=profile.log_frequency):
            emit_progress(
                progress_events,
                progress_callback,
                started_at_perf=run_started,
                archive=archive,
                phase_fevals=phase_fevals,
                event="phase2_seed_complete",
                profile=profile,
                completed=seed_idx + 1,
                total=profile.phase2_seed_count,
            )

    ranked = rank_single_objective_candidates(archive)
    polished_inputs = ranked[: profile.phase3_candidate_count]
    emit_progress(
        progress_events,
        progress_callback,
        started_at_perf=run_started,
        archive=archive,
        phase_fevals=phase_fevals,
        event="phase2_complete",
        profile=profile,
        completed=len(seed_candidates),
        total=profile.phase2_seed_count,
    )
    for polish_idx, candidate in enumerate(polished_inputs):
        local_algo = pg.compass_search(
            max_fevals=profile.phase3_compass_fevals,
            start_range=0.01,
            stop_range=1e-6,
        )
        algo3 = pg.algorithm(
            pg.mbh(
                algo=local_algo,
                stop=3 + profile.level,
                perturb=0.05,
                seed=base_seed + 40_000 + polish_idx,
            )
        )
        algo3.set_verbosity(0)
        algo3.set_seed(base_seed + 45_000 + polish_idx)
        pop = pg.population(problem, size=1, seed=base_seed + 50_000 + polish_idx)
        pop.set_x(0, np.array(candidate["x"], dtype=float))
        pop = algo3.evolve(pop)
        phase_fevals["phase3"] += pop.problem.get_fevals()
        archive.extend(
            mission_filter(
                collect_population_candidates(
                    problem=problem,
                    population=pop,
                    stage=f"phase3_polish_{polish_idx + 1}",
                )
            )
        )
        if should_emit_progress(
            step_index=polish_idx,
            total=len(polished_inputs),
            frequency=profile.log_frequency,
        ):
            emit_progress(
                progress_events,
                progress_callback,
                started_at_perf=run_started,
                archive=archive,
                phase_fevals=phase_fevals,
                event="phase3_polish_complete",
                profile=profile,
                completed=polish_idx + 1,
                total=len(polished_inputs),
            )

    deduped = deduplicate_candidates(archive)
    ranked = rank_single_objective_candidates(deduped)
    if not ranked:
        raise RuntimeError("No candidates satisfied the configured mission filters.")
    top_candidates = ranked[: min(top_n, profile.archive_top_n)]
    run_elapsed = time.perf_counter() - run_started
    emit_progress(
        progress_events,
        progress_callback,
        started_at_perf=run_started,
        archive=deduped,
        phase_fevals=phase_fevals,
        event="search_complete",
        profile=profile,
        completed=len(top_candidates),
        total=min(top_n, profile.archive_top_n),
    )

    return {
        "started_at_utc": started_at_utc,
        "runtime_seconds": run_elapsed,
        "base_seed": base_seed,
        "progress_events": progress_events,
        "evaluation_counts": {
            **phase_fevals,
            "total": sum(phase_fevals.values()),
        },
        "archive": deduped,
        "top_candidates": top_candidates,
        "best_candidate": top_candidates[0],
    }


def should_emit_progress(*, step_index: int, total: int, frequency: int) -> bool:
    freq = max(1, int(frequency))
    return ((step_index + 1) % freq == 0) or (step_index + 1 == total)


def emit_progress(
    progress_events: list[dict[str, Any]],
    progress_callback: ProgressCallback | None,
    *,
    started_at_perf: float,
    archive: list[dict[str, Any]],
    phase_fevals: dict[str, int],
    event: str,
    profile: ComputeProfile,
    completed: int | None = None,
    total: int | None = None,
) -> None:
    ranked = rank_single_objective_candidates(archive)
    payload: dict[str, Any] = {
        "event": event,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": time.perf_counter() - started_at_perf,
        "archive_size": len(archive),
        "fevals": {
            **phase_fevals,
            "total": sum(phase_fevals.values()),
        },
        "profile_level": profile.level,
    }
    if completed is not None:
        payload["completed"] = int(completed)
    if total is not None:
        payload["total"] = int(total)
    if ranked:
        best = ranked[0]
        payload["best_objective_m_s"] = float(best["objective"])
        if "c3_kms2" in best:
            payload["best_c3_kms2"] = float(best["c3_kms2"])
        if "total_tof_days" in best:
            payload["best_total_tof_days"] = float(best["total_tof_days"])
    progress_events.append(payload)
    if progress_callback is not None:
        progress_callback(payload)


def collect_archipelago_candidates(problem: pg.problem, archipelago: pg.archipelago, stage: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
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
) -> list[dict[str, Any]]:
    xs = population.get_x()
    fs = population.get_f()
    candidates: list[dict[str, Any]] = []
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


def deduplicate_candidates(candidates: list[dict[str, Any]], ndigits: int = 9) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[float, ...], dict[str, Any]] = {}
    for candidate in candidates:
        key = tuple(round(val, ndigits) for val in candidate["x"])
        incumbent = best_by_key.get(key)
        if incumbent is None or candidate["objective"] < incumbent["objective"]:
            best_by_key[key] = candidate
    return list(best_by_key.values())


def rank_single_objective_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feasible = [
        candidate
        for candidate in candidates
        if candidate["feasible"] and candidate.get("mission_feasible", True)
    ]
    return sorted(feasible, key=lambda item: item["objective"])


def save_search_results(
    run_dir: Path,
    *,
    config_payload: dict[str, Any],
    profile: ComputeProfile,
    search_result: dict[str, Any],
    reconstruct_candidate: Callable[[dict[str, Any]], dict[str, Any]],
    source_root: Path | None = None,
    git_metadata: dict[str, str | bool | None] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    details_dir = run_dir / "candidate_details"
    details_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "config.json", config_payload)
    write_json(run_dir / "compute_profile.json", profile.to_dict())
    write_json(
        run_dir / "run_summary.json",
        {
            "started_at_utc": search_result["started_at_utc"],
            "runtime_seconds": search_result["runtime_seconds"],
            "base_seed": search_result["base_seed"],
            "evaluation_counts": search_result["evaluation_counts"],
            "archive_size": len(search_result["archive"]),
            "top_candidate_count": len(search_result["top_candidates"]),
            "best_objective_m_s": search_result["best_candidate"]["objective"],
            **(git_metadata or get_git_metadata(source_root)),
        },
    )
    write_json(run_dir / "progress_events.json", search_result.get("progress_events", []))
    write_jsonl(run_dir / "all_candidates.jsonl", search_result["archive"])
    write_jsonl(run_dir / "top_candidates.jsonl", search_result["top_candidates"])

    top_summaries: list[dict[str, Any]] = []
    for rank_idx, candidate in enumerate(search_result["top_candidates"], start=1):
        detail = reconstruct_candidate(candidate)
        detail["rank"] = rank_idx
        detail["objective_m_s"] = candidate["objective"]
        detail_path = details_dir / f"rank_{rank_idx:03d}.json"
        write_json(detail_path, detail)
        top_summaries.append(
            {
                "rank": rank_idx,
                "objective_m_s": candidate["objective"],
                "objective_total_dv_kms": detail["summary"]["objective_total_dv_kms"],
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


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(to_jsonable(row)))
            stream.write("\n")


def to_jsonable(value: Any) -> Any:
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
