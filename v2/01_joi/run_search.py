from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from joi.config import build_search_profile, load_sequence_config
from joi.optimize import run_single_objective_search, save_search_results
from joi.problem import build_problem, ensure_results_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Earth-to-Jupiter JOI search runner.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML sequence config, for example configs/veega.toml",
    )
    parser.add_argument(
        "--compute-level",
        type=int,
        default=2,
        help="Integer from 1 to 10. Lower is quick local smoke-test, higher is heavier search.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where run folders will be created.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible searches.",
    )
    return parser.parse_args()


def run_search(
    *,
    config_path: Path,
    compute_level: int,
    results_dir: str | Path,
    seed: int,
    git_metadata: dict[str, str | bool | None] | None = None,
    progress_callback=None,
) -> tuple[Path, dict]:
    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parents[1]
    config = load_sequence_config(config_path.resolve())
    profile = build_search_profile(config, compute_level)

    problem, udp, sequence = build_problem(config)
    search_result = run_single_objective_search(
        problem=problem,
        udp=udp,
        sequence=sequence,
        config=config,
        profile=profile,
        base_seed=seed,
        progress_callback=progress_callback,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results_path = Path(results_dir)
    if results_path.is_absolute():
        results_base = results_path
    elif results_path.parts[:1] == ("v2",):
        results_base = repo_root / results_path
    else:
        results_base = base_dir / results_path
    run_dir = ensure_results_dir(results_base) / f"{timestamp}_{config.name}_L{profile.level}"
    save_search_results(
        run_dir=run_dir,
        config=config,
        profile=profile,
        search_result=search_result,
        udp=udp,
        sequence=sequence,
        source_root=base_dir,
        git_metadata=git_metadata,
    )
    return run_dir, search_result


def print_run_summary(run_dir: Path, search_result: dict, seed: int) -> None:
    best = search_result["best_candidate"]
    print(f"Saved run to: {run_dir}")
    print(f"Best objective: {best['objective'] / 1000.0:.4f} km/s")
    print(f"Saved top candidates: {len(search_result['top_candidates'])}")
    print(f"Saved total archive size: {len(search_result['archive'])}")
    print(f"Base seed: {seed}")


def main() -> None:
    args = parse_args()
    run_dir, search_result = run_search(
        config_path=Path(args.config),
        compute_level=args.compute_level,
        results_dir=args.results_dir,
        seed=args.seed,
    )

    print_run_summary(run_dir, search_result, args.seed)


if __name__ == "__main__":
    main()
