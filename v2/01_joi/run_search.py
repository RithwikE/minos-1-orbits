from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from joi.config import build_compute_profile, load_sequence_config
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


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    base_dir = Path(__file__).resolve().parent
    config = load_sequence_config(config_path)
    profile = build_compute_profile(args.compute_level)

    problem, udp, sequence = build_problem(config)
    search_result = run_single_objective_search(
        problem=problem,
        udp=udp,
        sequence=sequence,
        config=config,
        profile=profile,
        base_seed=args.seed,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ensure_results_dir(base_dir / args.results_dir) / f"{timestamp}_{config.name}_L{profile.level}"
    save_search_results(
        run_dir=run_dir,
        config=config,
        profile=profile,
        search_result=search_result,
        udp=udp,
        sequence=sequence,
    )

    best = search_result["best_candidate"]
    print(f"Saved run to: {run_dir}")
    print(f"Best objective: {best['objective'] / 1000.0:.4f} km/s")
    print(f"Saved top candidates: {len(search_result['top_candidates'])}")
    print(f"Saved total archive size: {len(search_result['archive'])}")
    print(f"Base seed: {args.seed}")


if __name__ == "__main__":
    main()
