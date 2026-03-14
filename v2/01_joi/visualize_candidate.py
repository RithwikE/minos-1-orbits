from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AU_METERS = 149597870700.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a saved JOI candidate archive file.")
    parser.add_argument(
        "--candidate-json",
        required=True,
        help="Path to a saved candidate detail JSON, e.g. results/.../candidate_details/rank_001.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots. Defaults to a sibling `plots` directory next to the candidate file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate_json).resolve()
    candidate = json.loads(candidate_path.read_text())

    if args.output_dir is None:
        output_dir = candidate_path.parent.parent / "plots"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_path = output_dir / f"{candidate_path.stem}_overview.png"
    orbital_path = output_dir / f"{candidate_path.stem}_orbital_elements.png"

    dense = candidate["dense_samples"]
    events = candidate["events"]
    summary = candidate["summary"]

    times_days = np.array(
        [sample["epoch_mjd2000"] - dense[0]["epoch_mjd2000"] for sample in dense],
        dtype=float,
    )
    positions_au = np.array(
        [[coord / AU_METERS for coord in sample["position_m"]] for sample in dense],
        dtype=float,
    )
    distances_au = np.array([sample["distance_to_sun_au"] for sample in dense], dtype=float)
    speeds_kms = np.array([sample["speed_kms"] for sample in dense], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax_xy, ax_xz, ax_dist, ax_speed = axes.flat

    ax_xy.plot(positions_au[:, 0], positions_au[:, 1], color="#005f73", linewidth=1.6)
    ax_xy.scatter([0], [0], color="gold", s=140, marker="*", zorder=5)
    ax_xy.set_xlabel("X (AU)")
    ax_xy.set_ylabel("Y (AU)")
    ax_xy.set_title("Heliocentric Trajectory: XY")
    ax_xy.set_aspect("equal", adjustable="box")

    ax_xz.plot(positions_au[:, 0], positions_au[:, 2], color="#0a9396", linewidth=1.6)
    ax_xz.scatter([0], [0], color="gold", s=140, marker="*", zorder=5)
    ax_xz.set_xlabel("X (AU)")
    ax_xz.set_ylabel("Z (AU)")
    ax_xz.set_title("Heliocentric Trajectory: XZ")
    ax_xz.set_aspect("equal", adjustable="box")

    ax_dist.plot(times_days, distances_au, color="#bb3e03", linewidth=1.6)
    ax_dist.set_xlabel("Days Since Launch")
    ax_dist.set_ylabel("Distance to Sun (AU)")
    ax_dist.set_title("Heliocentric Distance")

    ax_speed.plot(times_days, speeds_kms, color="#ae2012", linewidth=1.6)
    ax_speed.set_xlabel("Days Since Launch")
    ax_speed.set_ylabel("Speed (km/s)")
    ax_speed.set_title("Heliocentric Speed")

    for event in events:
        event_day = float(event["epoch_mjd2000"] - dense[0]["epoch_mjd2000"])
        event_position = np.array(event["spacecraft_position_m"], dtype=float) / AU_METERS
        label = f"{event['type']} {event['body'] or ''}".strip()
        ax_xy.scatter(event_position[0], event_position[1], s=20, color="#9b2226", zorder=6)
        ax_xz.scatter(event_position[0], event_position[2], s=20, color="#9b2226", zorder=6)
        ax_dist.axvline(event_day, color="#9b2226", alpha=0.15, linewidth=0.8)
        ax_speed.axvline(event_day, color="#9b2226", alpha=0.15, linewidth=0.8)
        if event["type"] in {"departure", "arrival"}:
            ax_xy.annotate(label, (event_position[0], event_position[1]), fontsize=8)

    fig.suptitle(
        f"{summary['sequence']} | TOF {summary['total_tof_years']:.2f} yr | "
        f"C3 {summary['c3_kms2']:.2f} km²/s² | DSM {summary['total_dsm_kms']:.3f} km/s | "
        f"V∞arr {summary['arrival_vinf_kms']:.3f} km/s",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(overview_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    orbital_keys = [
        ("semi_major_axis_m", "Semi-major axis (AU)", AU_METERS),
        ("eccentricity", "Eccentricity", 1.0),
        ("inclination_rad", "Inclination (deg)", np.pi / 180.0),
    ]
    for axis, (key, label, scale) in zip(axes, orbital_keys, strict=True):
        values = []
        for sample in dense:
            elements = sample.get("orbital_elements", {})
            raw = elements.get(key)
            if raw is None:
                values.append(np.nan)
            elif key == "semi_major_axis_m":
                values.append(raw / scale)
            elif key == "inclination_rad":
                values.append(raw / scale)
            else:
                values.append(raw)
        axis.plot(times_days, values, linewidth=1.5, color="#005f73")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Days Since Launch")
    fig.suptitle("Osculating Orbital Elements From Saved Dense Samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(orbital_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved overview plot: {overview_path}")
    print(f"Saved orbital element plot: {orbital_path}")


if __name__ == "__main__":
    main()
