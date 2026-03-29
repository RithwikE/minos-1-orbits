from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pykep as pk


AU_METERS = float(pk.AU)
REPO_ROOT = Path(__file__).resolve().parents[3]
TRAJECTORIES_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = TRAJECTORIES_DIR / "selected_candidates.json"
BODY_STYLE = {
    "sun": {"color": "#ffd166", "size": 210},
    "earth": {"color": "#4cc9f0", "size": 42},
    "venus": {"color": "#f4a261", "size": 34},
    "jupiter": {"color": "#e76f51", "size": 68},
}
SEGMENT_COLORS = [
    "#e63946",
    "#f4a261",
    "#2a9d8f",
    "#4361ee",
]


def short_sequence_label(label: str) -> str:
    parts = label.split(" ", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a static 3D chosen-trajectory plot.")
    parser.add_argument("--candidate-id", required=True, help="Candidate key from selected_candidates.json.")
    return parser.parse_args()


def load_candidate(candidate_id: str) -> tuple[dict, dict[str, str], Path]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if candidate_id not in manifest:
        raise KeyError(f"Unknown candidate id: {candidate_id}")
    entry = manifest[candidate_id]
    candidate_path = (REPO_ROOT / entry["candidate_json"]).resolve()
    candidate = json.loads(candidate_path.read_text())
    return candidate, entry, candidate_path


def to_au(vectors_m: np.ndarray) -> np.ndarray:
    return vectors_m / AU_METERS


def set_equal_limits(
    ax,
    points_au: np.ndarray,
    *,
    pad_fraction: float = 0.03,
) -> None:
    mins = points_au.min(axis=0)
    maxs = points_au.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    radius = 0.5 * span * (1.0 + pad_fraction)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.22)
    except TypeError:
        ax.set_box_aspect((1.0, 1.0, 1.0))


def find_dense_index(dense_epochs: np.ndarray, target_epoch: float) -> int:
    return int(np.argmin(np.abs(dense_epochs - target_epoch)))


def build_leg_segments(candidate: dict, dense_points_au: np.ndarray, dense_epochs: np.ndarray) -> list[dict]:
    events = candidate["events"]
    encounter_events = [event for event in events if event["type"] in {"departure", "flyby", "arrival"}]
    segments: list[dict] = []
    for idx in range(len(encounter_events) - 1):
        start_event = encounter_events[idx]
        end_event = encounter_events[idx + 1]
        start_idx = find_dense_index(dense_epochs, float(start_event["epoch_mjd2000"]))
        end_idx = find_dense_index(dense_epochs, float(end_event["epoch_mjd2000"]))
        if end_idx <= start_idx:
            continue
        start_label = "Earth" if start_event["type"] == "departure" else str(start_event.get("body") or "").title()
        end_label = "Jupiter" if end_event["type"] == "arrival" else str(end_event.get("body") or "").title()
        segments.append(
            {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "points": dense_points_au[start_idx : end_idx + 1],
                "label": f"{start_label} -> {end_label}",
            }
        )
    return segments


def label_with_outline(ax, x: float, y: float, z: float, text: str, color: str, fontsize: int) -> None:
    label = ax.text(x, y, z, text, color=color, fontsize=fontsize)
    label.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)])


def event_label_positions(candidate: dict) -> dict[float, np.ndarray]:
    offsets: dict[float, np.ndarray] = {}
    earth_flyby_count = 0
    dsm_count = 0
    for event in candidate["events"]:
        epoch = float(event["epoch_mjd2000"])
        event_type = str(event["type"])
        if event_type == "departure":
            offsets[epoch] = np.array([0.09, -0.02, 0.03])
        elif event_type == "arrival":
            offsets[epoch] = np.array([0.10, 0.03, 0.03])
        elif event_type == "flyby":
            body = str(event.get("body") or "")
            if body == "venus":
                offsets[epoch] = np.array([0.08, -0.06, 0.05])
            elif body == "earth":
                earth_flyby_count += 1
                offsets[epoch] = (
                    np.array([-0.16, 0.03, 0.06]) if earth_flyby_count == 1 else np.array([0.06, -0.10, 0.05])
                )
        elif event_type == "dsm":
            dsm_count += 1
            offsets[epoch] = (
                np.array([0.07, 0.03, 0.04])
                if dsm_count in {1, 2}
                else np.array([0.05, -0.04, 0.03])
            )
    return offsets


def main() -> None:
    args = parse_args()
    candidate, entry, _ = load_candidate(args.candidate_id)
    output_dir = TRAJECTORIES_DIR / args.candidate_id
    output_dir.mkdir(parents=True, exist_ok=True)

    dense = candidate["dense_samples"]
    dense_points_au = to_au(np.array([sample["position_m"] for sample in dense], dtype=float))
    dense_epochs = np.array([sample["epoch_mjd2000"] for sample in dense], dtype=float)
    events = candidate["events"]
    segments = build_leg_segments(candidate, dense_points_au, dense_epochs)
    label_offsets = event_label_positions(candidate)

    fig = plt.figure(figsize=(13.5, 8.6), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.grid(True, color="#d7dce2", linewidth=0.7, alpha=0.8)
    ax.xaxis.pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
    ax.yaxis.pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
    ax.zaxis.pane.set_facecolor((0.97, 0.98, 0.99, 1.0))

    for segment_idx, segment in enumerate(segments):
        color = SEGMENT_COLORS[segment_idx % len(SEGMENT_COLORS)]
        points = segment["points"]
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=1.55, label=segment["label"])

    all_points = [dense_points_au, np.zeros((1, 3))]
    for event in events:
        body = event.get("body")
        if body:
            point = np.array(event["planet_position_m"], dtype=float) / AU_METERS
            all_points.append(point.reshape(1, 3))
    set_equal_limits(ax, np.vstack(all_points))

    ax.scatter([0], [0], [0], color=BODY_STYLE["sun"]["color"], s=BODY_STYLE["sun"]["size"], marker="*", zorder=12)
    label_with_outline(ax, 0.0, 0.0, 0.0, " Sun", "#9a6b00", 10)

    dsm_counter = 0
    earth_flyby_counter = 0
    for event in events:
        point = np.array(event["spacecraft_position_m"], dtype=float) / AU_METERS
        event_type = str(event["type"])
        if event_type == "dsm":
            dsm_counter += 1
            ax.scatter(point[0], point[1], point[2], color="#111111", edgecolor="white", linewidth=0.6, s=24, zorder=11)
            dx, dy, dz = label_offsets.get(float(event["epoch_mjd2000"]), np.array([0.04, 0.02, 0.02]))
            label_with_outline(ax, point[0] + dx, point[1] + dy, point[2] + dz, f"DSM {dsm_counter}", "#111111", 8)
            continue
        body = str(event.get("body") or "")
        if body:
            style = BODY_STYLE[body]
            planet_point = np.array(event["planet_position_m"], dtype=float) / AU_METERS
            ax.scatter(planet_point[0], planet_point[1], planet_point[2], color=style["color"], s=style["size"], zorder=10)
            if event_type == "departure":
                label = "Launch Earth"
            elif event_type == "arrival":
                label = "Jupiter Arrival"
            elif body == "earth":
                earth_flyby_counter += 1
                label = f"Earth Flyby {earth_flyby_counter}"
            else:
                label = f"{body.title()} Flyby"
            dx, dy, dz = label_offsets.get(float(event["epoch_mjd2000"]), np.array([0.04, 0.02, 0.02]))
            label_with_outline(
                ax,
                planet_point[0] + dx,
                planet_point[1] + dy,
                planet_point[2] + dz,
                label,
                style["color"],
                9,
            )

    summary = candidate["summary"]
    ax.set_xlabel("X (AU)", labelpad=12)
    ax.set_ylabel("Y (AU)", labelpad=12)
    ax.set_zlabel("Z (AU)", labelpad=12)
    ax.view_init(elev=50, azim=30)
    ax.set_title(f"{short_sequence_label(summary['sequence'])} | {entry['id']}", pad=14)

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.96),
        fontsize=8,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
    )
    for handle in legend.legend_handles:
        handle.set_linewidth(3.0)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.93)
    output_path = output_dir / "static_3d_plot.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
