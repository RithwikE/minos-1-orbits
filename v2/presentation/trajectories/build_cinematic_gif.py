from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pykep as pk


AU_METERS = float(pk.AU)
REPO_ROOT = Path(__file__).resolve().parents[3]
TRAJECTORIES_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = TRAJECTORIES_DIR / "selected_candidates.json"
BODY_STYLE = {
    "sun": {"color": "#ffd166", "size": 120},
    "earth": {"color": "#4cc9f0", "size": 28},
    "venus": {"color": "#f4a261", "size": 26},
    "jupiter": {"color": "#e76f51", "size": 46},
}


def short_sequence_label(label: str) -> str:
    parts = label.split(" ", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a cinematic chosen-trajectory GIF.")
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


def collect_planet_tracks(candidate: dict, epochs_mjd2000: np.ndarray) -> dict[str, np.ndarray]:
    bodies: list[str] = []
    for event in candidate["events"]:
        body = event.get("body")
        if body and body not in bodies:
            bodies.append(body)
    tracks: dict[str, np.ndarray] = {}
    for body_name in bodies:
        planet = pk.planet.jpl_lp(body_name)
        rows = []
        for epoch_mjd2000 in epochs_mjd2000:
            position, _ = planet.eph(pk.epoch(float(epoch_mjd2000), "mjd2000"))
            rows.append(np.array(position, dtype=float))
        tracks[body_name] = to_au(np.vstack(rows))
    return tracks


def build_frame_plan(epochs_mjd2000: np.ndarray) -> tuple[np.ndarray, list[int]]:
    target_count = 132
    frame_indices = np.linspace(0, len(epochs_mjd2000) - 1, target_count, dtype=int)
    frame_indices = np.unique(frame_indices)
    if frame_indices[-1] != len(epochs_mjd2000) - 1:
        frame_indices = np.append(frame_indices, len(epochs_mjd2000) - 1)
    durations = [180] * len(frame_indices)
    if durations:
        durations[-1] = 1800
    return frame_indices, durations


def nearest_event_status(candidate: dict, epoch_mjd2000: float) -> str:
    nearest = None
    nearest_dt = None
    for event in candidate["events"]:
        dt = abs(float(event["epoch_mjd2000"]) - epoch_mjd2000)
        if nearest_dt is None or dt < nearest_dt:
            nearest = event
            nearest_dt = dt
    if nearest is None or nearest_dt is None:
        return ""
    event_type = str(nearest["type"])
    body = nearest.get("body")
    if event_type == "flyby" and nearest_dt > 30.0:
        return ""
    if event_type == "dsm" and nearest_dt > 20.0:
        return ""
    if event_type in {"departure", "arrival"} and nearest_dt > 18.0:
        return ""
    if event_type == "dsm":
        return "DSM"
    if body:
        return f"{event_type.title()} at {body.title()}"
    return event_type.title()


def figure_to_image(fig: plt.Figure) -> Image.Image:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, facecolor="black")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("P", palette=Image.ADAPTIVE)


def add_axes_style(ax: plt.Axes) -> None:
    ax.set_facecolor("black")
    ax.tick_params(colors="#b9c2cf", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#75808f")
    ax.xaxis.label.set_color("#d6dde6")
    ax.yaxis.label.set_color("#d6dde6")
    ax.grid(True, color="#2a3240", linewidth=0.6, alpha=0.65)


def render_frame(
    *,
    candidate_id: str,
    candidate: dict,
    frame_index: int,
    spacecraft_xy: np.ndarray,
    planet_tracks: dict[str, np.ndarray],
    epochs_mjd2000: np.ndarray,
    full_span: float,
) -> Image.Image:
    summary = candidate["summary"]
    fig = plt.figure(figsize=(9.2, 7.2), facecolor="black")
    ax = fig.add_axes([0.08, 0.10, 0.78, 0.80])
    inset = fig.add_axes([0.72, 0.67, 0.22, 0.22])

    add_axes_style(ax)
    inset.set_facecolor("#050607")
    inset.set_aspect("equal", adjustable="box")
    inset.tick_params(colors="#9aa4b2", labelsize=7)
    for spine in inset.spines.values():
        spine.set_color("#666f7a")
    inset.grid(True, color="#242b36", linewidth=0.5, alpha=0.5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-full_span, full_span)
    ax.set_ylim(-full_span, full_span)
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")

    current_xy = spacecraft_xy[frame_index]
    mission_day = float(epochs_mjd2000[frame_index] - epochs_mjd2000[0])

    for body_name, track in planet_tracks.items():
        style = BODY_STYLE[body_name]
        current_body_xy = track[frame_index, :2]
        ax.plot(track[:, 0], track[:, 1], color=style["color"], alpha=0.14, linewidth=0.95)
        ax.scatter(current_body_xy[0], current_body_xy[1], color=style["color"], s=style["size"] * 2.8, alpha=0.14)
        ax.scatter(current_body_xy[0], current_body_xy[1], color=style["color"], s=style["size"], alpha=0.95)

    ax.scatter([0], [0], color=BODY_STYLE["sun"]["color"], s=700, alpha=0.08, zorder=3)
    ax.scatter([0], [0], color=BODY_STYLE["sun"]["color"], s=BODY_STYLE["sun"]["size"], alpha=0.96, zorder=4)

    ax.plot(spacecraft_xy[:, 0], spacecraft_xy[:, 1], color="#495362", linewidth=0.8, alpha=0.25)
    ax.plot(spacecraft_xy[: frame_index + 1, 0], spacecraft_xy[: frame_index + 1, 1], color="#ffffff", linewidth=1.15)
    ax.scatter(current_xy[0], current_xy[1], color="#ffffff", s=30, zorder=12)

    status = nearest_event_status(candidate, float(epochs_mjd2000[frame_index]))
    title = f"{short_sequence_label(summary['sequence'])} | {candidate_id}"
    metrics = f"DSM {summary['total_dsm_kms']:.3f} km/s | TOF {summary['total_tof_years']:.2f} yr | C3 {summary['c3_kms2']:.1f}"
    ax.text(0.02, 0.98, title, transform=ax.transAxes, color="white", fontsize=15, fontweight="bold", ha="left", va="top")
    ax.text(0.02, 0.935, metrics, transform=ax.transAxes, color="#c9d1d9", fontsize=10, ha="left", va="top")
    status_text = f"Mission day: {mission_day:.0f}" if not status else f"Status: {status}\nMission day: {mission_day:.0f}"
    ax.text(
        0.02,
        0.06,
        status_text,
        transform=ax.transAxes,
        color="#ffd166",
        fontsize=11,
        ha="left",
        va="bottom",
        bbox={"facecolor": "#0f1318", "edgecolor": "#444c56", "pad": 8},
    )

    recent = spacecraft_xy[max(0, frame_index - 48) : frame_index + 1]
    inset.plot(recent[:, 0], recent[:, 1], color="#ffffff", linewidth=1.35)
    inset.scatter(current_xy[0], current_xy[1], color="#ffffff", s=18, zorder=12)

    nearby_distances = []
    for body_name, track in planet_tracks.items():
        nearby_distances.append(float(np.linalg.norm(track[frame_index, :2] - current_xy)))
    zoom_radius = min(1.0, max(0.18, min(nearby_distances) * 2.8))
    inset.set_xlim(current_xy[0] - zoom_radius, current_xy[0] + zoom_radius)
    inset.set_ylim(current_xy[1] - zoom_radius, current_xy[1] + zoom_radius)
    inset.set_xlabel("")
    inset.set_ylabel("")

    for body_name, track in planet_tracks.items():
        style = BODY_STYLE[body_name]
        current_body_xy = track[frame_index, :2]
        if np.linalg.norm(current_body_xy - current_xy) > zoom_radius * 2.6:
            continue
        inset.scatter(current_body_xy[0], current_body_xy[1], color=style["color"], s=style["size"] * 1.35, zorder=10)
        inset.text(
            current_body_xy[0],
            current_body_xy[1],
            f" {body_name.title()}",
            color=style["color"],
            fontsize=8,
            ha="left",
            va="center",
            clip_on=True,
        )

    return figure_to_image(fig)


def main() -> None:
    args = parse_args()
    candidate, entry, _ = load_candidate(args.candidate_id)
    output_dir = TRAJECTORIES_DIR / args.candidate_id
    output_dir.mkdir(parents=True, exist_ok=True)

    dense = candidate["dense_samples"]
    spacecraft_positions_m = np.array([sample["position_m"] for sample in dense], dtype=float)
    spacecraft_xy = to_au(spacecraft_positions_m)[:, :2]
    epochs_mjd2000 = np.array([sample["epoch_mjd2000"] for sample in dense], dtype=float)
    planet_tracks = collect_planet_tracks(candidate, epochs_mjd2000)
    frame_indices, durations = build_frame_plan(epochs_mjd2000)

    max_extent = max(
        float(np.max(np.abs(spacecraft_xy))),
        max(float(np.max(np.abs(track[:, :2]))) for track in planet_tracks.values()),
    )
    full_span = max_extent * 1.18

    frames = [
        render_frame(
            candidate_id=entry["id"],
            candidate=candidate,
            frame_index=int(frame_index),
            spacecraft_xy=spacecraft_xy,
            planet_tracks=planet_tracks,
            epochs_mjd2000=epochs_mjd2000,
            full_span=full_span,
        )
        for frame_index in frame_indices
    ]

    output_path = output_dir / "cinematic_trajectory.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(output_path)


if __name__ == "__main__":
    main()
