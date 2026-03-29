from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pykep as pk


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAJECTORIES_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = TRAJECTORIES_DIR / "selected_candidates.json"
BODY_COLORS = {
    "earth": "#4cc9f0",
    "venus": "#f4a261",
}
TRAJECTORY_COLOR = "#0f172a"
TEXT_COLOR = "#1f2937"


def outlined_text(ax: plt.Axes, x: float, y: float, text: str, *, color: str, fontsize: int, ha: str, va: str) -> None:
    artist = ax.text(x, y, text, color=color, fontsize=fontsize, ha=ha, va=va, clip_on=False)
    artist.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.95)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a three-panel flyby analysis figure.")
    parser.add_argument("--candidate-id", required=True, help="Candidate key from selected_candidates.json.")
    return parser.parse_args()


def load_candidate(candidate_id: str) -> tuple[dict, dict[str, str]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if candidate_id not in manifest:
        raise KeyError(f"Unknown candidate id: {candidate_id}")
    entry = manifest[candidate_id]
    candidate_path = (REPO_ROOT / entry["candidate_json"]).resolve()
    return json.loads(candidate_path.read_text()), entry


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def flyby_title(event: dict, earth_count: int) -> str:
    body = str(event["body"]).lower()
    if body == "venus":
        return "Venus Flyby"
    return f"Earth Flyby {earth_count}"


def hyperbola_points(rp_m: float, vinf_m_s: float, mu: float, radius_m: float) -> tuple[np.ndarray, float, float, float]:
    e = 1.0 + rp_m * vinf_m_s**2 / mu
    p = rp_m * (1.0 + e)
    nu_inf = math.acos(-1.0 / e)
    r_limit = max(11.0 * radius_m, 8.0 * rp_m)
    cos_nu_limit = np.clip((p / r_limit - 1.0) / e, -1.0, 1.0)
    nu_limit = min(nu_inf - 1e-4, math.acos(float(cos_nu_limit)))
    nus = np.linspace(-nu_limit, nu_limit, 900)
    rs = p / (1.0 + e * np.cos(nus))
    xy_km = np.column_stack((rs * np.cos(nus), rs * np.sin(nus))) / 1000.0
    return xy_km, e, p, nu_inf


def describe_side(periapsis_hat: np.ndarray, sun_hat: np.ndarray) -> tuple[str, float]:
    cosine = float(np.clip(np.dot(periapsis_hat, sun_hat), -1.0, 1.0))
    phase_deg = math.degrees(math.acos(cosine))
    side = "Light-side" if cosine > 0.0 else "Dark-side"
    return side, phase_deg


def compute_flyby_metrics(event: dict) -> dict[str, float | str | np.ndarray]:
    body_name = str(event["body"]).lower()
    body = pk.planet.jpl_lp(body_name)
    mu = float(body.mu_self)
    radius_m = float(body.radius)

    planet_velocity = np.array(event["planet_velocity_m_s"], dtype=float)
    heliocentric_in = np.array(event["spacecraft_velocity_in_m_s"], dtype=float)
    heliocentric_out = np.array(event["spacecraft_velocity_out_m_s"], dtype=float)
    vinf_in = heliocentric_in - planet_velocity
    vinf_out = heliocentric_out - planet_velocity
    vinf_mag = 0.5 * (np.linalg.norm(vinf_in) + np.linalg.norm(vinf_out))
    vinf_in_hat = unit(vinf_in)
    vinf_out_hat = unit(vinf_out)

    turn_angle_deg = math.degrees(math.acos(float(np.clip(np.dot(vinf_in_hat, vinf_out_hat), -1.0, 1.0))))
    rp_m = float(event["flyby_radius_m"])
    altitude_km = float(event["flyby_altitude_km"])
    xy_km, eccentricity, _, _ = hyperbola_points(rp_m, vinf_mag, mu, radius_m)
    b_mag_km = mu / vinf_mag**2 * math.sqrt(eccentricity**2 - 1.0) / 1000.0

    periapsis_hat = unit(vinf_in_hat - vinf_out_hat)
    transverse_hat = unit(vinf_in_hat + vinf_out_hat)
    plane_normal_hat = unit(np.cross(periapsis_hat, transverse_hat))
    sun_hat = unit(-np.array(event["planet_position_m"], dtype=float))
    sun_proj = sun_hat - np.dot(sun_hat, plane_normal_hat) * plane_normal_hat
    sun_proj_norm = np.linalg.norm(sun_proj)
    sun_plane_xy = None if sun_proj_norm < 1e-8 else np.array(
        [np.dot(unit(sun_proj), periapsis_hat), np.dot(unit(sun_proj), transverse_hat)],
        dtype=float,
    )
    side_label, phase_deg = describe_side(periapsis_hat, sun_hat)

    heliocentric_speed_in = np.linalg.norm(heliocentric_in) / 1000.0
    heliocentric_speed_out = np.linalg.norm(heliocentric_out) / 1000.0
    delta_heliocentric_speed = heliocentric_speed_out - heliocentric_speed_in
    delta_heliocentric_energy = (
        (np.linalg.norm(heliocentric_out) ** 2 - np.linalg.norm(heliocentric_in) ** 2) / 2.0 / 1.0e6
    )

    return {
        "body_name": body_name,
        "body_radius_km": radius_m / 1000.0,
        "xy_km": xy_km,
        "vinf_kms": vinf_mag / 1000.0,
        "turn_angle_deg": turn_angle_deg,
        "eccentricity": eccentricity,
        "b_mag_km": b_mag_km,
        "altitude_km": altitude_km,
        "side_label": side_label,
        "phase_deg": phase_deg,
        "sun_plane_xy": sun_plane_xy,
        "delta_heliocentric_speed": delta_heliocentric_speed,
        "delta_heliocentric_energy": delta_heliocentric_energy,
    }


def add_direction_arrow(ax: plt.Axes, xy_km: np.ndarray, start_frac: float, end_frac: float) -> None:
    start_idx = int(start_frac * (len(xy_km) - 1))
    end_idx = int(end_frac * (len(xy_km) - 1))
    start = xy_km[start_idx]
    end = xy_km[end_idx]
    ax.annotate(
        "",
        xy=(end[0], end[1]),
        xytext=(start[0], start[1]),
        arrowprops={"arrowstyle": "-|>", "color": TRAJECTORY_COLOR, "lw": 1.2, "mutation_scale": 12},
    )


def render_subplot(ax: plt.Axes, text_ax: plt.Axes, title: str, metrics: dict[str, float | str | np.ndarray]) -> None:
    body_name = str(metrics["body_name"])
    body_color = BODY_COLORS[body_name]
    xy_km = np.asarray(metrics["xy_km"])
    body_radius_km = float(metrics["body_radius_km"])

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d4d8dd", linewidth=0.6, alpha=0.7)
    ax.set_facecolor("white")

    ax.plot(xy_km[:, 0], xy_km[:, 1], color=TRAJECTORY_COLOR, linewidth=1.7, zorder=5)
    add_direction_arrow(ax, xy_km, 0.10, 0.18)
    add_direction_arrow(ax, xy_km, 0.82, 0.90)

    planet = patches.Circle((0.0, 0.0), body_radius_km, facecolor=body_color, edgecolor="#334155", linewidth=1.0, zorder=8)
    ax.add_patch(planet)

    sun_plane_xy = metrics["sun_plane_xy"]
    if sun_plane_xy is not None:
        axis_lim_guess = float(np.max(np.abs(xy_km))) * 1.05
        sun_dir = unit(np.asarray(sun_plane_xy))
        sun_end = sun_dir * axis_lim_guess * 0.78
        ax.annotate(
            "",
            xy=(sun_end[0], sun_end[1]),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "->", "color": "#b45309", "lw": 1.0},
        )
        label_pos = sun_dir * axis_lim_guess * 0.86
        outlined_text(
            ax,
            float(label_pos[0]),
            float(label_pos[1]),
            "to Sun",
            color="#b45309",
            fontsize=8,
            ha="left" if label_pos[0] >= 0 else "right",
            va="center",
        )

    axis_limit_km = max(float(np.max(np.abs(xy_km))), body_radius_km * 1.8) * 1.08
    ax.set_xlim(-axis_limit_km, axis_limit_km)
    ax.set_ylim(-axis_limit_km, axis_limit_km)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.tick_params(labelsize=8)

    text_ax.axis("off")
    text_lines = [
        f"Altitude: {metrics['altitude_km']:.0f} km",
        f"Vinf: {metrics['vinf_kms']:.3f} km/s",
        f"Turn angle: {metrics['turn_angle_deg']:.2f} deg",
        f"B-plane |B|: {metrics['b_mag_km']:.0f} km",
        f"Hyperbola e: {metrics['eccentricity']:.3f}",
        f"Periapsis side: {metrics['side_label']} (sun angle {metrics['phase_deg']:.1f} deg)",
        f"Heliocentric speed change: {metrics['delta_heliocentric_speed']:+.3f} km/s",
        f"Heliocentric energy change: {metrics['delta_heliocentric_energy']:+.1f} MJ/kg",
    ]
    text_ax.text(
        0.0,
        1.0,
        "\n".join(text_lines),
        ha="left",
        va="top",
        fontsize=9,
        color=TEXT_COLOR,
        family="monospace",
        linespacing=1.35,
    )


def main() -> None:
    args = parse_args()
    candidate, _entry = load_candidate(args.candidate_id)
    flyby_events = [event for event in candidate["events"] if event["type"] == "flyby"]
    if len(flyby_events) != 3:
        raise RuntimeError(f"Expected 3 flybys for this figure, found {len(flyby_events)}.")

    output_dir = TRAJECTORIES_DIR / args.candidate_id
    output_dir.mkdir(parents=True, exist_ok=True)

    titles = []
    earth_count = 0
    for event in flyby_events:
        if str(event["body"]).lower() == "earth":
            earth_count += 1
            titles.append(flyby_title(event, earth_count))
        else:
            titles.append(flyby_title(event, earth_count))

    metrics = [compute_flyby_metrics(event) for event in flyby_events]

    fig = plt.figure(figsize=(16, 8.8), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[4.3, 1.85], hspace=0.12, wspace=0.26)
    for col, (title, flyby_metrics) in enumerate(zip(titles, metrics, strict=True)):
        ax_plot = fig.add_subplot(grid[0, col])
        ax_text = fig.add_subplot(grid[1, col])
        render_subplot(ax_plot, ax_text, title, flyby_metrics)

    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.06, top=0.93, wspace=0.26, hspace=0.10)
    output_path = output_dir / "flyby_panels.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
