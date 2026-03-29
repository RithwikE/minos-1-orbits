from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pykep as pk


AU_METERS = float(pk.AU)
DAY2YEAR = float(pk.DAY2YEAR)
REPO_ROOT = Path(__file__).resolve().parents[3]
TRAJECTORIES_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = TRAJECTORIES_DIR / "selected_candidates.json"
HEADER_COLOR = "#b1935b"
ROW_A = "#f0ede7"
ROW_B = "#fbfaf8"
EDGE_COLOR = "#3b3428"
TEXT_COLOR = "#111111"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trajectory summary tables for a selected candidate.")
    parser.add_argument("--candidate-id", required=True, help="Candidate key from selected_candidates.json.")
    return parser.parse_args()


def load_candidate(candidate_id: str) -> tuple[dict, dict[str, str]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    if candidate_id not in manifest:
        raise KeyError(f"Unknown candidate id: {candidate_id}")
    entry = manifest[candidate_id]
    candidate_path = (REPO_ROOT / entry["candidate_json"]).resolve()
    return json.loads(candidate_path.read_text()), entry


def short_date(epoch_text: str) -> str:
    return datetime.strptime(epoch_text, "%Y-%b-%d %H:%M:%S.%f").strftime("%Y-%b-%d")


def wrap_deg(angle_rad: float) -> float:
    return float(np.degrees(angle_rad) % 360.0)


def orbital_elements_from_state(position_m: np.ndarray, velocity_m_s: np.ndarray) -> dict[str, float]:
    sma_m, ecc, inc_rad, raan_rad, argp_rad, _mean_anom_rad = pk.ic2par(position_m, velocity_m_s, pk.MU_SUN)
    return {
        "sma_au": float(sma_m / AU_METERS),
        "ecc": float(ecc),
        "raan_deg": wrap_deg(raan_rad),
        "incl_deg": wrap_deg(inc_rad),
        "argperi_deg": wrap_deg(argp_rad),
    }


def compact_encounter_name(event: dict, earth_flyby_ordinal: int) -> str:
    event_type = str(event["type"])
    body = str(event.get("body") or "")
    if event_type == "departure":
        return "Earth"
    if event_type == "arrival":
        return "Jupiter"
    if event_type == "flyby" and body == "venus":
        return "Venus"
    if event_type == "flyby" and body == "earth":
        return f"Earth {earth_flyby_ordinal}"
    raise ValueError(f"Unsupported encounter event for compact name: {event_type} {body}")


def event_name(event: dict, earth_flyby_count: int) -> str:
    event_type = str(event["type"])
    body = str(event.get("body") or "")
    if event_type == "departure":
        return "Earth Departure"
    if event_type == "arrival":
        return "Jupiter Arrival"
    if event_type == "flyby" and body == "venus":
        return "Venus Flyby"
    if event_type == "flyby" and body == "earth":
        return f"Earth Flyby {earth_flyby_count}"
    return event_type.title()


def classify_lambert_arc_type(
    start_position_m: np.ndarray,
    start_velocity_m_s: np.ndarray,
    end_position_m: np.ndarray,
    tof_days: float,
) -> str:
    """
    Lambert class naming follows Thorne's single-revolution convention:
    1/2 indicates transfer angle < or > pi, A/B indicates time below or above
    the minimum-energy elliptic time, and H indicates a hyperbolic arc.

    Sources:
    - J. D. Thorne, "Convergence Behavior of Series Solutions of the Lambert
      Problem", IDA Document NS D-5080, 2013.
    - D. Izzo, "Revisiting Lambert's problem", Celestial Mechanics and
      Dynamical Astronomy, 2015.
    """
    r1 = np.asarray(start_position_m, dtype=float)
    v1 = np.asarray(start_velocity_m_s, dtype=float)
    r2 = np.asarray(end_position_m, dtype=float)

    h_vec = np.cross(r1, v1)
    cross_r1_r2 = np.cross(r1, r2)
    cos_theta = np.clip(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)), -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if float(np.dot(h_vec, cross_r1_r2)) < 0.0:
        theta = float(2.0 * np.pi - theta)

    family = "1" if theta < np.pi else "2"

    specific_energy = float(np.dot(v1, v1) / 2.0 - pk.MU_SUN / np.linalg.norm(r1))
    if specific_energy >= 0.0:
        return f"{family}H"

    chord = float(np.linalg.norm(r2 - r1))
    semiperimeter = float((np.linalg.norm(r1) + np.linalg.norm(r2) + chord) / 2.0)
    lam_sq = max(0.0, 1.0 - chord / semiperimeter)
    lam = float(np.sqrt(lam_sq))
    if theta > np.pi:
        lam = -lam

    nondim_tof = float(np.sqrt(2.0 * pk.MU_SUN / semiperimeter**3) * (tof_days * pk.DAY2SEC))
    min_energy_tof = float(np.arccos(np.clip(lam, -1.0, 1.0)) + lam * np.sqrt(max(0.0, 1.0 - lam * lam)))
    branch = "A" if nondim_tof <= min_energy_tof + 1e-12 else "B"
    return f"{family}{branch}"


def build_leg_rows(candidate: dict) -> list[list[str]]:
    events = candidate["events"]
    earth_flyby_count = 0
    leg_rows: list[list[str]] = []
    dsm_index = 0

    for leg_index, leg in enumerate(candidate["legs"], start=1):
        start_event = events[2 * (leg_index - 1)]
        dsm_event = events[2 * (leg_index - 1) + 1]
        end_event = events[2 * leg_index]

        dsm_index += 1

        start_label = compact_encounter_name(start_event, earth_flyby_count)
        if end_event["type"] == "flyby" and str(end_event.get("body")) == "earth":
            end_label = compact_encounter_name(end_event, earth_flyby_count + 1)
        else:
            end_label = compact_encounter_name(end_event, earth_flyby_count)

        pre_position = np.array(start_event["spacecraft_position_m"], dtype=float)
        if start_event["type"] == "departure":
            pre_velocity = np.array(start_event["spacecraft_velocity_m_s"], dtype=float)
        else:
            pre_velocity = np.array(start_event["spacecraft_velocity_out_m_s"], dtype=float)
        pre_elements = orbital_elements_from_state(pre_position, pre_velocity)
        pre_arc_type = classify_lambert_arc_type(
            pre_position,
            pre_velocity,
            np.array(dsm_event["spacecraft_position_m"], dtype=float),
            float(leg["pre_dsm_duration_days"]),
        )
        leg_rows.append(
            [
                f"{start_label}-DSM {dsm_index}",
                pre_arc_type,
                f"{pre_elements['sma_au']:.4f}",
                f"{pre_elements['ecc']:.4f}",
                f"{pre_elements['raan_deg']:.3f}",
                f"{pre_elements['incl_deg']:.3f}",
                f"{pre_elements['argperi_deg']:.3f}",
            ]
        )

        post_position = np.array(dsm_event["spacecraft_position_m"], dtype=float)
        post_velocity = np.array(dsm_event["spacecraft_velocity_after_m_s"], dtype=float)
        post_elements = orbital_elements_from_state(post_position, post_velocity)
        post_arc_type = classify_lambert_arc_type(
            post_position,
            post_velocity,
            np.array(end_event["spacecraft_position_m"], dtype=float),
            float(leg["post_dsm_duration_days"]),
        )
        leg_rows.append(
            [
                f"DSM {dsm_index}-{end_label}",
                post_arc_type,
                f"{post_elements['sma_au']:.4f}",
                f"{post_elements['ecc']:.4f}",
                f"{post_elements['raan_deg']:.3f}",
                f"{post_elements['incl_deg']:.3f}",
                f"{post_elements['argperi_deg']:.3f}",
            ]
        )

        if end_event["type"] == "flyby" and str(end_event.get("body")) == "earth":
            earth_flyby_count += 1

    return leg_rows


def build_event_rows(candidate: dict) -> list[list[str]]:
    rows: list[list[str]] = []
    launch_epoch = float(candidate["events"][0]["epoch_mjd2000"])
    earth_flyby_count = 0
    dsm_count = 0

    for event in candidate["events"]:
        event_type = str(event["type"])
        if event_type == "flyby" and str(event.get("body")) == "earth":
            earth_flyby_count += 1
        if event_type == "dsm":
            dsm_count += 1

        if event_type == "dsm":
            name = f"DSM {dsm_count}"
        else:
            name = event_name(event, earth_flyby_count)

        tof_years = (float(event["epoch_mjd2000"]) - launch_epoch) * DAY2YEAR
        altitude_text = "--"
        if event_type == "flyby":
            altitude_text = f"{float(event['flyby_altitude_km']):,.0f}"

        delta_v_text = "--"
        if event_type == "dsm":
            delta_v_text = f"{float(event['delta_v_m_s']) / 1000.0:.3f}"

        rows.append(
            [
                name,
                short_date(str(event["epoch"])),
                f"{tof_years:.2f}",
                altitude_text,
                delta_v_text,
            ]
        )
    return rows


def draw_cell(
    ax: plt.Axes,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    facecolor: str,
    text: str,
    fontsize: float,
    weight: str = "normal",
    style: str = "normal",
    ha: str = "center",
    pad: float = 0.012,
) -> None:
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            facecolor=facecolor,
            edgecolor=EDGE_COLOR,
            linewidth=1.2,
        )
    )

    text_x = x0 + width / 2.0
    if ha == "left":
        text_x = x0 + pad
    elif ha == "right":
        text_x = x0 + width - pad

    ax.text(
        text_x,
        y0 + height / 2.0,
        text,
        ha=ha,
        va="center",
        color=TEXT_COLOR,
        fontsize=fontsize,
        fontweight=weight,
        fontstyle=style,
        family="DejaVu Sans",
    )


def render_table(
    output_path: Path,
    headers: list[str],
    rows: list[list[str]],
    *,
    figsize: tuple[float, float],
    col_widths: list[float],
    header_fontsize: float,
    body_fontsize: float,
    header_units: float = 1.18,
    row_units: float = 1.0,
    bold_first_column: bool = False,
    italicize_dsm_labels: bool = False,
) -> None:
    if len(headers) != len(col_widths):
        raise ValueError("Header count and column width count must match.")
    if not np.isclose(sum(col_widths), 1.0):
        raise ValueError("Column widths must sum to 1.0.")

    fig, ax = plt.subplots(figsize=figsize, dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    left = 0.02
    right = 0.02
    top = 0.03
    bottom = 0.03
    usable_width = 1.0 - left - right
    usable_height = 1.0 - top - bottom

    total_units = header_units + len(rows) * row_units
    header_height = usable_height * header_units / total_units
    row_height = usable_height * row_units / total_units
    x_edges = [left]
    for width_fraction in col_widths:
        x_edges.append(x_edges[-1] + usable_width * width_fraction)

    y = 1.0 - top - header_height
    for col_index, header in enumerate(headers):
        draw_cell(
            ax,
            x0=x_edges[col_index],
            y0=y,
            width=x_edges[col_index + 1] - x_edges[col_index],
            height=header_height,
            facecolor=HEADER_COLOR,
            text=header,
            fontsize=header_fontsize,
            weight="bold",
        )

    for row_index, row in enumerate(rows, start=1):
        y = 1.0 - top - header_height - row_index * row_height
        facecolor = ROW_A if row_index % 2 == 1 else ROW_B
        first_cell_is_dsm = str(row[0]).startswith("DSM")

        for col_index, value in enumerate(row):
            weight = "normal"
            style = "normal"
            ha = "left" if col_index == 0 else "center"

            if col_index == 0 and bold_first_column and not first_cell_is_dsm:
                weight = "bold"
            if col_index == 0 and italicize_dsm_labels and first_cell_is_dsm:
                style = "italic"

            draw_cell(
                ax,
                x0=x_edges[col_index],
                y0=y,
                width=x_edges[col_index + 1] - x_edges[col_index],
                height=row_height,
                facecolor=facecolor,
                text=str(value),
                fontsize=body_fontsize,
                weight=weight,
                style=style,
                ha=ha,
            )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    candidate, _entry = load_candidate(args.candidate_id)
    output_dir = TRAJECTORIES_DIR / args.candidate_id
    output_dir.mkdir(parents=True, exist_ok=True)

    leg_headers = ["Transfer", "Type", "SMA [AU]", "Ecc", "RAAN [deg]", "Incl [deg]", "Argperi [deg]"]
    leg_rows = build_leg_rows(candidate)
    render_table(
        output_dir / "leg_detail_table.png",
        leg_headers,
        leg_rows,
        figsize=(13.0, 4.8),
        col_widths=[0.24, 0.09, 0.13, 0.09, 0.15, 0.14, 0.16],
        header_fontsize=14.5,
        body_fontsize=13.0,
    )

    event_headers = ["Event", "Date", "TOF [yr]", "Altitude [km]", "DSM dV [km/s]"]
    event_rows = build_event_rows(candidate)
    render_table(
        output_dir / "event_timeline_table.png",
        event_headers,
        event_rows,
        figsize=(11.8, 5.4),
        col_widths=[0.29, 0.19, 0.12, 0.19, 0.21],
        header_fontsize=15.0,
        body_fontsize=13.2,
        bold_first_column=True,
        italicize_dsm_labels=True,
    )

    print(output_dir / "leg_detail_table.png")
    print(output_dir / "event_timeline_table.png")


if __name__ == "__main__":
    main()
