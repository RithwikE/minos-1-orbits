# conda run -n minos-orbits python v2/presentation/trade_study/final_trade_study.py

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import matplotlib
import tomllib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

FAMILY_STYLES = {
    "VEEGA": {"color": "#0f766e", "light": "#14b8a6", "marker": "o"},
    "DVEGA": {"color": "#c2410c", "light": "#f97316", "marker": "o"},
}

SCORING_FIELDS = [
    "total_dsm_kms",
    "total_tof_years",
    "c3_kms2",
    "arrival_vinf_kms",
    "departure_date",
]

SCORE_FIELD_TO_ROW_KEY = {
    "total_dsm_kms": "total_dsm_kms",
    "total_tof_years": "total_tof_years",
    "c3_kms2": "c3_kms2",
    "arrival_vinf_kms": "arrival_vinf_kms",
    "departure_date": "departure_date_ordinal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trade-study presentation figures."
    )
    parser.add_argument(
        "--veega-csv",
        default="v2/good_results/veega/veega_trade_study_candidates.csv",
        help="Path to the VEEGA candidate CSV.",
    )
    parser.add_argument(
        "--dvega-csv",
        default="v2/good_results/dvega/dvega_trade_study_candidates.csv",
        help="Path to the DVEGA candidate CSV.",
    )
    parser.add_argument(
        "--weights",
        default="v2/presentation/trade_study/weights.toml",
        help="Path to the trade-study weights TOML.",
    )
    parser.add_argument(
        "--output-dir",
        default="v2/presentation/trade_study/output",
        help="Directory for generated files.",
    )
    return parser.parse_args()


def load_weights(path: Path) -> dict:
    data = tomllib.loads(path.read_text())
    weights = {metric: float(data["weights"][metric]) for metric in SCORING_FIELDS}
    selection = data.get("selection", {})
    limits = data.get("limits", {})
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value.")
    normalized = {key: value / total for key, value in weights.items()}
    return {
        "weights": normalized,
        "limits": limits,
        "top_n": int(selection.get("top_n", 6)),
        "annotate_top_n": int(
            selection.get("annotate_top_n", selection.get("top_n", 6))
        ),
        "dedupe_exact_repeats": bool(selection.get("dedupe_exact_repeats", True)),
    }


def load_family_rows(path: Path, family: str) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = []
        for row in reader:
            parsed = dict(row)
            parsed["family"] = family
            parsed["rank_within_seed"] = int(parsed["rank_within_seed"])
            for metric in (
                "total_dsm_kms",
                "total_tof_years",
                "c3_kms2",
                "arrival_vinf_kms",
            ):
                parsed[metric] = float(parsed[metric])
            parsed["departure_date_ordinal"] = date.fromisoformat(
                parsed["departure_date"]
            ).toordinal()
            rows.append(parsed)
    return rows


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            row["total_dsm_kms"],
            row["total_tof_years"],
            row["c3_kms2"],
            row["arrival_vinf_kms"],
            row["family"],
            row["seed"],
            row["rank_within_seed"],
        ),
    )


def exact_signature(row: dict) -> tuple:
    return (
        row["family"],
        row["launch_epoch"],
        row["arrival_epoch"],
        round(row["total_dsm_kms"], 12),
        round(row["c3_kms2"], 12),
        round(row["arrival_vinf_kms"], 12),
        round(row["total_tof_years"], 12),
    )


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for row in sort_rows(rows):
        signature = exact_signature(row)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
    return deduped


def parse_optional_limit_number(raw_value) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value.strip().lower() == "none":
        return None
    return float(raw_value)


def parse_optional_limit_date(raw_value) -> int | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str) and raw_value.strip().lower() == "none":
        return None
    return date.fromisoformat(str(raw_value)).toordinal()


def apply_limits(rows: list[dict], limits: dict) -> list[dict]:
    dsm_max = parse_optional_limit_number(limits.get("total_dsm_kms_max"))
    tof_max = parse_optional_limit_number(limits.get("total_tof_years_max"))
    c3_max = parse_optional_limit_number(limits.get("c3_kms2_max"))
    vinf_max = parse_optional_limit_number(limits.get("arrival_vinf_kms_max"))
    departure_min = parse_optional_limit_date(limits.get("departure_date_min"))
    departure_max = parse_optional_limit_date(limits.get("departure_date_max"))

    filtered: list[dict] = []
    for row in rows:
        if dsm_max is not None and row["total_dsm_kms"] > dsm_max:
            continue
        if tof_max is not None and row["total_tof_years"] > tof_max:
            continue
        if c3_max is not None and row["c3_kms2"] > c3_max:
            continue
        if vinf_max is not None and row["arrival_vinf_kms"] > vinf_max:
            continue
        if departure_min is not None and row["departure_date_ordinal"] < departure_min:
            continue
        if departure_max is not None and row["departure_date_ordinal"] > departure_max:
            continue
        filtered.append(row)
    return filtered


def score_rows(rows: list[dict], weights: dict[str, float]) -> list[dict]:
    bounds = {
        field: (
            min(row[SCORE_FIELD_TO_ROW_KEY[field]] for row in rows),
            max(row[SCORE_FIELD_TO_ROW_KEY[field]] for row in rows),
        )
        for field in SCORING_FIELDS
    }

    scored: list[dict] = []
    for row in rows:
        scored_row = dict(row)
        weighted_total = 0.0
        for field in SCORING_FIELDS:
            row_key = SCORE_FIELD_TO_ROW_KEY[field]
            lower, upper = bounds[field]
            if upper <= lower:
                score = 1.0
            else:
                score = 1.0 - (row[row_key] - lower) / (upper - lower)
            scored_row[f"score_{field}"] = score
            weighted_total += weights[field] * score
        scored_row["score_total"] = weighted_total
        scored.append(scored_row)

    return sorted(
        scored,
        key=lambda row: (
            -row["score_total"],
            row["total_dsm_kms"],
            row["total_tof_years"],
        ),
    )


def label_top_rows(rows: list[dict]) -> dict[tuple, str]:
    return {
        exact_signature(row): f"C{index}" for index, row in enumerate(rows, start=1)
    }


def plot_scatter(
    raw_rows: list[dict],
    top_rows: list[dict],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7fafc")

    for family in ("VEEGA", "DVEGA"):
        family_rows = [row for row in raw_rows if row["family"] == family]
        style = FAMILY_STYLES[family]
        ax.scatter(
            [row["total_tof_years"] for row in family_rows],
            [row["total_dsm_kms"] for row in family_rows],
            s=64,
            c=style["light"],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.45,
            marker=style["marker"],
            label=f"{family} ({len(family_rows)} rows)",
        )

    labels = label_top_rows(top_rows)
    offsets = [
        (8, 8),
        (8, -18),
        (-28, 10),
        (-28, -16),
        (16, 20),
        (-18, 22),
        (16, -24),
        (-30, 24),
    ]
    for index, row in enumerate(top_rows):
        style = FAMILY_STYLES[row["family"]]
        ax.scatter(
            row["total_tof_years"],
            row["total_dsm_kms"],
            s=135,
            c=style["color"],
            edgecolors="white",
            linewidths=1.2,
            marker=style["marker"],
            zorder=4,
        )
        ax.annotate(
            labels[exact_signature(row)],
            (row["total_tof_years"], row["total_dsm_kms"]),
            xytext=offsets[index % len(offsets)],
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": style["color"],
                "edgecolor": "white",
                "linewidth": 0.8,
            },
        )

    ax.set_title("Trade Space Across All DVEGA and VEEGA Candidates", fontsize=17, pad=16)
    ax.set_xlabel("Total Time of Flight (years)", fontsize=12)
    ax.set_ylabel("Total DSM (km/s)", fontsize=12)
    ax.grid(alpha=0.18, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        fontsize=11,
        loc="lower left",
    )

    fig.tight_layout(pad=1.4)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def blend_hex(color_a: str, color_b: str, alpha: float) -> str:
    alpha = min(1.0, max(0.0, alpha))
    a = np.array([int(color_a[idx : idx + 2], 16) for idx in (1, 3, 5)], dtype=float)
    b = np.array([int(color_b[idx : idx + 2], 16) for idx in (1, 3, 5)], dtype=float)
    mixed = (1.0 - alpha) * a + alpha * b
    return "#" + "".join(f"{int(round(value)):02x}" for value in mixed)


def score_fill(score: float) -> str:
    return blend_hex("#f8d7da", "#d1fae5", score)


def plot_decision_matrix(
    top_rows: list[dict], weights: dict[str, float], limits: dict, output_path: Path
) -> None:
    labels = label_top_rows(top_rows)

    col_labels = [
        "Choice",
        "Family",
        "Seed",
        "Rank",
        f"Departure\n({weights['departure_date']:.0%})",
        f"DSM (km/s)\n({weights['total_dsm_kms']:.0%})",
        f"TOF (yr)\n({weights['total_tof_years']:.0%})",
        f"C3 (km²/s²)\n({weights['c3_kms2']:.0%})",
        f"Arr Vinf (km/s)\n({weights['arrival_vinf_kms']:.0%})",
        "Weighted Score",
    ]

    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []

    for row in top_rows:
        signature = exact_signature(row)
        cell_text.append(
            [
                labels[signature],
                row["family"],
                row["seed"],
                str(row["rank_within_seed"]),
                row["departure_date"],
                f"{row['total_dsm_kms']:.3f}",
                f"{row['total_tof_years']:.3f}",
                f"{row['c3_kms2']:.2f}",
                f"{row['arrival_vinf_kms']:.3f}",
                f"{row['score_total']:.3f}",
            ]
        )
        cell_colors.append(
            [
                "#ffffff",
                "#ecfeff" if row["family"] == "VEEGA" else "#fff7ed",
                "#ffffff",
                "#ffffff",
                score_fill(row["score_departure_date"]),
                score_fill(row["score_total_dsm_kms"]),
                score_fill(row["score_total_tof_years"]),
                score_fill(row["score_c3_kms2"]),
                score_fill(row["score_arrival_vinf_kms"]),
                score_fill(row["score_total"]),
            ]
        )

    # Explicit column widths read better here than pure text-based autosizing.
    # The metric headers are multiline and otherwise make C3 / Arr Vinf much wider
    # than the actual cell contents justify.
    col_widths = [
        0.082,  # Choice
        0.082,  # Family
        0.102,  # Seed
        0.058,  # Rank
        0.102,  # Departure
        0.077,  # DSM
        0.071,  # TOF
        0.060,  # C3
        0.087,  # Arr Vinf
        0.081,  # Weighted Score
    ]

    fig_height = 2.6 + 0.56 * len(top_rows)
    fig = plt.figure(figsize=(16, fig_height), facecolor="white")
    gs = GridSpec(2, 1, height_ratios=[9, 1.4], hspace=0.05, figure=fig)
    ax_table = fig.add_subplot(gs[0])
    ax_footer = fig.add_subplot(gs[1])
    ax_table.axis("off")
    ax_footer.axis("off")

    ax_table.set_title("Weighted Decision Matrix for Top Trade-Study Choices", fontsize=17, pad=14)

    table = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)

    for (row_index, col_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
            cell.set_edgecolor("#111827")
            cell.set_height(cell.get_height() * 1.18)
        else:
            cell.set_edgecolor("#e5e7eb")
            if col_index in {0, 1, 9}:
                cell.set_text_props(weight="bold")

    limits_text = (
        f"Ranking pool limits: DSM <= {limits.get('total_dsm_kms_max', 'none')} km/s   |   "
        f"TOF <= {limits.get('total_tof_years_max', 'none')} yr   |   "
        f"C3 <= {limits.get('c3_kms2_max', 'none')}   |   "
        f"Departure >= {limits.get('departure_date_min', 'none')}"
    )
    ax_footer.text(
        0.01,
        0.50,
        limits_text,
        ha="left",
        va="center",
        fontsize=10,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#d1d5db"},
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08, hspace=0.02)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_veega_zoom(
    veega_rows: list[dict],
    top_rows: list[dict],
    output_path: Path,
) -> None:
    unique_rows = dedupe_rows(veega_rows)
    in_view = [
        row
        for row in unique_rows
        if 6.5 <= row["total_tof_years"] <= 7.5 and 0.5 <= row["total_dsm_kms"] <= 1.0
    ]
    in_view = sorted(in_view, key=lambda row: (row["total_dsm_kms"], row["c3_kms2"], row["total_tof_years"]))
    if not in_view:
        raise RuntimeError("No VEEGA candidates fall inside the requested zoom bounds.")

    fig, ax = plt.subplots(figsize=(12.5, 7.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7fafc")

    ax.scatter(
        [row["total_tof_years"] for row in unique_rows],
        [row["total_dsm_kms"] for row in unique_rows],
        s=36,
        c="#cbd5e1",
        alpha=0.55,
        edgecolors="white",
        linewidths=0.25,
        label=f"All unique VEEGA candidates ({len(unique_rows)})",
    )

    c3_values = np.array([row["c3_kms2"] for row in in_view], dtype=float)
    scatter = ax.scatter(
        [row["total_tof_years"] for row in in_view],
        [row["total_dsm_kms"] for row in in_view],
        s=128,
        c=c3_values,
        cmap="viridis_r",
        edgecolors="#0f172a",
        linewidths=0.6,
        zorder=3,
        label=f"VEEGA candidates in zoom window ({len(in_view)})",
        vmin=float(np.min(c3_values)),
        vmax=float(np.max(c3_values)),
    )

    ranked_labels = label_top_rows(top_rows)
    offsets = [(8, 7), (8, -16), (-30, 8), (-30, -16), (12, 20), (-18, 20), (16, -24), (-22, 24)]
    for index, row in enumerate(in_view, start=1):
        top_label = ranked_labels.get(exact_signature(row))
        seed_num = row["seed"].removeprefix("seed")
        label = f"{seed_num}:{row['rank_within_seed']}, {row['c3_kms2']:.2f} C3"
        if top_label is not None:
            label += f" ({top_label})"
        ax.annotate(
            label,
            (row["total_tof_years"], row["total_dsm_kms"]),
            xytext=offsets[(index - 1) % len(offsets)],
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#0f172a",
                "edgecolor": "white",
                "linewidth": 0.8,
            },
        )

    ax.set_title("Zoomed VEEGA Cluster With In-Window C3 Shading", fontsize=17, pad=16)
    ax.set_xlabel("Total Time of Flight (years)", fontsize=12)
    ax.set_ylabel("Total DSM (km/s)", fontsize=12)
    ax.set_xlim(6.5, 7.5)
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.18, color="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, facecolor="white", edgecolor="#d1d5db", fontsize=11, loc="upper right")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("C3 (km²/s²) within zoom window", fontsize=11)

    fig.tight_layout(pad=1.4)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_ranked_csv(ranked_rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "family",
        "seed",
        "rank_within_seed",
        "departure_date",
        "total_dsm_kms",
        "total_tof_years",
        "c3_kms2",
        "arrival_vinf_kms",
        "score_total",
        "detail_json_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow(
                {
                    "family": row["family"],
                    "seed": row["seed"],
                    "rank_within_seed": row["rank_within_seed"],
                    "departure_date": row["departure_date"],
                    "total_dsm_kms": f"{row['total_dsm_kms']:.6f}",
                    "total_tof_years": f"{row['total_tof_years']:.6f}",
                    "c3_kms2": f"{row['c3_kms2']:.6f}",
                    "arrival_vinf_kms": f"{row['arrival_vinf_kms']:.6f}",
                    "score_total": f"{row['score_total']:.6f}",
                    "detail_json_path": row["detail_json_path"],
                }
            )


def main() -> None:
    args = parse_args()
    weights_cfg = load_weights(Path(args.weights).resolve())
    weights = weights_cfg["weights"]

    raw_rows = sort_rows(
        load_family_rows(Path(args.veega_csv).resolve(), "VEEGA")
        + load_family_rows(Path(args.dvega_csv).resolve(), "DVEGA")
    )
    ranked_input = (
        dedupe_rows(raw_rows) if weights_cfg["dedupe_exact_repeats"] else raw_rows
    )
    ranked_input = apply_limits(ranked_input, weights_cfg["limits"])
    if not ranked_input:
        raise RuntimeError(
            "No candidates remained after applying the TOML trade-study limits."
        )
    ranked_rows = score_rows(ranked_input, weights)
    top_rows = ranked_rows[: weights_cfg["top_n"]]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scatter_path = output_dir / "trade_space_scatter.png"
    matrix_path = output_dir / "weighted_decision_matrix.png"
    ranked_csv_path = output_dir / "ranked_trade_study_candidates.csv"
    veega_zoom_path = output_dir / "veega_zoom_scatter.png"

    plot_scatter(
        raw_rows=raw_rows,
        top_rows=top_rows[: weights_cfg["annotate_top_n"]],
        output_path=scatter_path,
    )
    plot_decision_matrix(top_rows, weights, weights_cfg["limits"], matrix_path)
    save_ranked_csv(ranked_rows, ranked_csv_path)
    plot_veega_zoom(
        [row for row in raw_rows if row["family"] == "VEEGA"],
        top_rows=top_rows,
        output_path=veega_zoom_path,
    )

    print(f"Saved scatter plot: {scatter_path}")
    print(f"Saved decision matrix: {matrix_path}")
    print(f"Saved ranked CSV: {ranked_csv_path}")
    print(f"Saved VEEGA zoom scatter: {veega_zoom_path}")


if __name__ == "__main__":
    main()
