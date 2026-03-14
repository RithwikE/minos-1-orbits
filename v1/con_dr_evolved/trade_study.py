# %% [markdown]
# # Trade Study Comparison Table
#
# Reads trade_study_results.csv and renders a color-coded table image.
# Each metric column is normalised across rows so colors reflect
# *relative* performance — close values get similar colors regardless
# of absolute magnitude.
#
# Usage:
#   python trade_study.py
#   python trade_study.py path/to/custom.csv

# %% Imports
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

# Columns to display and their display names
DISPLAY_COLS = {
    'sequence':          'Sequence',
    'launch_date':       'Launch',
    'tof_years':         'TOF (yr)',
    'c3_kms2':           'C3 (km²/s²)',
    'delivered_mass_kg': 'FH Mass (kg)',
    'total_dsm_kms':     'Σ DSM (km/s)',
    'vinf_arr_kms':      'V∞ arr (km/s)',
}

# Which columns to color-code, and whether lower or higher is better
#   'lower'  → green at min, red at max
#   'higher' → green at max, red at min
METRIC_DIRECTION = {
    'tof_years':         'lower',
    'c3_kms2':           'lower',
    'delivered_mass_kg': 'higher',
    'total_dsm_kms':     'lower',
    'vinf_arr_kms':      'lower',
}

# Colormap: RdYlGn goes Red → Yellow → Green
CMAP = cm.RdYlGn

# How much to compress the colormap range (avoids pure red / pure green
# at the extremes, keeping things readable).  0.15 means we use the
# colormap from 0.15 to 0.85 instead of 0.0 to 1.0.
CMAP_MARGIN = 0.15


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def color_for_value(val, col_min, col_max, direction):
    """Map a numeric value to an RGBA color using the global CMAP."""
    if col_max == col_min:
        return CMAP(0.5)  # all values identical → neutral

    # Normalise to [0, 1] where 0 = worst, 1 = best
    frac = (val - col_min) / (col_max - col_min)
    if direction == 'lower':
        score = 1.0 - frac   # low value = good = high score
    else:
        score = frac          # high value = good = high score

    # Compress into [margin, 1-margin] range of colormap
    cmap_val = CMAP_MARGIN + score * (1.0 - 2 * CMAP_MARGIN)
    return CMAP(cmap_val)


def format_cell(val, col):
    """Format a cell value for display."""
    if col == 'launch_date':
        # Trim to just the date portion
        return str(val)[:11].strip()
    if col == 'delivered_mass_kg':
        try:
            return f"{float(val):,.0f}"
        except (ValueError, TypeError):
            return str(val)
    if col in ('tof_years', 'c3_kms2', 'total_dsm_kms', 'vinf_arr_kms'):
        try:
            return f"{float(val):.2f}"
        except (ValueError, TypeError):
            return str(val)
    return str(val)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    # --- Locate CSV ---
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default: look in output/ subdirectory
        csv_path = os.path.join('output', 'trade_study_results.csv')

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Sequences: {list(df['sequence'])}\n")

    # --- Filter to display columns that exist in the CSV ---
    available = [c for c in DISPLAY_COLS if c in df.columns]
    display_names = [DISPLAY_COLS[c] for c in available]
    n_rows = len(df)
    n_cols = len(available)

    # --- Convert metric columns to numeric ---
    for col in METRIC_DIRECTION:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Compute column min/max for coloring ---
    col_ranges = {}
    for col in METRIC_DIRECTION:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                col_ranges[col] = (vals.min(), vals.max())

    # --- Build cell text and colors ---
    cell_text = []
    cell_colors = []

    for _, row in df.iterrows():
        row_text = []
        row_colors = []
        for col in available:
            val = row[col]
            text = format_cell(val, col)
            row_text.append(text)

            if col in METRIC_DIRECTION and col in col_ranges and pd.notna(val):
                cmin, cmax = col_ranges[col]
                rgba = color_for_value(float(val), cmin, cmax,
                                       METRIC_DIRECTION[col])
                row_colors.append(rgba)
            else:
                row_colors.append('white')

        cell_text.append(row_text)
        cell_colors.append(row_colors)

    # --- Render table as figure ---
    fig_width = max(12, n_cols * 1.8)
    fig_height = max(2.5, 0.6 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_title('Europa Lander — Interplanetary Transfer Trade Study',
                 fontsize=14, fontweight='bold', pad=20)

    table = ax.table(
        cellText=cell_text,
        colLabels=display_names,
        cellColours=cell_colors,
        colColours=['#D6E4F0'] * n_cols,
        loc='center',
        cellLoc='center',
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)  # row height

    # Bold the header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_text_props(fontweight='bold', fontsize=10)
        cell.set_height(0.08)

    # Bold the sequence column and adjust widths
    for i in range(n_rows):
        # Sequence column bold
        table[i + 1, 0].set_text_props(fontweight='bold')

    # Auto-size columns
    table.auto_set_column_width(list(range(n_cols)))

    # --- Add legend ---
    legend_y = -0.02
    fig.text(0.5, legend_y,
             '■ Green = better   ■ Red = worse   (each column normalised independently)',
             ha='center', va='top', fontsize=9, style='italic',
             color='#555555')

    # --- Save ---
    out_dir = os.path.dirname(csv_path)
    out_path = os.path.join(out_dir, 'trade_study_table.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {out_path}")
    plt.close()

    # --- Also print a text version to terminal ---
    print("\n" + "="*90)
    print("  TRADE STUDY COMPARISON")
    print("="*90)

    # Header
    widths = [max(len(display_names[j]),
                  max(len(cell_text[i][j]) for i in range(n_rows)))
              for j in range(n_cols)]
    header = "  ".join(display_names[j].center(widths[j]) for j in range(n_cols))
    print(header)
    print("  ".join("-" * widths[j] for j in range(n_cols)))

    # Rows
    for i in range(n_rows):
        row_str = "  ".join(cell_text[i][j].center(widths[j]) for j in range(n_cols))
        print(row_str)

    print("="*90)

    # --- Rank summary ---
    print("\nRankings by metric:")
    for col, direction in METRIC_DIRECTION.items():
        if col not in df.columns:
            continue
        ascending = (direction == 'lower')
        ranked = df[['sequence', col]].dropna().sort_values(col, ascending=ascending)
        vals = list(ranked.itertuples(index=False))
        rank_str = " > ".join(f"{row[0]} ({row[1]:.2f})" for row in vals)
        print(f"  {DISPLAY_COLS.get(col, col)}: {rank_str}")


if __name__ == '__main__':
    main()