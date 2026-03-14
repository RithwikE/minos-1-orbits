"""
Trajectory trade-study utilities for AAE450 Europa Lander — Minos-1 Orbits Team.

This module centralises shared logic so that individual sequence scripts
(01_direct_baseline.py, 02_dvega.py … 06_mga.py) stay short and consistent.

Key design decisions (see project docs for rationale):
  - Objective: minimize Σ(DSMs) only  (add_vinf_dep=False, add_vinf_arr=False)
  - Departure C3 is provided by Falcon Heavy — reported separately, not penalised
  - Arrival V∞ at Jupiter is NOT penalised — it determines JOI ΔV which is a
    separate tour-design problem (Ganymede/Callisto gravity assists reduce it).
    Including V∞_arr in the objective creates a perverse incentive where the
    optimizer spends DSM fuel to brake before Jupiter instead of arriving fast.
  - Post-processing computes delivered mass from C3 via Falcon Heavy curve fit
  - Arrival V∞ is extracted and reported for JOI ΔV estimation
"""

import os, csv, re, io, sys, datetime
import numpy as np
import pykep as pk
import pygmo as pg

# ─────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# STDOUT TEE (log to file + terminal simultaneously)
# ─────────────────────────────────────────────────────────────────────

class Tee:
    """Duplicate stdout to a file while still printing to terminal."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w')
        self.terminal = sys.__stdout__
    def write(self, s):
        self.terminal.write(s)
        self.file.write(s)
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        self.file.close()
        sys.stdout = self.terminal


def start_logging(script_num: str):
    """Call at the top of __main__ to log all output. Returns the Tee object.

    Usage:
        tee = start_logging('04')   # writes to output/04_output.txt
    """
    path = os.path.join(OUTPUT_DIR, f'{script_num}_output.txt')
    tee = Tee(path)
    sys.stdout = tee
    return tee

# ─────────────────────────────────────────────────────────────────────
# LAUNCH VEHICLE PERFORMANCE
# ─────────────────────────────────────────────────────────────────────
#
# Source: Silverbird Astronautics Launch Vehicle Performance Calculator
#         https://silverbirdastronautics.com/LVperform.html
#
# Configuration:
#   Launch Vehicle:       Falcon Heavy (expendable) w/long fairing
#   Launch Site:          Cape Canaveral / KSC
#   Perigee, km:          185
#   Declination, deg:     0
#   Trajectory:           Two-burn
#   Shutdown Mode:        GCS
#   Calibration:          Mixed
#   Upper Stage Disposal: ODMSP
#
# ┌───────────────┬──────────────────┬──────────────────────────────┐
# │ C3 (km²/s²)   │ Est. Payload (kg)│ 95% Confidence Interval (kg) │
# ├───────────────┼──────────────────┼──────────────────────────────┤
# │   0           │  20403           │  17450 – 23807               │
# │   5           │  18759           │  16010 – 21935               │
# │  10           │  17258           │  14690 – 20224               │
# │  15           │  15880           │  13479 – 18654               │
# │  20           │  14619           │  12367 – 17215               │
# │  25           │  13454           │  11341 – 15893               │
# │  30           │  12379           │  10396 – 14668               │
# │  35           │  11388           │   9525 – 13537               │
# │  40           │  10470           │   8716 – 12493               │
# └───────────────┴──────────────────┴──────────────────────────────┘

_FH_C3 = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40], dtype=float)

_FH_PAYLOAD_EST = np.array(
    [20403, 18759, 17258, 15880, 14619, 13454, 12379, 11388, 10470],
    dtype=float,
)

_FH_PAYLOAD_LO = np.array(
    [17450, 16010, 14690, 13479, 12367, 11341, 10396, 9525, 8716],
    dtype=float,
)

_FH_PAYLOAD_HI = np.array(
    [23807, 21935, 20224, 18654, 17215, 15893, 14668, 13537, 12493],
    dtype=float,
)

_FH_COEFFS_EST = np.polyfit(_FH_C3, _FH_PAYLOAD_EST, 2)
_FH_COEFFS_LO  = np.polyfit(_FH_C3, _FH_PAYLOAD_LO, 2)
_FH_COEFFS_HI  = np.polyfit(_FH_C3, _FH_PAYLOAD_HI, 2)


def falcon_heavy_payload(c3_kms2: float) -> dict:
    """
    Estimate Falcon Heavy expendable payload mass for a given departure C3.
    """
    in_range = 0.0 <= c3_kms2 <= 40.0
    est = float(np.polyval(_FH_COEFFS_EST, c3_kms2))
    lo  = float(np.polyval(_FH_COEFFS_LO, c3_kms2))
    hi  = float(np.polyval(_FH_COEFFS_HI, c3_kms2))
    return {
        'estimated_kg': est,
        'low_95_kg':    lo,
        'high_95_kg':   hi,
        'c3_kms2':      c3_kms2,
        'in_range':     in_range,
    }


# ─────────────────────────────────────────────────────────────────────
# 3-PHASE MGA-1DSM OPTIMISATION
# ─────────────────────────────────────────────────────────────────────

def run_mga_optimisation(
    udp,
    label: str = "sequence",
    # Phase 1 — broad search
    n_islands: int = 96,
    pop_size_1: int = 32,
    gen_1: int = 500,
    evolve_rounds_1: int = 8,
    n_seeds: int = 50,
    # Phase 2 — refine seeds
    pop_size_2: int = 32,
    gen_2: int = 1500,
    # Phase 3 — local polish
    compass_fevals: int = 200_000,
) -> tuple:
    """
    Run a 3-phase global optimisation on a pykep mga_1dsm problem.

    Returns
    -------
    best_x : np.ndarray   – best decision vector
    best_f : float         – best objective value (m/s)
    """
    prob = pg.problem(udp)
    print(f"\n{'='*60}")
    print(f"  OPTIMISING: {label}")
    print(f"{'='*60}")
    print(prob)

    # --- Phase 1: broad search ---
    print(f"\nPhase 1: {n_islands} islands × {pop_size_1} pop × {gen_1} gen × {evolve_rounds_1} rounds")
    algo1 = pg.algorithm(pg.sade(gen=gen_1))
    algo1.set_verbosity(0)
    archi = pg.archipelago(algo=algo1, prob=prob, n=n_islands, pop_size=pop_size_1)

    for rd in range(evolve_rounds_1):
        archi.evolve()
        archi.wait()
        best_so_far = min(isl.get_population().champion_f[0] for isl in archi)
        print(f"  Round {rd+1}/{evolve_rounds_1}: best = {best_so_far/1000:.4f} km/s")

    ranked = sorted(
        [(isl.get_population().champion_f[0], isl.get_population().champion_x)
         for isl in archi],
        key=lambda p: p[0],
    )
    seeds = ranked[:n_seeds]
    print(f"\nPhase 1 done. Top 5:")
    for i, (f, _) in enumerate(seeds[:5]):
        print(f"  #{i+1}: {f/1000:.4f} km/s")

    # --- Phase 2: refine seeds ---
    print(f"\nPhase 2: refining top {n_seeds} seeds ({gen_2} gen each)")
    algo2 = pg.algorithm(pg.sade(gen=gen_2))
    algo2.set_verbosity(0)

    best_f = float('inf')
    best_x = None
    for i, (_, x_seed) in enumerate(seeds):
        pop = pg.population(prob, size=pop_size_2)
        pop.set_x(0, x_seed)
        pop = algo2.evolve(pop)
        if pop.champion_f[0] < best_f:
            best_f = pop.champion_f[0]
            best_x = pop.champion_x.copy()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_seeds} done, best = {best_f/1000:.4f} km/s")

    # --- Phase 3: local polish ---
    print(f"\nPhase 3: Compass Search ({compass_fevals:,} fevals)")
    algo3 = pg.algorithm(pg.compass_search(
        max_fevals=compass_fevals, start_range=0.01, stop_range=1e-8,
    ))
    algo3.set_verbosity(0)
    pop = pg.population(prob, size=1)
    pop.set_x(0, best_x)
    pop = algo3.evolve(pop)

    best_f = pop.champion_f[0]
    best_x = pop.champion_x.copy()
    print(f"\n  ✓ Final objective: {best_f/1000:.4f} km/s")
    return best_x, best_f


# ─────────────────────────────────────────────────────────────────────
# RESULT EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def extract_mga_results(udp, best_x: np.ndarray, best_f: float,
                        sequence_label: str, seq_bodies: list,
                        seq: list = None,
                        ref: dict = None) -> dict:
    """
    Pull all trade-study columns from a converged mga_1dsm solution.

    Uses pykep's pretty() output to extract exact DSM magnitudes and
    arrival V∞ (guaranteed to match internal computation).
    """
    n_legs = len(seq_bodies) - 1

    # --- timing ---
    t0_mjd = best_x[0]
    vinf_dep_ms = best_x[3]
    vinf_dep_kms = vinf_dep_ms / 1000.0
    c3 = vinf_dep_kms ** 2

    # leg TOFs  (indices: 5, 9, 13, 17, … = 5 + 4*k for k=0..n_legs-1)
    tofs_days = [best_x[5 + 4*k] for k in range(n_legs)]
    total_tof = sum(tofs_days)

    # epochs
    cum_days = np.cumsum([0.0] + tofs_days)
    epochs = [pk.epoch(t0_mjd + d, 'mjd2000') for d in cum_days]

    launch_date = str(epochs[0])
    arrival_date = str(epochs[-1])

    # flyby bodies & dates
    flyby_names = seq_bodies[1:-1]
    flyby_dates = [str(epochs[i+1]) for i in range(len(flyby_names))]

    # flyby altitudes  (rp/safe_radius at indices 7, 11, 15, …)
    flyby_alts = []
    for k in range(len(flyby_names)):
        rp_ratio = best_x[7 + 4*k]
        body = seq[k+1]
        safe_r_km = body.safe_radius / 1000.0
        body_r_km = body.radius / 1000.0
        alt_km = rp_ratio * safe_r_km - body_r_km
        flyby_alts.append(alt_km)

    # --- ΔV breakdown from pretty() output ---
    obj_ms = best_f

    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        udp.pretty(best_x)
    finally:
        sys.stdout = old_stdout
    pretty_text = buf.getvalue()

    # Extract DSM magnitudes (m/s)
    dsm_matches = re.findall(r'DSM magnitude:\s*([\d.]+)\s*m/s', pretty_text)
    dsm_per_leg_ms = [float(m) for m in dsm_matches]
    total_dsm_ms = sum(dsm_per_leg_ms)

    # Extract arrival V∞ (m/s) — this is always printed even if not in objective
    vinf_arr_match = re.search(r'Arrival Vinf:\s*([\d.]+)\s*m/s', pretty_text)
    if vinf_arr_match:
        vinf_arr_kms = float(vinf_arr_match.group(1)) / 1000.0
    else:
        # Fallback: compute from Lambert
        try:
            arr_body = seq[-1]
            dep_body = seq[-2]
            r_arr, v_arr_planet = arr_body.eph(epochs[-1])
            r_dep_last, _ = dep_body.eph(epochs[-2])
            last_tof_sec = tofs_days[-1] * 86400.0
            lamb = pk.lambert_problem(r_dep_last, r_arr, last_tof_sec, pk.MU_SUN)
            v_sc_arr = np.array(lamb.get_v2()[0])
            vinf_arr_vec = v_sc_arr - np.array(v_arr_planet)
            vinf_arr_kms = np.linalg.norm(vinf_arr_vec) / 1000.0
        except Exception:
            vinf_arr_kms = float('nan')
            print("  ⚠ Could not extract arrival V∞")

    total_dsm_kms = total_dsm_ms / 1000.0
    dsm_per_leg_kms = [d / 1000.0 for d in dsm_per_leg_ms]

    # Extract departure V∞ direction for plotting
    vinf_dep_vec = _extract_vinf_dep_vec(best_x)

    # delivered mass
    fh = falcon_heavy_payload(c3)

    d = {
        'sequence':          sequence_label,
        'launch_date':       launch_date,
        'arrival_date':      arrival_date,
        'tof_years':         f"{total_tof / 365.25:.2f}",
        'c3_kms2':           f"{c3:.2f}",
        'vinf_dep_kms':      f"{vinf_dep_kms:.3f}",
        'delivered_mass_kg': f"{fh['estimated_kg']:.0f}",
        'total_dsm_kms':     f"{total_dsm_kms:.4f}",
        'dsm_per_leg_kms':   ';'.join(f"{d:.4f}" for d in dsm_per_leg_kms),
        'vinf_arr_kms':      f"{vinf_arr_kms:.3f}",
        'objective_kms':     f"{obj_ms / 1000.0:.4f}",
        'flyby_bodies':      ';'.join(flyby_names),
        'flyby_dates':       ';'.join(flyby_dates),
        'flyby_alts_km':     ';'.join(f"{a:.0f}" for a in flyby_alts),
        'leg_tofs_days':     ';'.join(f"{t:.1f}" for t in tofs_days),
        'ref_study':         (ref or {}).get('ref_study', ''),
        'ref_c3_kms2':       (ref or {}).get('ref_c3', ''),
        'ref_dsm_kms':       (ref or {}).get('ref_dsm', ''),
        'ref_vinf_arr_kms':  (ref or {}).get('ref_vinf_arr', ''),
        'ref_tof_years':     (ref or {}).get('ref_tof', ''),
        'ref_notes':         (ref or {}).get('ref_notes', ''),
    }
    return d


def _extract_vinf_dep_vec(best_x):
    """Extract departure V∞ vector (m/s) from decision vector."""
    vinf_ms = best_x[3]
    theta = 2 * np.pi * best_x[1]
    phi = np.arccos(2 * best_x[2] - 1) - np.pi / 2
    return np.array([
        vinf_ms * np.cos(phi) * np.cos(theta),
        vinf_ms * np.cos(phi) * np.sin(theta),
        vinf_ms * np.sin(phi),
    ])


# ─────────────────────────────────────────────────────────────────────
# TRAJECTORY PLOTTING  (uses propagate_lagrangian — no pykep plot bugs)
# ─────────────────────────────────────────────────────────────────────

def plot_trajectory(udp, best_x, seq, seq_bodies, title, save_path):
    """
    Plot an MGA-1DSM trajectory using propagate_lagrangian.
    Works for any sequence length.  Produces a 3D matplotlib plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pykep import propagate_lagrangian, AU

    n_legs = len(seq_bodies) - 1
    n_pts = 100

    # --- Parse decision vector ---
    t0_mjd = best_x[0]
    vinf_dep_ms = best_x[3]
    tofs_days = [best_x[5 + 4*k] for k in range(n_legs)]
    etas = [best_x[4 + 4*k] for k in range(n_legs)]

    # Cumulative time for epochs
    cum_days = np.cumsum([0.0] + tofs_days)
    epochs_mjd = [t0_mjd + d for d in cum_days]

    # Planet positions at each epoch
    planet_pos = []
    planet_vel = []
    for i, body in enumerate(seq):
        r, v = body.eph(pk.epoch(epochs_mjd[i], 'mjd2000'))
        planet_pos.append(np.array(r))
        planet_vel.append(np.array(v))

    # Departure velocity
    vinf_vec = _extract_vinf_dep_vec(best_x)
    v_sc = planet_vel[0] + vinf_vec

    # --- Reconstruct each leg: propagate to DSM, then Lambert to next body ---
    leg_arcs = []  # list of (pre_dsm_points, post_dsm_points, dsm_pos)

    r_current = planet_pos[0]
    v_current = v_sc

    for k in range(n_legs):
        tof_sec = tofs_days[k] * 86400.0
        eta = etas[k]
        dt_pre = eta * tof_sec
        dt_post = (1.0 - eta) * tof_sec

        r_target = planet_pos[k + 1]

        # Pre-DSM arc
        pre_pts = []
        for frac in np.linspace(0, 1, n_pts):
            r, _ = propagate_lagrangian(r_current, v_current, frac * dt_pre, pk.MU_SUN)
            pre_pts.append(np.array(r))

        # Position at DSM
        r_dsm, _ = propagate_lagrangian(r_current, v_current, dt_pre, pk.MU_SUN)

        # Post-DSM: Lambert from DSM position to next planet
        lamb = pk.lambert_problem(r_dsm, r_target, dt_post, pk.MU_SUN)
        v_post_dsm = np.array(lamb.get_v1()[0])
        v_arr = np.array(lamb.get_v2()[0])

        post_pts = []
        for frac in np.linspace(0, 1, n_pts):
            r, _ = propagate_lagrangian(r_dsm, v_post_dsm, frac * dt_post, pk.MU_SUN)
            post_pts.append(np.array(r))

        leg_arcs.append((pre_pts, post_pts, np.array(r_dsm)))

        # For next leg: use Lambert departure from next planet
        if k < n_legs - 1:
            # Post-flyby: approximate with Lambert for the next full leg
            next_tof_sec = tofs_days[k + 1] * 86400.0
            r_next_target = planet_pos[k + 2]
            try:
                lamb_next = pk.lambert_problem(r_target, r_next_target, next_tof_sec, pk.MU_SUN)
                v_current = np.array(lamb_next.get_v1()[0])
            except Exception:
                v_current = v_arr  # fallback
            r_current = r_target

    # --- Colours for legs ---
    leg_colors = ['#E91E63', '#4CAF50', '#9C27B0', '#FF9800', '#00BCD4', '#795548']

    # --- Plot ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Planet orbits (thin, transparent)
    unique_bodies = {}
    for i, (body, name) in enumerate(zip(seq, seq_bodies)):
        if name not in unique_bodies:
            unique_bodies[name] = body

    orbit_colors = {
        'earth': '#2196F3', 'venus': '#FF5722', 'jupiter': '#FF9800',
        'mars': '#F44336', 'saturn': '#9E9E9E',
    }
    for name, body in unique_bodies.items():
        col = orbit_colors.get(name, '#888888')
        try:
            period = body.compute_period(pk.epoch(t0_mjd, 'mjd2000'))
            t_orb = np.linspace(0, period, 200)
            xo, yo, zo = [], [], []
            for dt in t_orb:
                r, _ = body.eph(pk.epoch(t0_mjd + dt / 86400.0, 'mjd2000'))
                xo.append(r[0] / AU); yo.append(r[1] / AU); zo.append(r[2] / AU)
            ax.plot(xo, yo, zo, color=col, alpha=0.2, linewidth=1, label=f'{name.title()} orbit')
        except Exception:
            pass

    # Trajectory legs
    for k, (pre_pts, post_pts, r_dsm) in enumerate(leg_arcs):
        col = leg_colors[k % len(leg_colors)]
        label = f'Leg {k+1} ({seq_bodies[k][0].upper()}→{seq_bodies[k+1][0].upper()})'

        xs = [p[0] / AU for p in pre_pts]
        ys = [p[1] / AU for p in pre_pts]
        zs = [p[2] / AU for p in pre_pts]
        ax.plot(xs, ys, zs, color=col, linewidth=2, label=label)

        xs = [p[0] / AU for p in post_pts]
        ys = [p[1] / AU for p in post_pts]
        zs = [p[2] / AU for p in post_pts]
        ax.plot(xs, ys, zs, color=col, linewidth=2, linestyle='--')

        # DSM marker
        ax.scatter([r_dsm[0] / AU], [r_dsm[1] / AU], [r_dsm[2] / AU],
                   s=40, c='red', marker='x', zorder=5)

    # DSM legend entry
    ax.scatter([], [], [], s=40, c='red', marker='x', label='DSMs')

    # Encounter markers
    encounter_labels = [f'{seq_bodies[0].title()} dep.'] + \
                       [f'{seq_bodies[i+1].title()} flyby' for i in range(n_legs - 1)] + \
                       [f'{seq_bodies[-1].title()} arr.']
    for i, (pos, lab) in enumerate(zip(planet_pos, encounter_labels)):
        col = orbit_colors.get(seq_bodies[i], '#888888')
        ax.scatter([pos[0] / AU], [pos[1] / AU], [pos[2] / AU],
                   s=80, c=col, marker='o', label=lab, zorder=5)

    # Sun
    ax.scatter([0], [0], [0], s=200, c='gold', marker='*', label='Sun', zorder=10)

    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────
# CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    'sequence', 'launch_date', 'arrival_date', 'tof_years',
    'c3_kms2', 'vinf_dep_kms', 'delivered_mass_kg',
    'total_dsm_kms', 'dsm_per_leg_kms', 'vinf_arr_kms', 'objective_kms',
    'flyby_bodies', 'flyby_dates', 'flyby_alts_km', 'leg_tofs_days',
    'ref_study', 'ref_c3_kms2', 'ref_dsm_kms', 'ref_vinf_arr_kms',
    'ref_tof_years', 'ref_notes',
]

# Fixed row order — each sequence always occupies the same row.
SEQUENCE_ORDER = [
    '01 Direct',    # 01_direct_baseline.py
    '02 ΔV-EGA',    # 02_dvega.py
    '03 VEGA',      # 03_vega.py
    '04 VEEGA',     # 04_veega.py
    '05 VVEGA',     # 05_vvega.py
    '06 MGA',       # 06_mga.py
]

def save_to_csv(row: dict, csv_path: str = None):
    """
    Write a result row to the trade-study CSV, maintaining fixed row order.
    If the sequence already has a row, it is overwritten in place.
    """
    if csv_path is None:
        csv_path = os.path.join(OUTPUT_DIR, 'trade_study_results.csv')

    seq_label = row.get('sequence', '')

    # Read existing rows
    existing = {}
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing[r.get('sequence', '')] = r

    # Upsert
    existing[seq_label] = {col: row.get(col, '') for col in CSV_COLUMNS}

    # Write in fixed order
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for seq_name in SEQUENCE_ORDER:
            if seq_name in existing:
                writer.writerow(existing[seq_name])
        for seq_name, r in existing.items():
            if seq_name not in SEQUENCE_ORDER:
                writer.writerow(r)

    print(f"  → Saved '{seq_label}' to {csv_path}")


# ─────────────────────────────────────────────────────────────────────
# PRINTING HELPERS
# ─────────────────────────────────────────────────────────────────────

def print_summary(d: dict):
    """Pretty-print the trade-study result dict."""
    print(f"\n{'='*60}")
    print(f"  {d['sequence']} — TRADE STUDY SUMMARY")
    print(f"{'='*60}")
    print(f"  Launch:             {d['launch_date']}")
    print(f"  Arrival:            {d['arrival_date']}")
    print(f"  TOF:                {d['tof_years']} yr")
    print(f"  C3:                 {d['c3_kms2']} km²/s²")
    print(f"  V∞ dep:             {d['vinf_dep_kms']} km/s")
    print(f"  Delivered mass:     {d['delivered_mass_kg']} kg")
    print(f"  Total DSMs:         {d['total_dsm_kms']} km/s")
    if d.get('dsm_per_leg_kms'):
        print(f"  DSMs per leg:       {d['dsm_per_leg_kms']} km/s")
    print(f"  V∞ arrival (Jup):   {d['vinf_arr_kms']} km/s")
    print(f"  Objective (Σ DSMs): {d['objective_kms']} km/s")
    if d.get('flyby_bodies'):
        print(f"  Flybys:             {d['flyby_bodies']}")
        print(f"  Flyby dates:        {d['flyby_dates']}")
        print(f"  Flyby altitudes:    {d['flyby_alts_km']} km")
    print(f"  Leg TOFs:           {d['leg_tofs_days']} days")
    if d.get('ref_study'):
        print(f"  --- Reference ---")
        print(f"  Study:              {d['ref_study']}")
        print(f"  Ref C3:             {d['ref_c3_kms2']} km²/s²")
        print(f"  Ref DSM:            {d['ref_dsm_kms']} km/s")
        print(f"  Ref V∞ arr:         {d['ref_vinf_arr_kms']} km/s")
        print(f"  Ref TOF:            {d['ref_tof_years']} yr")
        print(f"  Notes:              {d['ref_notes']}")
    print(f"{'='*60}")


def print_bounds_check(best_x, udp, seq_bodies):
    """Print whether any decision variables are at their bounds."""
    n_legs = len(seq_bodies) - 1
    bounds = udp.get_bounds()
    lb, ub = np.array(bounds[0]), np.array(bounds[1])
    print("\nBounds check:")
    vinf_kms = best_x[3] / 1000.0
    vinf_lb = lb[3] / 1000.0
    vinf_ub = ub[3] / 1000.0
    at_bound = vinf_kms < vinf_lb + 0.01 or vinf_kms > vinf_ub - 0.01
    print(f"  V∞ dep: {vinf_lb:.1f} ≤ {vinf_kms:.2f} ≤ {vinf_ub:.1f} km/s  "
          f"{'⚠ AT BOUND' if at_bound else '✓'}")
    for k in range(n_legs):
        idx = 5 + 4*k
        t = best_x[idx]
        t_lb, t_ub = lb[idx], ub[idx]
        at_bound = t < t_lb + 1 or t > t_ub - 1
        print(f"  T{k+1}: {t_lb:.0f} ≤ {t:.0f} ≤ {t_ub:.0f} days  "
              f"{'⚠ AT BOUND' if at_bound else '✓'}")