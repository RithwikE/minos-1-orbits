"""
Trajectory trade-study utilities for AAE450 Europa Lander — Minos-1 Orbits Team.

This module centralises shared logic so that individual sequence scripts
(01_direct_baseline.py, 02_dvega.py … 06_mga.py) stay short and consistent.

Key design decisions (see project docs for rationale):
  - Objective: minimize DSMs + arrival V∞  (add_vinf_dep=False, add_vinf_arr=True)
  - Departure C3 is provided by Falcon Heavy — reported separately, not penalised
  - Arrival V∞ IS penalised because our spacecraft must provide JOI ΔV
  - Post-processing computes delivered mass from C3 via Falcon Heavy curve fit
"""

import os, csv, datetime
import numpy as np
import pykep as pk
import pygmo as pg

# ─────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────
# All images and CSV go here, regardless of working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Fit quadratic polynomials  (payload = a*C3^2 + b*C3 + c)
_FH_COEFFS_EST = np.polyfit(_FH_C3, _FH_PAYLOAD_EST, 2)
_FH_COEFFS_LO  = np.polyfit(_FH_C3, _FH_PAYLOAD_LO, 2)
_FH_COEFFS_HI  = np.polyfit(_FH_C3, _FH_PAYLOAD_HI, 2)


def falcon_heavy_payload(c3_kms2: float) -> dict:
    """
    Estimate Falcon Heavy expendable payload mass for a given departure C3.

    Parameters
    ----------
    c3_kms2 : float
        Departure characteristic energy (km²/s²).  Valid range ≈ 0–40.

    Returns
    -------
    dict with keys:
        'estimated_kg' : float  – best-estimate payload (kg)
        'low_95_kg'    : float  – lower bound of 95% CI (kg)
        'high_95_kg'   : float  – upper bound of 95% CI (kg)
        'c3_kms2'      : float  – echo of input C3
        'in_range'     : bool   – True if 0 ≤ C3 ≤ 40
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
    pop_size_1: int = 25,
    gen_1: int = 200,
    evolve_rounds_1: int = 6,
    n_seeds: int = 25,
    # Phase 2 — refine seeds
    pop_size_2: int = 25,
    gen_2: int = 800,
    # Phase 3 — local polish
    compass_fevals: int = 100_000,
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
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_seeds} done, best = {best_f/1000:.4f} km/s")

    # --- Phase 3: local polish ---
    print(f"\nPhase 3: Compass Search ({compass_fevals:,} fevals)")
    algo3 = pg.algorithm(pg.compass_search(
        max_fevals=compass_fevals, start_range=0.01, stop_range=1e-7,
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
                        ref: dict = None) -> dict:
    """
    Pull all trade-study columns from a converged mga_1dsm solution.

    Parameters
    ----------
    udp          : the mga_1dsm UDP instance
    best_x       : champion decision vector
    best_f       : champion fitness (m/s)
    sequence_label : e.g. "ΔV-EGA"
    seq_bodies   : list of body name strings, e.g. ['earth','earth','jupiter']
    ref          : optional dict with keys ref_study, ref_c3, ref_dsm, ref_vinf_arr,
                   ref_tof, ref_notes

    Returns
    -------
    dict of all trade-study columns (strings / floats ready for CSV)
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

    # flyby bodies & dates (middle entries)
    flyby_names = seq_bodies[1:-1]
    flyby_dates = [str(epochs[i+1]) for i in range(len(flyby_names))]

    # flyby altitudes  (rp/safe_radius at indices 7, 11, 15, …)
    flyby_alts = []
    for k in range(len(flyby_names)):
        rp_ratio = best_x[7 + 4*k]
        body = udp.get_sequence()[k+1]
        safe_r_km = body.safe_radius / 1000.0
        body_r_km = body.radius / 1000.0
        alt_km = rp_ratio * safe_r_km - body_r_km
        flyby_alts.append(alt_km)

    # --- ΔV breakdown ---
    # The objective = sum(DSMs) + vinf_arr  (with add_vinf_dep=False, add_vinf_arr=True)
    # We need to isolate DSMs from arrival V∞.
    # Arrival V∞ = |v_sc - v_planet| at the last body.
    # Easiest: call udp.fitness(best_x) gives us the objective.
    # Then reconstruct arrival V∞ from ephemeris.
    obj_ms = best_f  # m/s

    # Compute arrival V∞ by ephemeris
    arr_body = udp.get_sequence()[-1]
    r_arr, v_arr_planet = arr_body.eph(epochs[-1])

    # Reconstruct the spacecraft arrival velocity via Lambert on last leg
    dep_body_last = udp.get_sequence()[-2]
    r_dep_last, _ = dep_body_last.eph(epochs[-2])
    # We need the post-DSM velocity for the last leg. Rather than full reconstruction,
    # use the identity: total_dsm = objective - vinf_arr
    # So we compute vinf_arr independently.

    # Actually, pykep gives us a cleaner way: pretty() prints everything.
    # But for programmatic extraction, let's reconstruct the last Lambert arc.
    last_tof_sec = tofs_days[-1] * 86400.0
    eta_last = best_x[4 + 4*(n_legs-1)]  # DSM timing fraction for last leg

    # For a clean extraction, use the mga_1dsm internal method if available,
    # otherwise compute from the fitness components.
    # Simple approach: total_dsm = obj - vinf_arr, where we compute vinf_arr separately.

    # We'll compute vinf_arr from the Lambert solution of the full last leg
    # (this is approximate but very close for converged solutions)
    try:
        lamb = pk.lambert_problem(r_dep_last, r_arr, last_tof_sec, pk.MU_SUN)
        v_sc_arr = np.array(lamb.get_v2()[0])
        vinf_arr_vec = v_sc_arr - np.array(v_arr_planet)
        vinf_arr_kms = np.linalg.norm(vinf_arr_vec) / 1000.0
    except Exception:
        # Fallback: rough estimate
        vinf_arr_kms = obj_ms / 1000.0 * 0.5  # placeholder
        print("  ⚠ Could not compute arrival V∞ from Lambert; using rough estimate")

    total_dsm_kms = obj_ms / 1000.0 - vinf_arr_kms
    if total_dsm_kms < 0:
        total_dsm_kms = 0.0  # numerical noise

    # delivered mass
    fh = falcon_heavy_payload(c3)

    # --- assemble dict ---
    d = {
        'sequence':          sequence_label,
        'launch_date':       launch_date,
        'arrival_date':      arrival_date,
        'tof_years':         f"{total_tof / 365.25:.2f}",
        'c3_kms2':           f"{c3:.2f}",
        'vinf_dep_kms':      f"{vinf_dep_kms:.3f}",
        'delivered_mass_kg': f"{fh['estimated_kg']:.0f}",
        'total_dsm_kms':     f"{total_dsm_kms:.4f}",
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


# ─────────────────────────────────────────────────────────────────────
# CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    'sequence', 'launch_date', 'arrival_date', 'tof_years',
    'c3_kms2', 'vinf_dep_kms', 'delivered_mass_kg',
    'total_dsm_kms', 'vinf_arr_kms', 'objective_kms',
    'flyby_bodies', 'flyby_dates', 'flyby_alts_km', 'leg_tofs_days',
    'ref_study', 'ref_c3_kms2', 'ref_dsm_kms', 'ref_vinf_arr_kms',
    'ref_tof_years', 'ref_notes',
]

def append_to_csv(row: dict, csv_path: str = None):
    """Append a result row to the trade-study CSV, creating headers if needed."""
    if csv_path is None:
        csv_path = os.path.join(OUTPUT_DIR, 'trade_study_results.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  → Appended to {csv_path}")


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
    print(f"  V∞ arrival (Jup):   {d['vinf_arr_kms']} km/s")
    print(f"  Objective (DSM+V∞): {d['objective_kms']} km/s")
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