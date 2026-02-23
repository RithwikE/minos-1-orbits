# %% [markdown]
# # 03 — VEGA: Earth → Venus → Earth → Jupiter
#
# Venus-Earth gravity assist. The Venus flyby pumps up the spacecraft's
# heliocentric energy, setting up an Earth return flyby at higher V∞
# that then sends the spacecraft toward Jupiter.
#
# **Sequence:** Earth → Venus → Earth → Jupiter  (3 legs, 2 flybys)
#
# **Objective:** minimize Σ(DSMs) only
#   - add_vinf_dep = False  (C3 provided by Falcon Heavy)
#   - add_vinf_arr = False  (arrival V∞ reported separately)
#
# VEGA typically requires lower C3 than ΔV-EGA (~15–20 km²/s²) since
# Venus provides a free energy boost. TOF is usually ~4–6 years.
# The 2012 Europa study notes VEGA delivers ~1740 kg (Delta IV Heavy)
# for 4.4 yr flights, underperforming VEEGA but beating ΔV-EGA on mass.

# %% Cell 1 — Imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pykep as pk
import pygmo as pg

from utils import (
    falcon_heavy_payload, run_mga_optimisation,
    extract_mga_results, save_to_csv, print_summary, print_bounds_check,
    plot_trajectory, OUTPUT_DIR, start_logging
)

# %% Cell 2 — Define problem
seq = [
    pk.planet.jpl_lp('earth'),    # departure
    pk.planet.jpl_lp('venus'),    # flyby 1
    pk.planet.jpl_lp('earth'),    # flyby 2
    pk.planet.jpl_lp('jupiter'),  # arrival
]
seq_names = ['earth', 'venus', 'earth', 'jupiter']

udp = pk.trajopt.mga_1dsm(
    seq=seq,
    t0=[pk.epoch_from_string('2030-01-01 00:00:00'),
        pk.epoch_from_string('2038-01-01 00:00:00')],
    tof=[[100, 500],      # Leg 1: E→V (days) — inner transfer, ~0.3–1.4 yr
                           #   type I (~150d) or type II (~350-450d) both valid
         [100, 1000],     # Leg 2: V→E (days) — return to Earth, ~0.3–2.7 yr
                           #   widened from 700; phasing may need longer arcs
         [500, 2200]],    # Leg 3: E→J (days) — outer transfer, ~1.4–6.0 yr
    vinf=[1.5, 7.0],      # V∞ dep bounds (km/s) → C3 2.25–49 km²/s²
                           # wide range: let optimizer find natural VEGA C3
    add_vinf_dep=False,    # C3 from launch vehicle, not penalised
    add_vinf_arr=False,    # arrival V∞ NOT penalised (JOI is separate)
    tof_encoding='direct',
    multi_objective=False,
)

ref = {
    'ref_study':    '2012 Europa Study (VEGA class)',
    'ref_c3':       '15–20',
    'ref_dsm':      '~0.1–0.5 (after Venus flyby)',
    'ref_vinf_arr': '~6–7 (at Ganymede G0)',
    'ref_tof':      '4.4–5.4',
    'ref_notes':    'VEGA underperforms VEEGA on mass but shorter TOF; Delta IV Heavy baseline',
}

# %% Cell 3 — Main execution (guarded for macOS multiprocessing)
if __name__ == '__main__':
    tee = start_logging('03')

    print(f"pykep {pk.__version__}  |  pygmo {pg.__version__}")

    # 3 legs → 14-dimensional search space, significantly harder than 2-leg
    # Need aggressive compute to find proper VEGA phasing windows
    best_x, best_f = run_mga_optimisation(
        udp, label="VEGA (E-V-E-J)",
        n_islands=128, pop_size_1=48, gen_1=800, evolve_rounds_1=12, n_seeds=80,
        pop_size_2=48, gen_2=2500,
        compass_fevals=500_000,
    )

    # --- pykep pretty output ---
    print("\n" + "="*60)
    print("  VEGA — pykep pretty() output")
    print("="*60)
    udp.pretty(best_x)

    # --- Extract trade-study metrics ---
    results = extract_mga_results(udp, best_x, best_f,
                                  sequence_label='03 VEGA',
                                  seq_bodies=seq_names,
                                  seq=seq,
                                  ref=ref)
    print_summary(results)
    print_bounds_check(best_x, udp, seq_names)

    # --- Trajectory plot ---
    c3 = float(results['c3_kms2'])
    dsm = float(results['total_dsm_kms'])
    vinf_arr = float(results['vinf_arr_kms'])
    tof = float(results['tof_years'])
    title = (f'VEGA: Earth → Venus → Earth → Jupiter\n'
             f'Launch {results["launch_date"][:11]}  |  TOF {tof:.2f} yr  |  '
             f'C3 = {c3:.1f} km²/s²  |  ΣDSM = {dsm:.2f} km/s  |  '
             f'V∞_arr = {vinf_arr:.1f} km/s')
    plot_trajectory(udp, best_x, seq, seq_names, title,
                    save_path=f'{OUTPUT_DIR}/03_trajectory_vega.png')

    # --- Save to CSV ---
    save_to_csv(results)

    # --- Diagnostics ---
    n_legs = len(seq_names) - 1
    print("\nDecision vector breakdown:")
    print(f"  t0 (mjd2000):       {best_x[0]:.2f}")
    print(f"  u, v (V∞ dir):      {best_x[1]:.4f}, {best_x[2]:.4f}")
    print(f"  V∞ dep:             {best_x[3]:.1f} m/s = {best_x[3]/1000:.3f} km/s")
    for k in range(n_legs):
        i_eta = 4 + 4*k
        i_T   = 5 + 4*k
        print(f"  Leg {k+1}: η={best_x[i_eta]:.4f}, T={best_x[i_T]:.1f} d ({best_x[i_T]/365.25:.2f} yr)")
        if k < n_legs - 1:
            i_beta = 6 + 4*k
            i_rp   = 7 + 4*k
            print(f"         β={best_x[i_beta]:.4f} rad, rp/safe_r={best_x[i_rp]:.4f}")

    print(f"\n  Objective (m/s): {best_f:.1f}")
    print(f"  Objective (km/s): {best_f/1000:.4f}")
    print(f"  → This = Σ(DSMs) only. Arrival V∞ = {vinf_arr:.3f} km/s (NOT included).")
    print(f"  → C3 = {c3:.1f} → FH delivers {results['delivered_mass_kg']} kg")