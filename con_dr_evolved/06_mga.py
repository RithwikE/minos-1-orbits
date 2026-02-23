# %% [markdown]
# # 06 — MGA (Mars): Earth → Mars → Jupiter
#
# Mars gravity assist. Mars lies between Earth and Jupiter, making it a
# natural waypoint. The Mars flyby can redirect the spacecraft toward
# Jupiter while adding some energy from Mars's orbital velocity.
#
# **Sequence:** Earth → Mars → Jupiter  (2 legs, 1 flyby)
#
# **Objective:** minimize Σ(DSMs) only
#   - add_vinf_dep = False  (C3 provided by Falcon Heavy)
#   - add_vinf_arr = False  (arrival V∞ reported separately)
#
# Mars flybys for Jupiter missions are opportunistic — they depend on
# favourable Mars-Jupiter alignment. The 2016/2018 Europa Lander study
# used Earth+Mars gravity assists for their 2026/2028 launch windows.
# Mars provides less ΔV boost than Venus (smaller mass) but doesn't
# require going sunward first, potentially saving flight time.

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
    plot_trajectory, OUTPUT_DIR,
)

# %% Cell 2 — Define problem
seq = [
    pk.planet.jpl_lp('earth'),    # departure
    pk.planet.jpl_lp('mars'),     # flyby
    pk.planet.jpl_lp('jupiter'),  # arrival
]
seq_names = ['earth', 'mars', 'jupiter']

udp = pk.trajopt.mga_1dsm(
    seq=seq,
    t0=[pk.epoch_from_string('2030-01-01 00:00:00'),
        pk.epoch_from_string('2038-01-01 00:00:00')],
    tof=[[150, 800],      # Leg 1: E→M (days) — ~0.4–2.2 yr
                           #   Hohmann ~259d; type II transfers longer
         [200, 2500]],    # Leg 2: M→J (days) — ~0.5–6.8 yr
                           #   depends heavily on alignment
    vinf=[1.5, 7.0],      # V∞ dep bounds (km/s) → C3 2.25–49 km²/s²
    add_vinf_dep=False,    # C3 from launch vehicle, not penalised
    add_vinf_arr=False,    # arrival V∞ NOT penalised (JOI is separate)
    tof_encoding='direct',
    multi_objective=False,
)

ref = {
    'ref_study':    '2018 JPL Europa Lander (EGA+Mars, 2026 launch)',
    'ref_c3':       '~20–30 (varies with alignment window)',
    'ref_dsm':      '~0.1–0.5',
    'ref_vinf_arr': '~6–7',
    'ref_tof':      '5–7 (Earth-Mars-Jupiter portion)',
    'ref_notes':    'Mars GA opportunistic; alignment dependent; SLS baseline in JPL study',
}

# %% Cell 3 — Main execution
if __name__ == '__main__':

    print(f"pykep {pk.__version__}  |  pygmo {pg.__version__}")

    # 2 legs → 10-dimensional, same as ΔV-EGA
    # But Mars alignment windows are narrow, so still need good coverage
    best_x, best_f = run_mga_optimisation(
        udp, label="MGA-Mars (E-M-J)",
        n_islands=96, pop_size_1=32, gen_1=500, evolve_rounds_1=8, n_seeds=50,
        pop_size_2=32, gen_2=1500,
        compass_fevals=200_000,
    )

    # --- pykep pretty output ---
    print("\n" + "="*60)
    print("  MGA (Mars) — pykep pretty() output")
    print("="*60)
    udp.pretty(best_x)

    # --- Extract trade-study metrics ---
    results = extract_mga_results(udp, best_x, best_f,
                                  sequence_label='06 MGA',
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
    title = (f'MGA: Earth → Mars → Jupiter\n'
             f'Launch {results["launch_date"][:11]}  |  TOF {tof:.2f} yr  |  '
             f'C3 = {c3:.1f} km²/s²  |  ΣDSM = {dsm:.2f} km/s  |  '
             f'V∞_arr = {vinf_arr:.1f} km/s')
    plot_trajectory(udp, best_x, seq, seq_names, title,
                    save_path=f'{OUTPUT_DIR}/06_trajectory_mga.png')

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