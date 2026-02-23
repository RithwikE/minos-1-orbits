# %% [markdown]
# # 02 — ΔV-EGA: Earth → (DSM) → Earth → Jupiter
#
# Single Earth gravity assist with a deep space maneuver near aphelion.
# The 2016 Europa Lander SDT study baselined this class of trajectory
# (C3 ≈ 25–30 km²/s², DSM ≈ few hundred m/s, TOF ≈ 4.5–5 yr).
#
# **Sequence:** Earth → Earth → Jupiter  (2 legs, 1 flyby)
#
# **Objective:** minimize Σ(DSMs) only
#   - add_vinf_dep = False  (C3 provided by Falcon Heavy, not penalised)
#   - add_vinf_arr = False  (arrival V∞ reported separately, not penalised)
#
# **V∞ bounds:** [3.0, 5.5] km/s → C3 ∈ [9, 30.25] km²/s²
#   At C3=30, FH delivers ~12,400 kg. Prevents optimizer from using
#   unrealistically high departure energy that delivers too little mass.

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
    pk.planet.jpl_lp('earth'),
    pk.planet.jpl_lp('earth'),
    pk.planet.jpl_lp('jupiter'),
]
seq_names = ['earth', 'earth', 'jupiter']

udp = pk.trajopt.mga_1dsm(
    seq=seq,
    t0=[pk.epoch_from_string('2030-01-01 00:00:00'),
        pk.epoch_from_string('2038-01-01 00:00:00')],
    tof=[[300, 1100],     # Leg 1: E→E  ~0.8–3.0 yr (2:1 resonance ≈ 730 d)
         [400, 2200]],    # Leg 2: E→J  ~1.1–6.0 yr
    vinf=[3.0, 5.5],      # V∞ dep bounds (km/s) → C3 9–30.25 km²/s²
                           # At C3=30, FH delivers ~12,400 kg; at C3=9, ~17,700 kg
    add_vinf_dep=False,    # C3 from launch vehicle, not penalised
    add_vinf_arr=False,    # arrival V∞ NOT penalised (JOI is separate)
    tof_encoding='direct',
    multi_objective=False,
)

ref = {
    'ref_study':    '2016 SDT ΔV-EGA',
    'ref_c3':       '25–30',
    'ref_dsm':      '~0.3 (aphelion DSM)',
    'ref_vinf_arr': '~6 (at Ganymede G0)',
    'ref_tof':      '4.5–5.0',
    'ref_notes':    'SLS launch; 2:1 resonance ΔV-EGA; C3 provided by LV',
}

# %% Cell 3 — Main execution (guarded for macOS multiprocessing)
if __name__ == '__main__':
    tee = start_logging('02')

    print(f"pykep {pk.__version__}  |  pygmo {pg.__version__}")

    # --- Run 3-phase optimisation (increased compute) ---
    best_x, best_f = run_mga_optimisation(
        udp, label="ΔV-EGA (E-E-J)",
        n_islands=96, pop_size_1=32, gen_1=500, evolve_rounds_1=8, n_seeds=50,
        pop_size_2=32, gen_2=1500,
        compass_fevals=200_000,
    )

    # --- pykep pretty output ---
    print("\n" + "="*60)
    print("  ΔV-EGA — pykep pretty() output")
    print("="*60)
    udp.pretty(best_x)

    # --- Extract trade-study metrics ---
    results = extract_mga_results(udp, best_x, best_f,
                                  sequence_label='02 ΔV-EGA',
                                  seq_bodies=seq_names,
                                  seq=seq,
                                  ref=ref)
    print_summary(results)
    print_bounds_check(best_x, udp, seq_names)

    # --- Trajectory plot (custom, using propagate_lagrangian) ---
    c3 = float(results['c3_kms2'])
    dsm = float(results['total_dsm_kms'])
    vinf_arr = float(results['vinf_arr_kms'])
    tof = float(results['tof_years'])
    title = (f'ΔV-EGA: Earth → Earth → Jupiter\n'
             f'Launch {results["launch_date"][:11]}  |  TOF {tof:.2f} yr  |  '
             f'C3 = {c3:.1f} km²/s²  |  ΣDSM = {dsm:.2f} km/s  |  '
             f'V∞_arr = {vinf_arr:.1f} km/s')
    plot_trajectory(udp, best_x, seq, seq_names, title,
                    save_path=f'{OUTPUT_DIR}/02_trajectory_dvega.png')

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