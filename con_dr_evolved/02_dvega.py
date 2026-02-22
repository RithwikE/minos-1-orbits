# %% [markdown]
# # 02 — ΔV-EGA: Earth → (DSM) → Earth → Jupiter
#
# Single Earth gravity assist with a deep space maneuver near aphelion.
# The 2016 Europa Lander SDT study baselined this class of trajectory
# (C3 ≈ 25–30 km²/s², DSM ≈ few hundred m/s, TOF ≈ 4.5–5 yr).
#
# **Sequence:** Earth → Earth → Jupiter  (2 legs, 1 flyby)
#
# **Objective:** minimize Σ(DSMs) + arrival V∞ at Jupiter
#   - add_vinf_dep = False  (C3 provided by Falcon Heavy, not penalised)
#   - add_vinf_arr = True   (spacecraft must supply JOI ΔV)
#
# **IMPORTANT:** pykep mga_1dsm stores V∞_dep in m/s internally.

# %% Cell 1 — Imports
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pykep as pk
import pygmo as pg

from utils import (
    falcon_heavy_payload, run_mga_optimisation,
    extract_mga_results, append_to_csv, print_summary, print_bounds_check,
    OUTPUT_DIR,
)

print(f"pykep {pk.__version__}  |  pygmo {pg.__version__}")

# %% Cell 2 — Define problem
#
# ΔV-EGA: Earth → Earth → Jupiter
#   Leg 1 (E→E): Launch, DSM near aphelion, return to Earth. ~1–3 yr.
#   Leg 2 (E→J): Post-flyby cruise to Jupiter. ~1.5–5.5 yr.
#
# Launch window 2030-2038 covers multiple Earth-Jupiter synodic opportunities.
# V∞ up to 7 km/s → C3 up to 49 km²/s² (FH delivers ~10,500 kg there).

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
    tof=[[300, 1100],     # Leg 1: E→E  ~0.8–3.0 yr
         [400, 2200]],    # Leg 2: E→J  ~1.1–6.0 yr
    vinf=[0.5, 7.0],      # V∞ dep bounds (km/s) → C3 0.25–49 km²/s²
    add_vinf_dep=False,    # ← KEY FIX: C3 from launch vehicle, not penalised
    add_vinf_arr=True,     # ← spacecraft must kill arrival V∞ via JOI
    tof_encoding='direct',
    multi_objective=False,
)

# %% Cell 3 — Run 3-phase optimisation
best_x, best_f = run_mga_optimisation(
    udp, label="ΔV-EGA (E-E-J)",
    # Phase 1: broad — more islands since this is the template run
    n_islands=96, pop_size_1=25, gen_1=250, evolve_rounds_1=6, n_seeds=25,
    # Phase 2: refine
    pop_size_2=25, gen_2=800,
    # Phase 3: local
    compass_fevals=100_000,
)

# %% Cell 4 — Pretty-print pykep's own summary
print("\n" + "="*60)
print("  ΔV-EGA — pykep pretty() output")
print("="*60)
udp.pretty(best_x)

# %% Cell 5 — Extract trade-study metrics
ref = {
    'ref_study':    '2016 SDT ΔV-EGA',
    'ref_c3':       '25–30',
    'ref_dsm':      '~0.3 (aphelion DSM)',
    'ref_vinf_arr': '~6 (at Ganymede G0)',
    'ref_tof':      '4.5–5.0',
    'ref_notes':    'SLS launch; 2:1 resonance ΔV-EGA; C3 provided by LV',
}

results = extract_mga_results(udp, best_x, best_f,
                              sequence_label='ΔV-EGA',
                              seq_bodies=seq_names,
                              ref=ref)
print_summary(results)
print_bounds_check(best_x, udp, seq_names)

# %% Cell 6 — Trajectory plot (pykep built-in)
fig, ax = plt.subplots(figsize=(10, 10))
udp.plot(best_x, ax=ax)

c3 = float(results['c3_kms2'])
obj = float(results['objective_kms'])
tof = float(results['tof_years'])
ax.set_title(f'ΔV-EGA: Earth → Earth → Jupiter\n'
             f'Launch {results["launch_date"]}  |  TOF {tof:.2f} yr  |  '
             f'C3 = {c3:.1f} km²/s²  |  Obj = {obj:.2f} km/s',
             fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_trajectory_dvega.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/02_trajectory_dvega.png")

# %% Cell 7 — Append to CSV
append_to_csv(results)

# %% Cell 8 — Diagnostics
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
print(f"  → This = Σ(DSMs) + V∞_arr (departure V∞ NOT included)")