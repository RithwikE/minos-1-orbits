# %% [markdown]
# # 01 — Direct Earth → Jupiter Transfer (Baseline)
#
# Establishes the baseline: direct Hohmann-class transfer from Earth to Jupiter
# with no gravity assists. Demonstrates that direct transfer requires prohibitively
# high C3, motivating gravity-assist trajectories.
#
# **Method:** Lambert solver (pykep) over a grid of launch dates and TOFs.

# %% Cell 1 — Imports and setup
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import pykep as pk

from utils import falcon_heavy_payload, save_to_csv, print_summary, OUTPUT_DIR

earth   = pk.planet.jpl_lp('earth')
jupiter = pk.planet.jpl_lp('jupiter')
print(f"pykep version: {pk.__version__}")

# %% Cell 2 — Define porkchop grid
# Launch window: 2030–2038, TOF: 1–5 years (Hohmann ≈ 2.73 yr)
launch_start = pk.epoch_from_string('2030-01-01 00:00:00').mjd2000
launch_end   = pk.epoch_from_string('2038-01-01 00:00:00').mjd2000

tof_min = 1.0 * 365.25
tof_max = 5.0 * 365.25

n_launch, n_tof = 300, 200
launch_dates = np.linspace(launch_start, launch_end, n_launch)
tofs = np.linspace(tof_min, tof_max, n_tof)
print(f"Grid: {n_launch} × {n_tof} = {n_launch * n_tof:,} Lambert problems")

# %% Cell 3 — Compute porkchop
C3_grid       = np.full((n_tof, n_launch), np.nan)
vinf_arr_grid = np.full((n_tof, n_launch), np.nan)

for i, t0 in enumerate(launch_dates):
    r_e, v_e = earth.eph(pk.epoch(t0, 'mjd2000'))
    for j, tof in enumerate(tofs):
        r_j, v_j = jupiter.eph(pk.epoch(t0 + tof, 'mjd2000'))
        try:
            lamb = pk.lambert_problem(r_e, r_j, tof * pk.DAY2SEC, pk.MU_SUN, False)
            vd = np.array(lamb.get_v1()[0]) - np.array(v_e)
            va = np.array(lamb.get_v2()[0]) - np.array(v_j)
            C3_grid[j, i]       = (np.linalg.norm(vd) / 1000.0) ** 2
            vinf_arr_grid[j, i] = np.linalg.norm(va) / 1000.0
        except Exception:
            pass
    if (i + 1) % 60 == 0:
        print(f"  {100*(i+1)//n_launch}% complete")

print("Done.")

# %% Cell 4 — Find minimum C3
idx = np.nanargmin(C3_grid)
j_best, i_best = np.unravel_index(idx, C3_grid.shape)

best_c3       = C3_grid[j_best, i_best]
best_vinf_dep = np.sqrt(best_c3)
best_vinf_arr = vinf_arr_grid[j_best, i_best]
best_launch   = launch_dates[i_best]
best_tof      = tofs[j_best]

launch_ep = pk.epoch(best_launch, 'mjd2000')
arrive_ep = pk.epoch(best_launch + best_tof, 'mjd2000')

fh = falcon_heavy_payload(best_c3)

print("=" * 55)
print("  DIRECT TRANSFER — MINIMUM C3 SOLUTION")
print("=" * 55)
print(f"  Launch:           {launch_ep}")
print(f"  Arrival:          {arrive_ep}")
print(f"  TOF:              {best_tof:.0f} days ({best_tof/365.25:.2f} yr)")
print(f"  C3:               {best_c3:.2f} km²/s²")
print(f"  V∞ dep:           {best_vinf_dep:.3f} km/s")
print(f"  V∞ arr (Jupiter): {best_vinf_arr:.3f} km/s")
print(f"  FH payload:       {fh['estimated_kg']:.0f} kg (in range: {fh['in_range']})")
print("=" * 55)

# %% Cell 5 — Porkchop plot (C3)
fig, ax = plt.subplots(figsize=(14, 8))
L_yr = 2000 + launch_dates / 365.25
T_yr = tofs / 365.25
Lm, Tm = np.meshgrid(L_yr, T_yr)

c3_plot = np.clip(C3_grid, 0, 250)
levels = np.arange(70, 260, 10)
cf = ax.contourf(Lm, Tm, c3_plot, levels=levels, cmap='RdYlGn_r', extend='both')
cs = ax.contour(Lm, Tm, c3_plot, levels=[80, 100, 120, 150, 200],
                colors='k', linewidths=0.8, alpha=0.5)
ax.clabel(cs, fmt='%.0f', fontsize=8)
plt.colorbar(cf, ax=ax, label='Departure C3 (km²/s²)')

bx = 2000 + best_launch / 365.25
by = best_tof / 365.25
ax.plot(bx, by, 'w*', markersize=20, markeredgecolor='k', markeredgewidth=1.5, zorder=10)
ax.annotate(f'Min C3 = {best_c3:.1f}\n{launch_ep}\nTOF = {by:.2f} yr',
            xy=(bx, by), xytext=(bx+0.5, by+0.4), fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.9),
            arrowprops=dict(arrowstyle='->'))

ax.set_xlabel('Launch Date (year)')
ax.set_ylabel('Time of Flight (years)')
ax.set_title('Earth → Jupiter Direct — Departure C3 (km²/s²)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_porkchop_c3.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/01_porkchop_c3.png")

# %% Cell 6 — Porkchop plot (arrival V∞)
fig, ax = plt.subplots(figsize=(14, 8))
vplot = np.clip(vinf_arr_grid, 0, 20)
cf = ax.contourf(Lm, Tm, vplot, levels=np.arange(4, 20, 1), cmap='viridis_r', extend='both')
cs = ax.contour(Lm, Tm, vplot, levels=[5, 6, 7, 8, 10, 12], colors='w', linewidths=0.8)
ax.clabel(cs, fmt='%.0f', fontsize=8)
plt.colorbar(cf, ax=ax, label='V∞ at Jupiter (km/s)')
ax.plot(bx, by, 'r*', markersize=20, markeredgecolor='k', markeredgewidth=1.5, zorder=10)
ax.set_xlabel('Launch Date (year)')
ax.set_ylabel('Time of Flight (years)')
ax.set_title('Earth → Jupiter Direct — Jupiter Arrival V∞ (km/s)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_porkchop_vinf.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/01_porkchop_vinf.png")

# %% Cell 7 — Append to CSV
row = {
    'sequence':          'Direct',
    'launch_date':       str(launch_ep),
    'arrival_date':      str(arrive_ep),
    'tof_years':         f"{best_tof/365.25:.2f}",
    'c3_kms2':           f"{best_c3:.2f}",
    'vinf_dep_kms':      f"{best_vinf_dep:.3f}",
    'delivered_mass_kg': f"{fh['estimated_kg']:.0f}",
    'total_dsm_kms':     '0.0000',
    'dsm_per_leg_kms':   '',
    'vinf_arr_kms':      f"{best_vinf_arr:.3f}",
    'objective_kms':     f"{best_vinf_arr:.3f}",
    'flyby_bodies':      '',
    'flyby_dates':       '',
    'flyby_alts_km':     '',
    'leg_tofs_days':     f"{best_tof:.1f}",
    'ref_study':         'N/A',
    'ref_c3_kms2':       '~77',
    'ref_dsm_kms':       '0',
    'ref_vinf_arr_kms':  '~6',
    'ref_tof_years':     '~2.7',
    'ref_notes':         'Hohmann min-energy; C3≈77 rules out all LVs at mission mass',
}
save_to_csv(row)
print_summary(row)

# %% Cell 8 — Final summary
print(f"\nSolutions with C3 < 100: {np.sum(C3_grid < 100):,}")
print(f"Solutions with C3 <  80: {np.sum(C3_grid < 80):,}")
print(f"Solutions with C3 <  40: {np.sum(C3_grid < 40):,}")
print("\nCONCLUSION: Minimum C3 ≈ 77 km²/s². Falcon Heavy cannot deliver mission")
print("mass at this C3. Gravity assists are required.")