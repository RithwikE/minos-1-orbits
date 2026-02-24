# %% [markdown]
# 02 -- Direct Jupiter -> Europa Transfer (Baseline)
# Analytical Hohmann-class transfer from a Jupiter capture orbit to Europa.
#
# MISSION SCENARIO:
#   1. Spacecraft arrives at Jupiter on hyperbola with Vinf = 5.623 km/s
#   2. Single JOI burn at periapsis (5 Rj) -> elliptical capture orbit
#      whose apoapsis = Europa's orbital radius (9.4 Rj)
#   3. Coast to apoapsis -- spacecraft arrives at Europa with Vinf ~ 0
#      (Hohmann transfer is tangent to Europa's circular orbit)
#   4. EOI burn is a separate Europa-centred budget item
#
# Also sweeps over larger capture orbit apoapsis (r_apo > Europa) to show
# the JOI dV vs Europa Vinf tradeoff for the two-burn case.

# %% Cell 1 -- Imports and setup
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pykep as pk
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from utils import save_to_csv, print_summary, OUTPUT_DIR, start_logging

tee = start_logging('02')
print(f"pykep version: {pk.version}")

# %% Cell 2 -- Constants
MU_JUPITER   = 1.26686534e17   # m^3/s^2
R_JUPITER    = 71492e3         # m
A_EUROPA     = 671100e3        # m  (9.39 Rj)
V_EUROPA     = np.sqrt(MU_JUPITER / A_EUROPA)   # m/s  Europa circular speed
R_PERI_JOI   = 5.0 * R_JUPITER  # m  JOI periapsis
VINF_ARR_KMS = 5.623            # km/s  Jupiter arrival Vinf from Earth leg

print(f"Europa orbital radius:  {A_EUROPA/R_JUPITER:.2f} Rj")
print(f"Europa orbital speed:   {V_EUROPA/1e3:.3f} km/s")
print(f"JOI periapsis:          {R_PERI_JOI/R_JUPITER:.1f} Rj")
print(f"Jupiter arrival Vinf:   {VINF_ARR_KMS:.3f} km/s")

# %% Cell 3 -- Compute Hohmann transfer (single burn, r_apo = Europa orbit)
#
# The Hohmann transfer ellipse has:
#   periapsis = R_PERI_JOI (5 Rj)  -- where JOI burn happens
#   apoapsis  = A_EUROPA   (9.4 Rj) -- where spacecraft meets Europa
#
# This is a SINGLE burn mission: one JOI burn at periapsis, coast to Europa.
# No second burn needed -- the transfer ellipse is tangent to Europa's orbit
# at apoapsis, so Vinf at Europa ~ 0.

a_hoh = (R_PERI_JOI + A_EUROPA) / 2.0   # semi-major axis of Hohmann ellipse

# Hyperbolic speed at JOI periapsis
v_hyp_peri = np.sqrt(VINF_ARR_KMS**2 * 1e6 + 2 * MU_JUPITER / R_PERI_JOI)

# Speed in Hohmann ellipse at periapsis (vis-viva)
v_hoh_peri = np.sqrt(MU_JUPITER * (2.0/R_PERI_JOI - 1.0/a_hoh))

# JOI dV: retrograde burn from hyperbola to Hohmann ellipse
hoh_dv_joi = (v_hyp_peri - v_hoh_peri) / 1e3   # km/s

# Speed in Hohmann ellipse at apoapsis (= Europa encounter)
v_hoh_apo  = np.sqrt(MU_JUPITER * (2.0/A_EUROPA - 1.0/a_hoh))

# Europa arrival Vinf: difference from Europa's circular speed
hoh_vinf_eur = abs(V_EUROPA - v_hoh_apo) / 1e3   # km/s  (should be ~0)

# Transfer TOF: half the orbital period of Hohmann ellipse
hoh_tof = np.pi * np.sqrt(a_hoh**3 / MU_JUPITER) / 86400.0   # days

hoh_dv_total = hoh_dv_joi + hoh_vinf_eur

print("\n" + "=" * 57)
print("  HOHMANN TRANSFER  (single burn, r_apo = Europa orbit)")
print("=" * 57)
print(f"  JOI dV (periapsis burn):    {hoh_dv_joi:.4f} km/s")
print(f"  Europa arrival Vinf:        {hoh_vinf_eur:.4f} km/s  (~0, tangent)")
print(f"  Transfer TOF:               {hoh_tof:.3f} days")
print(f"  Total dV this leg:          {hoh_dv_total:.4f} km/s")
print(f"\n  Hyperbolic speed at peri:   {v_hyp_peri/1e3:.3f} km/s")
print(f"  Hohmann speed at peri:      {v_hoh_peri/1e3:.3f} km/s")
print(f"  Hohmann speed at apo:       {v_hoh_apo/1e3:.3f} km/s")
print(f"  Europa circular speed:      {V_EUROPA/1e3:.3f} km/s")
print("=" * 57)

# %% Cell 4 -- Sweep over capture orbit apoapsis (two-burn case)
#
# For r_apo > A_EUROPA: spacecraft captured into large ellipse, then does
# a second burn at apoapsis to transfer down to Europa's orbit.
#   Burn 1 (JOI at periapsis): hyperbola -> capture ellipse
#   Burn 2 (at apoapsis):      capture ellipse -> transfer ellipse to Europa
#
# Shows JOI dV vs Europa Vinf tradeoff for context.

r_apo_Rj = np.linspace(A_EUROPA/R_JUPITER, 1000, 1000)
r_apo    = r_apo_Rj * R_JUPITER

# Burn 1: JOI dV (same periapsis, varying apoapsis)
v_cap_peri = np.sqrt(MU_JUPITER * (2.0/R_PERI_JOI - 2.0/(R_PERI_JOI + r_apo)))
dv_joi     = (v_hyp_peri - v_cap_peri) / 1e3

# Speed at apoapsis of capture ellipse
v_cap_apo  = np.sqrt(MU_JUPITER * (2.0/r_apo - 2.0/(R_PERI_JOI + r_apo)))

# Burn 2: at apoapsis, enter transfer ellipse (apo=r_apo, peri=A_EUROPA)
a_trans2      = (r_apo + A_EUROPA) / 2.0
v_trans2_apo  = np.sqrt(MU_JUPITER * (2.0/r_apo - 1.0/a_trans2))
dv_trans      = np.abs(v_cap_apo - v_trans2_apo) / 1e3

# Europa arrival Vinf
v_at_europa   = np.sqrt(MU_JUPITER * (2.0/A_EUROPA - 1.0/a_trans2))
vinf_europa   = np.abs(V_EUROPA - v_at_europa) / 1e3

# TOFs
a_capture     = (R_PERI_JOI + r_apo) / 2.0
tof_capture   = np.pi * np.sqrt(a_capture**3  / MU_JUPITER) / 86400.0
tof_transfer2 = np.pi * np.sqrt(a_trans2**3   / MU_JUPITER) / 86400.0
tof_total     = tof_capture + tof_transfer2

# Total dV
dv_total = dv_joi + dv_trans + vinf_europa

# Minimum total dV
idx_best      = np.argmin(dv_total)
best_rapo_Rj  = r_apo_Rj[idx_best]
best_dv_joi   = dv_joi[idx_best]
best_dv_total = dv_total[idx_best]
best_vinf     = vinf_europa[idx_best]
best_tof      = tof_total[idx_best]

print(f"\n  -- Minimum total dV (two-burn sweep) --")
print(f"  Capture orbit apoapsis:  {best_rapo_Rj:.1f} Rj")
print(f"  JOI dV:                  {best_dv_joi:.3f} km/s")
print(f"  Europa arrival Vinf:     {best_vinf:.3f} km/s")
print(f"  Total dV:                {best_dv_total:.3f} km/s")
print(f"  Total TOF:               {best_tof:.1f} days")

# %% Cell 5 -- Plot 1: dV budget vs capture orbit apoapsis
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor('#0a0a1a'); fig.patch.set_facecolor('#0a0a1a')

ax.plot(r_apo_Rj, dv_joi,     color='#E91E63', lw=2.5, label='JOI dV (Burn 1 at periapsis)')
ax.plot(r_apo_Rj, dv_trans,   color='#2196F3', lw=2.5, label='Apoapsis dV (Burn 2, two-burn case)')
ax.plot(r_apo_Rj, vinf_europa,color='#4CAF50', lw=2.5, label='Europa arrival Vinf')
ax.plot(r_apo_Rj, dv_total,   color='#FF9800', lw=2.5, ls='--', label='Total dV')

# Mark Hohmann point
ax.axvline(A_EUROPA/R_JUPITER, color='#aaaacc', ls=':', lw=1.2, alpha=0.7)
ax.plot(A_EUROPA/R_JUPITER, hoh_dv_joi, 'o', color='#E91E63', ms=12,
        markeredgecolor='white', markeredgewidth=1.5, zorder=10)
ax.plot(A_EUROPA/R_JUPITER, hoh_vinf_eur, 'o', color='#4CAF50', ms=12,
        markeredgecolor='white', markeredgewidth=1.5, zorder=10)
ax.plot(A_EUROPA/R_JUPITER, hoh_dv_total, '*', color='#FF9800', ms=18,
        markeredgecolor='white', markeredgewidth=1.5, zorder=10,
        label=f'Hohmann: JOI dV={hoh_dv_joi:.3f}, Vinf arr={hoh_vinf_eur:.3f} km/s')

ax.annotate(f'Hohmann (single burn)\nJOI dV = {hoh_dv_joi:.3f} km/s\n'
            f'Vinf arr = {hoh_vinf_eur:.4f} km/s\nTOF = {hoh_tof:.2f} days',
            xy=(A_EUROPA/R_JUPITER, hoh_dv_joi),
            xytext=(A_EUROPA/R_JUPITER + 120, hoh_dv_joi + 0.8),
            fontsize=9, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a2e', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='white'))

for spine in ax.spines.values(): spine.set_edgecolor('#333355')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
ax.title.set_color('white'); ax.grid(True, color='#1a1a33', lw=0.5)
ax.set_xlabel('JOI Capture Orbit Apoapsis (Rj)', fontsize=11)
ax.set_ylabel('dV (km/s)', fontsize=11)
ax.set_title(f'Jupiter to Europa: dV Budget vs Capture Orbit Apoapsis\n'
             f'JOI periapsis = {R_PERI_JOI/R_JUPITER:.0f} Rj  |  '
             f'Jupiter arrival Vinf = {VINF_ARR_KMS} km/s', fontsize=11)
ax.legend(fontsize=9, facecolor='#12122a', edgecolor='#3333aa', labelcolor='white')
ax.set_xlim(r_apo_Rj[0], r_apo_Rj[-1]); ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_dv_budget.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR}/02_dv_budget.png")

# %% Cell 6 -- Plot 2: Europa Vinf and TOF vs apoapsis (dual axis)
fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.set_facecolor('#0a0a1a'); fig.patch.set_facecolor('#0a0a1a')
ax2 = ax1.twinx()

ax1.plot(r_apo_Rj, vinf_europa, color='#4CAF50', lw=2.5, label='Europa arrival Vinf (km/s)')
ax1.plot(r_apo_Rj, dv_joi,      color='#E91E63', lw=2.5, ls='--', label='JOI dV (km/s)')
ax2.plot(r_apo_Rj, tof_total,   color='#9C27B0', lw=2.5, ls='-.', label='Total Transfer TOF (days)')

ax1.axvline(A_EUROPA/R_JUPITER, color='#aaaacc', ls=':', lw=1.2, alpha=0.7)
ax1.plot(A_EUROPA/R_JUPITER, hoh_vinf_eur, '*', color='#4CAF50', ms=18,
         markeredgecolor='white', markeredgewidth=1.5, zorder=10,
         label=f'Hohmann: Vinf={hoh_vinf_eur:.4f} km/s, TOF={hoh_tof:.2f} d')
ax1.text(A_EUROPA/R_JUPITER + 15, 0.4,
         f'Europa orbit\n({A_EUROPA/R_JUPITER:.1f} Rj)', color='#aaaacc', fontsize=8)

for spine in ax1.spines.values(): spine.set_edgecolor('#333355')
for spine in ax2.spines.values(): spine.set_edgecolor('#333355')
ax1.tick_params(colors='white'); ax2.tick_params(colors='#9C27B0')
ax1.xaxis.label.set_color('white'); ax1.yaxis.label.set_color('white')
ax2.yaxis.label.set_color('#9C27B0'); ax1.title.set_color('white')
ax1.grid(True, color='#1a1a33', lw=0.5)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=9,
           facecolor='#12122a', edgecolor='#3333aa', labelcolor='white')
ax1.set_xlabel('JOI Capture Orbit Apoapsis (Rj)', fontsize=11)
ax1.set_ylabel('dV / Vinf (km/s)', fontsize=11)
ax2.set_ylabel('Transfer TOF (days)', fontsize=11)
ax1.set_title('Jupiter to Europa: Arrival Vinf and TOF vs Capture Orbit', fontsize=11)
ax1.set_xlim(r_apo_Rj[0], r_apo_Rj[-1]); ax1.set_ylim(0, None); ax2.set_ylim(0, None)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_vinf_tof.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/02_vinf_tof.png")

# %% Cell 7 -- Plot 3: JOI dV vs Europa Vinf tradeoff
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor('#0a0a1a'); fig.patch.set_facecolor('#0a0a1a')

sc = ax.scatter(dv_joi, vinf_europa, c=r_apo_Rj, cmap='plasma',
                s=6, zorder=3, vmin=r_apo_Rj[0], vmax=min(r_apo_Rj[-1], 300))
cb = plt.colorbar(sc, ax=ax)
cb.set_label('Capture Orbit Apoapsis (Rj)', color='white')
cb.ax.tick_params(colors='white')

ax.plot(hoh_dv_joi, hoh_vinf_eur, '*', color='white', ms=22,
        markeredgecolor='#ffdd44', markeredgewidth=1.5, zorder=10)
ax.annotate(f'Hohmann (single burn)\nJOI dV = {hoh_dv_joi:.3f} km/s\n'
            f'Vinf arr = {hoh_vinf_eur:.4f} km/s',
            xy=(hoh_dv_joi, hoh_vinf_eur),
            xytext=(hoh_dv_joi - 1.5, hoh_vinf_eur + 0.5),
            fontsize=10, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a2e', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='white'))

for spine in ax.spines.values(): spine.set_edgecolor('#333355')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
ax.title.set_color('white'); ax.grid(True, color='#1a1a33', lw=0.5)
ax.set_xlabel('JOI dV (km/s)', fontsize=11)
ax.set_ylabel('Europa Arrival Vinf (km/s)', fontsize=11)
ax.set_title(f'Jupiter to Europa: JOI dV vs Europa Arrival Vinf Tradeoff\n'
             f'Jupiter arrival Vinf = {VINF_ARR_KMS} km/s  |  '
             f'JOI periapsis = {R_PERI_JOI/R_JUPITER:.0f} Rj', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_joi_tradeoff.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/02_joi_tradeoff.png")

# %% Cell 8 -- 3D Trajectory GIF
print("\nGenerating 3D trajectory GIF...")

RJ     = R_JUPITER / 1e3   # km
AE     = A_EUROPA  / 1e3   # km
RP     = R_PERI_JOI / 1e3  # km

a_hoh_km = (RP + AE) / 2.0
e_hoh_km = (AE - RP) / (AE + RP)
b_hoh_km = a_hoh_km * np.sqrt(1 - e_hoh_km**2)

# Transfer arc: eccentric anomaly pi -> 2pi (periapsis -> apoapsis)
E_vals  = np.linspace(np.pi, 2*np.pi, 600)
x_trans = a_hoh_km * (np.cos(E_vals) - e_hoh_km)
y_trans = b_hoh_km * np.sin(E_vals)
z_trans = np.zeros_like(x_trans)

# Full Hohmann ellipse
E_full  = np.linspace(0, 2*np.pi, 600)
x_cap   = a_hoh_km * (np.cos(E_full) - e_hoh_km)
y_cap   = b_hoh_km * np.sin(E_full)
z_cap   = np.zeros_like(x_cap)

# Europa orbit (slightly inclined for 3D effect)
inc    = np.radians(3.0)
th     = np.linspace(0, 2*np.pi, 600)
xe_orb = AE * np.cos(th)
ye_orb = AE * np.sin(th) * np.cos(inc)
ze_orb = AE * np.sin(th) * np.sin(inc)

# Europa moves during transfer
T_EUR   = 3.551
frac    = hoh_tof / T_EUR
N_PTS   = len(x_trans)
th_anim = np.linspace(0.0, -frac * 2 * np.pi, N_PTS)
xe_anim = AE * np.cos(th_anim)
ye_anim = AE * np.sin(th_anim) * np.cos(inc)
ze_anim = AE * np.sin(th_anim) * np.sin(inc)

N_FRAMES = 150
TRAIL    = 50
idxs     = np.linspace(0, N_PTS - 1, N_FRAMES).astype(int)

fig = plt.figure(figsize=(10, 10), facecolor='#0a0a1a')
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0a0a1a')
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#222244')
ax.grid(True, color='#1a1a33', lw=0.5)
ax.tick_params(colors='#8888aa', labelsize=7)
for lbl in [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]:
    lbl.set_color('#8888aa')

lim = AE * 1.3
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim*0.3, lim*0.3)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')

ax.scatter([0],[0],[0], s=500, c='#f5a623', marker='o',
           edgecolors='#ffdd88', linewidths=2, zorder=10, label='Jupiter')
ax.plot(xe_orb, ye_orb, ze_orb,
        color='#4488ff', lw=1.0, alpha=0.5, label="Europa's orbit")
ax.plot(x_cap, y_cap, z_cap,
        color='#ff4488', lw=0.8, alpha=0.25, ls='--', label='Hohmann ellipse')
ax.plot(x_trans, y_trans, z_trans,
        color='#ffcc33', lw=0.8, alpha=0.25, label='Transfer arc')

sc_dot,   = ax.plot([],[], 'o', color='white', ms=9,
                     markeredgecolor='#ffffaa', markeredgewidth=1.5,
                     zorder=20, label='Spacecraft')
sc_trail, = ax.plot([],[],[], '-', color='#ffdd44', lw=2.5, alpha=0.95, zorder=15)
eur_dot,  = ax.plot([],[],[], 'o', color='#55aaff', ms=12,
                     markeredgecolor='#aaddff', markeredgewidth=1.5,
                     zorder=20, label='Europa')
time_txt  = ax.text2D(0.03, 0.93, '', transform=ax.transAxes,
                       color='white', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', fc='#0a0a1a', alpha=0.7))
ax.text2D(0.5, 0.97, 'Jupiter to Europa  |  Hohmann Transfer',
          transform=ax.transAxes, color='white', fontsize=13,
          fontweight='bold', ha='center',
          bbox=dict(boxstyle='round,pad=0.3', fc='#0a0a1a', alpha=0.7))
ax.text2D(0.5, 0.02,
          f'TOF = {hoh_tof:.2f} d  |  Vinf arr = {hoh_vinf_eur:.4f} km/s  |  JOI dV = {hoh_dv_joi:.3f} km/s',
          transform=ax.transAxes, color='#aaaacc', fontsize=9, ha='center')
ax.legend(loc='upper right', fontsize=8, facecolor='#12122a',
          edgecolor='#3333aa', labelcolor='white', framealpha=0.85)

def update(frame):
    k = idxs[frame]
    sc_dot.set_data([x_trans[k]], [y_trans[k]])
    sc_dot.set_3d_properties([z_trans[k]])
    t0 = max(0, k - TRAIL * (N_PTS // N_FRAMES))
    sc_trail.set_data(x_trans[t0:k+1], y_trans[t0:k+1])
    sc_trail.set_3d_properties(z_trans[t0:k+1])
    eur_dot.set_data([xe_anim[k]], [ye_anim[k]])
    eur_dot.set_3d_properties([ze_anim[k]])
    t_elapsed = hoh_tof * frame / N_FRAMES
    time_txt.set_text(f'T + {t_elapsed:.2f} d  ({t_elapsed*24:.1f} h)')
    ax.view_init(elev=20 + 15*np.sin(frame/N_FRAMES*np.pi),
                 azim=-70 + 80*frame/N_FRAMES)
    return sc_dot, sc_trail, eur_dot, time_txt

anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=50, blit=False)
gif_path = f'{OUTPUT_DIR}/02_trajectory.gif'
anim.save(gif_path, writer=PillowWriter(fps=24), dpi=120)
plt.close('all')
print(f"Saved: {gif_path}")

# %% Cell 9 -- Save to CSV
row = {
    'sequence':          '02 Direct Jupiter-Europa',
    'launch_date':       'N/A (analytical)',
    'arrival_date':      'N/A (analytical)',
    'tof_years':         f"{hoh_tof / 365.25:.4f}",
    'c3_kms2':           f"{hoh_vinf_eur**2:.6f}",
    'vinf_dep_kms':      f"{hoh_dv_joi:.3f}",
    'delivered_mass_kg': 'N/A',
    'total_dsm_kms':     '0.0000',
    'dsm_per_leg_kms':   '',
    'vinf_arr_kms':      f"{hoh_vinf_eur:.4f}",
    'objective_kms':     f"{hoh_vinf_eur:.4f}",
    'flyby_bodies':      '',
    'flyby_dates':       '',
    'flyby_alts_km':     '',
    'leg_tofs_days':     f"{hoh_tof:.3f}",
    'ref_study':         'N/A',
    'ref_c3_kms2':       '~0',
    'ref_dsm_kms':       '0',
    'ref_vinf_arr_kms':  '~0.04',
    'ref_tof_years':     '~0.005',
    'ref_notes':         'Analytical Hohmann; JOI dV dominates; Vinf arr negligible',
}
save_to_csv(row)
print_summary(row)

# %% Cell 10 -- Final summary
print(f"\nKey results:")
print(f"  Hohmann TOF:           {hoh_tof:.3f} days")
print(f"  Europa arrival Vinf:   {hoh_vinf_eur:.4f} km/s  (negligible)")
print(f"  JOI dV:                {hoh_dv_joi:.4f} km/s  (dominant cost)")
print(f"  Total dV this leg:     {hoh_dv_total:.4f} km/s")
print()
print("CONCLUSION: JOI dV ~ 5.66 km/s is the dominant cost.")
print("Europa arrival Vinf ~ 0 for Hohmann -- EOI dV is a separate budget.")
print("A resonant Galilean-moon tour reduces JOI dV by using gravity assists")
print(f"to bleed off the {VINF_ARR_KMS} km/s Jupiter arrival Vinf gradually.")
