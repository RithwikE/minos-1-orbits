# %% [markdown]
# # 04 — VEEGA Deep Dive: Earth → Venus → Earth → Earth → Jupiter
#
# Venus-Earth-Earth gravity assist — the 2012 Europa Lander study baseline.
# This is the comprehensive deep-dive version with:
#   - Tightened V∞ bounds to push C3 toward reference ~15 km²/s²
#   - Full 2030-2038 launch window (safe TOF caps prevent ephemeris overflow)
#   - Heavier compute budget
#   - Animated GIF of the trajectory
#
# **Sequence:** Earth → Venus → Earth → Earth → Jupiter  (4 legs, 3 flybys)
#
# **Objective:** minimize Σ(DSMs) only
#   - add_vinf_dep = False  (C3 provided by Falcon Heavy)
#   - add_vinf_arr = False  (arrival V∞ reported separately)
#
# **2012 reference:** C3 ≈ 15 km²/s², DSM ≈ 0-100 m/s, TOF ≈ 6.4 yr,
# V∞ arrival ≈ 7.4 km/s (at Ganymede G0). Delta IV Heavy baseline.

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
    plot_trajectory, OUTPUT_DIR, start_logging,
    _extract_vinf_dep_vec,
)

# %% Cell 2 — Define problem

seq = [
    pk.planet.jpl_lp('earth'),    # departure
    pk.planet.jpl_lp('venus'),    # flyby 1
    pk.planet.jpl_lp('earth'),    # flyby 2
    pk.planet.jpl_lp('earth'),    # flyby 3
    pk.planet.jpl_lp('jupiter'),  # arrival
]
seq_names = ['earth', 'venus', 'earth', 'earth', 'jupiter']

# --- Bounds rationale ---
# V∞ dep: [2.5, 5.5] -> C3 = [6.25, 30.25] km²/s²
#   Pushes optimizer toward reference C3 ~ 15 (V∞ ~ 3.87)
#   Previous run found V∞ = 6.72 (C3 = 45) with wider bounds
#
# TOF caps chosen so max total = 3500d = 9.58 yr
#   -> worst case: 2038 + 9.58 = 2047.6 < 2050 ephemeris limit
#
# Leg bounds:
#   E->V: [100, 400]   inner transfer (Hohmann ~150d, type-II ~300d)
#   V->E: [100, 700]   return to Earth (ref ~500d)
#   E->E: [300, 900]   resonant orbit: 2:1 (~365d) or 1:1 (~730d)
#   E->J: [400, 1500]  outer transfer to Jupiter (~2-4 yr)

udp = pk.trajopt.mga_1dsm(
    seq=seq,
    t0=[pk.epoch_from_string('2030-01-01 00:00:00'),
        pk.epoch_from_string('2038-01-01 00:00:00')],
    tof=[[100, 400],      # Leg 1: E->V
         [100, 700],      # Leg 2: V->E
         [300, 900],      # Leg 3: E->E (resonant)
         [400, 1500]],    # Leg 4: E->J
    vinf=[2.5, 5.5],      # V∞ dep (km/s) -> C3 6.25-30.25 km²/s²
    add_vinf_dep=False,
    add_vinf_arr=False,
    tof_encoding='direct',
    multi_objective=False,
)

ref = {
    'ref_study':    '2012 Europa Study (VEEGA baseline)',
    'ref_c3':       '~15 (14.2 exact)',
    'ref_dsm':      '0-0.1 (optimal day zero; up to 0.1 over launch period)',
    'ref_vinf_arr': '~7.4 (V∞ at Ganymede G0, pre-JOI)',
    'ref_tof':      '6.36',
    'ref_notes':    'Delta IV Heavy; V flyby 3184km, E flybys 11764/3336km; DSM on E-V leg',
}


# %% Cell 3 — Trajectory reconstruction for GIF

def compute_legs_data(best_x, seq, seq_names):
    """
    Reconstruct full trajectory from mga_1dsm decision vector.
    Returns a list of leg dicts with positions/velocities for propagation.
    """
    from pykep import propagate_lagrangian, lambert_problem, AU, MU_SUN

    n_legs = len(seq_names) - 1
    t0_mjd = best_x[0]

    # Parse TOFs and etas
    tofs_days = [best_x[5 + 4*k] for k in range(n_legs)]
    etas      = [best_x[4 + 4*k] for k in range(n_legs)]

    # Cumulative epochs
    cum_days = np.cumsum([0.0] + tofs_days)
    epochs_mjd = [t0_mjd + d for d in cum_days]

    # Planet states at each epoch
    planet_pos = []
    planet_vel = []
    for i, body in enumerate(seq):
        r, v = body.eph(pk.epoch(epochs_mjd[i], 'mjd2000'))
        planet_pos.append(np.array(r))
        planet_vel.append(np.array(v))

    # Departure velocity
    vinf_vec = _extract_vinf_dep_vec(best_x)
    v_sc = planet_vel[0] + vinf_vec

    legs = []
    r_current = planet_pos[0]
    v_current = v_sc

    for k in range(n_legs):
        tof_sec = tofs_days[k] * 86400.0
        eta = etas[k]
        dt_pre  = eta * tof_sec
        dt_post = (1.0 - eta) * tof_sec
        r_target = planet_pos[k + 1]

        # Propagate to DSM
        r_dsm, v_pre_dsm = propagate_lagrangian(
            r_current, v_current, dt_pre, MU_SUN)

        # Lambert from DSM to next planet
        lamb = lambert_problem(r_dsm, r_target, dt_post, MU_SUN)
        v_post_dsm = np.array(lamb.get_v1()[0])
        v_arr      = np.array(lamb.get_v2()[0])

        dsm_mag = np.linalg.norm(v_post_dsm - np.array(v_pre_dsm))

        legs.append({
            'r_start':     np.array(r_current),
            'v_start':     np.array(v_current),
            'r_dsm':       np.array(r_dsm),
            'v_pre_dsm':   np.array(v_pre_dsm),
            'v_post_dsm':  v_post_dsm,
            'r_end':       np.array(r_target),
            'v_arr':       v_arr,
            'dt_pre':      dt_pre,
            'dt_post':     dt_post,
            'dsm_mag':     dsm_mag,
            't_start_mjd': epochs_mjd[k],
            't_end_mjd':   epochs_mjd[k + 1],
        })

        # Set up next leg (approximate post-flyby velocity via Lambert)
        if k < n_legs - 1:
            next_tof_sec = tofs_days[k + 1] * 86400.0
            r_next = planet_pos[k + 2]
            try:
                lamb_next = lambert_problem(r_target, r_next, next_tof_sec, MU_SUN)
                v_current = np.array(lamb_next.get_v1()[0])
            except Exception:
                v_current = v_arr
            r_current = r_target

    return legs, t0_mjd, epochs_mjd


def make_trajectory_gif(best_x, best_f, seq, seq_names, results, save_path):
    """
    Create an animated GIF of the VEEGA trajectory.
    """
    from pykep import propagate_lagrangian, AU
    from matplotlib.animation import FuncAnimation, PillowWriter

    legs, t0_mjd, epochs_mjd = compute_legs_data(best_x, seq, seq_names)

    fps = 12
    days_per_frame = 12
    max_frames = 500

    # Segment boundaries in seconds (pre-DSM, post-DSM for each leg)
    seg_bounds = [0.0]
    for ld in legs:
        seg_bounds.append(seg_bounds[-1] + ld['dt_pre'])
        seg_bounds.append(seg_bounds[-1] + ld['dt_post'])

    t_total = seg_bounds[-1]
    sec_per_frame = days_per_frame * 86400.0
    n_frames = int(min(max_frames, np.ceil(t_total / sec_per_frame))) + 1
    t_grid = np.linspace(0.0, t_total, n_frames)

    # Unique planet objects for ephemeris
    planet_map = {}
    for name, body in zip(seq_names, seq):
        if name not in planet_map:
            planet_map[name] = body
    tracked = list(planet_map.keys())  # e.g. ['earth', 'venus', 'jupiter']

    # Precompute positions
    sc_pos = np.zeros((n_frames, 3))
    pl_pos = {name: np.zeros((n_frames, 3)) for name in tracked}

    for k, t in enumerate(t_grid):
        # Find segment
        for seg_idx in range(len(seg_bounds) - 1):
            if t <= seg_bounds[seg_idx + 1] or seg_idx == len(seg_bounds) - 2:
                break

        leg_idx = seg_idx // 2
        is_post_dsm = seg_idx % 2 == 1
        dt_local = t - seg_bounds[seg_idx]
        ld = legs[leg_idx]

        if is_post_dsm:
            r, _ = propagate_lagrangian(ld['r_dsm'], ld['v_post_dsm'],
                                        dt_local, pk.MU_SUN)
        else:
            r, _ = propagate_lagrangian(ld['r_start'], ld['v_start'],
                                        dt_local, pk.MU_SUN)
        sc_pos[k] = np.array(r) / AU

        # Planet positions
        abs_mjd = t0_mjd + t / 86400.0
        ep = pk.epoch(abs_mjd, 'mjd2000')
        for name in tracked:
            rp, _ = planet_map[name].eph(ep)
            pl_pos[name][k] = np.array(rp) / AU

    # --- Animate ---
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    planet_colors = {'earth': '#2196F3', 'venus': '#FF5722', 'jupiter': '#FF9800'}
    trail_len = 80

    (sc_line,) = ax.plot([], [], [], linewidth=2, color='#E91E63', label='Spacecraft')
    (sc_dot,)  = ax.plot([], [], [], 'o', markersize=5, color='#E91E63')

    planet_trails = {}
    planet_dots = {}
    for name in tracked:
        col = planet_colors.get(name, '#888888')
        tr, = ax.plot([], [], [], linewidth=1, alpha=0.6, color=col,
                      label=name.title())
        pt, = ax.plot([], [], [], 'o', markersize=6 if name != 'venus' else 5,
                      color=col)
        planet_trails[name] = tr
        planet_dots[name] = pt

    ax.scatter([0], [0], [0], s=200, marker='*', color='gold',
              label='Sun', zorder=10)

    # Axis limits
    pad = 0.3
    all_x = np.concatenate([sc_pos[:, 0]] + [pl_pos[n][:, 0] for n in tracked])
    all_y = np.concatenate([sc_pos[:, 1]] + [pl_pos[n][:, 1] for n in tracked])
    all_z = np.concatenate([sc_pos[:, 2]] + [pl_pos[n][:, 2] for n in tracked])
    ax.set_xlim(all_x.min()-pad, all_x.max()+pad)
    ax.set_ylim(all_y.min()-pad, all_y.max()+pad)
    ax.set_zlim(all_z.min()-pad, all_z.max()+pad)

    ax.set_xlabel('X (AU)'); ax.set_ylabel('Y (AU)'); ax.set_zlabel('Z (AU)')
    ax.legend(fontsize=8, loc='upper left')

    c3 = float(results['c3_kms2'])
    dsm = float(results['total_dsm_kms'])
    vinf_arr = float(results['vinf_arr_kms'])
    tof = float(results['tof_years'])
    ax.set_title(
        f'VEEGA: Earth -> Venus -> Earth -> Earth -> Jupiter\n'
        f'Launch {results["launch_date"][:11]}  |  TOF {tof:.2f} yr  |  '
        f'C3 = {c3:.1f} km²/s²  |  DSM = {dsm:.3f} km/s  |  '
        f'V_inf_arr = {vinf_arr:.1f} km/s', fontsize=11)

    def init():
        for obj in [sc_line, sc_dot]:
            obj.set_data([], []); obj.set_3d_properties([])
        for name in tracked:
            planet_trails[name].set_data([], [])
            planet_trails[name].set_3d_properties([])
            planet_dots[name].set_data([], [])
            planet_dots[name].set_3d_properties([])
        return [sc_line, sc_dot] + [planet_trails[n] for n in tracked] + \
               [planet_dots[n] for n in tracked]

    def update(i):
        # Spacecraft
        sc_line.set_data(sc_pos[:i+1, 0], sc_pos[:i+1, 1])
        sc_line.set_3d_properties(sc_pos[:i+1, 2])
        sc_dot.set_data([sc_pos[i, 0]], [sc_pos[i, 1]])
        sc_dot.set_3d_properties([sc_pos[i, 2]])
        # Planets
        j0 = max(0, i - trail_len)
        for name in tracked:
            pp = pl_pos[name]
            planet_trails[name].set_data(pp[j0:i+1, 0], pp[j0:i+1, 1])
            planet_trails[name].set_3d_properties(pp[j0:i+1, 2])
            planet_dots[name].set_data([pp[i, 0]], [pp[i, 1]])
            planet_dots[name].set_3d_properties([pp[i, 2]])
        return [sc_line, sc_dot] + [planet_trails[n] for n in tracked] + \
               [planet_dots[n] for n in tracked]

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                         interval=1000//fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  Saved GIF: {save_path}")


# %% Cell 4 — Extra diagnostics

def print_extra_diagnostics(results, best_x, seq, seq_names):
    """Print additional mission design info beyond the standard summary."""
    c3 = float(results['c3_kms2'])
    vinf_dep = float(results['vinf_dep_kms'])
    vinf_arr = float(results['vinf_arr_kms'])
    tof = float(results['tof_years'])
    mass = float(results['delivered_mass_kg'])

    print("\n" + "="*60)
    print("  EXTRA DIAGNOSTICS")
    print("="*60)

    # --- LEO departure ---
    mu_earth = 3.986e5   # km^3/s^2
    r_leo = 6378 + 200   # km (200 km parking orbit)
    v_circ = np.sqrt(mu_earth / r_leo)
    v_dep = np.sqrt(vinf_dep**2 + 2 * mu_earth / r_leo)
    dv_leo = v_dep - v_circ
    print(f"\n  LEO Departure (200 km parking orbit):")
    print(f"    V_circular  = {v_circ:.3f} km/s")
    print(f"    V_departure = {v_dep:.3f} km/s")
    print(f"    dV_depart   = {dv_leo:.3f} km/s  (Oberth-boosted TLI-like burn)")
    print(f"    Note: launch vehicle provides this, not spacecraft propulsion")

    # --- JOI estimate ---
    mu_jup = 1.267e8     # km^3/s^2
    r_peri_jup = 71492 * 12.8  # periapsis at 12.8 Rj (per 2012 study)
    v_hyp = np.sqrt(vinf_arr**2 + 2 * mu_jup / r_peri_jup)
    # 200-day orbit: semi-major axis from period
    T_200d = 200 * 86400  # seconds
    a_200d = (mu_jup * (T_200d / (2 * np.pi))**2)**(1/3)  # km
    v_capture = np.sqrt(mu_jup * (2/r_peri_jup - 1/a_200d))
    dv_joi = v_hyp - v_capture
    print(f"\n  JOI Estimate (200-day capture orbit, periapsis 12.8 Rj):")
    print(f"    V_inf arrival  = {vinf_arr:.3f} km/s")
    print(f"    V_hyperbolic   = {v_hyp:.3f} km/s (at periapsis)")
    print(f"    V_capture      = {v_capture:.3f} km/s (200-day orbit at periapsis)")
    print(f"    dV_JOI         = {dv_joi:.3f} km/s")
    print(f"    With G0 flyby: ~{max(0, dv_joi - 0.4):.3f} km/s (saving ~0.4 km/s)")

    # --- Mission timeline ---
    n_legs = len(seq_names) - 1
    tofs_days = [best_x[5 + 4*k] for k in range(n_legs)]
    cum = np.cumsum([0.0] + tofs_days)
    t0_mjd = best_x[0]

    print(f"\n  Mission Timeline:")
    events = ['Departure'] + \
             [f'{seq_names[k+1].title()} flyby' for k in range(n_legs-1)] + \
             ['Jupiter arrival']
    for i, event in enumerate(events):
        ep = pk.epoch(t0_mjd + cum[i], 'mjd2000')
        print(f"    {event:20s}  {str(ep)[:20]}  (T+{cum[i]/365.25:.2f} yr)")

    # --- V-infinity at each flyby ---
    print(f"\n  Flyby V_inf (from Lambert reconstruction):")
    legs, _, _ = compute_legs_data(best_x, seq, seq_names)
    for k in range(n_legs - 1):
        ld = legs[k]
        ep_fb = pk.epoch(ld['t_end_mjd'], 'mjd2000')
        _, v_planet = seq[k+1].eph(ep_fb)
        v_inf_in = np.linalg.norm(ld['v_arr'] - np.array(v_planet))
        body_name = seq_names[k+1]
        print(f"    {body_name.title():8s}: V_inf = {v_inf_in/1000:.3f} km/s")

    # --- Mass budget context ---
    dsm_total = float(results['total_dsm_kms'])
    print(f"\n  Mass Budget Context:")
    print(f"    FH delivers {mass:,.0f} kg at C3 = {c3:.1f} km^2/s^2")
    print(f"    Spacecraft DSM budget: {dsm_total:.3f} km/s")
    print(f"    JOI budget (est.): ~{dv_joi:.2f} km/s (before G0 flyby savings)")
    if mass >= 15000:
        print(f"    15,000 kg wet mass target MET (margin: {mass - 15000:,.0f} kg)")
    else:
        print(f"    15,000 kg wet mass target NOT MET (deficit: {15000 - mass:,.0f} kg)")


# %% Cell 5 — Main execution
if __name__ == '__main__':

    tee = start_logging('04')
    print(f"pykep {pk.__version__}  |  pygmo {pg.__version__}")

    # 4 legs, 18 dimensions. Deep-dive compute budget.
    best_x, best_f = run_mga_optimisation(
        udp, label="VEEGA (E-V-E-E-J)",
        n_islands=160, pop_size_1=64, gen_1=1000, evolve_rounds_1=14,
        n_seeds=100,
        pop_size_2=48, gen_2=3000,
        compass_fevals=750_000,
    )

    # --- pykep pretty output ---
    print("\n" + "="*60)
    print("  VEEGA — pykep pretty() output")
    print("="*60)
    udp.pretty(best_x)

    # --- Extract trade-study metrics ---
    results = extract_mga_results(udp, best_x, best_f,
                                  sequence_label='04 VEEGA',
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
    title = (f'VEEGA: Earth -> Venus -> Earth -> Earth -> Jupiter\n'
             f'Launch {results["launch_date"][:11]}  |  TOF {tof:.2f} yr  |  '
             f'C3 = {c3:.1f} km^2/s^2  |  DSM = {dsm:.3f} km/s  |  '
             f'V_inf_arr = {vinf_arr:.1f} km/s')
    plot_trajectory(udp, best_x, seq, seq_names, title,
                    save_path=f'{OUTPUT_DIR}/04_trajectory_veega.png')

    # --- Save to CSV ---
    save_to_csv(results)

    # --- Decision vector diagnostics ---
    n_legs = len(seq_names) - 1
    print("\nDecision vector breakdown:")
    print(f"  t0 (mjd2000):       {best_x[0]:.2f}")
    print(f"  u, v (V_inf dir):   {best_x[1]:.4f}, {best_x[2]:.4f}")
    print(f"  V_inf dep:          {best_x[3]:.1f} m/s = {best_x[3]/1000:.3f} km/s")
    for k in range(n_legs):
        i_eta = 4 + 4*k
        i_T   = 5 + 4*k
        print(f"  Leg {k+1}: eta={best_x[i_eta]:.4f}, T={best_x[i_T]:.1f} d "
              f"({best_x[i_T]/365.25:.2f} yr)")
        if k < n_legs - 1:
            i_beta = 6 + 4*k
            i_rp   = 7 + 4*k
            print(f"         beta={best_x[i_beta]:.4f} rad, rp/safe_r={best_x[i_rp]:.4f}")

    print(f"\n  Objective (m/s): {best_f:.1f}")
    print(f"  Objective (km/s): {best_f/1000:.4f}")
    print(f"  -> This = sum(DSMs) only. Arrival V_inf = {vinf_arr:.3f} km/s (NOT included).")
    print(f"  -> C3 = {c3:.1f} -> FH delivers {results['delivered_mass_kg']} kg")

    # --- Extra diagnostics ---
    print_extra_diagnostics(results, best_x, seq, seq_names)

    # --- Animated GIF (last, wrapped in try/except) ---
    print("\n" + "="*60)
    print("  GENERATING TRAJECTORY GIF...")
    print("="*60)
    try:
        make_trajectory_gif(
            best_x, best_f, seq, seq_names, results,
            save_path=f'{OUTPUT_DIR}/04_trajectory_veega.gif',
        )
        print("  GIF generation complete.")
    except Exception as e:
        print(f"  WARNING: GIF generation failed: {e}")
        import traceback
        traceback.print_exc()
        print("  (All other outputs were saved successfully.)")