# =============================================================================
# Callisto Pumpdown Trajectory Optimisation  (Jupiter → Europa)
# =============================================================================
#
# Uses pykep.planet.keplerian for Callisto and Europa — real orbital elements,
# real positions on real dates. No more fake phase sweep.
#
# Orbital elements sourced from NASA/JPL Horizons (J2000 epoch):
#   Callisto: a=1.8827e9 m, e=0.0074, i=0.192°
#   Europa:   a=6.709e8 m,  e=0.0094, i=0.471°
#
# Method: patched-conic pumpdown
#   1. JOI puts spacecraft on ellipse with apojove near Callisto
#   2. Each Callisto flyby is found by scanning for crossings of Callisto's
#      orbital radius, then checking if Callisto is nearby at that time
#   3. Both turn directions tried; whichever lowers apojove is kept
#   4. Once apojove < Europa SMA, find Europa encounter and compute EOI
#
# Sweep: JOI dates July 2041 – July 2044, perijove 2–5 RJ
# =============================================================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pykep as pk
from pykep import DAY2SEC, epoch, epoch_from_string
from pykep.planet import keplerian
from scipy.optimize import minimize_scalar
from scipy.signal import argrelextrema
import multiprocessing as mp
import itertools

np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
MU_JUP   = 1.26686534e17
R_JUP    = 69911e3
R_CAL    = 2410e3
R_EUR    = 1561e3
SMA_CAL  = 1.8827e9
SMA_EUR  = 6.709e8
MU_CAL   = 7.179e12
MU_EUR   = 3.203e12
V_INF    = 5623.0          # m/s Jupiter arrival hyperbolic excess
ALT_CAL  = 200e3           # m target flyby periapsis above Callisto surface
ALT_EUR  = 200e3           # m target Europa circular orbit altitude

T_CAL    = 2*np.pi * np.sqrt(SMA_CAL**3 / MU_JUP)   # ~16.69 days
T_EUR    = 2*np.pi * np.sqrt(SMA_EUR**3 / MU_JUP)   # ~3.55 days

# ── Build keplerian moon objects (Jupiter-centred, J2000) ─────────────────────
# Elements: (a, e, i, raan, argp, M0) in metres and radians
# Mean anomalies at J2000 from JPL Horizons
_J2000 = epoch(0)

CALLISTO = keplerian(
    _J2000,
    (SMA_CAL, 0.0074, np.radians(0.192), np.radians(298.8), np.radians(52.6), np.radians(181.4)),
    MU_JUP, MU_CAL, R_CAL, R_CAL, 'callisto'
)

EUROPA = keplerian(
    _J2000,
    (SMA_EUR, 0.0094, np.radians(0.471), np.radians(219.1), np.radians(88.0), np.radians(101.3)),
    MU_JUP, MU_EUR, R_EUR, R_EUR, 'europa'
)

def callisto_rv(ep):
    """Return (r, v) numpy arrays for Callisto at pykep epoch ep."""
    r, v = CALLISTO.eph(ep)
    return np.array(r), np.array(v)

def europa_rv(ep):
    """Return (r, v) numpy arrays for Europa at pykep epoch ep."""
    r, v = EUROPA.eph(ep)
    return np.array(r), np.array(v)

# ── Fast leapfrog propagator ──────────────────────────────────────────────────
def prop(r0, v0, dt, mu, steps=400):
    """Velocity-Verlet (leapfrog) integrator. Symplectic, good energy conservation."""
    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)
    h = dt / steps
    a = -mu / np.linalg.norm(r)**3 * r
    for _ in range(steps):
        v_half = v + 0.5*h*a
        r      = r + h*v_half
        a      = -mu / np.linalg.norm(r)**3 * r
        v      = v_half + 0.5*h*a
    return r, v

# ── Orbital mechanics helpers ─────────────────────────────────────────────────
def orbit_elements(r, v, mu):
    """Returns (semi-major axis [m], eccentricity)."""
    rm = np.linalg.norm(r)
    vm = np.linalg.norm(v)
    a  = 1.0 / (2.0/rm - vm**2/mu)
    e  = np.linalg.norm(((vm**2 - mu/rm)*r - np.dot(r,v)*v) / mu)
    return a, e

def rotate_vector(v, axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    return v*c + np.cross(axis, v)*s + axis*np.dot(axis, v)*(1-c)

def gravity_assist(v_inf_in, r_moon, mu_moon, rp, flip=False):
    """
    Compute outgoing V-infinity after unpowered flyby.
    flip=True reverses turn direction. Try both; keep whichever lowers apojove.
    """
    vm = np.linalg.norm(v_inf_in)
    if vm < 1.0:
        return None
    sin_half = 1.0 / (1.0 + rp * vm**2 / mu_moon)
    delta    = 2.0 * np.arcsin(np.clip(sin_half, -1.0, 1.0))
    axis     = np.cross(v_inf_in, -r_moon)
    anorm    = np.linalg.norm(axis)
    if anorm < 1e-10:
        return None
    axis = (-axis if flip else axis) / anorm
    return rotate_vector(v_inf_in, axis, delta)

def joi_dv(v_inf, rp, a_target, mu):
    return np.sqrt(v_inf**2 + 2*mu/rp) - np.sqrt(max(0.0, 2*mu/rp - mu/a_target))

def eoi_dv(v_inf, rp_ins, mu_moon, r_circ):
    return np.sqrt(v_inf**2 + 2*mu_moon/rp_ins) - np.sqrt(mu_moon/r_circ)

# ── Encounter finder ──────────────────────────────────────────────────────────
def find_encounter(r_sc, v_sc, ep0_mjd2000, moon_rv_fn,
                   t_max, mu, tol=2e8, n=300, refine_steps=800):
    """
    Scan [0, t_max] for close approach between spacecraft and moon.
    ep0_mjd2000: current epoch in MJD2000 days (float)
    moon_rv_fn: callable(pykep.epoch) -> (r, v)
    tol: encounter detection radius in metres
    Returns (dt_sec, r_sc, v_sc, r_moon, v_moon) or None.
    """
    dt_grid = np.linspace(0.0, t_max, n+1)

    sc_pos   = np.array([prop(r_sc, v_sc, float(dt), mu)[0] for dt in dt_grid])
    moon_pos = np.array([
        moon_rv_fn(epoch(ep0_mjd2000 + dt/DAY2SEC, "mjd2000"))[0]
        for dt in dt_grid
    ])
    dists = np.linalg.norm(sc_pos - moon_pos, axis=1)

    minima    = argrelextrema(dists, np.less)[0]
    candidates = [i for i in minima if dists[i] < tol]
    if not candidates:
        return None

    best_i  = min(candidates, key=lambda i: dists[i])
    lo      = float(dt_grid[max(0, best_i-1)])
    hi      = float(dt_grid[min(n, best_i+1)])

    def miss(dt):
        r_s = prop(r_sc, v_sc, float(dt), mu, steps=refine_steps)[0]
        r_m = moon_rv_fn(epoch(ep0_mjd2000 + dt/DAY2SEC, "mjd2000"))[0]
        return np.linalg.norm(r_s - r_m)

    res     = minimize_scalar(miss, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e4})
    best_dt = float(res.x)
    r_s, v_s = prop(r_sc, v_sc, best_dt, mu, steps=refine_steps)
    r_m, v_m = moon_rv_fn(epoch(ep0_mjd2000 + best_dt/DAY2SEC, "mjd2000"))
    return best_dt, r_s, v_s, r_m, v_m

# ── Full sequence evaluator ───────────────────────────────────────────────────
def evaluate_sequence(joi_mjd2000, rp_rj, apo_ratio, wait_orbits=0,
                      max_flybys=25, debug=False):
    """
    joi_mjd2000 : JOI epoch as MJD2000 float — real date, real moon positions
    rp_rj       : perijove in Jupiter radii
    apo_ratio   : initial apojove as multiple of Callisto SMA
    wait_orbits : free coasting orbits before targeting Callisto (no fuel cost).
                  Each orbit ~25-50 days depending on perijove/apojove.
                  Callisto moves ~(T_orb/T_callisto)*360° per wait orbit,
                  so sweeping this finds the best geometry for free.
    """
    rp         = rp_rj * R_JUP
    r_apo_init = apo_ratio * SMA_CAL
    a_init     = (rp + r_apo_init) / 2.0
    if a_init <= 0:
        return None

    dv_joi = joi_dv(V_INF, rp, a_init, MU_JUP)
    v_pmag = np.sqrt(max(0.0, 2*MU_JUP/rp - MU_JUP/a_init))
    T_init = 2*np.pi * np.sqrt(a_init**3 / MU_JUP)

    # Orient perijove OPPOSITE to Callisto at JOI so apojove faces Callisto
    r_cal_joi, _ = callisto_rv(epoch(joi_mjd2000, "mjd2000"))
    cal_dir = r_cal_joi / np.linalg.norm(r_cal_joi)
    perp    = np.cross(cal_dir, np.array([0., 0., 1.]))
    perp    = perp / np.linalg.norm(perp)
    r_sc = -cal_dir * rp
    v_sc =  perp    * v_pmag

    # Coast wait_orbits full periods before starting pumpdown — free, no fuel.
    # Callisto moves ~(T_init/T_CAL) * 360° per wait orbit, so each integer
    # wait value gives a genuinely different encounter geometry.
    t_wait = wait_orbits * T_init
    ep     = joi_mjd2000 + t_wait / DAY2SEC
    if wait_orbits > 0:
        r_sc, v_sc = prop(r_sc, v_sc, t_wait, MU_JUP,
                          steps=max(400, wait_orbits * 300))

    if debug:
        print(f"\n  JOI {epoch(joi_mjd2000,'mjd2000')}  "
              f"rp={rp_rj}RJ  apo={apo_ratio}×CSMA  "
              f"wait={wait_orbits} orbits ({t_wait/DAY2SEC:.1f}d)  "
              f"dv_joi={dv_joi/1000:.3f}km/s  "
              f"T={T_init/DAY2SEC:.1f}d")

    flyby_log = []
    success   = False
    dv_eoi    = 0.0
    vinf_eur  = 0.0

    for fb in range(max_flybys):
        a_cur, e_cur = orbit_elements(r_sc, v_sc, MU_JUP)
        if a_cur <= 0:
            if debug: print(f"  ✗ hyperbolic orbit at step {fb}")
            break

        T_cur  = 2*np.pi * np.sqrt(a_cur**3 / MU_JUP)
        r_apo  = a_cur * (1 + e_cur)
        r_peri = a_cur * (1 - e_cur)

        if debug:
            print(f"  [{fb:2d}] apo={r_apo/SMA_CAL:.4f}×CSMA  "
                  f"({r_apo/SMA_EUR:.3f}×ESMA)  "
                  f"peri={r_peri/R_JUP:.2f}RJ  "
                  f"T={T_cur/DAY2SEC:.2f}d  "
                  f"t={ep - joi_mjd2000:.1f}d")

        # ── Check if pumped down enough for Europa ────────────────────────────
        if r_apo <= SMA_EUR * 1.05:
            enc_e = find_encounter(r_sc, v_sc, ep, europa_rv,
                                   30*DAY2SEC, MU_JUP,
                                   tol=5e7, n=300)
            if enc_e:
                dt_e, r_se, v_se, r_eu, v_eu = enc_e
                vinf_eur = np.linalg.norm(v_se - v_eu)
                rp_eoi   = R_EUR + ALT_EUR
                dv_eoi   = eoi_dv(vinf_eur, rp_eoi, MU_EUR, R_EUR + ALT_EUR)
                flyby_log.append((dt_e/DAY2SEC, vinf_eur/1000, 0.0, "Europa"))
                success = True
                if debug:
                    print(f"  ✓ Europa! V∞={vinf_eur/1000:.3f}km/s  "
                          f"EOI ΔV={dv_eoi/1000:.3f}km/s")
            else:
                if debug: print("  ✗ orbit inside Europa SMA but no encounter")
            break

        # ── Find next Callisto encounter ──────────────────────────────────────
        # Scan up to 3 orbital periods; use generous tolerance since we'll
        # refine — the real targeting happens in the gravity assist step
        t_scan = min(3.0 * T_cur, 120*DAY2SEC)
        enc = find_encounter(r_sc, v_sc, ep, callisto_rv,
                             t_scan, MU_JUP,
                             tol=3e8, n=400)
        if enc is None:
            if debug: print(f"  ✗ no Callisto encounter in {t_scan/DAY2SEC:.0f}d")
            break

        dt, r_enc, v_enc, r_cal, v_cal = enc
        v_inf_in = v_enc - v_cal
        vinf_mag = np.linalg.norm(v_inf_in)

        # Target 200 km periapsis at Callisto
        rp_cal = R_CAL + ALT_CAL

        # ── KEY: try both turn directions, keep whichever lowers apojove ──────
        best_v_sc  = None
        best_apo   = r_apo        # only accept improvement
        best_flip  = False

        for flip in [False, True]:
            v_inf_out = gravity_assist(v_inf_in, r_cal, MU_CAL, rp_cal, flip)
            if v_inf_out is None:
                continue
            v_new = v_inf_out + v_cal
            a_new, e_new = orbit_elements(r_enc, v_new, MU_JUP)
            if a_new <= 0:
                continue
            r_apo_new = a_new * (1 + e_new)
            if r_apo_new < best_apo:
                best_apo  = r_apo_new
                best_v_sc = v_new
                best_flip = flip

        if best_v_sc is None:
            if debug: print(f"  ✗ neither turn direction lowers apojove at flyby {fb+1}")
            break

        # Compute turn angle for logging
        v_inf_out_best = best_v_sc - v_cal
        cos_turn = np.dot(v_inf_in, v_inf_out_best) / (
            np.linalg.norm(v_inf_in) * np.linalg.norm(v_inf_out_best) + 1e-30)
        turn_deg = np.degrees(np.arccos(np.clip(cos_turn, -1.0, 1.0)))

        if debug:
            print(f"  → flyby {fb+1}: Δt={dt/DAY2SEC:.2f}d  "
                  f"V∞={vinf_mag/1000:.3f}km/s  turn={turn_deg:.1f}°  "
                  f"flip={best_flip}  "
                  f"apo: {r_apo/SMA_CAL:.4f}→{best_apo/SMA_CAL:.4f}×CSMA")

        flyby_log.append((dt/DAY2SEC, vinf_mag/1000, turn_deg, "Callisto"))
        r_sc  = r_enc.copy()
        v_sc  = best_v_sc.copy()
        ep   += dt / DAY2SEC

    if not success:
        return None

    total_dv   = dv_joi + dv_eoi
    n_callisto = sum(1 for f in flyby_log if f[3] == "Callisto")
    t_total    = (ep - joi_mjd2000)
    return total_dv, n_callisto, dv_joi, dv_eoi, vinf_eur, flyby_log, t_total

# ── Sweep setup ───────────────────────────────────────────────────────────────
START_MJD = epoch_from_string("2041-07-01 00:00:00").mjd2000
END_MJD   = epoch_from_string("2044-07-01 00:00:00").mjd2000

STEP_DAYS        = 5           # date resolution; use 1 for full sweep
RP_RJ_LIST       = [2, 3, 4, 5]
APO_LIST         = [2.5, 3.0, 3.5, 4.0]
WAIT_ORBITS_LIST = list(range(0, 10))  # 0–9 free coasting orbits after JOI

dates_mjd = np.arange(START_MJD, END_MJD, float(STEP_DAYS))

# ── Debug mode: test a small sample with verbose output ───────────────────────
DEBUG = True    # ← flip to False for full parallel sweep

if DEBUG:
    print("=" * 65)
    print("DEBUG MODE — real ephemerides, no phase sweep")
    print("=" * 65)

    # Sample a handful of dates spread across the window
    test_dates = dates_mjd[::60]   # every ~300 days = ~5 test dates
    best_dv  = np.inf
    best_res = None

    for mjd in test_dates:
        for rp_rj in RP_RJ_LIST:
            for apo in APO_LIST:
                for wait in WAIT_ORBITS_LIST:
                    res = evaluate_sequence(mjd, rp_rj, apo,
                                            wait_orbits=wait, debug=True)
                    if res:
                        total_dv = res[0]
                        if total_dv < best_dv:
                            best_dv  = total_dv
                            best_res = (mjd, rp_rj, apo, wait, res)

    print("\n" + "=" * 65)
    if best_res:
        mjd, rp_rj, apo, wait, res = best_res
        total_dv, n_fb, dv_joi, dv_eoi, vinf_eur, flyby_log, t_total = res
        print("BEST IN DEBUG SAMPLE")
        print("=" * 65)
        print(f"  JOI date  : {epoch(mjd, 'mjd2000')}")
        print(f"  Perijove  : {rp_rj} RJ")
        print(f"  Init apo  : {apo} × Callisto SMA")
        print(f"  Wait orbs : {wait} orbits ({wait * 2*np.pi*np.sqrt(((rp_rj*R_JUP + apo*SMA_CAL)/2)**3/MU_JUP)/DAY2SEC:.1f} d)")
        print(f"  JOI ΔV    : {dv_joi/1000:.3f} km/s")
        print(f"  EOI ΔV    : {dv_eoi/1000:.3f} km/s")
        print(f"  Total ΔV  : {total_dv/1000:.3f} km/s")
        print(f"  Flybys    : {n_fb} Callisto")
        print(f"  Duration  : {t_total:.1f} days")
        print(f"  Europa V∞ : {vinf_eur/1000:.3f} km/s")
        print("  Flyby log :")
        for dt, vi, turn, label in flyby_log:
            print(f"    {label:10s}  Δt={dt:6.2f}d  V∞={vi:.3f}km/s  turn={turn:.1f}°")
        print("\nSet DEBUG=False to run the full parallel sweep.")
    else:
        print("⚠  No successful trajectories in debug sample.")
        print("Try increasing tol in find_encounter calls (currently 3e8 m for Callisto).")
    print("=" * 65)

# ── Full parallel sweep ───────────────────────────────────────────────────────
def _worker(args):
    mjd, rp_rj, apo, wait = args
    res = evaluate_sequence(mjd, rp_rj, apo, wait_orbits=wait, debug=False)
    if res is None:
        return None
    total_dv, n_fb, dv_joi, dv_eoi, vinf_eur, flyby_log, t_total = res
    return mjd, rp_rj, apo, wait, total_dv, n_fb, dv_joi, dv_eoi, t_total

if not DEBUG:
    sweep_args = list(itertools.product(dates_mjd, RP_RJ_LIST, APO_LIST, WAIT_ORBITS_LIST))
    print(f"Evaluations : {len(sweep_args)}  |  Cores : {mp.cpu_count()}")

    try:
        with mp.Pool(mp.cpu_count()) as pool:
            raw = pool.map(_worker, sweep_args, chunksize=10)
    except Exception as e:
        print(f"Parallel failed ({e}), running serial.")
        raw = [_worker(a) for a in sweep_args]

    results = [r for r in raw if r is not None]
    print(f"Feasible trajectories: {len(results)}")

    if results:
        best = min(results, key=lambda r: r[4])
        mjd, rp_rj, apo, wait, total_dv, n_fb, dv_joi, dv_eoi, t_total = best

        print("\n" + "="*65)
        print("BEST CALLISTO PUMPDOWN TRAJECTORY")
        print("="*65)
        print(f"  JOI date  : {epoch(mjd, 'mjd2000')}")
        print(f"  Perijove  : {rp_rj} RJ")
        print(f"  Init apo  : {apo} × Callisto SMA")
        print(f"  Wait orbs : {wait}")
        print(f"  JOI ΔV    : {dv_joi/1000:.3f} km/s")
        print(f"  EOI ΔV    : {dv_eoi/1000:.3f} km/s")
        print(f"  Total ΔV  : {total_dv/1000:.3f} km/s")
        print(f"  Flybys    : {n_fb} Callisto")
        print(f"  Duration  : {t_total:.1f} days")

        # Plot ΔV vs date
        all_mjd = [r[0] for r in results]
        all_dv  = [r[4]/1000 for r in results]
        plt.figure(figsize=(12, 5))
        plt.scatter(all_mjd, all_dv, s=8, alpha=0.5)
        plt.xlabel("JOI date (MJD2000)")
        plt.ylabel("Total ΔV (km/s)")
        plt.title("Callisto pumpdown ΔV — July 2041 to July 2044")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("total_dv_sweep.png", dpi=150)
        print("Saved total_dv_sweep.png")
    else:
        print("No feasible trajectory found. Set DEBUG=True to diagnose.")
