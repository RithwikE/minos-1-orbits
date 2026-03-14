from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import re

import numpy as np
import pykep as pk

from .config import SequenceConfig


def parse_decision_vector(udp, x: np.ndarray, n_legs: int) -> dict:
    # pykep accepts constructor vinf bounds in km/s, but in the instantiated
    # chromosome and internal decoders the departure vinf is handled in m/s.
    tofs_days, vinfx, vinfy, vinfz = udp._decode_times_and_vinf(x.tolist())
    vinf_dep_vec = np.array([vinfx, vinfy, vinfz], dtype=float)
    parsed = {
        "t0_mjd2000": float(x[0]),
        "u": float(x[1]),
        "v": float(x[2]),
        "vinf_dep_m_s": float(np.linalg.norm(vinf_dep_vec)),
        "vinf_dep_vector_m_s": vinf_dep_vec.tolist(),
        "etas": [float(x[4 + 4 * idx]) for idx in range(n_legs)],
        "tofs_days": [float(value) for value in tofs_days],
        "betas": [float(x[6 + 4 * idx]) for idx in range(n_legs - 1)],
        "rp_safe_radius_ratios": [float(x[7 + 4 * idx]) for idx in range(n_legs - 1)],
    }
    if udp._tof_encoding == "alpha":
        parsed["tof_alpha_parameters"] = [float(x[5 + 4 * idx]) for idx in range(n_legs)]
        parsed["tof_total_days"] = float(x[-1])
    elif udp._tof_encoding == "eta":
        parsed["tof_eta_parameters"] = [float(x[5 + 4 * idx]) for idx in range(n_legs)]
        parsed["tof_total_days"] = float(sum(tofs_days))
        parsed["tof_eta_limit_days"] = float(udp._tof)
    else:
        parsed["tof_total_days"] = float(sum(tofs_days))
    return parsed


def pretty_output(udp, x: np.ndarray) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        udp.pretty(x)
    return buffer.getvalue()


def pretty_metrics(udp, x: np.ndarray) -> dict:
    text = pretty_output(udp, x)
    dsm_values = [float(val) for val in re.findall(r"DSM magnitude:\s*([\d.]+)\s*m/s", text)]
    arrival_match = re.search(r"Arrival Vinf:\s*([\d.]+)\s*m/s", text)
    tof_match = re.search(r"Total mission time:\s*([\d.]+)\s*years", text)
    return {
        "pretty_text": text,
        "dsm_magnitudes_m_s": dsm_values,
        "arrival_vinf_m_s": float(arrival_match.group(1)) if arrival_match else None,
        "total_tof_years": float(tof_match.group(1)) if tof_match else None,
    }


def reconstruct_candidate(udp, sequence: list, config: SequenceConfig, x: np.ndarray) -> dict:
    n_legs = len(sequence) - 1
    parsed = parse_decision_vector(udp, x, n_legs)
    t0_mjd2000 = parsed["t0_mjd2000"]
    tofs_days = parsed["tofs_days"]
    etas = parsed["etas"]
    betas = parsed["betas"]
    rp_ratios = parsed["rp_safe_radius_ratios"]

    epochs_mjd2000 = [t0_mjd2000]
    for tof_days in tofs_days:
        epochs_mjd2000.append(epochs_mjd2000[-1] + tof_days)

    planet_states = []
    for body, epoch_mjd2000 in zip(sequence, epochs_mjd2000):
        epoch = pk.epoch(epoch_mjd2000, "mjd2000")
        position, velocity = body.eph(epoch)
        planet_states.append(
            {
                "epoch": epoch,
                "position_m": np.array(position, dtype=float),
                "velocity_m_s": np.array(velocity, dtype=float),
            }
        )

    dv_values, lamberts, _, ballistic_legs, ballistic_epochs = udp._compute_dvs(x.tolist())
    total_dv_values_m_s = [float(val) for val in dv_values]
    pretty = pretty_metrics(udp, x)
    dsm_values_exact = pretty["dsm_magnitudes_m_s"]
    if len(dsm_values_exact) != n_legs:
        dsm_values_exact = total_dv_values_m_s[:n_legs]
    arrival_vinf_exact = None
    if lamberts:
        final_arrival_velocity = np.array(lamberts[-1].get_v2()[0], dtype=float)
        arrival_vinf_exact = float(
            np.linalg.norm(final_arrival_velocity - planet_states[-1]["velocity_m_s"])
        )

    legs = []
    events = [
        {
            "type": "departure",
            "body": config.bodies[0],
            "epoch": str(planet_states[0]["epoch"]),
            "epoch_mjd2000": t0_mjd2000,
            "spacecraft_position_m": np.array(ballistic_legs[0][0], dtype=float).tolist(),
            "spacecraft_velocity_m_s": np.array(ballistic_legs[0][1], dtype=float).tolist(),
            "planet_position_m": planet_states[0]["position_m"].tolist(),
            "planet_velocity_m_s": planet_states[0]["velocity_m_s"].tolist(),
            "vinf_m_s": parsed["vinf_dep_m_s"],
            "vinf_vector_m_s": parsed["vinf_dep_vector_m_s"],
        }
    ]

    for leg_idx in range(n_legs):
        tof_seconds = tofs_days[leg_idx] * pk.DAY2SEC
        dt_pre = etas[leg_idx] * tof_seconds

        target_planet_state = planet_states[leg_idx + 1]
        target_position = target_planet_state["position_m"]
        target_velocity = target_planet_state["velocity_m_s"]
        start_state_index = 0 if leg_idx == 0 else 2 * leg_idx
        dsm_state_index = 2 * leg_idx + 1
        start_position = np.array(ballistic_legs[start_state_index][0], dtype=float)
        start_velocity = np.array(ballistic_legs[start_state_index][1], dtype=float)
        dsm_position = np.array(ballistic_legs[dsm_state_index][0], dtype=float)
        post_dsm_velocity = np.array(ballistic_legs[dsm_state_index][1], dtype=float)
        _, pre_dsm_velocity = pk.propagate_lagrangian(
            start_position,
            start_velocity,
            dt_pre,
            pk.MU_SUN,
        )
        arrival_velocity = np.array(lamberts[leg_idx].get_v2()[0], dtype=float)

        dsm_magnitude_m_s = dsm_values_exact[leg_idx]
        arrival_vinf_vec = arrival_velocity - target_velocity
        rp_ratio = rp_ratios[leg_idx] if leg_idx < len(rp_ratios) else None
        beta = betas[leg_idx] if leg_idx < len(betas) else None

        leg_record = {
            "leg_index": leg_idx + 1,
            "from_body": config.bodies[leg_idx],
            "to_body": config.bodies[leg_idx + 1],
            "start_epoch": str(planet_states[leg_idx]["epoch"]),
            "end_epoch": str(target_planet_state["epoch"]),
            "tof_days": tofs_days[leg_idx],
            "dsm_fraction": etas[leg_idx],
            "pre_dsm_duration_days": dt_pre * pk.SEC2DAY,
            "post_dsm_duration_days": tofs_days[leg_idx] - dt_pre * pk.SEC2DAY,
            "dsm_position_m": np.array(dsm_position, dtype=float).tolist(),
            "pre_dsm_velocity_m_s": np.array(pre_dsm_velocity, dtype=float).tolist(),
            "post_dsm_velocity_m_s": post_dsm_velocity.tolist(),
            "arrival_velocity_m_s": arrival_velocity.tolist(),
            "arrival_vinf_m_s": float(np.linalg.norm(arrival_vinf_vec)),
            "dsm_magnitude_m_s": dsm_magnitude_m_s,
            "flyby_beta_rad": beta,
            "flyby_rp_planet_radius_ratio": rp_ratio,
        }
        legs.append(leg_record)

        events.append(
            {
                "type": "dsm",
                "body": None,
                "epoch": str(pk.epoch(ballistic_epochs[dsm_state_index], "mjd2000")),
                "epoch_mjd2000": float(ballistic_epochs[dsm_state_index]),
                "spacecraft_position_m": np.array(dsm_position, dtype=float).tolist(),
                "spacecraft_velocity_before_m_s": np.array(pre_dsm_velocity, dtype=float).tolist(),
                "spacecraft_velocity_after_m_s": post_dsm_velocity.tolist(),
                "delta_v_m_s": dsm_magnitude_m_s,
            }
        )

        encounter_type = "arrival" if leg_idx == n_legs - 1 else "flyby"
        event = {
            "type": encounter_type,
            "body": config.bodies[leg_idx + 1],
            "epoch": str(target_planet_state["epoch"]),
            "epoch_mjd2000": epochs_mjd2000[leg_idx + 1],
            "spacecraft_position_m": target_position.tolist(),
            "spacecraft_velocity_m_s": arrival_velocity.tolist(),
            "planet_position_m": target_position.tolist(),
            "planet_velocity_m_s": target_velocity.tolist(),
            "vinf_m_s": float(np.linalg.norm(arrival_vinf_vec)),
        }
        if encounter_type == "flyby":
            body = sequence[leg_idx + 1]
            rp_m = float(rp_ratio * body.radius)
            outgoing_velocity = np.array(ballistic_legs[2 * (leg_idx + 1)][1], dtype=float)
            event["flyby_beta_rad"] = beta
            event["flyby_radius_m"] = rp_m
            event["flyby_altitude_km"] = (rp_m - body.radius) / 1000.0
            event["spacecraft_velocity_in_m_s"] = arrival_velocity.tolist()
            event["spacecraft_velocity_out_m_s"] = outgoing_velocity.tolist()
        events.append(event)

    total_dsm_m_s = float(sum(dsm_values_exact))
    arrival_vinf_m_s = float(arrival_vinf_exact if arrival_vinf_exact is not None else legs[-1]["arrival_vinf_m_s"])
    total_tof_days = float(sum(tofs_days))
    dense_samples = sample_candidate_ephemeris(
        udp=udp,
        x=x,
        t0_mjd2000=t0_mjd2000,
        total_tof_days=total_tof_days,
        sample_count=max(2, config.dense_output_samples_per_leg * n_legs + 1),
    )

    return {
        "decision_vector": [float(val) for val in x.tolist()],
        "decision_vector_breakdown": parsed,
        "summary": {
            "sequence": config.label,
            "launch_epoch": str(planet_states[0]["epoch"]),
            "arrival_epoch": str(planet_states[-1]["epoch"]),
            "total_tof_days": total_tof_days,
            "total_tof_years": total_tof_days * pk.DAY2YEAR,
            "vinf_dep_kms": parsed["vinf_dep_m_s"] / 1000.0,
            "c3_kms2": (parsed["vinf_dep_m_s"] / 1000.0) ** 2,
            "objective_total_dv_kms": float(sum(total_dv_values_m_s)) / 1000.0,
            "total_dsm_kms": total_dsm_m_s / 1000.0,
            "dsm_per_leg_kms": [value / 1000.0 for value in dsm_values_exact],
            "arrival_vinf_kms": arrival_vinf_m_s / 1000.0,
            "arrival_cost_kms": total_dv_values_m_s[-1] / 1000.0 if config.add_vinf_arr else 0.0,
            "pretty_total_tof_years": pretty["total_tof_years"],
        },
        "pretty_output": pretty["pretty_text"],
        "events": events,
        "legs": legs,
        "dense_samples": dense_samples,
    }


def sample_candidate_ephemeris(
    udp,
    x: np.ndarray,
    t0_mjd2000: float,
    total_tof_days: float,
    sample_count: int,
) -> list[dict]:
    eph = udp.get_eph_function(x)
    samples = []
    for offset_days in np.linspace(0.0, total_tof_days, sample_count):
        epoch = pk.epoch(t0_mjd2000 + float(offset_days), "mjd2000")
        # The public doc example shows a pykep.epoch input, but this installed
        # implementation expects the raw mjd2000 float and performs its own checks.
        position, velocity = eph(float(epoch.mjd2000))
        position = np.array(position, dtype=float)
        velocity = np.array(velocity, dtype=float)
        orbital_elements = {}
        try:
            sma_m, ecc, inc_rad, raan_rad, argp_rad, mean_anom_rad = pk.ic2par(position, velocity, pk.MU_SUN)
            orbital_elements = {
                "semi_major_axis_m": float(sma_m),
                "eccentricity": float(ecc),
                "inclination_rad": float(inc_rad),
                "raan_rad": float(raan_rad),
                "arg_periapsis_rad": float(argp_rad),
                "mean_anomaly_rad": float(mean_anom_rad),
            }
        except Exception:
            orbital_elements = {}

        samples.append(
            {
                "epoch": str(epoch),
                "epoch_mjd2000": float(epoch.mjd2000),
                "position_m": position.tolist(),
                "velocity_m_s": velocity.tolist(),
                "distance_to_sun_au": float(np.linalg.norm(position) / pk.AU),
                "speed_kms": float(np.linalg.norm(velocity) / 1000.0),
                "orbital_elements": orbital_elements,
            }
        )
    return samples
