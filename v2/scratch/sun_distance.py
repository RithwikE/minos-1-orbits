from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pykep as pk


@dataclass(slots=True)
class Segment:
    name: str
    start_epoch_mjd2000: float
    duration_days: float
    start_position_m: np.ndarray
    start_velocity_m_s: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the closest heliocentric approach for a saved V2 candidate."
    )
    parser.add_argument(
        "--candidate-json",
        default="v2/good_results/veega/veega_batches/seed42/candidate_details/rank_001.json",
        help="Path to a saved candidate detail JSON.",
    )
    parser.add_argument(
        "--coarse-samples-per-segment",
        type=int,
        default=4096,
        help="Initial sample count per ballistic segment before local refinement.",
    )
    return parser.parse_args()


def load_candidate(path: Path) -> dict:
    return json.loads(path.read_text())


def build_segments(candidate: dict) -> list[Segment]:
    events = candidate["events"]
    legs = candidate["legs"]

    departure_event = events[0]
    flyby_events = [event for event in events if event["type"] == "flyby"]
    if len(flyby_events) != len(legs) - 1:
        raise ValueError("Candidate archive does not have the expected number of flyby events.")

    segments: list[Segment] = []
    for leg_idx, leg in enumerate(legs):
        if leg_idx == 0:
            pre_start_position = np.array(departure_event["spacecraft_position_m"], dtype=float)
            pre_start_velocity = np.array(departure_event["spacecraft_velocity_m_s"], dtype=float)
            pre_start_epoch = float(departure_event["epoch_mjd2000"])
        else:
            prior_flyby = flyby_events[leg_idx - 1]
            pre_start_position = np.array(prior_flyby["spacecraft_position_m"], dtype=float)
            pre_start_velocity = np.array(prior_flyby["spacecraft_velocity_out_m_s"], dtype=float)
            pre_start_epoch = float(prior_flyby["epoch_mjd2000"])

        pre_duration_days = float(leg["pre_dsm_duration_days"])
        segments.append(
            Segment(
                name=(
                    f"leg {leg['leg_index']} pre-DSM "
                    f"{leg['from_body']}->{leg['to_body']}"
                ),
                start_epoch_mjd2000=pre_start_epoch,
                duration_days=pre_duration_days,
                start_position_m=pre_start_position,
                start_velocity_m_s=pre_start_velocity,
            )
        )

        post_duration_days = float(leg["post_dsm_duration_days"])
        segments.append(
            Segment(
                name=(
                    f"leg {leg['leg_index']} post-DSM "
                    f"{leg['from_body']}->{leg['to_body']}"
                ),
                start_epoch_mjd2000=pre_start_epoch + pre_duration_days,
                duration_days=post_duration_days,
                start_position_m=np.array(leg["dsm_position_m"], dtype=float),
                start_velocity_m_s=np.array(leg["post_dsm_velocity_m_s"], dtype=float),
            )
        )

    return segments


def distance_to_sun_m(position_m: np.ndarray) -> float:
    return float(np.linalg.norm(position_m))


def segment_distance_m(segment: Segment, time_seconds: float) -> float:
    position_m, _ = pk.propagate_lagrangian(
        segment.start_position_m,
        segment.start_velocity_m_s,
        time_seconds,
        pk.MU_SUN,
    )
    return distance_to_sun_m(np.array(position_m, dtype=float))


def refine_local_minimum(segment: Segment, left_s: float, right_s: float, iterations: int = 64) -> tuple[float, float]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    c = right_s - (right_s - left_s) * inv_phi
    d = left_s + (right_s - left_s) * inv_phi
    fc = segment_distance_m(segment, c)
    fd = segment_distance_m(segment, d)

    for _ in range(iterations):
        if fc < fd:
            right_s = d
            d = c
            fd = fc
            c = right_s - (right_s - left_s) * inv_phi
            fc = segment_distance_m(segment, c)
        else:
            left_s = c
            c = d
            fc = fd
            d = left_s + (right_s - left_s) * inv_phi
            fd = segment_distance_m(segment, d)

    best_time_s = 0.5 * (left_s + right_s)
    best_distance_m = segment_distance_m(segment, best_time_s)
    return best_time_s, best_distance_m


def find_segment_minimum(segment: Segment, coarse_samples: int) -> dict:
    duration_seconds = segment.duration_days * pk.DAY2SEC
    times_s = np.linspace(0.0, duration_seconds, coarse_samples + 1)
    distances_m = np.array([segment_distance_m(segment, float(t)) for t in times_s], dtype=float)
    best_index = int(np.argmin(distances_m))

    left_index = max(0, best_index - 1)
    right_index = min(len(times_s) - 1, best_index + 1)
    best_time_s, best_distance_m = refine_local_minimum(
        segment,
        float(times_s[left_index]),
        float(times_s[right_index]),
    )
    best_epoch_mjd2000 = segment.start_epoch_mjd2000 + best_time_s * pk.SEC2DAY

    return {
        "segment": segment,
        "time_since_segment_start_days": best_time_s * pk.SEC2DAY,
        "epoch_mjd2000": best_epoch_mjd2000,
        "epoch": str(pk.epoch(best_epoch_mjd2000, "mjd2000")),
        "distance_m": best_distance_m,
        "distance_au": best_distance_m / pk.AU,
        "distance_solar_radii": best_distance_m / 695700000.0,
    }


def find_global_minimum(candidate: dict, coarse_samples: int) -> dict:
    segment_results = [
        find_segment_minimum(segment, coarse_samples)
        for segment in build_segments(candidate)
    ]
    return min(segment_results, key=lambda item: item["distance_m"])


def find_bracketing_events(candidate: dict, epoch_mjd2000: float) -> tuple[dict | None, dict | None]:
    events = sorted(candidate["events"], key=lambda event: float(event["epoch_mjd2000"]))
    previous_event = None
    next_event = None
    for event in events:
        event_epoch = float(event["epoch_mjd2000"])
        if event_epoch <= epoch_mjd2000:
            previous_event = event
        if event_epoch >= epoch_mjd2000:
            next_event = event
            break
    return previous_event, next_event


def format_event(event: dict | None) -> str:
    if event is None:
        return "none"
    body = f" {event['body']}" if event.get("body") else ""
    return f"{event['type']}{body} at {event['epoch']}"


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate_json).resolve()
    candidate = load_candidate(candidate_path)

    best = find_global_minimum(candidate, args.coarse_samples_per_segment)
    previous_event, next_event = find_bracketing_events(candidate, best["epoch_mjd2000"])

    dense_samples = candidate.get("dense_samples", [])
    dense_min = None
    if dense_samples:
        dense_min = min(dense_samples, key=lambda sample: float(sample["distance_to_sun_au"]))

    print(f"candidate: {candidate_path}")
    print(f"sequence: {candidate['summary']['sequence']}")
    print(f"closest distance to sun: {best['distance_au']:.9f} AU")
    print(f"closest distance to sun: {best['distance_m'] / 1000.0:,.1f} km")
    print(f"closest distance to sun: {best['distance_solar_radii']:.3f} solar radii")
    print(f"epoch: {best['epoch']}")
    print(f"segment: {best['segment'].name}")
    print(f"time into segment: {best['time_since_segment_start_days']:.6f} days")
    print(f"previous event: {format_event(previous_event)}")
    print(f"next event: {format_event(next_event)}")

    if dense_min is not None:
        print(
            "saved dense-sample minimum: "
            f"{float(dense_min['distance_to_sun_au']):.9f} AU at {dense_min['epoch']}"
        )


if __name__ == "__main__":
    main()
