from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined trade-study CSV from per-seed top candidate summaries.")
    parser.add_argument("--batch-dir", required=True, help="Directory containing per-seed result folders.")
    parser.add_argument("--output", required=True, help="CSV path to write.")
    return parser.parse_args()


def parse_departure_date(launch_epoch: str) -> str:
    parsed = datetime.strptime(launch_epoch, "%Y-%b-%d %H:%M:%S.%f")
    return parsed.date().isoformat()


def build_rows(batch_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
        summaries = json.loads((seed_dir / "top_candidate_summaries.json").read_text())
        for candidate in summaries:
            launch_epoch = candidate["launch_epoch"]
            rows.append(
                {
                    "seed": seed_dir.name,
                    "rank_within_seed": str(candidate["rank"]),
                    "departure_date": parse_departure_date(launch_epoch),
                    "launch_epoch": launch_epoch,
                    "arrival_epoch": candidate["arrival_epoch"],
                    "total_dsm_kms": f"{candidate['total_dsm_kms']:.12f}",
                    "objective_total_dv_kms": f"{candidate['objective_total_dv_kms']:.12f}",
                    "c3_kms2": f"{candidate['c3_kms2']:.12f}",
                    "arrival_vinf_kms": f"{candidate['arrival_vinf_kms']:.12f}",
                    "total_tof_years": f"{candidate['total_tof_years']:.12f}",
                    "detail_file": candidate["detail_file"],
                    "detail_json_path": str((seed_dir / "candidate_details" / candidate["detail_file"]).resolve()),
                }
            )
    rows.sort(
        key=lambda row: (
            float(row["objective_total_dv_kms"]),
            float(row["c3_kms2"]),
            row["seed"],
            int(row["rank_within_seed"]),
        )
    )
    return rows


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    output_path = Path(args.output).resolve()
    rows = build_rows(batch_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "rank_within_seed",
        "departure_date",
        "launch_epoch",
        "arrival_epoch",
        "total_dsm_kms",
        "objective_total_dv_kms",
        "c3_kms2",
        "arrival_vinf_kms",
        "total_tof_years",
        "detail_file",
        "detail_json_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
