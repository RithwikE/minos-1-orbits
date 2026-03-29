from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
V2_ROOT = REPO_ROOT / "v2"
GOOD_RESULTS_DIR = V2_ROOT / "good_results"
BACKGROUND_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BACKGROUND_DIR / "output"


@dataclass(slots=True)
class RunRecord:
    run_dir: Path
    family_key: str
    family_label: str
    config_name: str
    config_path: str
    seed: int
    compute_level: int
    phase1_islands: int
    phase1_rounds: int
    phase2_seed_count: int
    phase3_candidate_count: int
    execution_mode: str
    runtime_seconds: float
    eval_phase1: int
    eval_phase2: int
    eval_phase3: int
    eval_total: int
    archive_size: int
    top_candidate_count: int
    archive_problem_feasible_count: int
    archive_mission_feasible_count: int
    archive_fully_feasible_count: int
    detail_file_count: int
    detail_event_count: int
    detail_leg_count: int
    detail_dense_sample_count: int
    best_objective_m_s: float
    best_total_dsm_kms: float
    best_total_tof_years: float
    best_c3_kms2: float
    best_arrival_vinf_kms: float
    started_at_utc: str
    submitted_at_utc: str | None
    batch_job_id: str | None


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def iter_run_dirs() -> list[Path]:
    return sorted(path.parent for path in GOOD_RESULTS_DIR.glob("*/*_batches/seed*/run_summary.json"))


def count_archive_candidates(path: Path) -> tuple[int, int, int]:
    problem_feasible = 0
    mission_feasible = 0
    fully_feasible = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            candidate = json.loads(line)
            feasible = bool(candidate.get("feasible", False))
            mission = bool(candidate.get("mission_feasible", False))
            if feasible:
                problem_feasible += 1
            if mission:
                mission_feasible += 1
            if feasible and mission:
                fully_feasible += 1
    return problem_feasible, mission_feasible, fully_feasible


def count_detail_payloads(details_dir: Path) -> tuple[int, int, int, int]:
    detail_files = sorted(details_dir.glob("rank_*.json"))
    event_count = 0
    leg_count = 0
    dense_sample_count = 0
    for path in detail_files:
        payload = load_json(path)
        event_count += len(payload.get("events", []))
        leg_count += len(payload.get("legs", []))
        dense_sample_count += len(payload.get("dense_samples", []))
    return len(detail_files), event_count, leg_count, dense_sample_count


def build_run_record(run_dir: Path) -> RunRecord:
    run_summary = load_json(run_dir / "run_summary.json")
    config = load_json(run_dir / "config.json")
    compute_profile = load_json(run_dir / "compute_profile.json")
    cloud_job = load_json(run_dir / "cloud_job.json")
    top_summaries = load_json(run_dir / "top_candidate_summaries.json")
    best_summary = top_summaries[0]
    archive_problem_feasible, archive_mission_feasible, archive_fully_feasible = count_archive_candidates(
        run_dir / "all_candidates.jsonl"
    )
    detail_file_count, detail_event_count, detail_leg_count, detail_dense_sample_count = count_detail_payloads(
        run_dir / "candidate_details"
    )
    return RunRecord(
        run_dir=run_dir,
        family_key=str(config.get("name")),
        family_label=str(config.get("label")),
        config_name=str(config.get("name")),
        config_path=str(cloud_job.get("config_path", "")),
        seed=int(run_summary["base_seed"]),
        compute_level=int(compute_profile["level"]),
        phase1_islands=int(compute_profile["phase1_islands"]),
        phase1_rounds=int(compute_profile["phase1_rounds"]),
        phase2_seed_count=int(compute_profile["phase2_seed_count"]),
        phase3_candidate_count=int(compute_profile["phase3_candidate_count"]),
        execution_mode=str(cloud_job.get("execution_mode", "unknown")),
        runtime_seconds=float(run_summary["runtime_seconds"]),
        eval_phase1=int(run_summary["evaluation_counts"]["phase1"]),
        eval_phase2=int(run_summary["evaluation_counts"]["phase2"]),
        eval_phase3=int(run_summary["evaluation_counts"]["phase3"]),
        eval_total=int(run_summary["evaluation_counts"]["total"]),
        archive_size=int(run_summary["archive_size"]),
        top_candidate_count=int(run_summary["top_candidate_count"]),
        archive_problem_feasible_count=archive_problem_feasible,
        archive_mission_feasible_count=archive_mission_feasible,
        archive_fully_feasible_count=archive_fully_feasible,
        detail_file_count=detail_file_count,
        detail_event_count=detail_event_count,
        detail_leg_count=detail_leg_count,
        detail_dense_sample_count=detail_dense_sample_count,
        best_objective_m_s=float(run_summary["best_objective_m_s"]),
        best_total_dsm_kms=float(best_summary["total_dsm_kms"]),
        best_total_tof_years=float(best_summary["total_tof_years"]),
        best_c3_kms2=float(best_summary["c3_kms2"]),
        best_arrival_vinf_kms=float(best_summary["arrival_vinf_kms"]),
        started_at_utc=str(run_summary["started_at_utc"]),
        submitted_at_utc=cloud_job.get("submitted_at_utc"),
        batch_job_id=cloud_job.get("batch_job_id"),
    )


def format_int(value: int) -> str:
    return f"{value:,}"


def format_hours(seconds: float) -> str:
    return f"{seconds / 3600.0:.2f}"


def format_millions(value: int) -> str:
    return f"{value / 1_000_000.0:.1f}"


def format_billions(value: int) -> str:
    return f"{value / 1_000_000_000.0:.3f}"


def format_kms(value: float) -> str:
    return f"{value:.3f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    divider = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def aggregate(records: list[RunRecord]) -> dict:
    family_groups: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        family_groups[record.family_key].append(record)

    totals = {
        "run_count": len(records),
        "family_count": len(family_groups),
        "seed_count": len({record.seed for record in records}),
        "families": sorted(family_groups),
        "seeds": sorted({record.seed for record in records}),
        "runtime_seconds": sum(record.runtime_seconds for record in records),
        "evaluation_total": sum(record.eval_total for record in records),
        "evaluation_phase1": sum(record.eval_phase1 for record in records),
        "evaluation_phase2": sum(record.eval_phase2 for record in records),
        "evaluation_phase3": sum(record.eval_phase3 for record in records),
        "archive_size": sum(record.archive_size for record in records),
        "archive_problem_feasible_count": sum(record.archive_problem_feasible_count for record in records),
        "archive_mission_feasible_count": sum(record.archive_mission_feasible_count for record in records),
        "archive_fully_feasible_count": sum(record.archive_fully_feasible_count for record in records),
        "top_candidate_count": sum(record.top_candidate_count for record in records),
        "detail_file_count": sum(record.detail_file_count for record in records),
        "detail_event_count": sum(record.detail_event_count for record in records),
        "detail_leg_count": sum(record.detail_leg_count for record in records),
        "detail_dense_sample_count": sum(record.detail_dense_sample_count for record in records),
    }

    per_family = []
    for family_key in sorted(family_groups):
        group = sorted(family_groups[family_key], key=lambda item: item.seed)
        best = min(group, key=lambda item: item.best_total_dsm_kms)
        per_family.append(
            {
                "family_key": family_key,
                "family_label": group[0].family_label,
                "run_count": len(group),
                "seeds": ",".join(str(item.seed) for item in group),
                "runtime_seconds": sum(item.runtime_seconds for item in group),
                "evaluation_total": sum(item.eval_total for item in group),
                "archive_size": sum(item.archive_size for item in group),
                "archive_fully_feasible_count": sum(item.archive_fully_feasible_count for item in group),
                "detail_dense_sample_count": sum(item.detail_dense_sample_count for item in group),
                "best_total_dsm_kms": best.best_total_dsm_kms,
                "best_total_tof_years": best.best_total_tof_years,
                "best_arrival_vinf_kms": best.best_arrival_vinf_kms,
            }
        )

    return {"totals": totals, "per_family": per_family}


def write_csv(records: list[RunRecord]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "campaign_runs.csv"
    fieldnames = [
        "family_key",
        "family_label",
        "seed",
        "compute_level",
        "execution_mode",
        "config_path",
        "runtime_seconds",
        "eval_phase1",
        "eval_phase2",
        "eval_phase3",
        "eval_total",
        "archive_size",
        "archive_problem_feasible_count",
        "archive_mission_feasible_count",
        "archive_fully_feasible_count",
        "top_candidate_count",
        "detail_file_count",
        "detail_event_count",
        "detail_leg_count",
        "detail_dense_sample_count",
        "best_total_dsm_kms",
        "best_total_tof_years",
        "best_c3_kms2",
        "best_arrival_vinf_kms",
        "batch_job_id",
        "started_at_utc",
        "submitted_at_utc",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item.family_key, item.seed)):
            writer.writerow(
                {
                    "family_key": record.family_key,
                    "family_label": record.family_label,
                    "seed": record.seed,
                    "compute_level": record.compute_level,
                    "execution_mode": record.execution_mode,
                    "config_path": record.config_path,
                    "runtime_seconds": f"{record.runtime_seconds:.6f}",
                    "eval_phase1": record.eval_phase1,
                    "eval_phase2": record.eval_phase2,
                    "eval_phase3": record.eval_phase3,
                    "eval_total": record.eval_total,
                    "archive_size": record.archive_size,
                    "archive_problem_feasible_count": record.archive_problem_feasible_count,
                    "archive_mission_feasible_count": record.archive_mission_feasible_count,
                    "archive_fully_feasible_count": record.archive_fully_feasible_count,
                    "top_candidate_count": record.top_candidate_count,
                    "detail_file_count": record.detail_file_count,
                    "detail_event_count": record.detail_event_count,
                    "detail_leg_count": record.detail_leg_count,
                    "detail_dense_sample_count": record.detail_dense_sample_count,
                    "best_total_dsm_kms": f"{record.best_total_dsm_kms:.12f}",
                    "best_total_tof_years": f"{record.best_total_tof_years:.12f}",
                    "best_c3_kms2": f"{record.best_c3_kms2:.12f}",
                    "best_arrival_vinf_kms": f"{record.best_arrival_vinf_kms:.12f}",
                    "batch_job_id": record.batch_job_id or "",
                    "started_at_utc": record.started_at_utc,
                    "submitted_at_utc": record.submitted_at_utc or "",
                }
            )


def write_totals_json(aggregate_payload: dict) -> None:
    path = OUTPUT_DIR / "campaign_totals.json"
    path.write_text(json.dumps(aggregate_payload, indent=2))


def write_slide_bullets(records: list[RunRecord], totals: dict) -> None:
    family_profiles = {record.family_key: record for record in records if record.seed == 42}
    dvega = family_profiles["dvega"]
    veega = family_profiles["veega"]
    path = BACKGROUND_DIR / "slide_ready_bullets.md"
    lines = [
        "# Key Facts About The Background Work",
        "",
        "## What optimization method we used",
        f"- The active search codepath is `v2/01_joi`, which builds a `pykep.trajopt.mga_1dsm` Earth-to-Jupiter transfer problem and solves it with `pygmo`.",
        "- The optimizer is single-objective: it minimizes total deep-space maneuver (DSM) magnitude, then compares time of flight, launch `C3`, and Jupiter arrival `V_inf` afterward in the trade study.",
        "- Mission constraints are enforced as hard filters rather than blended into the objective: the archived campaign used `C3 <= 15 km^2/s^2` and `total TOF <= 3652.5 days`.",
        "- The search itself has three phases: multi-island `sade` exploration, seeded `sade` refinement, and `mbh(compass_search)` polishing of the best candidates.",
        f"- The curated DVEGA campaign ran at compute level {dvega.compute_level} with {dvega.phase1_islands} islands over {dvega.phase1_rounds} exploration rounds, {dvega.phase2_seed_count} seeded refinements, and {dvega.phase3_candidate_count} local-polish candidates; VEEGA used {veega.phase1_islands} islands over {veega.phase1_rounds} rounds, {veega.phase2_seed_count} seeded refinements, and {veega.phase3_candidate_count} local-polish candidates.",
        "",
        "## What the cloud workflow did for us",
        "- The same packaged runner was used for local Docker tests and AWS Batch jobs, so cloud runs reused the exact search/archive codepath instead of a separate script.",
        "- `aws-submit` uploaded the exact mission config to S3, submitted one seed per Batch job, and recorded job metadata for reproducibility.",
        "- Inside the container, the runner emitted heartbeat progress, wrote `cloud_job.json`, and synced the finished run directory back to S3 for later fetch.",
        "- The cloud ledger workflow (`aws-status`, `aws-fetch`, `aws-sync-ledger`, `aws-watch-ledger`) kept active and finished jobs visible without manually tracking job IDs.",
        "",
        "## What scale of search we actually ran",
        f"- The curated `good_results` campaign contains {totals['run_count']} AWS Batch runs across {totals['family_count']} trajectory families and {totals['seed_count']} seeds per family.",
        f"- Across those six runs, the search recorded {format_int(totals['evaluation_total'])} objective evaluations and {format_hours(totals['runtime_seconds'])} hours of wall-clock runtime.",
        f"- The run archives contain {format_int(totals['archive_size'])} deduplicated candidate trajectories in `all_candidates.jsonl`, with {format_int(totals['archive_fully_feasible_count'])} that are both optimizer-feasible and mission-feasible.",
        f"- Each run saved the top 100 detailed candidates, giving {format_int(totals['detail_file_count'])} fully reconstructed candidate files across the campaign.",
        "",
        "## What data we saved for each run and candidate",
        "- Every run directory saves the resolved config, compute profile, run summary, structured progress log, full candidate archive, top-candidate shortlist, and cloud metadata.",
        "- Every detailed candidate file stores the decision vector, decoded timing/flyby parameters, mission summary metrics, event timeline, per-leg transfer data, and dense heliocentric samples.",
        f"- The archived top-candidate set in `good_results` contains {format_int(totals['detail_event_count'])} event records, {format_int(totals['detail_leg_count'])} leg records, and {format_int(totals['detail_dense_sample_count'])} dense trajectory samples ready for later analysis without rerunning optimization.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_campaign_summary(records: list[RunRecord], aggregate_payload: dict) -> None:
    totals = aggregate_payload["totals"]
    per_family = aggregate_payload["per_family"]

    family_rows = []
    for family in per_family:
        family_rows.append(
            [
                family["family_label"],
                str(family["run_count"]),
                family["seeds"],
                format_hours(family["runtime_seconds"]),
                format_millions(family["evaluation_total"]),
                format_int(family["archive_size"]),
                format_int(family["archive_fully_feasible_count"]),
                format_kms(family["best_total_dsm_kms"]),
                f"{family['best_total_tof_years']:.2f}",
            ]
        )

    run_rows = []
    for record in sorted(records, key=lambda item: (item.family_key, item.seed)):
        run_rows.append(
            [
                record.family_label,
                str(record.seed),
                format_hours(record.runtime_seconds),
                format_millions(record.eval_total),
                format_int(record.archive_size),
                format_int(record.archive_fully_feasible_count),
                format_kms(record.best_total_dsm_kms),
                f"{record.best_total_tof_years:.2f}",
            ]
        )

    path = BACKGROUND_DIR / "campaign_summary.md"
    lines = [
        "# Search Campaign Summary",
        "",
        "## Totals",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Runs in curated campaign", str(totals["run_count"])],
                ["Trajectory families compared", str(totals["family_count"])],
                ["Seeds used", ", ".join(str(seed) for seed in totals["seeds"])],
                ["Compute level used", "10 for all curated campaign runs"],
                ["Execution mode", "AWS Batch for all curated runs"],
                ["Total wall-clock runtime", f"{format_hours(totals['runtime_seconds'])} hours"],
                ["Total objective evaluations", format_int(totals["evaluation_total"])],
                ["Phase 1 evaluations", format_int(totals["evaluation_phase1"])],
                ["Phase 2 evaluations", format_int(totals["evaluation_phase2"])],
                ["Phase 3 evaluations", format_int(totals["evaluation_phase3"])],
                ["Archived candidate trajectories", format_int(totals["archive_size"])],
                ["Fully feasible archived candidates", format_int(totals["archive_fully_feasible_count"])],
                ["Detailed top-candidate files", format_int(totals["detail_file_count"])],
                ["Saved event records in detailed files", format_int(totals["detail_event_count"])],
                ["Saved leg records in detailed files", format_int(totals["detail_leg_count"])],
                ["Saved dense trajectory samples", format_int(totals["detail_dense_sample_count"])],
            ],
        ),
        "",
        "## By Family",
        "",
        markdown_table(
            [
                "Family",
                "Runs",
                "Seeds",
                "Runtime (h)",
                "Evaluations (M)",
                "Archived candidates",
                "Fully feasible archive",
                "Best DSM (km/s)",
                "Best TOF (yr)",
            ],
            family_rows,
        ),
        "",
        "## By Run",
        "",
        markdown_table(
            [
                "Family",
                "Seed",
                "Runtime (h)",
                "Evaluations (M)",
                "Archived candidates",
                "Fully feasible archive",
                "Best DSM (km/s)",
                "Best TOF (yr)",
            ],
            run_rows,
        ),
        "",
        "## Notes",
        "",
        "- `Archived candidate trajectories` comes from the per-run `all_candidates.jsonl` counts saved in each run summary.",
        "- `Fully feasible archived candidates` counts candidates that are both optimizer-feasible and mission-feasible in the saved JSONL archive.",
        "- Runtime is the recorded wall time from `run_summary.json`; it is not multiplied by Batch vCPU allocation.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_workflow_markdown() -> None:
    path = BACKGROUND_DIR / "workflow_at_a_glance.md"
    lines = [
        "# Workflow At A Glance",
        "",
        "![Workflow overview](output/workflow_overview.svg)",
        "",
        markdown_table(
            ["Stage", "What happens", "Primary saved artifact"],
            [
                [
                    "Config",
                    "Define sequence, departure window, `C3` cap, TOF bounds, search overrides, and optional incumbent seeds.",
                    "`v2/01_joi/configs/*.toml`",
                ],
                [
                    "Search",
                    "Build the JOI `mga_1dsm` problem and run the three-phase `pygmo` search campaign.",
                    "`progress_events.json` and `run_summary.json`",
                ],
                [
                    "Archive",
                    "Save all deduplicated candidates plus a reconstructed top-100 shortlist for each run.",
                    "`all_candidates.jsonl`, `top_candidates.jsonl`, `candidate_details/*.json`",
                ],
                [
                    "Trade study",
                    "Flatten cross-seed candidate summaries into sortable tables for family comparison.",
                    "`v2/good_results/*/*_trade_study_candidates.csv`",
                ],
                [
                    "Chosen analysis",
                    "Use the saved candidate detail file for plots, DSM tables, and follow-on mission analysis without rerunning search.",
                    "`candidate_details/rank_###.json`",
                ],
            ],
        ),
        "",
        "The workflow visual is meant to be slide-ready. The table below it is there in case a slide needs a text version.",
    ]
    path.write_text("\n".join(lines) + "\n")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #14213d; }",
        ".title { font-size: 34px; font-weight: 700; }",
        ".subtitle { font-size: 18px; fill: #44556b; }",
        ".box { fill: #f8fafc; stroke: #cbd5e1; stroke-width: 2; rx: 18; ry: 18; }",
        ".step { font-size: 24px; font-weight: 700; }",
        ".body { font-size: 17px; fill: #334155; }",
        ".metric { font-size: 38px; font-weight: 700; }",
        ".metric-label { font-size: 18px; fill: #475569; }",
        ".small { font-size: 15px; fill: #64748b; }",
        "</style>",
    ]


def write_workflow_svg() -> None:
    width = 1500
    height = 720
    boxes = [
        (70, 190, 240, 160, "1. Config", "Sequence, bounds,\nsearch overrides,\nseed incumbents"),
        (350, 190, 240, 160, "2. Search", "pykep problem +\n3-phase pygmo\noptimization"),
        (630, 190, 240, 160, "3. Archive", "All candidates,\nrun summary,\ntop details"),
        (910, 190, 240, 160, "4. Trade Study", "Cross-seed tables\nand family\ncomparison"),
        (1190, 190, 240, 160, "5. Chosen Analysis", "Trajectory plots,\nDSM tables,\nfollow-on analysis"),
    ]
    lines = svg_header(width, height)
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    lines.append('<text class="title" x="70" y="70">Earth-to-Jupiter Search Workflow</text>')
    lines.append(
        '<text class="subtitle" x="70" y="105">Config -> search -> archive -> trade study -> chosen trajectory analysis</text>'
    )
    for idx in range(len(boxes) - 1):
        x1 = boxes[idx][0] + boxes[idx][2]
        x2 = boxes[idx + 1][0]
        y = 270
        lines.append(
            f'<line x1="{x1 + 8}" y1="{y}" x2="{x2 - 8}" y2="{y}" stroke="#94a3b8" stroke-width="6" stroke-linecap="round"/>'
        )
        lines.append(
            f'<polygon points="{x2 - 8},{y} {x2 - 28},{y - 12} {x2 - 28},{y + 12}" fill="#94a3b8"/>'
        )

    for x, y, w, h, title, body in boxes:
        lines.append(f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}"/>')
        lines.append(f'<text class="step" x="{x + 18}" y="{y + 38}">{title}</text>')
        for line_idx, text_line in enumerate(body.split("\n")):
            lines.append(f'<text class="body" x="{x + 18}" y="{y + 80 + 24 * line_idx}">{text_line}</text>')

    artifact_text = [
        ("Config files", "Mission family, TOF bounds, C3 cap, campaign budget"),
        ("Search outputs", "Progress checkpoints and evaluation counts while the run is active"),
        ("Detailed archive", "Reconstructed events, legs, and dense samples for top candidates"),
    ]
    lines.append('<text class="subtitle" x="70" y="440">Why this workflow mattered</text>')
    for idx, (label, detail) in enumerate(artifact_text):
        y = 490 + idx * 62
        lines.append(f'<circle cx="90" cy="{y - 7}" r="8" fill="#fca311"/>')
        lines.append(f'<text class="step" x="115" y="{y}">{label}</text>')
        lines.append(f'<text class="body" x="320" y="{y}">{detail}</text>')
    lines.append("</svg>")
    (OUTPUT_DIR / "workflow_overview.svg").write_text("\n".join(lines) + "\n")


def rect_card(x: int, y: int, w: int, h: int, metric: str, label: str) -> list[str]:
    return [
        f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}"/>',
        f'<text class="metric" x="{x + 22}" y="{y + 62}">{metric}</text>',
        f'<text class="metric-label" x="{x + 22}" y="{y + 98}">{label}</text>',
    ]


def write_campaign_scale_svg(records: list[RunRecord], aggregate_payload: dict) -> None:
    totals = aggregate_payload["totals"]
    per_family = aggregate_payload["per_family"]
    width = 1400
    height = 860
    lines = svg_header(width, height)
    lines.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    lines.append('<text class="title" x="70" y="70">Curated Search Campaign Scale</text>')
    lines.append(
        '<text class="subtitle" x="70" y="105">Numbers below are computed directly from the saved `good_results` archives.</text>'
    )
    cards = [
        (str(totals["run_count"]), "AWS Batch runs"),
        (str(totals["family_count"]), "trajectory families"),
        (f"{format_billions(totals['evaluation_total'])}B", "objective evaluations"),
        (f"{format_hours(totals['runtime_seconds'])} h", "recorded wall time"),
        (format_int(totals["archive_size"]), "archived candidates"),
        (format_int(totals["detail_file_count"]), "reconstructed top candidates"),
    ]
    card_positions = [
        (70, 150),
        (290, 150),
        (510, 150),
        (820, 150),
        (1040, 150),
        (70, 320),
    ]
    card_sizes = [(190, 130), (190, 130), (270, 130), (190, 130), (270, 130), (220, 130)]
    for (metric, label), (x, y), (w, h) in zip(cards, card_positions, card_sizes, strict=True):
        lines.extend(rect_card(x, y, w, h, metric, label))

    lines.append('<text class="subtitle" x="70" y="520">Per-family breakdown</text>')
    bar_x = 70
    bar_y = 560
    bar_height = 34
    bar_gap = 84
    max_eval = max(item["evaluation_total"] for item in per_family)
    for idx, family in enumerate(per_family):
        y = bar_y + idx * bar_gap
        bar_width = int(760 * family["evaluation_total"] / max_eval)
        fill = "#fca311" if family["family_key"] == "veega" else "#4f772d"
        lines.append(f'<text class="step" x="{bar_x}" y="{y - 14}">{family["family_label"]}</text>')
        lines.append(f'<rect class="box" x="{bar_x}" y="{y}" width="760" height="{bar_height}"/>')
        lines.append(
            f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{fill}" rx="18" ry="18"/>'
        )
        lines.append(
            f'<text class="body" x="{bar_x + 790}" y="{y + 25}">{format_millions(family["evaluation_total"])}M evals, {format_hours(family["runtime_seconds"])} h, best DSM {format_kms(family["best_total_dsm_kms"])} km/s</text>'
        )

    lines.append('<text class="subtitle" x="70" y="760">Detailed archive payload</text>')
    lines.append(
        f'<text class="body" x="70" y="795">{format_int(totals["detail_event_count"])} saved events, {format_int(totals["detail_leg_count"])} saved legs, and {format_int(totals["detail_dense_sample_count"])} dense trajectory samples across the 600 detailed candidate files.</text>'
    )
    lines.append("</svg>")
    (OUTPUT_DIR / "campaign_scale.svg").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = [build_run_record(run_dir) for run_dir in iter_run_dirs()]
    aggregate_payload = aggregate(records)
    write_csv(records)
    write_totals_json(aggregate_payload)
    write_slide_bullets(records, aggregate_payload["totals"])
    write_campaign_summary(records, aggregate_payload)
    write_workflow_markdown()
    write_workflow_svg()
    write_campaign_scale_svg(records, aggregate_payload)
    print(BACKGROUND_DIR)


if __name__ == "__main__":
    main()
