from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Callable

from .runtime import build_job_metadata, env_git_metadata


RunSearchFn = Callable[..., tuple[Path, dict]]
PrintSummaryFn = Callable[[Path, dict, int], None]


@dataclass(slots=True)
class ContainerJobTarget:
    name: str
    source_root: Path
    run_search: RunSearchFn
    print_run_summary: PrintSummaryFn


class JobHeartbeatReporter:
    def __init__(
        self,
        *,
        name: str,
        compute_level: int,
        seed: int,
        config_label: str | None,
        s3_output_prefix: str | None,
    ) -> None:
        self._name = name
        self._compute_level = compute_level
        self._seed = seed
        self._config_label = config_label
        self._s3_output_prefix = s3_output_prefix.rstrip("/") if s3_output_prefix else None
        self._heartbeat_path = Path("/tmp/joi-batch-heartbeat.json")
        self._last_upload_monotonic = 0.0

    def emit(self, progress_event: dict) -> None:
        summary = {
            "event": progress_event.get("event"),
            "elapsed_seconds": round(float(progress_event.get("elapsed_seconds", 0.0)), 2),
            "completed": progress_event.get("completed"),
            "total": progress_event.get("total"),
            "best_objective_km_s": (
                round(float(progress_event["best_objective_m_s"]) / 1000.0, 6)
                if "best_objective_m_s" in progress_event
                else None
            ),
            "best_c3_kms2": progress_event.get("best_c3_kms2"),
            "best_total_tof_days": progress_event.get("best_total_tof_days"),
            "archive_size": progress_event.get("archive_size"),
            "fevals_total": progress_event.get("fevals", {}).get("total"),
        }
        print(f"Progress: {json.dumps(summary, sort_keys=True)}", flush=True)
        self._write_payload(status="running", progress_event=progress_event)

    def finalize(
        self,
        *,
        status: str,
        run_dir: Path | None = None,
        uploaded_prefix: str | None = None,
        error: str | None = None,
        search_result: dict | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "status": status,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        if run_dir is not None:
            payload["run_dir"] = str(run_dir)
        if uploaded_prefix is not None:
            payload["uploaded_s3_prefix"] = uploaded_prefix
        if error is not None:
            payload["error"] = error
        if search_result is not None:
            payload["best_objective_m_s"] = float(search_result["best_candidate"]["objective"])
            payload["runtime_seconds"] = float(search_result["runtime_seconds"])
            payload["evaluation_counts"] = search_result["evaluation_counts"]
        self._write_payload(extra_payload=payload, force_upload=True)

    def _write_payload(
        self,
        *,
        status: str | None = None,
        progress_event: dict | None = None,
        extra_payload: dict[str, object] | None = None,
        force_upload: bool = False,
    ) -> None:
        payload: dict[str, object] = {
            "target": self._name,
            "compute_level": self._compute_level,
            "seed": self._seed,
            "config_label": self._config_label,
            "batch_job_id": os.getenv("AWS_BATCH_JOB_ID"),
            "status": status or "running",
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }
        if progress_event is not None:
            payload["progress"] = progress_event
        if extra_payload is not None:
            payload.update(extra_payload)
        self._heartbeat_path.write_text(json.dumps(payload, indent=2))
        self._upload_if_needed(force=force_upload)

    def _upload_if_needed(self, *, force: bool) -> None:
        if self._s3_output_prefix is None:
            return
        now = time.monotonic()
        if not force and (now - self._last_upload_monotonic) < 60.0:
            return
        try:
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    str(self._heartbeat_path),
                    f"{self._s3_output_prefix}/heartbeat.json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self._last_upload_monotonic = now
        except subprocess.CalledProcessError as exc:
            print(f"Heartbeat upload warning: {exc.stderr.strip() or exc}", flush=True)


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Container path to the sequence config TOML.")
    config_group.add_argument("--config-s3-uri", help="S3 URI to the sequence config TOML.")
    parser.add_argument("--compute-level", type=int, required=True, help="Compute level from 1 to 10.")
    parser.add_argument("--results-dir", required=True, help="Container directory for output run folders.")
    parser.add_argument("--seed", type=int, required=True, help="Base random seed.")
    parser.add_argument(
        "--metadata-filename",
        default="cloud_job.json",
        help="Filename to write alongside the saved run directory.",
    )
    parser.add_argument(
        "--execution-mode",
        default="docker-local",
        help="Execution mode label recorded in the cloud metadata.",
    )
    parser.add_argument(
        "--s3-output-prefix",
        default=None,
        help="If set, upload the final run directory under this S3 prefix.",
    )
    return parser.parse_args()


def materialize_config(args: argparse.Namespace) -> tuple[Path, str | None]:
    if args.config is not None:
        return Path(args.config).resolve(), None

    if args.config_s3_uri is None:
        raise RuntimeError("Either --config or --config-s3-uri must be provided.")

    local_dir = Path("/tmp/joi-batch-config")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / Path(args.config_s3_uri).name
    subprocess.run(["aws", "s3", "cp", args.config_s3_uri, str(local_path)], check=True)
    return local_path, args.config_s3_uri


def upload_results(run_dir: Path, s3_output_prefix: str) -> str:
    final_prefix = f"{s3_output_prefix.rstrip('/')}/{run_dir.name}"
    subprocess.run(["aws", "s3", "sync", str(run_dir), final_prefix], check=True)
    return final_prefix


def run_container_job_main(target: ContainerJobTarget) -> None:
    args = parse_args(f"Run a {target.name} search inside a container job.")
    config_path, config_s3_uri = materialize_config(args)
    git_metadata = env_git_metadata()
    requested_s3_output_prefix = args.s3_output_prefix or os.getenv("JOI_S3_OUTPUT_PREFIX")
    reporter = JobHeartbeatReporter(
        name=target.name,
        compute_level=args.compute_level,
        seed=args.seed,
        config_label=os.getenv("JOI_CONFIG_LABEL"),
        s3_output_prefix=requested_s3_output_prefix,
    )
    try:
        run_dir, search_result = target.run_search(
            config_path=config_path,
            compute_level=args.compute_level,
            results_dir=args.results_dir,
            seed=args.seed,
            git_metadata=git_metadata,
            progress_callback=reporter.emit,
        )
    except Exception as exc:
        reporter.finalize(status="failed", error=str(exc))
        raise

    metadata = build_job_metadata(
        config_path=config_path,
        compute_level=args.compute_level,
        seed=args.seed,
        execution_mode=args.execution_mode,
        source_root=target.source_root,
        config_label=os.getenv("JOI_CONFIG_LABEL"),
        image_ref=os.getenv("JOI_IMAGE_REF"),
        submitted_at_utc=os.getenv("JOI_SUBMITTED_AT_UTC"),
        git_metadata=git_metadata,
        batch_job_id=os.getenv("AWS_BATCH_JOB_ID"),
        s3_output_prefix=requested_s3_output_prefix,
    )
    if config_s3_uri is not None:
        metadata["config_s3_uri"] = config_s3_uri
    metadata_path = run_dir / args.metadata_filename
    metadata_path.write_text(json.dumps(metadata, indent=2))

    uploaded_prefix = None
    if requested_s3_output_prefix is not None:
        uploaded_prefix = upload_results(run_dir, requested_s3_output_prefix)
        metadata["uploaded_s3_prefix"] = uploaded_prefix
        metadata_path.write_text(json.dumps(metadata, indent=2))

    reporter.finalize(
        status="succeeded",
        run_dir=run_dir,
        uploaded_prefix=uploaded_prefix,
        search_result=search_result,
    )

    target.print_run_summary(run_dir, search_result, args.seed)
    print(f"Saved container metadata: {metadata_path}")
    if uploaded_prefix is not None:
        print(f"Uploaded run to: {uploaded_prefix}")
