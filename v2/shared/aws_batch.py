from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import tomllib

from .runtime import get_git_metadata


DEFAULT_INSTANCE_TYPES = [
    "c7i.2xlarge",
    "c7i.4xlarge",
    "c7i.8xlarge",
    "c7i.12xlarge",
    "c7a.2xlarge",
    "c7a.4xlarge",
    "c7a.8xlarge",
    "c7a.12xlarge",
    "c6i.2xlarge",
    "c6i.4xlarge",
    "c6i.8xlarge",
    "c6i.12xlarge",
]


@dataclass(slots=True)
class AwsBatchSettings:
    region: str
    project_name: str
    stack_name: str
    batch_job_queue: str
    batch_job_definition: str
    artifacts_bucket: str
    artifacts_prefix: str = "joi-runs"
    job_name_prefix: str = "joi"
    submission_dir: str = "v2/shared/cloud-submissions"
    active_job_ledger_path: str = "v2/cloud_runs_active.csv"
    completed_job_ledger_path: str = "v2/cloud_runs_done.csv"
    job_vcpus: int = 8
    job_memory_mib: int = 16384
    max_vcpus: int = 64
    instance_types: list[str] = field(default_factory=lambda: list(DEFAULT_INSTANCE_TYPES))
    vpc_id: str | None = None
    subnet_ids: list[str] | None = None
    associate_app_registry: bool = False
    app_registry_application: str | None = None
    app_registry_application_arn: str | None = None
    cloudformation_template: str = "v2/shared/cloud/aws_batch_stack.yaml"
    ecr_image_tag: str = "latest"

    def to_dict(self) -> dict:
        return asdict(self)


def load_aws_batch_settings(path: str | Path) -> AwsBatchSettings:
    cfg_path = Path(path)
    raw = tomllib.loads(cfg_path.read_text())
    aws_cfg = raw["aws"]
    subnet_ids = aws_cfg.get("subnet_ids")
    instance_types = aws_cfg.get("instance_types")
    legacy_ledger_path = aws_cfg.get("job_ledger_path")
    return AwsBatchSettings(
        region=aws_cfg["region"],
        project_name=aws_cfg.get("project_name", "minos-joi"),
        stack_name=aws_cfg.get("stack_name", "minos-joi-batch"),
        batch_job_queue=aws_cfg["batch_job_queue"],
        batch_job_definition=aws_cfg["batch_job_definition"],
        artifacts_bucket=aws_cfg["artifacts_bucket"],
        artifacts_prefix=aws_cfg.get("artifacts_prefix", "joi-runs"),
        job_name_prefix=aws_cfg.get("job_name_prefix", "joi"),
        submission_dir=aws_cfg.get("submission_dir", "v2/shared/cloud-submissions"),
        active_job_ledger_path=aws_cfg.get(
            "active_job_ledger_path",
            legacy_ledger_path or "v2/cloud_runs_active.csv",
        ),
        completed_job_ledger_path=aws_cfg.get(
            "completed_job_ledger_path",
            "v2/cloud_runs_done.csv",
        ),
        job_vcpus=int(aws_cfg.get("job_vcpus", 8)),
        job_memory_mib=int(aws_cfg.get("job_memory_mib", 16_384)),
        max_vcpus=int(aws_cfg.get("max_vcpus", 64)),
        instance_types=list(instance_types) if instance_types is not None else list(DEFAULT_INSTANCE_TYPES),
        vpc_id=aws_cfg.get("vpc_id"),
        subnet_ids=list(subnet_ids) if subnet_ids is not None else None,
        associate_app_registry=bool(aws_cfg.get("associate_app_registry", False)),
        app_registry_application=aws_cfg.get("app_registry_application"),
        app_registry_application_arn=aws_cfg.get("app_registry_application_arn"),
        cloudformation_template=aws_cfg.get("cloudformation_template", "v2/shared/cloud/aws_batch_stack.yaml"),
        ecr_image_tag=aws_cfg.get("ecr_image_tag", "latest"),
    )


def ensure_clean_worktree(repo_root: Path, allow_dirty: bool) -> None:
    git_metadata = get_git_metadata(repo_root)
    if git_metadata["git_is_dirty"] and not allow_dirty:
        raise RuntimeError(
            "Refusing AWS submission from a dirty worktree. "
            "Commit or stash changes, or pass --allow-dirty to override."
        )


def aws_cli_json(args: list[str], *, region: str, cwd: Path) -> dict | list:
    command = ["aws", "--region", region, *args, "--output", "json"]
    proc = subprocess.run(command, check=True, capture_output=True, text=True, cwd=cwd)
    return json.loads(proc.stdout)


def aws_cli_run(args: list[str], *, region: str, cwd: Path) -> None:
    command = ["aws", "--region", region, *args]
    subprocess.run(command, check=True, cwd=cwd)


def utc_timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def build_job_name(prefix: str, config_name: str, compute_level: int, seed: int) -> str:
    slug = utc_timestamp_slug().lower()
    return f"{prefix}-{config_name}-l{compute_level}-s{seed}-{slug}"


def s3_uri(bucket: str, key: str) -> str:
    clean_key = key.lstrip("/")
    return f"s3://{bucket}/{clean_key}"


def build_submission_prefix(
    settings: AwsBatchSettings,
    *,
    job_name: str,
) -> str:
    base = settings.artifacts_prefix.strip("/")
    return s3_uri(settings.artifacts_bucket, f"{base}/{job_name}")


def ensure_submission_dir(repo_root: Path, submission_dir: str) -> Path:
    path = repo_root / submission_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_submission_record(
    *,
    repo_root: Path,
    submission_dir: str,
    job_id: str,
    payload: dict,
) -> Path:
    record_dir = ensure_submission_dir(repo_root, submission_dir)
    path = record_dir / f"{job_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_submission_record(
    *,
    repo_root: Path,
    submission_dir: str,
    job_id: str,
) -> dict | None:
    path = repo_root / submission_dir / f"{job_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def extract_log_stream(job_description: dict) -> str | None:
    attempts = job_description.get("attempts", [])
    if attempts:
        container = attempts[-1].get("container", {})
        log_stream = container.get("logStreamName")
        if log_stream:
            return log_stream
    container = job_description.get("container", {})
    return container.get("logStreamName")


LEDGER_FIELDS = [
    "job_id",
    "job_name",
    "trajectory",
    "status",
    "next_action",
    "result_location",
    "submitted_at_utc",
    "started_at_utc",
    "stopped_at_utc",
    "last_checked_utc",
    "compute_level",
    "seed",
    "job_vcpus",
    "job_memory_mib",
    "image_ref",
    "job_definition",
    "config_path",
    "config_s3_uri",
    "s3_output_prefix",
    "log_stream",
    "fetched_to",
]

TERMINAL_JOB_STATUSES = {"SUCCEEDED", "FAILED"}


def epoch_ms_to_iso8601(epoch_ms: int | None) -> str:
    if epoch_ms is None:
        return ""
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=UTC).isoformat()


def ensure_job_ledger(repo_root: Path, ledger_path: str) -> Path:
    path = repo_root / ledger_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
            writer.writeheader()
    return path


def read_job_ledger_rows(repo_root: Path, ledger_path: str) -> list[dict[str, str]]:
    path = repo_root / ledger_path
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def derive_trajectory(row: dict[str, str]) -> str:
    config_path = row.get("config_path", "").strip()
    if config_path:
        return Path(config_path).stem
    job_name = row.get("job_name", "").strip()
    parts = job_name.split("-")
    if len(parts) >= 2:
        return parts[1]
    return ""


def derive_next_action(row: dict[str, str]) -> str:
    status = row.get("status", "").strip()
    fetched_to = row.get("fetched_to", "").strip()
    if status == "SUCCEEDED":
        return "review_local" if fetched_to else "fetch"
    if status == "FAILED":
        return "inspect_failure"
    return "wait"


def normalize_ledger_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {field: str(row.get(field, "")) for field in LEDGER_FIELDS}
    normalized["trajectory"] = normalized["trajectory"] or derive_trajectory(normalized)
    normalized["next_action"] = derive_next_action(normalized)
    normalized["result_location"] = normalized["fetched_to"] or normalized["s3_output_prefix"]
    return normalized


def write_job_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def is_completed_row(row: dict[str, str]) -> bool:
    status = row.get("status", "").strip()
    if status == "SUCCEEDED":
        return bool(row.get("fetched_to", "").strip())
    return status in TERMINAL_JOB_STATUSES


def upsert_job_ledger_row(
    *,
    repo_root: Path,
    active_ledger_path: str,
    completed_ledger_path: str,
    row: dict[str, str],
) -> tuple[Path, Path]:
    active_path = ensure_job_ledger(repo_root, active_ledger_path)
    completed_path = ensure_job_ledger(repo_root, completed_ledger_path)

    existing_rows = read_job_ledger_rows(repo_root, active_ledger_path) + read_job_ledger_rows(repo_root, completed_ledger_path)
    normalized = normalize_ledger_row(row)
    replaced = False
    for idx, existing in enumerate(existing_rows):
        if existing.get("job_id") == normalized["job_id"]:
            merged = {field: existing.get(field, "") for field in LEDGER_FIELDS}
            for field, value in normalized.items():
                if value != "":
                    merged[field] = value
            existing_rows[idx] = merged
            replaced = True
            break
    if not replaced:
        existing_rows.append(normalized)

    existing_rows.sort(key=lambda item: item.get("submitted_at_utc", ""), reverse=True)
    active_rows = [normalize_ledger_row(item) for item in existing_rows if not is_completed_row(item)]
    completed_rows = [normalize_ledger_row(item) for item in existing_rows if is_completed_row(item)]
    write_job_ledger(active_path, active_rows)
    write_job_ledger(completed_path, completed_rows)
    return active_path, completed_path
