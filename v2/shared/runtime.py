from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import os
from pathlib import Path
import socket
import subprocess


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def resolve_repo_root(start_path: Path) -> Path | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(start_path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = proc.stdout.strip()
    return Path(output) if output else None


def get_git_metadata(source_root: Path | None) -> dict[str, str | bool | None]:
    if source_root is None:
        return {
            "git_commit": None,
            "git_branch": None,
            "git_is_dirty": None,
        }

    repo_root = resolve_repo_root(source_root)
    if repo_root is None:
        return {
            "git_commit": None,
            "git_branch": None,
            "git_is_dirty": None,
        }

    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        branch = subprocess.run(
            ["git", "-C", str(repo_root), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": None,
            "git_branch": None,
            "git_is_dirty": None,
        }

    return {
        "git_commit": commit.stdout.strip() or None,
        "git_branch": branch.stdout.strip() or None,
        "git_is_dirty": bool(dirty.stdout.strip()),
    }


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def env_git_metadata(prefix: str = "JOI_SOURCE_") -> dict[str, str | bool | None] | None:
    commit = os.getenv(f"{prefix}GIT_COMMIT")
    branch = os.getenv(f"{prefix}GIT_BRANCH")
    dirty = parse_optional_bool(os.getenv(f"{prefix}GIT_IS_DIRTY"))
    if commit is None and branch is None and dirty is None:
        return None
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_is_dirty": dirty,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_path_label(config_path: Path, source_root: Path | None) -> str:
    if source_root is None:
        return str(config_path)
    try:
        return str(config_path.relative_to(source_root))
    except ValueError:
        return str(config_path)


def build_job_metadata(
    *,
    config_path: Path,
    compute_level: int,
    seed: int,
    execution_mode: str,
    source_root: Path | None,
    config_label: str | None = None,
    image_ref: str | None = None,
    submitted_at_utc: str | None = None,
    git_metadata: dict[str, str | bool | None] | None = None,
    batch_job_id: str | None = None,
    s3_output_prefix: str | None = None,
) -> dict[str, str | int | bool | None]:
    metadata = dict(git_metadata or get_git_metadata(source_root))
    metadata.update(
        {
            "execution_mode": execution_mode,
            "image_ref": image_ref,
            "submitted_at_utc": submitted_at_utc,
            "container_hostname": socket.gethostname(),
            "config_path": config_label or config_path_label(config_path, source_root),
            "config_sha256": file_sha256(config_path),
            "compute_level": compute_level,
            "seed": seed,
            "batch_job_id": batch_job_id,
            "s3_output_prefix": s3_output_prefix,
            "metadata_recorded_at_utc": utc_now_iso(),
        }
    )
    return metadata
