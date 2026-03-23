from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import time

from .aws_batch import (
    AwsBatchSettings,
    aws_cli_json,
    aws_cli_run,
    build_job_name,
    build_submission_prefix,
    epoch_ms_to_iso8601,
    ensure_clean_worktree,
    extract_log_stream,
    ensure_submission_dir,
    load_aws_batch_settings,
    read_job_ledger_rows,
    read_submission_record,
    upsert_job_ledger_row,
    write_submission_record,
)
from .runtime import get_git_metadata, resolve_repo_root


CONTAINER_CONFIG_DIR = Path("/job-config")
CONTAINER_RESULTS_DIR = Path("/job-results")
DEFAULT_PLACEHOLDER_IMAGE = "public.ecr.aws/docker/library/python:3.12"


@dataclass(slots=True)
class CloudCliTarget:
    name: str
    repo_root_hint: Path
    dockerfile_path: str
    default_image_tag: str
    default_docker_results_dir: str
    default_fetch_results_dir: str


def require_repo_root(repo_root_hint: Path) -> Path:
    repo_root = resolve_repo_root(repo_root_hint)
    if repo_root is None:
        raise RuntimeError("Could not resolve the git repo root for the v2 cloud workflow.")
    return repo_root


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd, env=env)


def parse_args(target: CloudCliTarget) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{target.name} cloud tooling for local Docker and AWS Batch workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("docker-build", help="Build the local runner image.")
    build_parser.add_argument("--image-tag", default=target.default_image_tag, help="Docker image tag.")
    build_parser.add_argument("--platform", default="linux/amd64", help="Target container platform.")
    build_parser.add_argument("--no-cache", action="store_true", help="Disable Docker layer cache during build.")

    run_parser = subparsers.add_parser("docker-run", help="Run the search inside the local Docker image.")
    run_parser.add_argument("--image-tag", default=target.default_image_tag, help="Docker image tag.")
    run_parser.add_argument("--config", required=True, help="Host path to the sequence config TOML.")
    run_parser.add_argument("--compute-level", type=int, default=1, help="Compute level from 1 to 10.")
    run_parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    run_parser.add_argument(
        "--host-results-dir",
        default=target.default_docker_results_dir,
        help="Host directory that will receive the run folders.",
    )
    run_parser.add_argument(
        "--execution-mode",
        default="docker-local",
        help="Execution mode label to store in container metadata.",
    )

    stack_parser = subparsers.add_parser("aws-deploy-stack", help="Create or update the AWS Batch stack.")
    stack_parser.add_argument("--aws-config", required=True, help="Path to the AWS Batch TOML config.")
    stack_parser.add_argument(
        "--image-uri",
        default=DEFAULT_PLACEHOLDER_IMAGE,
        help="Image URI used in the Batch job definition for this deployment.",
    )

    push_parser = subparsers.add_parser("aws-push-image", help="Push the local Docker image to the stack ECR repo.")
    push_parser.add_argument("--aws-config", required=True, help="Path to the AWS Batch TOML config.")
    push_parser.add_argument("--image-tag", default=target.default_image_tag, help="Local Docker image tag.")
    push_parser.add_argument("--remote-tag", default=None, help="Remote ECR tag. Defaults to config ecr_image_tag.")
    push_parser.add_argument(
        "--build",
        action="store_true",
        help="Build the local image before pushing it.",
    )

    submit_parser = subparsers.add_parser("aws-submit", help="Upload config and submit an AWS Batch job.")
    submit_parser.add_argument("--aws-config", required=True, help="Path to the AWS Batch TOML config.")
    submit_parser.add_argument("--config", required=True, help="Path to the sequence config TOML.")
    submit_parser.add_argument("--compute-level", type=int, required=True, help="Compute level from 1 to 10.")
    submit_parser.add_argument("--seed", type=int, required=True, help="Base random seed.")
    submit_parser.add_argument("--allow-dirty", action="store_true", help="Allow submission from a dirty worktree.")
    submit_parser.add_argument(
        "--job-vcpus",
        type=int,
        default=None,
        help="Override the Batch container vCPU reservation for this submission.",
    )
    submit_parser.add_argument(
        "--job-memory-mib",
        type=int,
        default=None,
        help="Override the Batch container memory reservation in MiB for this submission.",
    )

    campaign_parser = subparsers.add_parser(
        "aws-submit-campaign",
        help="Submit multiple AWS Batch jobs for the same config across several seeds.",
    )
    campaign_parser.add_argument("--aws-config", required=True, help="Path to the AWS Batch TOML config.")
    campaign_parser.add_argument("--config", required=True, help="Path to the sequence config TOML.")
    campaign_parser.add_argument("--compute-level", type=int, required=True, help="Compute level from 1 to 10.")
    campaign_parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated integer seeds, for example 42,314,2718",
    )
    campaign_parser.add_argument("--allow-dirty", action="store_true", help="Allow submission from a dirty worktree.")
    campaign_parser.add_argument(
        "--job-vcpus",
        type=int,
        default=None,
        help="Override the Batch container vCPU reservation for each submission.",
    )
    campaign_parser.add_argument(
        "--job-memory-mib",
        type=int,
        default=None,
        help="Override the Batch container memory reservation in MiB for each submission.",
    )

    status_parser = subparsers.add_parser("aws-status", help="Inspect an AWS Batch job.")
    status_parser.add_argument("--aws-config", required=True, help="Path to the AWS Batch TOML config.")
    status_parser.add_argument("--job-id", required=True, help="AWS Batch job ID.")

    fetch_parser = subparsers.add_parser("aws-fetch", help="Download a completed AWS Batch run from S3.")
    fetch_parser.add_argument("--aws-config", required=True, help="Path to the AWS Batch TOML config.")
    fetch_parser.add_argument("--job-id", required=True, help="AWS Batch job ID.")
    fetch_parser.add_argument(
        "--output-dir",
        default=target.default_fetch_results_dir,
        help="Directory that will receive the downloaded run directory.",
    )

    sync_parser = subparsers.add_parser("aws-sync-ledger", help="Refresh the local CSV ledger from submission records.")
    sync_parser.add_argument("--aws-config", required=True, help="Path to the AWS Batch TOML config.")

    watch_parser = subparsers.add_parser(
        "aws-watch-ledger",
        help="Poll AWS Batch and keep the active/done ledgers up to date until stopped.",
    )
    watch_parser.add_argument("--aws-config", required=True, help="Path to the AWS Batch TOML config.")
    watch_parser.add_argument("--interval-seconds", type=int, default=60, help="Polling interval in seconds.")
    watch_parser.add_argument(
        "--until-no-active",
        action="store_true",
        help="Exit automatically once the active ledger is empty.",
    )
    return parser.parse_args()


def docker_build(target: CloudCliTarget, args: argparse.Namespace) -> None:
    repo_root = require_repo_root(target.repo_root_hint)
    command = [
        "docker",
        "build",
        "--platform",
        args.platform,
        "-f",
        target.dockerfile_path,
        "-t",
        args.image_tag,
    ]
    if args.no_cache:
        command.append("--no-cache")
    command.append(".")
    run_checked(command, cwd=repo_root)


def docker_run(target: CloudCliTarget, args: argparse.Namespace) -> None:
    repo_root = require_repo_root(target.repo_root_hint)
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    host_results_dir = Path(args.host_results_dir).resolve()
    host_results_dir.mkdir(parents=True, exist_ok=True)
    git_metadata = get_git_metadata(repo_root)
    submitted_at_utc = datetime.now(UTC).isoformat()
    container_config_path = CONTAINER_CONFIG_DIR / config_path.name
    try:
        config_label = str(config_path.relative_to(repo_root))
    except ValueError:
        config_label = str(config_path)

    env = os.environ.copy()
    env.update(
        {
            "JOI_SOURCE_GIT_COMMIT": str(git_metadata["git_commit"] or ""),
            "JOI_SOURCE_GIT_BRANCH": str(git_metadata["git_branch"] or ""),
            "JOI_SOURCE_GIT_IS_DIRTY": "" if git_metadata["git_is_dirty"] is None else str(git_metadata["git_is_dirty"]).lower(),
            "JOI_CONFIG_LABEL": config_label,
            "JOI_IMAGE_REF": args.image_tag,
            "JOI_SUBMITTED_AT_UTC": submitted_at_utc,
        }
    )

    command = [
        "docker",
        "run",
        "--rm",
        "-e",
        "JOI_SOURCE_GIT_COMMIT",
        "-e",
        "JOI_SOURCE_GIT_BRANCH",
        "-e",
        "JOI_SOURCE_GIT_IS_DIRTY",
        "-e",
        "JOI_CONFIG_LABEL",
        "-e",
        "JOI_IMAGE_REF",
        "-e",
        "JOI_SUBMITTED_AT_UTC",
        "-v",
        f"{config_path.parent}:{CONTAINER_CONFIG_DIR}:ro",
        "-v",
        f"{host_results_dir}:{CONTAINER_RESULTS_DIR}",
        args.image_tag,
        "--config",
        str(container_config_path),
        "--compute-level",
        str(args.compute_level),
        "--results-dir",
        str(CONTAINER_RESULTS_DIR),
        "--seed",
        str(args.seed),
        "--execution-mode",
        args.execution_mode,
    ]
    run_checked(command, cwd=repo_root, env=env)


def stack_outputs(settings: AwsBatchSettings, repo_root: Path) -> dict[str, str]:
    stacks = aws_cli_json(
        ["cloudformation", "describe-stacks", "--stack-name", settings.stack_name],
        region=settings.region,
        cwd=repo_root,
    )["Stacks"]
    if not stacks:
        raise RuntimeError(f"CloudFormation stack {settings.stack_name} not found.")
    outputs = stacks[0].get("Outputs", [])
    return {item["OutputKey"]: item["OutputValue"] for item in outputs}


def deploy_stack(settings: AwsBatchSettings, repo_root: Path, image_uri: str) -> None:
    if settings.vpc_id is None or not settings.subnet_ids:
        raise RuntimeError("AWS config is missing vpc_id or subnet_ids for stack deployment.")

    template_path = repo_root / settings.cloudformation_template
    command = [
        "aws",
        "--region",
        settings.region,
        "cloudformation",
        "deploy",
        "--stack-name",
        settings.stack_name,
        "--template-file",
        str(template_path),
        "--capabilities",
        "CAPABILITY_IAM",
        "--parameter-overrides",
        f"ProjectName={settings.project_name}",
        f"VpcId={settings.vpc_id}",
        f"SubnetIds={','.join(settings.subnet_ids)}",
        f"ImageUri={image_uri}",
        f"JobVcpus={settings.job_vcpus}",
        f"JobMemoryMiB={settings.job_memory_mib}",
        f"MaxvCpus={settings.max_vcpus}",
        f"InstanceTypes={','.join(settings.instance_types)}",
    ]
    if settings.artifacts_bucket:
        command.append(f"ArtifactsBucketName={settings.artifacts_bucket}")
    if settings.associate_app_registry and settings.app_registry_application:
        command.append(f"AppRegistryApplication={settings.app_registry_application}")
    tags = [f"Project={settings.project_name}"]
    if settings.app_registry_application_arn:
        tags.append(f"awsApplication={settings.app_registry_application_arn}")
    command.extend(["--tags", *tags])
    run_checked(command, cwd=repo_root)


def aws_deploy_stack(target: CloudCliTarget, args: argparse.Namespace) -> None:
    repo_root = require_repo_root(target.repo_root_hint)
    settings = load_aws_batch_settings(args.aws_config)
    deploy_stack(settings, repo_root, args.image_uri)
    outputs = stack_outputs(settings, repo_root)
    print(json.dumps({"stack_name": settings.stack_name, "outputs": outputs}, indent=2))


def ecr_repository_uri(settings: AwsBatchSettings, repo_root: Path) -> str:
    outputs = stack_outputs(settings, repo_root)
    try:
        return outputs["EcrRepositoryUri"]
    except KeyError as exc:
        raise RuntimeError("Stack outputs did not include EcrRepositoryUri.") from exc


def active_job_definition_arn(settings: AwsBatchSettings, repo_root: Path) -> str:
    outputs = stack_outputs(settings, repo_root)
    try:
        return outputs["BatchJobDefinitionName"]
    except KeyError as exc:
        raise RuntimeError("Stack outputs did not include BatchJobDefinitionName.") from exc


def active_job_definition_description(settings: AwsBatchSettings, repo_root: Path) -> dict:
    job_definition_arn = active_job_definition_arn(settings, repo_root)
    response = aws_cli_json(
        ["batch", "describe-job-definitions", "--job-definitions", job_definition_arn],
        region=settings.region,
        cwd=repo_root,
    )
    job_definitions = response.get("jobDefinitions", [])
    if not job_definitions:
        raise RuntimeError(f"No AWS Batch job definition found for {job_definition_arn}.")
    return job_definitions[0]


def aws_push_image(target: CloudCliTarget, args: argparse.Namespace) -> None:
    repo_root = require_repo_root(target.repo_root_hint)
    settings = load_aws_batch_settings(args.aws_config)

    if args.build:
        build_args = argparse.Namespace(image_tag=args.image_tag, platform="linux/amd64", no_cache=False)
        docker_build(target, build_args)

    repo_uri = ecr_repository_uri(settings, repo_root)
    remote_tag = args.remote_tag or settings.ecr_image_tag
    remote_image = f"{repo_uri}:{remote_tag}"
    login_password = subprocess.run(
        ["aws", "--region", settings.region, "ecr", "get-login-password"],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", repo_uri],
        check=True,
        cwd=repo_root,
        input=login_password.stdout,
        text=True,
    )
    run_checked(["docker", "tag", args.image_tag, remote_image], cwd=repo_root)
    run_checked(["docker", "push", remote_image], cwd=repo_root)
    deploy_stack(settings, repo_root, remote_image)
    print(json.dumps({"repo_uri": repo_uri, "remote_image": remote_image}, indent=2))


def build_container_overrides(
    *,
    config_path: Path,
    config_s3_uri: str,
    compute_level: int,
    image_ref: str,
    job_vcpus: int,
    job_memory_mib: int,
    seed: int,
    s3_output_prefix: str,
    repo_root: Path,
) -> dict:
    git_metadata = get_git_metadata(repo_root)
    try:
        config_label = str(config_path.resolve().relative_to(repo_root))
    except ValueError:
        config_label = str(config_path.resolve())

    environment = [
        {"name": "JOI_SOURCE_GIT_COMMIT", "value": str(git_metadata["git_commit"] or "")},
        {"name": "JOI_SOURCE_GIT_BRANCH", "value": str(git_metadata["git_branch"] or "")},
        {
            "name": "JOI_SOURCE_GIT_IS_DIRTY",
            "value": "" if git_metadata["git_is_dirty"] is None else str(git_metadata["git_is_dirty"]).lower(),
        },
        {"name": "JOI_CONFIG_LABEL", "value": config_label},
        {"name": "JOI_IMAGE_REF", "value": image_ref},
        {"name": "JOI_SUBMITTED_AT_UTC", "value": datetime.now(UTC).isoformat()},
        {"name": "JOI_S3_OUTPUT_PREFIX", "value": s3_output_prefix},
    ]
    return {
        "vcpus": job_vcpus,
        "memory": job_memory_mib,
        "command": [
            "--config-s3-uri",
            config_s3_uri,
            "--compute-level",
            str(compute_level),
            "--results-dir",
            "/tmp/joi-batch-results",
            "--seed",
            str(seed),
            "--execution-mode",
            "aws-batch",
        ],
        "environment": environment,
    }


def aws_submit(target: CloudCliTarget, args: argparse.Namespace) -> None:
    repo_root = require_repo_root(target.repo_root_hint)
    settings = load_aws_batch_settings(args.aws_config)
    ensure_clean_worktree(repo_root, allow_dirty=args.allow_dirty)
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    job_vcpus = args.job_vcpus or settings.job_vcpus
    job_memory_mib = args.job_memory_mib or settings.job_memory_mib
    job_definition = active_job_definition_description(settings, repo_root)
    submission = submit_job(
        settings=settings,
        repo_root=repo_root,
        config_path=config_path,
        compute_level=args.compute_level,
        seed=args.seed,
        job_vcpus=job_vcpus,
        job_memory_mib=job_memory_mib,
        job_definition=job_definition,
    )
    print(json.dumps(submission, indent=2))


def parse_seed_list(raw: str) -> list[int]:
    seeds = [item.strip() for item in raw.split(",")]
    parsed = [int(item) for item in seeds if item]
    if not parsed:
        raise ValueError("At least one seed must be provided.")
    return parsed


def submit_job(
    *,
    settings: AwsBatchSettings,
    repo_root: Path,
    config_path: Path,
    compute_level: int,
    seed: int,
    job_vcpus: int,
    job_memory_mib: int,
    job_definition: dict,
) -> dict[str, str | int]:
    active_job_definition = job_definition["jobDefinitionArn"]
    image_ref = job_definition["containerProperties"]["image"]
    job_name = build_job_name(settings.job_name_prefix, config_path.stem, compute_level, seed)
    submission_prefix = build_submission_prefix(settings, job_name=job_name)
    config_s3_uri = f"{submission_prefix}/input/{config_path.name}"
    aws_cli_run(["s3", "cp", str(config_path), config_s3_uri], region=settings.region, cwd=repo_root)
    overrides = build_container_overrides(
        config_path=config_path,
        config_s3_uri=config_s3_uri,
        compute_level=compute_level,
        image_ref=image_ref,
        job_vcpus=job_vcpus,
        job_memory_mib=job_memory_mib,
        seed=seed,
        s3_output_prefix=f"{submission_prefix}/output",
        repo_root=repo_root,
    )
    response = aws_cli_json(
        [
            "batch",
            "submit-job",
            "--job-name",
            job_name,
            "--job-queue",
            settings.batch_job_queue,
            "--job-definition",
            active_job_definition,
            "--container-overrides",
            json.dumps(overrides),
        ],
        region=settings.region,
        cwd=repo_root,
    )
    job_id = response["jobId"]
    record = {
        "submitted_at_utc": datetime.now(UTC).isoformat(),
        "job_name": job_name,
        "job_id": job_id,
        "config_path": str(config_path),
        "config_s3_uri": config_s3_uri,
        "s3_output_prefix": f"{submission_prefix}/output",
        "compute_level": compute_level,
        "seed": seed,
        "image_ref": image_ref,
        "job_definition": active_job_definition,
        "job_vcpus": job_vcpus,
        "job_memory_mib": job_memory_mib,
        "settings": settings.to_dict(),
    }
    record_path = write_submission_record(
        repo_root=repo_root,
        submission_dir=settings.submission_dir,
        job_id=job_id,
        payload=record,
    )
    upsert_job_ledger_row(
        repo_root=repo_root,
        active_ledger_path=settings.active_job_ledger_path,
        completed_ledger_path=settings.completed_job_ledger_path,
        row={
            "job_id": job_id,
            "job_name": job_name,
            "trajectory": config_path.stem,
            "status": "SUBMITTED",
            "submitted_at_utc": record["submitted_at_utc"],
            "started_at_utc": "",
            "stopped_at_utc": "",
            "last_checked_utc": record["submitted_at_utc"],
            "compute_level": str(compute_level),
            "seed": str(seed),
            "job_vcpus": str(job_vcpus),
            "job_memory_mib": str(job_memory_mib),
            "image_ref": image_ref,
            "job_definition": active_job_definition,
            "config_path": str(config_path),
            "config_s3_uri": config_s3_uri,
            "s3_output_prefix": record["s3_output_prefix"],
            "log_stream": "",
            "fetched_to": "",
        },
    )
    return {"job_id": job_id, "job_name": job_name, "record": str(record_path), **record}


def aws_submit_campaign(target: CloudCliTarget, args: argparse.Namespace) -> None:
    repo_root = require_repo_root(target.repo_root_hint)
    settings = load_aws_batch_settings(args.aws_config)
    ensure_clean_worktree(repo_root, allow_dirty=args.allow_dirty)
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    seeds = parse_seed_list(args.seeds)
    job_vcpus = args.job_vcpus or settings.job_vcpus
    job_memory_mib = args.job_memory_mib or settings.job_memory_mib
    job_definition = active_job_definition_description(settings, repo_root)
    submissions = [
        submit_job(
            settings=settings,
            repo_root=repo_root,
            config_path=config_path,
            compute_level=args.compute_level,
            seed=seed,
            job_vcpus=job_vcpus,
            job_memory_mib=job_memory_mib,
            job_definition=job_definition,
        )
        for seed in seeds
    ]
    print(json.dumps({"submitted_jobs": submissions}, indent=2))


def get_job_description(*, settings: AwsBatchSettings, job_id: str, repo_root: Path) -> dict:
    jobs = aws_cli_json(["batch", "describe-jobs", "--jobs", job_id], region=settings.region, cwd=repo_root)["jobs"]
    if not jobs:
        raise RuntimeError(f"No AWS Batch job found for ID {job_id}.")
    return jobs[0]


def aws_status(target: CloudCliTarget, args: argparse.Namespace) -> None:
    repo_root = require_repo_root(target.repo_root_hint)
    settings = load_aws_batch_settings(args.aws_config)
    job = get_job_description(settings=settings, job_id=args.job_id, repo_root=repo_root)
    record = read_submission_record(repo_root=repo_root, submission_dir=settings.submission_dir, job_id=args.job_id)
    payload = {
        "job_id": job["jobId"],
        "job_name": job["jobName"],
        "status": job["status"],
        "status_reason": job.get("statusReason"),
        "created_at_epoch_ms": job.get("createdAt"),
        "started_at_epoch_ms": job.get("startedAt"),
        "stopped_at_epoch_ms": job.get("stoppedAt"),
        "job_queue": job.get("jobQueue"),
        "job_definition": job.get("jobDefinition"),
        "log_stream": extract_log_stream(job),
        "s3_output_prefix": record.get("s3_output_prefix") if record else None,
        "config_s3_uri": record.get("config_s3_uri") if record else None,
        "record_found": record is not None,
    }
    checked_at_utc = datetime.now(UTC).isoformat()
    upsert_job_ledger_row(
        repo_root=repo_root,
        active_ledger_path=settings.active_job_ledger_path,
        completed_ledger_path=settings.completed_job_ledger_path,
        row={
            "job_id": job["jobId"],
            "job_name": job["jobName"],
            "status": job["status"],
            "submitted_at_utc": record.get("submitted_at_utc", "") if record else "",
            "started_at_utc": epoch_ms_to_iso8601(job.get("startedAt")),
            "stopped_at_utc": epoch_ms_to_iso8601(job.get("stoppedAt")),
            "last_checked_utc": checked_at_utc,
            "compute_level": str(record.get("compute_level", "")) if record else "",
            "seed": str(record.get("seed", "")) if record else "",
            "job_vcpus": str(record.get("job_vcpus", "")) if record else "",
            "job_memory_mib": str(record.get("job_memory_mib", "")) if record else "",
            "image_ref": str(record.get("image_ref", "")) if record else "",
            "job_definition": str(job.get("jobDefinition", "")),
            "config_path": str(record.get("config_path", "")) if record else "",
            "config_s3_uri": str(record.get("config_s3_uri", "")) if record else "",
            "s3_output_prefix": str(record.get("s3_output_prefix", "")) if record else "",
            "log_stream": str(payload["log_stream"] or ""),
            "fetched_to": "",
        },
    )
    print(json.dumps(payload, indent=2))


def aws_fetch(target: CloudCliTarget, args: argparse.Namespace) -> None:
    repo_root = require_repo_root(target.repo_root_hint)
    settings = load_aws_batch_settings(args.aws_config)
    record = read_submission_record(repo_root=repo_root, submission_dir=settings.submission_dir, job_id=args.job_id)
    if record is None:
        raise RuntimeError("No local submission record found for this job ID.")
    job = get_job_description(settings=settings, job_id=args.job_id, repo_root=repo_root)
    if job["status"] != "SUCCEEDED":
        raise RuntimeError(f"Job {args.job_id} is not complete. Current status: {job['status']}")
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / args.job_id
    destination.mkdir(parents=True, exist_ok=True)
    aws_cli_run(["s3", "cp", "--recursive", record["s3_output_prefix"], str(destination)], region=settings.region, cwd=repo_root)
    upsert_job_ledger_row(
        repo_root=repo_root,
        active_ledger_path=settings.active_job_ledger_path,
        completed_ledger_path=settings.completed_job_ledger_path,
        row={
            "job_id": args.job_id,
            "job_name": job["jobName"],
            "status": job["status"],
            "submitted_at_utc": str(record.get("submitted_at_utc", "")),
            "started_at_utc": epoch_ms_to_iso8601(job.get("startedAt")),
            "stopped_at_utc": epoch_ms_to_iso8601(job.get("stoppedAt")),
            "last_checked_utc": datetime.now(UTC).isoformat(),
            "compute_level": str(record.get("compute_level", "")),
            "seed": str(record.get("seed", "")),
            "job_vcpus": str(record.get("job_vcpus", "")),
            "job_memory_mib": str(record.get("job_memory_mib", "")),
            "image_ref": str(record.get("image_ref", "")),
            "job_definition": str(job.get("jobDefinition", "")),
            "config_path": str(record.get("config_path", "")),
            "config_s3_uri": str(record.get("config_s3_uri", "")),
            "s3_output_prefix": str(record.get("s3_output_prefix", "")),
            "log_stream": str(extract_log_stream(job) or ""),
            "fetched_to": str(destination),
        },
    )
    print(json.dumps({"job_id": args.job_id, "source": record["s3_output_prefix"], "destination": str(destination)}, indent=2))


def aws_sync_ledger(target: CloudCliTarget, args: argparse.Namespace) -> None:
    repo_root = require_repo_root(target.repo_root_hint)
    settings = load_aws_batch_settings(args.aws_config)
    submission_dir = ensure_submission_dir(repo_root, settings.submission_dir)
    count = 0
    for record_path in sorted(submission_dir.glob("*.json")):
        record = json.loads(record_path.read_text())
        job = get_job_description(settings=settings, job_id=record["job_id"], repo_root=repo_root)
        upsert_job_ledger_row(
            repo_root=repo_root,
            active_ledger_path=settings.active_job_ledger_path,
            completed_ledger_path=settings.completed_job_ledger_path,
            row={
                "job_id": record["job_id"],
                "job_name": record.get("job_name", job["jobName"]),
                "trajectory": Path(str(record.get("config_path", ""))).stem,
                "status": job["status"],
                "submitted_at_utc": str(record.get("submitted_at_utc", "")),
                "started_at_utc": epoch_ms_to_iso8601(job.get("startedAt")),
                "stopped_at_utc": epoch_ms_to_iso8601(job.get("stoppedAt")),
                "last_checked_utc": datetime.now(UTC).isoformat(),
                "compute_level": str(record.get("compute_level", "")),
                "seed": str(record.get("seed", "")),
                "job_vcpus": str(record.get("job_vcpus", "")),
                "job_memory_mib": str(record.get("job_memory_mib", "")),
                "image_ref": str(record.get("image_ref", "")),
                "job_definition": str(job.get("jobDefinition", record.get("job_definition", ""))),
                "config_path": str(record.get("config_path", "")),
                "config_s3_uri": str(record.get("config_s3_uri", "")),
                "s3_output_prefix": str(record.get("s3_output_prefix", "")),
                "log_stream": str(extract_log_stream(job) or ""),
                "fetched_to": "",
            },
        )
        count += 1
    print(
        json.dumps(
            {
                "active_job_ledger": str(repo_root / settings.active_job_ledger_path),
                "completed_job_ledger": str(repo_root / settings.completed_job_ledger_path),
                "synced_jobs": count,
            },
            indent=2,
        )
    )


def aws_watch_ledger(target: CloudCliTarget, args: argparse.Namespace) -> None:
    repo_root = require_repo_root(target.repo_root_hint)
    settings = load_aws_batch_settings(args.aws_config)
    interval_seconds = max(5, int(args.interval_seconds))

    while True:
        aws_sync_ledger(target, args)
        active_rows = read_job_ledger_rows(repo_root, settings.active_job_ledger_path)
        summary = {
            "checked_at_utc": datetime.now(UTC).isoformat(),
            "active_jobs": len(active_rows),
            "active_job_ids": [row.get("job_id", "") for row in active_rows],
            "active_job_ledger": str(repo_root / settings.active_job_ledger_path),
            "completed_job_ledger": str(repo_root / settings.completed_job_ledger_path),
        }
        print(json.dumps(summary, indent=2))
        if args.until_no_active and not active_rows:
            return
        time.sleep(interval_seconds)


def main_for_target(target: CloudCliTarget) -> None:
    args = parse_args(target)
    if args.command == "docker-build":
        docker_build(target, args)
        return
    if args.command == "docker-run":
        docker_run(target, args)
        return
    if args.command == "aws-deploy-stack":
        aws_deploy_stack(target, args)
        return
    if args.command == "aws-push-image":
        aws_push_image(target, args)
        return
    if args.command == "aws-submit":
        aws_submit(target, args)
        return
    if args.command == "aws-submit-campaign":
        aws_submit_campaign(target, args)
        return
    if args.command == "aws-status":
        aws_status(target, args)
        return
    if args.command == "aws-fetch":
        aws_fetch(target, args)
        return
    if args.command == "aws-sync-ledger":
        aws_sync_ledger(target, args)
        return
    if args.command == "aws-watch-ledger":
        aws_watch_ledger(target, args)
        return
    raise RuntimeError(f"Unhandled command: {args.command}")
