# AWS Batch Deployment Plan for `v2/01_joi`

## Summary

- Use `AWS Batch` with an `EC2`-backed managed compute environment.
- Package the `v2/01_joi` runner as a Docker image, push it to ECR, run one Batch job per search, and store final artifacts in S3.
- First-version user flow: local CLI `submit -> status -> fetch`, with no requirement for the laptop to stay on after submission.
- Keep the existing optimizer interface intact: the cloud job still runs `run_search.py` with `--config`, `--compute-level`, `--seed`, and writes the same result tree, then uploads that tree to S3.
- Local evidence: a `veega` level 1 probe in `minos-orbits` completed in `6.43s` and wrote about `9.4 MB` for 25 saved candidates. Based on the current compute-profile scaling, level 10 is roughly `884x` the level 1 workload. That is an inference, not a guarantee, but it is enough to treat this as a long-running CPU batch job.

## Key Changes

### Execution Model

- Reject Lambda for this workload. AWS documents a hard maximum runtime of `900 seconds (15 minutes)`, which is too small for the current batch shape.
- Reject Fargate as the default. It is viable for some batch jobs, but its task sizing is capped at `16 vCPU / 120 GB`, while ECS-on-EC2 tasks can go much higher. Start with EC2-backed Batch for more headroom and simpler tuning.
- Use `On-Demand` only for the first version.

### Cloud Infrastructure

- Create one AWS Batch job queue and one managed EC2 compute environment.
- Allow compute-optimized x86 instance families, with `c7i` as the primary target and `c7a` / `c6i` as acceptable fallbacks later.
- Initial job definition target: `32 vCPU`, `64 GiB` memory, Linux `x86_64`.
- Create one S3 bucket for run artifacts and one ECR repository for runner images.
- Send stdout and stderr to CloudWatch Logs.

### Packaging and Reproducibility

- Build a Linux container image that installs the native `pykep` / `pygmo` stack and runs the existing `v2/01_joi/run_search.py`.
- Submission flow builds or selects an ECR image, records `git SHA`, `dirty/clean` state, config path, compute level, seed, and submission timestamp, then submits the Batch job.
- Default policy: block submission from a dirty worktree unless the user passes an explicit override. If override is used, save a manifest marking the run as dirty so results are still attributable.

### Job Contract

- Container entrypoint accepts: `config`, `compute_level`, `seed`, `s3_output_prefix`.
- Job writes results to local scratch first, then uploads the full run directory to S3 at the end.
- Add a small run metadata file to the uploaded artifact set containing Batch job ID, image digest or tag, git SHA, dirty flag, hostname, UTC timestamps, and the final S3 prefix.
- Do not redesign the optimizer yet. No checkpoint/restart in v1; if the job fails, rerun from seed.

### Developer and User Interface

- Add one local cloud CLI with three subcommands: `submit`, `status`, `fetch`.
- `submit` uploads or references the config, submits the Batch job, and prints the Batch job ID plus S3 destination.
- `status` reads Batch state and prints the CloudWatch log stream plus the expected S3 prefix.
- `fetch` downloads the completed run directory from S3 to a local output path.

## Public Interfaces

- New CLI surface:
  - `cloud submit --config <path> --compute-level <1-10> --seed <int>`
  - `cloud status --job-id <id>`
  - `cloud fetch --job-id <id> --output-dir <path>`
- New run metadata artifact in each uploaded result set:
  - `cloud_job.json` with Batch job ID, queue, image reference, git SHA, dirty flag, seed, compute level, and S3 prefix.
- No changes to the existing `run_search.py` arguments or result directory structure beyond adding cloud metadata.

## Test Plan

- Local container smoke test: image builds, imports `pykep` and `pygmo`, and runs a level 1 search to `/tmp`.
- Batch smoke test: submit one `dvega` or `veega` level 1 job, verify `SUCCEEDED`, confirm CloudWatch logs and S3 artifact upload.
- Retrieval test: `fetch` reproduces the remote run directory locally and includes `run_summary.json`, candidate archives, and candidate detail files.
- Reproducibility test: submit the same config, level, and seed twice from the same image SHA and confirm identical or near-identical summaries, subject to underlying library determinism.
- Failure-path test: submit a job with an invalid config path or force a container error, confirm `FAILED` status is surfaced and logs remain accessible.
- Scale test: run at least one level 3 cloud job before level 10, then one level 10 benchmark on the initial `32 vCPU / 64 GiB` definition and inspect runtime, CPU saturation, and memory headroom.

## Assumptions and Defaults

- Scope is `v2/01_joi` only.
- AWS is the target cloud.
- First version optimizes for a single-user research workflow, not a multi-user platform.
- Results are needed only after job completion in v1; no live partial-result streaming or mid-run checkpointing.
- The main optimization logic may keep changing, so cloud submissions must be tied to an image digest and source state rather than "whatever is on the server".
- If level 10 runtime or memory is materially worse than the current inference suggests, the next upgrade should be checkpointing plus optional Batch array jobs for seeded sweeps, not Lambda decomposition.

## Sources

- AWS Lambda timeout: <https://docs.aws.amazon.com/lambda/latest/dg/configuration-timeout.html>
- AWS Batch overview: <https://docs.aws.amazon.com/batch/latest/userguide/what-is-batch.html>
- AWS Batch array jobs: <https://docs.aws.amazon.com/batch/latest/userguide/array_jobs.html>
- AWS Batch job timeouts: <https://docs.aws.amazon.com/batch/latest/userguide/job_timeouts.html>
- ECS task sizing for Fargate vs EC2: <https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-cpu-memory-error.html>
- S3 strong consistency: <https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html>
- EC2 C7i instance family reference: <https://aws.amazon.com/ec2/instance-types/c7i/>
