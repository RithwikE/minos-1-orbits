# Earth To Jupiter Search

This directory is for the Earth-to-Jupiter search only. It ends at Jupiter arrival and reports the quantities needed for later JOI analysis.

## Main Goals

- prioritize physically correct and repeatable results over clever code structure
- keep the implementation straightforward and close to precedent
- support at least the current `02 ΔV-EGA` and `04 VEEGA` sequence families
- save enough data that we do not need to rerun a search just to extract one more metric later
- make the search efficient enough to scale to larger local or cloud runs

## What `v2` Must Do

For each sequence, `v2` should let us define:

- departure window, initially something like `2030-01-01` to `2038-01-01`
- launcher limit, likely through a maximum departure `V_inf` or `C3`
- total time of flight limit, initially about `10 years`
- per-leg time-of-flight bounds when they are justified
- flyby sequence and any altitude safety rules
- optimizer settings and random seeds

Those values belong in simple config files, but the exact bounds still need to be justified from references and testing. The current `v1` bounds are only starting points.

## What To Save

Do not save only the single best solution.

Each run should save:

- the exact search config
- code version and run metadata
- optimizer settings, seeds, wall time, and evaluation counts
- all final individuals from every island / restart, with their decision vectors and fitness values
- the top `N` feasible solutions ranked by the primary objective
- any Pareto front if we run a multi-objective search

For each saved candidate in the final shortlist, also save reconstructed trajectory data so we can inspect it later without rerunning the optimizer:

- launch, flyby, DSM, and arrival events
- state vectors at each event
- leg-by-leg transfer details
- derived mission metrics such as `C3`, `V_inf` at departure and arrival, DSM totals, flyby altitudes, and TOFs
- dense propagated samples versus time so we can inspect distance, speed, heliocentric state, and other post-processed values later

The default should be to save a lot. If storage becomes a problem, we can reduce it later.

## Current Recommendation

Start with a single-objective search that minimizes total DSM magnitude, while enforcing hard mission limits through bounds and filters.

Reason:

- weighted sums across unrelated quantities like DSM, TOF, `C3`, and arrival `V_inf` are hard to justify physically
- a single clear objective makes the optimizer easier to debug and compare
- mission limits can remove obviously unacceptable solutions without distorting the objective

That does not mean we should keep only one answer. The right workflow is:

1. run a strong single-objective search
2. save a wide archive of good feasible candidates
3. compare the top candidates on secondary metrics like TOF, launch `C3`, and Jupiter arrival `V_inf`

If we want true trade studies rather than a single ranking, then we should also run a multi-objective search and save the Pareto front.

## Key Learnings From Official `pykep` / `pygmo` Docs

- `pykep.trajopt.mga_1dsm` supports both single-objective and multi-objective formulations.
- In the official `pykep` docs, the built-in multi-objective form is described as optimizing total `DV` and total time of flight.
- `pykep` offers different time-of-flight encodings. The docs describe direct leg-by-leg encoding as easier for the optimizer, while alpha / eta encodings reduce the amount of prior leg-bound knowledge required.
- `pykep` generally uses SI internally, but `mga_1dsm` has an important interface exception: the constructor `vinf` bounds are provided in `km/s`, while the instantiated chromosome stores that same quantity in `m/s`. This must be handled carefully when reconstructing `C3` and reporting `V_inf`.
- `pykep` trajectory problems provide built-in helpers such as `pretty()` and `plot()`, and the official examples also show `get_eph_function()` for reconstruction and analysis.
- `pygmo` populations expose the full decision vectors and fitness arrays, not just the champion. That means saving top `N` candidates or entire final populations is straightforward.
- `pygmo` also provides multi-objective tools such as `nsga2`, `moead`, and non-dominated sorting utilities, so saving Pareto fronts is supported directly.
- `pygmo` archipelagos are built for parallel optimization and are a natural fit for CPU scaling.

## Practical Implication

For this problem, the likely best path is:

1. use `pykep.mga_1dsm` with a clean single-objective setup first
2. save many feasible candidates, not just the champion
3. reconstruct and store dense trajectory data for the shortlist
4. if needed, add a separate multi-objective mode for `DSM` versus `TOF`

If we later decide that arrival `V_inf` or launch `C3` must be optimized directly as additional objectives, we may need a small custom `pygmo` UDP wrapper around the `pykep` problem rather than relying only on the built-in `multi_objective=True` mode.

## Cloud Direction

This problem looks much more CPU-friendly than GPU-friendly.

The likely order is:

1. make local runs reproducible
2. benchmark parallel archipelago / restart runs on CPU
3. move large sweeps to EC2 or similar CPU instances

GPU or Colab should not be the default assumption for this type of optimization.

## Immediate Next Step

Build the first simple `v2` runner around one sequence, make it save a full candidate archive, and verify that we can reconstruct any saved candidate later without rerunning the search.

## Local Docker Workflow

The first cloud-integration step is a local Docker-packaged runner that uses the same `run_search.py` entrypoint and output structure as native runs.

Build the image from the repo root:

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py docker-build
```

Run a low-compute smoke test and save artifacts on the host:

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py docker-run \
  --config v2/01_joi/configs/veega.toml \
  --compute-level 1 \
  --seed 42 \
  --host-results-dir v2/01_joi/docker-results
```

That container writes the normal run outputs plus `cloud_job.json`, which records the execution mode, config hash, image tag, and source git metadata. The AWS Batch implementation should reuse this packaged runtime rather than inventing a separate code path.

The Dockerized optimizer is operationally equivalent to the native runner, but it is not bitwise reproducible today. In testing with the same config and seed, the Docker run and native run produced the same artifact structure but different objective values and evaluation counts. Treat the current seed as reproducible only within a stable runtime environment.

## AWS Batch Workflow

Provisioning inputs and submission settings live under `cloud/`.

- `../shared/cloud/aws_batch_stack.yaml`: shared CloudFormation stack for S3, ECR, Batch, IAM, and logs
- `../shared/search.py`: shared compute-profile and search-archive logic intended to be reused by later `02_eoi` work
- `cloud/aws_batch.example.toml`: example local CLI config for submit/status/fetch
- `cloud/aws_batch.toml`: current JOI deployment config for this project/account
- `../cloud_runs_active.csv`: top-level visible ledger of jobs that still need attention
- `../cloud_runs_done.csv`: top-level visible ledger of jobs that are fetched or otherwise finished

Current deployment defaults for this project:

- region: `us-east-1`
- default VPC: `vpc-0effb451482334845`
- initial Batch subnets:
  - `subnet-0b9caec9b0b07141a` in `us-east-1a`
  - `subnet-0286fb01046a4ea32` in `us-east-1b`
  - `subnet-0bf07a7ead9677dd6` in `us-east-1c`
- AWS account: `165159921038`

Existing project grouping target:

- AppRegistry application ARN: `arn:aws:servicecatalog:us-east-1:165159921038:/applications/058u2kszdbja1o7gr8tddmfx94`
- integrated AppRegistry resource group ARN: `arn:aws:resource-groups:us-east-1:165159921038:group/AWS_AppRegistry_Application-rithwik-senior-design-project`

Recommended grouping model for this project:

- keep one dedicated CloudFormation stack for the JOI cloud infrastructure
- keep one dedicated S3 bucket and one dedicated ECR repo created by that stack
- associate the stack/resources to the existing AppRegistry application for console grouping
- treat the CloudFormation stack as the primary deletion boundary

Important: AppRegistry grouping is metadata and visibility, not the actual deletion mechanism. Also, the `awsApplication` tag must use the AppRegistry application ARN, not an arbitrary resource-group ARN.

Current deployment config is expected to keep live AppRegistry association enabled when the AWS identity has `servicecatalog:AssociateResource`. If that permission is missing, stack deploys may need to disable association temporarily rather than removing the AppRegistry identifiers entirely.

Current live expectation:

- `cloud/aws_batch.toml` should keep `associate_app_registry = true` when the IAM identity has AppRegistry write permissions
- every deployed stack/resource should carry the `awsApplication` tag and also be associated to the AppRegistry application when possible
- do not create one-off AWS resources outside the stack for this project unless the user explicitly asks for that

Example stack deployment:

```bash
aws cloudformation deploy \
  --stack-name minos-joi-batch \
  --template-file v2/shared/cloud/aws_batch_stack.yaml \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    ProjectName=minos-joi \
    VpcId=vpc-xxxxxxxx \
    SubnetIds=subnet-aaaaaaa,subnet-bbbbbbb \
    ImageUri=<account>.dkr.ecr.<region>.amazonaws.com/minos-joi-runner:latest \
    JobVcpus=8 \
    JobMemoryMiB=16384 \
    InstanceTypes=c7i.2xlarge,c7i.4xlarge,c7i.8xlarge,c7i.12xlarge
```

Once the stack outputs and `cloud/aws_batch.toml` are filled in, the local CLI supports:

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-submit \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --config v2/01_joi/configs/veega.toml \
  --compute-level 3 \
  --seed 42 \
  --job-vcpus 8 \
  --job-memory-mib 16384
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-status \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --job-id <job-id>
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-fetch \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --job-id <job-id> \
  --output-dir v2/01_joi/fetched-results
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-sync-ledger \
  --aws-config v2/01_joi/cloud/aws_batch.toml
```

`aws-submit` uploads the exact config TOML to S3, records a local submission manifest, and submits a Batch job whose container downloads the config, runs the search, writes `cloud_job.json`, and syncs the completed run directory back to S3.
Those commands also maintain the top-level ledgers `../cloud_runs_active.csv` and `../cloud_runs_done.csv` so job IDs, statuses, fetch locations, and the next human action do not need to be tracked manually.

Ledger interpretation:

- `v2/cloud_runs_active.csv` is the file to watch during active cloud work
- `next_action = wait` means the job is still queued or running
- `next_action = fetch` means the job succeeded remotely and should be downloaded with `aws-fetch`
- `next_action = review_local` means the results are already pulled locally
- `next_action = inspect_failure` means the remote job failed and needs investigation

## Cleanup

The intended cleanup boundary is the CloudFormation stack `minos-joi-batch`.

Typical teardown order:

1. delete or keep any fetched local result copies under `v2/01_joi/fetched-results/`
2. empty the S3 artifacts bucket if the stack delete reports non-empty bucket errors
3. delete ECR images if the stack delete reports repository-not-empty errors
4. delete the CloudFormation stack
5. delete the AppRegistry application only if the whole project is finished and you no longer want the grouping metadata

Do not leave ad hoc Batch, ECR, IAM, or S3 resources outside the stack unless there is a deliberate reason.

## Iteration Workflow

For future agent turns, the intended cloud iteration loop is:

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py docker-build
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py docker-run \
  --config v2/01_joi/configs/veega.toml \
  --compute-level 1 \
  --seed 42 \
  --host-results-dir v2/01_joi/docker-results
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-deploy-stack \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --image-uri 165159921038.dkr.ecr.us-east-1.amazonaws.com/minos-joi-runner:latest
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-push-image \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --image-tag joi-search:local
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-submit \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --config v2/01_joi/configs/veega.toml \
  --compute-level 3 \
  --seed 42 \
  --job-vcpus 8 \
  --job-memory-mib 16384 \
  --allow-dirty
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-status \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --job-id <job-id>
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-fetch \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --job-id <job-id> \
  --output-dir v2/01_joi/fetched-results
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-sync-ledger \
  --aws-config v2/01_joi/cloud/aws_batch.toml
```

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-watch-ledger \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --interval-seconds 60 \
  --until-no-active
```

The main human workflow is:

1. check `v2/cloud_runs_active.csv`
2. if `next_action` is `wait`, leave it alone and check again later
3. if `next_action` is `fetch`, run `aws-fetch`
4. after fetch, the row moves to `v2/cloud_runs_done.csv`

Important: the CSV files are not magically live on their own. They update when one of the cloud CLI commands runs. Use `aws-watch-ledger` if you want the files to refresh continuously while your machine is on.

Near-term architecture note:

- `01_joi` should stay mission-specific to Earth-to-Jupiter search work.
- A future `02_eoi` track is expected soon.
- Shared cloud/runtime/optimization infrastructure should eventually move into a separate shared location rather than being duplicated between `01_joi` and `02_eoi`.

The deployment plan for the full Batch path is saved in `docs/aws_batch_deployment_plan.md`.

## VEEGA Recovery Campaign

The generic `compute_level = 10` profile is not guaranteed to exceed the old hand-tuned `v1` VEEGA search budget. The first attempt to fix that by blindly scaling every phase up (`configs/veega_strong.toml`) turned into a long opaque run and is no longer the recommended path.

For regression-recovery or "beat v1" runs under the current `C3 <= 15` mission definition, use:

- `configs/veega_campaign.toml`

That config does two things:

- raises the search budget above the previous successful `v2` level-10 run without exploding into a multi-day monolith
- seeds the search with the legacy VEEGA incumbent stored in `incumbents/veega_v1_rank1.json`
- emits phase progress to CloudWatch and writes a live `heartbeat.json` to the job's S3 output prefix while the job is running

Recommended submit pattern is a multi-seed campaign:

```bash
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-submit-campaign \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --config v2/01_joi/configs/veega_campaign.toml \
  --compute-level 10 \
  --seeds 42,314,2718 \
  --job-vcpus 16 \
  --job-memory-mib 32768 \
  --allow-dirty
```

This gives better search diversity, lower per-job risk, and clearer cost control than one giant job. The top-level ledgers still track each submission separately:

- `v2/cloud_runs_active.csv`
- `v2/cloud_runs_done.csv`

Each running Batch job now emits CloudWatch lines like:

- current phase / step count
- elapsed time
- current best objective
- best C3 / TOF seen so far

The same information is mirrored to `heartbeat.json` under the run's S3 output prefix so progress can be checked even before the final artifacts are uploaded.
