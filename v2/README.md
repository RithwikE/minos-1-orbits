# v2 Architecture

`v2/` is the current codebase for trajectory search, result archiving, and cloud execution. It replaces `v1/` for active work. Right now the only implemented mission package is `v2/01_joi`, which searches Earth-to-Jupiter transfer sequences up to Jupiter arrival. `v1/` should be treated as historical reference unless you explicitly need an old result for comparison.

## Current Status

- Active optimization codepath: `v2/01_joi`
- Supported trajectory families in code today:
  - `02 Delta-V EGA` via `v2/01_joi/configs/dvega*.toml`
  - `04 VEEGA` via `v2/01_joi/configs/veega*.toml`
- Shared infrastructure already lives in `v2/shared` so later work such as `02_eoi` can reuse the same search, runtime, and cloud plumbing.

## Directory Layout

```text
v2/
├── 01_joi/         Earth-to-Jupiter optimization package and mission-specific CLI entrypoints
├── good_results/   Curated fetched or copied result archives plus trade-study helpers
├── scratch/        One-off analysis scripts that operate on saved artifacts
└── shared/         Reusable search, container, runtime, and AWS Batch infrastructure
```

### `01_joi/`

`01_joi/` is the mission-specific layer. It owns the search problem definition, config parsing, candidate reconstruction, and the thin entrypoints that bind the JOI package to the shared infrastructure.

Important files:

- `run_search.py`
  Runs a local search from a TOML config and writes a timestamped run directory.
- `run_container_job.py`
  Container entrypoint used by Docker and AWS Batch jobs.
- `cloud_cli.py`
  JOI-specific wrapper around the shared Docker and AWS Batch CLI.
- `visualize_candidate.py`
  Plots saved `candidate_details/*.json` archives without rerunning optimization.
- `Dockerfile`
  Runtime image for local container tests and Batch jobs.
- `configs/*.toml`
  Mission configs and campaign configs.
- `incumbents/*.json`
  Seed candidates used to warm-start bounded recovery campaigns.
- `joi/problem.py`
  Builds the `pykep.trajopt.mga_1dsm` UDP and the `pygmo` problem.
- `joi/config.py`
  Parses TOML configs, validates trajectory bounds, and applies search-profile overrides.
- `joi/optimize.py`
  JOI-specific mission filtering, incumbent loading, and result saving.
- `joi/postprocess.py`
  Reconstructs saved candidates into events, legs, dense samples, and derived metrics.

### `shared/`

`shared/` exists so future mission packages do not duplicate generic execution infrastructure.

Important files:

- `search.py`
  Defines compute profiles and the generic three-phase single-objective search workflow:
  1. multi-island `sade` exploration
  2. seeded `sade` refinement
  3. `mbh(compass_search)` polishing
- `cloud_cli.py`
  Shared implementation of `docker-build`, `docker-run`, `aws-deploy-stack`, `aws-push-image`, `aws-submit`, `aws-submit-campaign`, `aws-status`, `aws-fetch`, `aws-sync-ledger`, and `aws-watch-ledger`.
- `container_job.py`
  Shared container runner, metadata recording, progress heartbeat writing, and optional S3 upload.
- `aws_batch.py`
  AWS Batch settings parsing, submission record helpers, and active/done ledger maintenance.
- `runtime.py`
  Git metadata, config hashing, and repo-root helpers.
- `cloud/aws_batch_stack.yaml`
  CloudFormation stack for Batch, IAM, S3, ECR, and related resources.

### `good_results/`

`good_results/` is not the live output location for fresh searches. It is a curated archive area for runs worth keeping around for trade studies and comparison.

Current contents:

- per-seed fetched/copied run folders under `good_results/veega/veega_batches/` and `good_results/dvega/dvega_batches/`
- combined trade-study CSVs such as `good_results/veega/veega_trade_study_candidates.csv`
- `build_trade_study_csv.py` for flattening the per-seed `top_candidate_summaries.json` files into a sortable CSV

### `scratch/`

`scratch/` is for ad hoc analysis that should not be treated as stable architecture. Scripts here are expected to consume saved archives rather than become core dependencies.

## Local Search Workflow

The main local entrypoint is:

```bash
conda run -n minos-orbits python v2/01_joi/run_search.py \
  --config v2/01_joi/configs/veega.toml \
  --compute-level 2 \
  --seed 42
```

This does the following:

1. Loads a sequence config from `configs/*.toml`
2. Builds the `pykep` multiple-gravity-assist problem
3. Builds a compute profile from the requested level, optionally overridden by the config
4. Runs the three-phase `pygmo` search
5. Applies mission filters, currently including hard checks on total TOF and launch `C3`
6. Saves a run archive containing raw candidates, reconstructed top candidates, summaries, and progress logs

## Config Model

Each mission TOML defines:

- sequence name and label
- body sequence
- departure window
- launch-energy cap through `max_c3_kms2`
- mission duration cap through `max_total_tof_days`
- per-leg TOF bounds
- archive size and dense-sampling density
- optional search overrides
- optional seed candidate files

Notable configs in current use:

- `v2/01_joi/configs/veega.toml`
  Baseline broad VEEGA search.
- `v2/01_joi/configs/veega_campaign.toml`
  Current bounded VEEGA recovery config. This is the preferred config when trying to beat the prior VEEGA result under `C3 <= 15`.
- `v2/01_joi/configs/veega_strong.toml`
  Historical oversized VEEGA config. Keep as reference; do not use it as the default unattended rerun path.
- `v2/01_joi/configs/dvega.toml`
  Baseline DVEGA search.
- `v2/01_joi/configs/dvega_campaign.toml`
  Current bounded DVEGA campaign config seeded from the best feasible known `v2` incumbent.

## Search Architecture

The search stack is intentionally split between shared and mission-specific logic.

Shared responsibilities:

- compute-profile scaling
- candidate collection from populations and archipelagos
- candidate deduplication and ranking
- progress-event emission
- generic result writing

JOI-specific responsibilities:

- `mga_1dsm` problem construction
- decoding the decision vector and mission metrics correctly
- mission feasibility filters such as `C3` and total TOF
- reconstructing events, flybys, DSMs, and dense heliocentric samples for saved candidates

The current search is single-objective only and minimizes total DSM magnitude. Secondary metrics such as TOF, `C3`, and arrival `V_inf` are recovered from the saved candidate archive rather than optimized directly.

## Saved Run Format

Every run folder under `results/`, `docker-results/`, `fetched-results/`, or curated locations such as `good_results/...` follows the same basic structure:

```text
<run_dir>/
├── all_candidates.jsonl
├── candidate_details/
│   ├── rank_001.json
│   ├── rank_002.json
│   └── ...
├── compute_profile.json
├── config.json
├── progress_events.json
├── run_summary.json
├── top_candidates.jsonl
└── top_candidate_summaries.json
```

When the run came from Docker or AWS Batch, `cloud_job.json` is also written.

### What each artifact means

- `config.json`
  The resolved config payload used for the run.
- `compute_profile.json`
  The numeric search budget after level scaling and overrides.
- `run_summary.json`
  Runtime, evaluation counts, seed, archive sizes, and git metadata.
- `progress_events.json`
  Structured progress checkpoints from the search phases.
- `all_candidates.jsonl`
  Deduplicated feasible and infeasible archived candidates from the search process.
- `top_candidates.jsonl`
  Ranked shortlist used for detailed reconstruction.
- `top_candidate_summaries.json`
  Flat summary table for the saved shortlist.
- `candidate_details/rank_###.json`
  Fully reconstructed candidate archive with:
  - decision vector and parsed fields
  - mission summary metrics
  - `events` for departure, DSMs, flybys, and arrival
  - `legs` with segment-level details
  - `dense_samples` for heliocentric position, speed, sun distance, and osculating elements
- `cloud_job.json`
  Execution metadata including config hash, git state, image ref, execution mode, Batch job id, and upload location

## Candidate Analysis Workflow

The code is set up so most analysis can happen from saved artifacts alone.

Examples:

- Plot a candidate:

```bash
conda run -n minos-orbits python v2/01_joi/visualize_candidate.py \
  --candidate-json v2/good_results/veega/veega_batches/seed42/candidate_details/rank_001.json
```

- Build a cross-seed trade-study CSV:

```bash
conda run -n minos-orbits python v2/good_results/build_trade_study_csv.py \
  --batch-dir v2/good_results/veega/veega_batches \
  --output v2/good_results/veega/veega_trade_study_candidates.csv
```

Ad hoc analyses such as `v2/scratch/sun_distance.py` should follow the same pattern: operate on saved candidate archives instead of requiring a rerun.

## Docker and Cloud Workflow

The JOI mission package exposes the shared cloud tooling through `v2/01_joi/cloud_cli.py`.

Common commands:

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
conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-submit-campaign \
  --aws-config v2/01_joi/cloud/aws_batch.toml \
  --config v2/01_joi/configs/veega_campaign.toml \
  --compute-level 3 \
  --seeds 42,314,2718
```

The shared cloud layer is built around these assumptions:

- AWS region: `us-east-1`
- primary lifecycle boundary: one CloudFormation stack
- AppRegistry association is desired for grouping, but stack ownership remains the deletion boundary
- top-level job ledgers live at:
  - `v2/cloud_runs_active.csv`
  - `v2/cloud_runs_done.csv`

Ledger semantics:

- `next_action = wait`
  Job is queued or running.
- `next_action = fetch`
  Job succeeded remotely and should be downloaded.
- `next_action = review_local`
  Results were fetched locally.
- `next_action = inspect_failure`
  Job failed and needs inspection.

The ledgers only update when a cloud CLI command runs. Use `aws-watch-ledger` if you want continuous refresh while the machine is on.

## Current Mission Guidance

- Prefer `v2/01_joi/configs/veega_campaign.toml` plus `aws-submit-campaign` for VEEGA recovery work.
- Prefer `v2/01_joi/configs/dvega_campaign.toml` plus the same multi-seed campaign pattern for DVEGA recovery work.
- Avoid using `veega_strong.toml` as the default path for unattended reruns unless there is an explicit decision to spend the additional budget.
- Do not create cloud resources outside the dedicated CloudFormation stack and AppRegistry grouping context for this project.

## Forward Architecture

The intended next expansion is `v2/02_eoi` for Europa orbit insertion. When `01_joi` and `02_eoi` need the same infrastructure:

- keep mission-specific problem definitions inside their numbered package
- move reusable search, runtime, packaging, and cloud logic into `v2/shared`
- avoid cloning generic code into both mission directories

That split is already visible in the current `01_joi` implementation and should remain the default architecture rule for future `v2` work.
