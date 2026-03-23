# Project Instructions

## Python Environment

- Do not use the system Python for this repo.
- For any Python, `pip`, test, or script command, use the Conda environment named `minos-orbits`.
- Prefer `conda run -n minos-orbits <command>` for one-off commands because shell state does not persist between tool calls.
- If a command requires activation semantics, run it through a shell that first activates Conda and then `conda activate minos-orbits`.

## Current Focus

- `v2/01_joi` is the active optimization codepath for current work.
- Treat `v1` as historical reference unless the user explicitly asks for it.

## AWS Context

- Default AWS region for this project is `us-east-1`.
- Current AWS account in use for deployment work is `165159921038`.
- Target default VPC for the first Batch deployment is `vpc-0effb451482334845`.
- Preferred default subnets for the first Batch deployment are:
  - `subnet-0b9caec9b0b07141a` in `us-east-1a`
  - `subnet-0286fb01046a4ea32` in `us-east-1b`
  - `subnet-0bf07a7ead9677dd6` in `us-east-1c`
- The user wants cloud resources grouped so they can be deleted cleanly after the project.
- Existing AWS AppRegistry application:
  - name: `rithwik-senior-design-project`
  - application ARN: `arn:aws:servicecatalog:us-east-1:165159921038:/applications/058u2kszdbja1o7gr8tddmfx94`
  - integrated resource-group ARN: `arn:aws:resource-groups:us-east-1:165159921038:group/AWS_AppRegistry_Application-rithwik-senior-design-project`
- Prefer associating the CloudFormation stack/resources to the AppRegistry application instead of relying on ad hoc resource groups.
- The AppRegistry application is grouping metadata, not the primary deletion mechanism. Primary lifecycle control should remain a dedicated CloudFormation stack plus dedicated S3/ECR resources.
- Do not treat arbitrary resource-group ARNs as the `awsApplication` tag value. For AppRegistry-aware grouping, use the actual application ARN.
- Prefer saving project-specific AWS identifiers in repo docs/config files, not environment variables, unless the value is secret.
- Never create AWS resources for this project without attaching them to the dedicated CloudFormation stack and the AppRegistry application context. Avoid zombie resources.
- The user wants a top-level visible run ledger:
  - active runs: `v2/cloud_runs_active.csv`
  - done runs: `v2/cloud_runs_done.csv`
  - `next_action` is the human-facing column:
    - `wait`: still running or queued
    - `fetch`: succeeded remotely and should be retrieved
    - `review_local`: results have been fetched locally
    - `inspect_failure`: job failed and needs inspection
- Keep those ledgers current via the cloud CLI commands `aws-submit`, `aws-status`, `aws-fetch`, and `aws-sync-ledger`.
- The ledgers do not update by themselves unless a command touches AWS.
- If the user wants a continuously refreshed ledger while their machine is on, use:
  - `conda run -n minos-orbits python v2/01_joi/cloud_cli.py aws-watch-ledger --aws-config v2/01_joi/cloud/aws_batch.toml --interval-seconds 60`
  - add `--until-no-active` to stop automatically when all active jobs are finished
- Before considering the cloud path complete, verify:
  - the CloudFormation stack exists and is the lifecycle boundary
  - the stack/resources carry the `awsApplication` tag for this application
  - Batch/ECR/S3 resources are discoverable through the project grouping
- When cleaning up, delete the CloudFormation stack first, then delete any remaining S3 objects and ECR images only if they block stack deletion.

## Forward Architecture

- `01_joi` is Earth-to-Jupiter specific.
- `02_eoi` for Europa orbit insertion is expected soon.
- When both tracks need the same cloud, optimization, or packaging logic, prefer moving shared code into a third shared location rather than duplicating it under `01_joi` and `02_eoi`.
- Important regression note:
  - `v2/01_joi/configs/veega.toml` is the normal mission config, but it uses the generic compute-profile scaling.
  - `v2/01_joi/configs/veega_strong.toml` is historical; it was intentionally oversized and proved too expensive/opaque for unattended reruns.
  - `v2/01_joi/configs/veega_campaign.toml` is the current bounded regression-recovery config for VEEGA.
  - It seeds the search with `v2/01_joi/incumbents/veega_v1_rank1.json`, emits progress to CloudWatch, and is intended to be run as a multi-seed campaign instead of one huge monolithic job.
  - Prefer `aws-submit-campaign` with several seeds and moderate per-job resource reservations over a single giant `aws-submit` when the goal is to beat the old `v1` VEEGA result under the `C3 <= 15` mission definition.
  - Avoid submitting single VEEGA jobs whose nominal budget is an order of magnitude above the prior successful `v2` level-10 run unless the user explicitly approves the expected time/cost.
  - `v2/01_joi/configs/dvega_campaign.toml` is the bounded DVEGA campaign config.
  - It seeds from `v2/01_joi/incumbents/dvega_v2_rank1.json` because the old `v1` DVEGA winner is not directly usable under `C3 <= 15`.
  - Use the same multi-seed Batch pattern for DVEGA as for VEEGA unless the user asks for a different search strategy.
