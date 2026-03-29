# Key Facts About The Background Work

## What optimization method we used
- The active search codepath is `v2/01_joi`, which builds a `pykep.trajopt.mga_1dsm` Earth-to-Jupiter transfer problem and solves it with `pygmo`.
- The optimizer is single-objective: it minimizes total deep-space maneuver (DSM) magnitude, then compares time of flight, launch `C3`, and Jupiter arrival `V_inf` afterward in the trade study.
- Mission constraints are enforced as hard filters rather than blended into the objective: the archived campaign used `C3 <= 15 km^2/s^2` and `total TOF <= 3652.5 days`.
- The search itself has three phases: multi-island `sade` exploration, seeded `sade` refinement, and `mbh(compass_search)` polishing of the best candidates.
- The curated DVEGA campaign ran at compute level 10 with 80 islands over 10 exploration rounds, 80 seeded refinements, and 16 local-polish candidates; VEEGA used 64 islands over 12 rounds, 96 seeded refinements, and 16 local-polish candidates.

## What the cloud workflow did for us
- The same packaged runner was used for local Docker tests and AWS Batch jobs, so cloud runs reused the exact search/archive codepath instead of a separate script.
- `aws-submit` uploaded the exact mission config to S3, submitted one seed per Batch job, and recorded job metadata for reproducibility.
- Inside the container, the runner emitted heartbeat progress, wrote `cloud_job.json`, and synced the finished run directory back to S3 for later fetch.
- The cloud ledger workflow (`aws-status`, `aws-fetch`, `aws-sync-ledger`, `aws-watch-ledger`) kept active and finished jobs visible without manually tracking job IDs.

## What scale of search we actually ran
- The curated `good_results` campaign contains 6 AWS Batch runs across 2 trajectory families and 3 seeds per family.
- Across those six runs, the search recorded 1,026,119,268 objective evaluations and 17.12 hours of wall-clock runtime.
- The run archives contain 234,994 deduplicated candidate trajectories in `all_candidates.jsonl`, with 234,994 that are both optimizer-feasible and mission-feasible.
- Each run saved the top 100 detailed candidates, giving 600 fully reconstructed candidate files across the campaign.

## What data we saved for each run and candidate
- Every run directory saves the resolved config, compute profile, run summary, structured progress log, full candidate archive, top-candidate shortlist, and cloud metadata.
- Every detailed candidate file stores the decision vector, decoded timing/flyby parameters, mission summary metrics, event timeline, per-leg transfer data, and dense heliocentric samples.
- The archived top-candidate set in `good_results` contains 4,200 event records, 1,800 leg records, and 231,000 dense trajectory samples ready for later analysis without rerunning optimization.
