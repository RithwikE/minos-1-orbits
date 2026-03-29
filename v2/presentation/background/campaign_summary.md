# Search Campaign Summary

## Totals

| Metric | Value |
| --- | --- |
| Runs in curated campaign | 6 |
| Trajectory families compared | 2 |
| Seeds used | 42, 314, 2718 |
| Compute level used | 10 for all curated campaign runs |
| Execution mode | AWS Batch for all curated runs |
| Total wall-clock runtime | 17.12 hours |
| Total objective evaluations | 1,026,119,268 |
| Phase 1 evaluations | 146,529,888 |
| Phase 2 evaluations | 65,898,912 |
| Phase 3 evaluations | 813,690,468 |
| Archived candidate trajectories | 234,994 |
| Fully feasible archived candidates | 234,994 |
| Detailed top-candidate files | 600 |
| Saved event records in detailed files | 4,200 |
| Saved leg records in detailed files | 1,800 |
| Saved dense trajectory samples | 231,000 |

## By Family

| Family | Runs | Seeds | Runtime (h) | Objective evals (M) | Archived candidates | Fully feasible archive | Best DSM (km/s) | Best TOF (yr) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 02 Delta-V EGA | 3 | 42,314,2718 | 4.60 | 410.8 | 118,613 | 118,613 | 1.961 | 5.94 |
| 04 VEEGA | 3 | 42,314,2718 | 12.52 | 615.4 | 116,381 | 116,381 | 0.613 | 7.15 |

## By Run

| Family | Seed | Runtime (h) | Objective evals (M) | Archived candidates | Fully feasible archive | Best DSM (km/s) | Best TOF (yr) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 02 Delta-V EGA | 42 | 1.28 | 117.5 | 39,627 | 39,627 | 1.961 | 5.94 |
| 02 Delta-V EGA | 314 | 1.71 | 151.5 | 39,410 | 39,410 | 1.961 | 5.94 |
| 02 Delta-V EGA | 2718 | 1.61 | 141.7 | 39,576 | 39,576 | 1.961 | 5.94 |
| 04 VEEGA | 42 | 4.23 | 206.4 | 38,837 | 38,837 | 0.613 | 7.15 |
| 04 VEEGA | 314 | 4.23 | 210.8 | 38,718 | 38,718 | 0.701 | 6.81 |
| 04 VEEGA | 2718 | 4.06 | 198.2 | 38,826 | 38,826 | 1.050 | 6.99 |

## Notes

- `Archived candidate trajectories` comes from the per-run `all_candidates.jsonl` counts saved in each run summary.
- `Objective evaluations` means calls into the optimizer's trajectory objective / feasibility evaluation. It is much larger than the archive because most evaluations are intermediate search work, not saved candidate records.
- `Fully feasible archived candidates` counts candidates that are both optimizer-feasible and mission-feasible in the saved JSONL archive.
- The archive is a deduplicated subset of candidates collected from the phase outputs, not a record of every objective evaluation the optimizer performed.
- Runtime is the recorded wall time from `run_summary.json`; it is not multiplied by Batch vCPU allocation.
