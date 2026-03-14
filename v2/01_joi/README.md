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
