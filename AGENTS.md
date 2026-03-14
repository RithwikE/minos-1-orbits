# Project Instructions

## Python Environment

- Do not use the system Python for this repo.
- For any Python, `pip`, test, or script command, use the Conda environment named `minos-orbits`.
- Prefer `conda run -n minos-orbits <command>` for one-off commands because shell state does not persist between tool calls.
- If a command requires activation semantics, run it through a shell that first activates Conda and then `conda activate minos-orbits`.

## Current Focus

- `v2/01_joi` is the active optimization codepath for current work.
- Treat `v1` as historical reference unless the user explicitly asks for it.
