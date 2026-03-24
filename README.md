# minos-1-orbits

This repo contains trajectory-search and analysis work for the MINOS mission studies. Active development is in `v2/`. Treat `v1/` as historical reference unless a task explicitly asks for it.

## Environment

Use the Conda environment `minos-orbits` for all Python work in this repo.

Typical setup:

```bash
brew install miniforge
conda init bash
conda create -n minos-orbits python=3.11
conda activate minos-orbits
conda install pykep
```

Rules:

- Do not use the system Python for this repo.
- For one-off commands, prefer `conda run -n minos-orbits <command>`.
- If you need project architecture, workflow, or file layout details for the current codebase, start with [v2/README.md](v2/README.md).

## Where To Look

- Current architecture and workflows:
  [v2/README.md](v2/README.md)
- Earth-to-Jupiter mission package:
  [v2/01_joi](v2/01_joi)
- Shared cloud and search infrastructure:
  [v2/shared](v2/shared)
