# Tools

- Pykep
  brew install miniforge
  conda init bash
  conda create -n minos-orbits python=3.11
  conda activate minos-orbits # DO THIS IN EVERY NEW TERMINAL
  conda install pykep

# Environment Rule

- Use the `minos-orbits` Conda environment for this repo.
- Do not use the machine's default `python`.
- For one-off commands, prefer `conda run -n minos-orbits <command>`.
