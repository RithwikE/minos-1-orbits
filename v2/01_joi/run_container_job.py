from __future__ import annotations

import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from shared.container_job import ContainerJobTarget, run_container_job_main
from run_search import print_run_summary, run_search


def main() -> None:
    target = ContainerJobTarget(
        name="JOI",
        source_root=Path(__file__).resolve().parent,
        run_search=run_search,
        print_run_summary=print_run_summary,
    )
    run_container_job_main(target)


if __name__ == "__main__":
    main()
