from __future__ import annotations

import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from shared.cloud_cli import CloudCliTarget, main_for_target


def main() -> None:
    target = CloudCliTarget(
        name="JOI",
        repo_root_hint=Path(__file__).resolve().parent,
        dockerfile_path="v2/01_joi/Dockerfile",
        default_image_tag="joi-search:local",
        default_docker_results_dir="v2/01_joi/docker-results",
        default_fetch_results_dir="v2/01_joi/fetched-results",
    )
    main_for_target(target)


if __name__ == "__main__":
    main()
