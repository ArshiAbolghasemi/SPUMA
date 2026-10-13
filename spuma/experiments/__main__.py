"""Run the SPUMA experiment: generate -> score -> train -> evaluate (agents use their configured tools).

    uv run python -m spuma.experiments                                  # every phase
    uv run python -m spuma.experiments --phase train --phase evaluate   # from saved artifacts
    uv run python -m spuma.experiments --limit 5                        # smoke run: 5 problems per split

(`scripts/run-experiments.sh` wraps this and forwards the same flags.)
"""

import argparse
import logging
import os

from spuma.common.logs import set_run_id, setup_logging
from spuma.config import SETTINGS_FILE_ENV, get_settings
from spuma.experiments.runner import PHASES, ExperimentRunner

logger = logging.getLogger("spuma.experiments")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run-experiments", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="path to the config file (default: config/config-qwen-math.yaml)")
    parser.add_argument("--phase", action="append", choices=PHASES, help="run only this phase (repeatable)")
    parser.add_argument("--limit", type=int, help="use only the first N problems of each split")
    args = parser.parse_args()

    if args.config:
        os.environ[SETTINGS_FILE_ENV] = args.config  # read by the first get_settings() call
    setup_logging(get_settings().experiment.name)
    runner = ExperimentRunner()
    set_run_id(runner.run_id)
    runner.run(phases=args.phase, limit=args.limit)


if __name__ == "__main__":
    main()
