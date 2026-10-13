"""Run SPUMA experiments: generate -> score -> train -> evaluate, for each tool-use variant.

    uv run python -m spuma.experiments --list
    uv run python -m spuma.experiments                                  # every variant, every phase
    uv run python -m spuma.experiments --variant no_tools --variant both_tools
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
    parser.add_argument("--variant", action="append", help="run only this variant (repeatable)")
    parser.add_argument("--phase", action="append", choices=PHASES, help="run only this phase (repeatable)")
    parser.add_argument("--limit", type=int, help="use only the first N problems of each split")
    parser.add_argument("--list", action="store_true", help="list variants and their per-agent allowed tools")
    args = parser.parse_args()

    if args.config:
        os.environ[SETTINGS_FILE_ENV] = args.config  # read by the first get_settings() call
    settings = get_settings()
    setup_logging(settings.experiment.name)
    if args.list:
        for name in settings.experiment.variants:
            agents = get_settings(name).agents.enabled
            logger.info("variant %s: %s", name, {a.id: a.allowed_tools for a in agents})
        return
    runner = ExperimentRunner()
    set_run_id(runner.run_id)
    runner.run(variants=args.variant, phases=args.phase, limit=args.limit)


if __name__ == "__main__":
    main()
