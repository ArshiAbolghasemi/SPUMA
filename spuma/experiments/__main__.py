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

from spuma.common.logs import set_run_id, setup_logging
from spuma.config import load_settings
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

    config = load_settings(args.config)
    setup_logging(config.logging, config.experiment.name)
    if args.list:
        for name in config.experiment.variants:
            v = config.with_variant(name)
            logger.info("variant %s: %s", name, {a.id: a.allowed_tools for a in v.agents.enabled})
        return
    runner = ExperimentRunner(config)
    set_run_id(runner.run_id)
    runner.run(variants=args.variant, phases=args.phase, limit=args.limit)


if __name__ == "__main__":
    main()
