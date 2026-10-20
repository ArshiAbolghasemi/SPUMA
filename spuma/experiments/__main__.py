"""Run SPUMA experiments: generate -> score -> train -> evaluate. Without --experiment, every registered one runs.

    uv run python -m spuma.experiments                                       # all experiments, every phase
    uv run python -m spuma.experiments --experiment math_ma_qwen2_5_7b_feedbackloop  # one experiment, by name
    uv run python -m spuma.experiments --phase train --phase evaluate        # from saved artifacts
    uv run python -m spuma.experiments --experiment math_ma_qwen2_5_7b_feedbackloop --limit 5  # smoke run: 5 per split
    uv run python -m spuma.experiments --list                                # registered experiments

Each experiment runs with its own settings file (SPUMA_* environment overrides apply to every experiment).
(`scripts/run-experiments.sh` wraps this and forwards the same flags.)
"""

import argparse
import logging
import sys

from spuma.experiments.registry import EXPERIMENTS
from spuma.experiments.runner import PHASES

logger = logging.getLogger("spuma.experiments")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-experiments", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment", action="append", choices=sorted(EXPERIMENTS), help="run only this experiment (repeatable)"
    )
    parser.add_argument("--phase", action="append", choices=PHASES, help="run only this phase (repeatable)")
    parser.add_argument("--limit", type=int, help="use only the first N problems of each split")
    parser.add_argument("--list", action="store_true", help="list the registered experiments and exit")
    args = parser.parse_args(argv)

    if args.list:
        for e in EXPERIMENTS.values():
            sys.stdout.write(f"{e.name}\t{e.config_file}\t{e.description}\n")
        return 0
    failed = []
    for name in args.experiment or list(EXPERIMENTS):
        try:
            EXPERIMENTS[name].run(phases=args.phase, limit=args.limit)
        except Exception:  # noqa: BLE001
            logger.exception("experiment failed", extra={"experiment": name})
            failed.append(name)
    if failed:
        logger.error("experiments failed", extra={"failed": failed})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
