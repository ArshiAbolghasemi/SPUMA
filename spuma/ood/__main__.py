"""OOD evaluation of a frozen SPN on another dataset. Without --ood, every registered OOD evaluation runs.

    uv run python -m spuma.ood                                                         # all, every phase
    uv run python -m spuma.ood --ood ood_morehopqa_math_ma_qwen_feedbackloop           # one, by name
    uv run python -m spuma.ood --ood ood_morehopqa_math_ma_qwen_feedbackloop --limit 5 # smoke run: 5 problems
    uv run python -m spuma.ood --phase evaluate                                        # re-evaluate saved features
    uv run python -m spuma.ood --list                                                  # registered OOD evaluations

(`scripts/run-ood.sh` wraps this and forwards the same flags.)
"""

import argparse
import logging
import sys

from spuma.ood.config import load_ood_config
from spuma.ood.evaluation import OODEvaluation
from spuma.ood.registry import OOD_EVALUATIONS

logger = logging.getLogger("spuma.ood")

PHASES = ("generate", "score", "evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-ood", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ood", action="append", choices=sorted(OOD_EVALUATIONS), help="run only this one (repeatable)"
    )
    parser.add_argument("--phase", action="append", choices=PHASES, help="run only this phase (repeatable)")
    parser.add_argument("--limit", type=int, help="use only the first N OOD problems")
    parser.add_argument("--list", action="store_true", help="list the registered OOD evaluations and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name, path in OOD_EVALUATIONS.items():
            sys.stdout.write(f"{name}\t{path}\n")
        return 0
    failed = []
    for name in args.ood or list(OOD_EVALUATIONS):
        try:
            OODEvaluation(load_ood_config(OOD_EVALUATIONS[name])).run(phases=args.phase, limit=args.limit)
        except Exception:  # noqa: BLE001 - one failing evaluation must not stop the others
            logger.exception("ood evaluation failed", extra={"ood": name})
            failed.append(name)
    if failed:
        logger.error("ood evaluations failed", extra={"failed": failed})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
