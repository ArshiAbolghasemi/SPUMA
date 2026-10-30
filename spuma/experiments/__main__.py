"""Run SPUMA experiments: generate -> score -> train -> evaluate. Without --experiment, every registered one runs.

    uv run python -m spuma.experiments                                       # all experiments, every phase
    uv run python -m spuma.experiments --experiment math_ma_qwen2_5_7b_feedbackloop  # one experiment, by name
    uv run python -m spuma.experiments --phase train --phase evaluate        # from saved artifacts
    uv run python -m spuma.experiments --experiment math_ma_qwen2_5_7b_feedbackloop --limit 5  # smoke run: 5 per split
    uv run python -m spuma.experiments --architecture sequential --dataset musique  # by architecture and/or dataset
    uv run python -m spuma.experiments --list                                # registered experiments

Each experiment runs with its own settings file (SPUMA_* environment overrides apply to every experiment).
(`scripts/run-experiments.sh` wraps this and forwards the same flags.)
"""

import argparse
import json
import logging
import sys

from spuma.config import PROJECT_ROOT
from spuma.experiments.registry import ARCHITECTURES, EXPERIMENTS
from spuma.experiments.runner import PHASES

logger = logging.getLogger("spuma.experiments")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-experiments", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment", action="append", choices=sorted(EXPERIMENTS), help="run only this experiment (repeatable)"
    )
    parser.add_argument(
        "--architecture", action="append", choices=ARCHITECTURES, help="only experiments of this architecture"
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted({e.dataset for e in EXPERIMENTS.values()}),
        help="only experiments on this dataset (repeatable)",
    )
    parser.add_argument("--phase", action="append", choices=PHASES, help="run only this phase (repeatable)")
    parser.add_argument("--limit", type=int, help="use only the first N problems of each split")
    parser.add_argument("--list", action="store_true", help="list the registered experiments and exit")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="write the consolidated CHMM-study table of the selected experiments (artifacts/comparison.md) and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        for e in EXPERIMENTS.values():
            sys.stdout.write(f"{e.name}\t{e.architecture}\t{e.dataset}\t{e.config_file}\t{e.description}\n")
        return 0
    names = [
        name
        for name in args.experiment or list(EXPERIMENTS)
        if (not args.architecture or EXPERIMENTS[name].architecture in args.architecture)
        and (not args.dataset or EXPERIMENTS[name].dataset in args.dataset)
    ]
    if not names:
        parser.error("no registered experiment matches --experiment/--architecture/--dataset")
    if args.compare:
        write_comparison(names)
        return 0
    failed = []
    for name in names:
        try:
            EXPERIMENTS[name].run(phases=args.phase, limit=args.limit)
        except Exception:  # noqa: BLE001
            logger.exception("experiment failed", extra={"experiment": name})
            failed.append(name)
    if failed:
        logger.error("experiments failed", extra={"failed": failed})
    return 1 if failed else 0


def write_comparison(names: list[str]) -> None:
    from spuma.chmm.experiment import METRIC_NAMES, consolidate  # noqa: PLC0415

    rows = consolidate(names)
    head = ["Dataset", "Architecture", "Protocol", "Method", "Alpha", *(m.upper() for m in METRIC_NAMES)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        alpha = "—" if r["alpha"] is None else f"{r['alpha']:g}"
        values = [
            "—" if (r["calibration_metrics_from_platt"] and m in ("brier", "ece")) or r[m] != r[m] else f"{r[m]:.4f}"
            for m in METRIC_NAMES
        ]
        lines.append(
            "| " + " | ".join([r["dataset"], r["architecture"], r["protocol"], r["method"], alpha, *values]) + " |"
        )
    out = PROJECT_ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(rows, indent=1))
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    sys.stdout.write(
        f"{len(rows)} rows from {len({r['experiment'] for r in rows})} experiments -> {out / 'comparison.md'}\n"
    )


if __name__ == "__main__":
    sys.exit(main())
