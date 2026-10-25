"""OOD evaluation of a frozen SPN: run the source experiment's agents and uncertainty on a new dataset, feed the
aggregated features to the trained risk model, and score its error probabilities against the OOD labels.

    1. settings: the source experiment's settings file with `dataset:` and `experiment.name/output_dir`
       replaced. Before `ood.agents` is applied, its generation and feature fingerprints must equal the
       checkpoint's `generation-<fp>/scored-<fp>/` directories, so workflow, agents, sampling, rounds,
       uncertainty, aggregation and feature order are exactly those the SPN was trained on. Then each agent's
       role and prompts are replaced by `ood.agents` (the task changes, the pipeline does not) and the result
       is saved as <output>/settings.yaml.
    2. generate + score: the unchanged `ExperimentRunner` phases on the OOD test problems (per-step MCSP/SE/NLL/
       length, per-problem features, steps.jsonl).
    3. evaluate: the checkpoint is loaded (never fitted, its SHA-256 checked before and after), P(Y=1|z) is
       predicted per problem, and `ood.metrics` are computed with problem-level bootstrap intervals; the
       in-distribution test results next to the checkpoint are reported alongside when present.

Nothing here trains, tunes or recalibrates; OOD labels are read only by the grader and the metrics.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from spuma.chmm.experiment import CHMMAggregation
from spuma.common.config import to_plain
from spuma.common.logs import set_run_id, setup_logging
from spuma.config import SETTINGS_FILE_ENV, get_settings
from spuma.evaluation.metrics import binary_entropy, evaluate
from spuma.evaluation.risk import risk_model_spec
from spuma.experiments.runner import ExperimentRunner
from spuma.models.engines import release_backends
from spuma.ood.config import OODConfig

logger = logging.getLogger(__name__)

RANKING = ("auroc", "auprc", "auarc", "aurc")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(path: Path, prefix: str) -> str | None:
    return next((p.name.removeprefix(prefix) for p in path.parents if p.name.startswith(prefix)), None)


class OODEvaluation:
    def __init__(self, config: OODConfig, runner_kwargs: dict[str, Any] | None = None) -> None:
        self.config = config
        self.root = config.output_dir / config.name
        self.runner_kwargs = runner_kwargs or {}

    def activate(self) -> None:
        """Install the source settings with the OOD dataset as the process-wide settings and check that they
        reproduce the checkpoint's pipeline."""
        c = self.config
        if not c.checkpoint.is_file():
            raise FileNotFoundError(f"ood.checkpoint {c.checkpoint} not found")
        raw = yaml.safe_load(c.source_config.read_text())
        raw["dataset"] = to_plain(c.dataset)
        raw["experiment"] = {**raw.get("experiment", {}), "name": c.name, "output_dir": str(c.output_dir)}
        self.root.mkdir(parents=True, exist_ok=True)
        settings = self._install(raw)
        expected = (_fingerprint(c.checkpoint, "generation-"), _fingerprint(c.checkpoint, "scored-"))
        actual = (settings.generation_fingerprint(), settings.scoring_fingerprint())
        if None not in expected and expected != actual:
            raise ValueError(
                f"source settings do not reproduce the checkpoint's pipeline: generation/scored fingerprints "
                f"{actual} != checkpoint's {expected}"
            )
        if c.agents:
            items = {a["id"]: a for a in raw["agents"]["items"] if a.get("enabled", True)}
            if set(c.agents) != set(items):
                raise ValueError(f"ood.agents {sorted(c.agents)} must name the source agents {sorted(items)}")
            for agent_id, prompt in c.agents.items():
                items[agent_id].update({k: v for k, v in to_plain(prompt).items() if v is not None})
            self._install(raw)

    def _install(self, raw: dict[str, Any]) -> Any:
        settings_file = self.root / "settings.yaml"
        settings_file.write_text(yaml.safe_dump(raw, sort_keys=False))
        os.environ[SETTINGS_FILE_ENV] = str(settings_file)
        get_settings.cache_clear()
        return get_settings()

    def run(self, phases: list[str] | None = None, limit: int | None = None) -> dict[str, Any] | None:
        phases = phases or ["generate", "score", "evaluate"]
        self.activate()
        setup_logging(self.config.name)
        runner = ExperimentRunner(**self.runner_kwargs)
        set_run_id(runner.run_id)
        logger.info("ood evaluation", extra={"config": to_plain(self.config), "phases": phases, "limit": limit})
        try:
            if "generate" in phases or "score" in phases:
                runner.run(phases=[p for p in ("generate", "score") if p in phases], limit=limit)
            return self.evaluate(runner) if "evaluate" in phases else None
        finally:
            release_backends()

    def evaluate(self, runner: ExperimentRunner) -> dict[str, Any]:
        c = self.config
        settings = get_settings()
        before = sha256(c.checkpoint)
        model = risk_model_spec(c.risk_model).load(c.checkpoint)
        order = settings.features.order
        if list(model.feature_names) != list(order):
            raise ValueError(f"checkpoint features {model.feature_names} != settings features.order {order}")
        rows = [r for r in runner.store.read_json(f"{runner.scored_dir()}/features.json") if r["split"] == "test"]
        if not rows:
            raise RuntimeError("no OOD feature rows: run the generate and score phases first")
        Z = np.array([[r["features"][f] for f in order] for r in rows], dtype=float)
        y = np.array([r["label"] for r in rows], dtype=int)
        p = model.predict_error_proba(Z)
        h = binary_entropy(p)
        groups = np.array([r["problem_id"] for r in rows])
        ev = settings.evaluation
        boot = (
            {
                "iterations": ev.bootstrap.iterations,
                "confidence": ev.bootstrap.confidence,
                "seed": settings.project.seed,
            }
            if ev.bootstrap.enabled
            else None
        )
        ranking = [m for m in c.metrics if m in RANKING]
        result: dict[str, Any] = {
            "run_id": runner.run_id,
            "dataset": to_plain(c.dataset),
            "checkpoint": {"path": str(c.checkpoint), "risk_model": c.risk_model, "sha256": before},
            "n": len(y),
            "accuracy": float(1 - y.mean()),
            "parse_failures": int(sum(r["parse_failure"] for r in rows)),
            "models": {
                c.risk_model: evaluate(y, p, c.metrics, groups, boot, ev.ece_bins),
                f"{c.risk_model}_posterior_entropy": evaluate(y, h, ranking, groups, boot, ev.ece_bins),
            },
            "predictions": [
                {"problem_id": r["problem_id"], "label": int(t), "p_incorrect": float(a), "posterior_entropy": float(b)}
                for r, t, a, b in zip(rows, y, p, h, strict=True)
            ],
        }
        result["id_comparison"] = self.compare_with_id(result["models"][c.risk_model], Z, order)
        chmm = None
        if settings.chmm.enabled:
            aggregation = CHMMAggregation(runner)
            # the source experiment's chmm directory: same settings, so the same fingerprint, next to models/
            source = c.checkpoint.parent.parent / Path(aggregation.out).name
            if source.is_dir():
                chmm = aggregation.evaluate(frozen_from=source)
                result["chmm"] = {"output": aggregation.out, "frozen_from": str(source)}
            else:
                logger.warning(
                    "no trained CHMM next to the checkpoint; CHMM methods skipped", extra={"path": str(source)}
                )
        after = sha256(c.checkpoint)
        if after != before:
            raise RuntimeError(f"checkpoint {c.checkpoint} changed during OOD evaluation")
        out = f"{runner.scored_dir()}/ood_evaluation.json"
        runner.store.write_json(out, result)
        summary = {
            "n": result["n"],
            "accuracy": result["accuracy"],
            "models": {m: {k: r["value"] for k, r in res.items()} for m, res in result["models"].items()},
            "id_comparison": result["id_comparison"]["metrics"] if result["id_comparison"] else None,
        }
        if chmm is not None:
            summary["chmm"] = {m: {k: r["value"] for k, r in res.items()} for m, res in chmm["methods"].items()}
        runner.store.write_json("ood_summary.json", summary)
        logger.info("ood evaluation done", extra={"output": out, "summary": summary})
        return result

    def compare_with_id(
        self, ood: dict[str, dict[str, float]], Z: np.ndarray, order: list[str]
    ) -> dict[str, Any] | None:
        """OOD minus in-distribution test values per metric, and the shift of each feature's mean."""
        c = self.config
        scored = c.checkpoint.parent.parent
        path = c.id_evaluation or scored / "evaluation.json"
        if not path.is_file():
            logger.warning("no in-distribution evaluation to compare with", extra={"path": str(path)})
            return None
        id_eval = json.loads(path.read_text())
        id_test = id_eval["splits"]["test"]
        id_model = id_test["models"].get(c.risk_model, {})
        metrics = {
            m: {"id": id_model[m]["value"], "ood": ood[m]["value"], "delta": ood[m]["value"] - id_model[m]["value"]}
            for m in c.metrics
            if m in id_model
        }
        features = None
        id_features = scored / "features.json"
        if id_features.is_file():
            rows = [r for r in json.loads(id_features.read_text()) if r["split"] == "test"]
            id_Z = np.array([[r["features"][f] for f in order] for r in rows], dtype=float)
            features = {
                f: {
                    "id_mean": float(id_Z[:, j].mean()),
                    "ood_mean": float(Z[:, j].mean()),
                    "id_std": float(id_Z[:, j].std()),
                    "ood_std": float(Z[:, j].std()),
                }
                for j, f in enumerate(order)
            }
        return {
            "source": str(path),
            "id_n": id_test["n"],
            "id_accuracy": id_test["accuracy"],
            "metrics": metrics,
            "features": features,
        }
