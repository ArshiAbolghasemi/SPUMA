"""Experiment execution: generate -> score -> train -> evaluate, phase by phase over the tool-use variants.

Phases run for every variant before the next phase starts. With a hybrid backend setup
(`model.backend: vllm`, `model.scoring_backend: transformers`) the generation engines are released
after the generate phase, so the scoring engine never shares the GPU with them.

    artifacts/<experiment>/
      config.json                      sanitized config snapshot
      splits.json                      problem ids per split (shared by every variant)
      runs/<run_id>.json               reproducibility metadata of each invocation
      <variant>-<gen fp>/
        trajectories/<problem>.json    raw trajectory (exact contexts, tokens, tool records, generation-time
                                       scores, semantic-entropy samples) + grade
        scored-<scoring fp>/
          steps/<problem>.json         trajectory with step metrics and semantic-entropy samples
          step_metrics.json            per-step metric rows
          features.json                Z (8 features), label Y, parse failure, tool metadata
          models/<name>.pt             risk-model checkpoints (evaluation.risk_models; today: spn)
          evaluation.json              risk models on validation/test (+ cross-variant)
      summary.json                     test metrics of every variant side by side

Fingerprints in directory names (generation: model/backend/sampling/agents/tools/SE settings;
scoring: scoring engine + UQ/aggregation/feature settings) mean a config change never reuses stale
artifacts; with `experiment.resume`, existing artifacts are reused instead of recomputed.
"""

import logging
import platform
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

import numpy as np

from spuma.agents.trajectory import Trajectory
from spuma.agents.workflow import build_workflow
from spuma.artifacts import FileArtifactStore
from spuma.common.config import to_plain
from spuma.common.utils import safe_name
from spuma.config import PROJECT_ROOT, get_settings
from spuma.dataset.base import AnswerGrader, DatasetProvider, Problem
from spuma.dataset.registry import get_dataset, get_grader
from spuma.evaluation.metrics import binary_entropy, evaluate
from spuma.evaluation.risk import risk_model_spec
from spuma.features import extract_features, tool_metadata
from spuma.models.backend import ModelBackend, get_backend, release_backends
from spuma.models.config import ModelConfig
from spuma.models.validation import validate_backends
from spuma.tools.registry import ToolExecutor, ToolProvider, build_tool_registry, default_providers
from spuma.tools.worker import IsolatedRunner
from spuma.uncertainty.estimators import (
    SemanticSampler,
    build_estimators,
    build_token_scorers,
    score_trajectory,
    step_metric_rows,
)

logger = logging.getLogger(__name__)

PHASES = ("generate", "score", "train", "evaluate")
SPLITS = ("train", "validation", "test")


class ExperimentRunner:
    """Runs the experiment described by `get_settings()`; tool-use variants via `get_settings(variant)`."""

    def __init__(
        self,
        dataset: DatasetProvider | None = None,
        backend_factory: Callable[[ModelConfig, str], ModelBackend] = get_backend,
        providers: dict[str, ToolProvider] | None = None,
        grader: AnswerGrader | None = None,
        release: Callable[[], None] = release_backends,
    ) -> None:
        config = get_settings()
        self.release = release
        self.dataset = dataset or get_dataset()
        self.backend_factory = backend_factory
        self._runner = IsolatedRunner() if providers is None else None
        self.providers = providers if providers is not None else default_providers(self._runner)
        self.grader = grader or get_grader()
        self.store = FileArtifactStore(PROJECT_ROOT / config.experiment.output_dir / config.experiment.name)
        self.run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"

    # --- layout ----------------------------------------------------------------------------------

    def variant_dir(self, variant: str) -> str:
        return f"{variant}-{get_settings(variant).generation_fingerprint()}"

    def scored_dir(self, variant: str) -> str:
        return f"{self.variant_dir(variant)}/scored-{get_settings().scoring_fingerprint()}"

    # --- driver ----------------------------------------------------------------------------------

    def run(self, variants: list[str] | None = None, phases: list[str] | None = None, limit: int | None = None) -> None:
        configured = list(get_settings().experiment.variants)
        variants = variants or configured
        phases = phases or list(PHASES)
        unknown = set(variants) - set(configured)
        if unknown:
            raise ValueError(f"unknown variants {sorted(unknown)}; configured: {configured}")
        # Fail before loading any model if a backend cannot provide what this experiment needs.
        plans = {v: validate_backends(v) for v in variants}
        self.store.write_json("config.json", get_settings().sanitized())
        self.store.write_json(f"runs/{self.run_id}.json", self.metadata(variants, phases, limit, plans))
        try:
            if "generate" in phases:
                self.write_splits(limit)
                for variant in variants:
                    logger.info("generate", extra={"variant": variant, "dir": self.variant_dir(variant)})
                    self.generate(variant)
                if self._hybrid():
                    logger.info("releasing generation engines before scoring (hybrid backends)")
                    self.release()
            if "score" in phases:
                for variant in variants:
                    self.score(variant)
            if "train" in phases:
                for variant in variants:
                    self.train(variant)
            if "evaluate" in phases:  # after every variant is scored, so cross-variant tests see them all
                for variant in variants:
                    self.evaluate(variant, variants)
                self.summarize(variants)
        finally:
            if self._runner:
                self._runner.close()

    def _hybrid(self) -> bool:
        return any(not m.scores_at_generation for m in get_settings().all_models().values())

    def metadata(
        self, variants: list[str], phases: list[str], limit: int | None, plans: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT, check=False
            ).stdout.strip()
        except OSError:
            commit = ""
        packages = [
            "torch",
            "transformers",
            "vllm",
            "langgraph",
            "langchain-core",
            "spflow",
            "sympy",
            "math-verify",
        ]
        se = get_settings().uncertainty.semantic_entropy
        return {
            "run_id": self.run_id,
            "started": datetime.now(UTC).isoformat(),
            "argv": sys.argv,
            "variants": variants,
            "phases": phases,
            "limit": limit,
            "seed": get_settings().project.seed,
            "git_commit": commit,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: _version(p) for p in packages},
            "semantic_entropy_policy": {
                "n": se.n,
                "top_k": None,
                "temperature": se.temperature,
                "top_p": se.top_p,
                "tool_call_policy": se.tool_call_policy,
                "clustering": "uqlm",
                "nli_model": se.nli_model_name,
            },
            "variant_dirs": {v: self.variant_dir(v) for v in variants},
            "backends": {
                "models": {k: to_plain(m) for k, m in get_settings().all_models().items()},
                "plan": plans,
            },
        }

    # --- phases ----------------------------------------------------------------------------------

    def write_splits(self, limit: int | None) -> None:
        splits = self.dataset.load()
        if limit is not None:
            splits = {k: v[:limit] for k, v in splits.items()}
        self.store.write_json("splits.json", {k: [asdict(p) for p in v] for k, v in splits.items()})

    def problems(self) -> dict[str, list[Problem]]:
        if not self.store.exists("splits.json"):
            raise RuntimeError("no splits.json: run the generate phase first")
        return {k: [Problem(**p) for p in v] for k, v in self.store.read_json("splits.json").items()}

    def _backends(self) -> dict[str, ModelBackend]:
        """Generation engine per agent (shared by agents on the same model)."""
        config = get_settings()
        return {
            a.id: self.backend_factory(config.model_for(a), config.model_for(a).backend) for a in config.agents.enabled
        }

    def generate(self, variant: str) -> None:
        backends = self._backends()
        se_enabled = get_settings().uncertainty.semantic_entropy.enabled
        samplers = {a: SemanticSampler(b) for a, b in backends.items()} if se_enabled else {}
        registry = build_tool_registry(self.providers)
        first = next(iter(backends.values()))
        executor = ToolExecutor(registry, truncate=first.truncate)
        workflow = build_workflow(registry, backends.__getitem__, executor, variant)
        self.store.write_json(f"{self.variant_dir(variant)}/agents.json", [a.describe() for a in workflow.agents])
        for split, problems in self.problems().items():
            for i, problem in enumerate(problems):
                rel = f"{self.variant_dir(variant)}/trajectories/{safe_name(problem.problem_id)}.json"
                if get_settings().experiment.resume and self.store.exists(rel):
                    record = self.store.read_json(rel)
                    trajectory = Trajectory.model_validate(record["trajectory"])
                    if not any(
                        samplers[t.agent_id].needs_samples(s)
                        for t in trajectory.agents
                        for s in t.steps
                        if t.agent_id in samplers
                    ):
                        continue
                    grade_dict = record["grade"]
                else:
                    # Only the id and the problem text enter the workflow; the reference stays here.
                    trajectory = workflow.run(problem.problem_id, problem.problem)
                    grade_dict = self.grader.grade(problem, trajectory.final_answer_text).to_dict()
                for trace in trajectory.agents:
                    if trace.agent_id in samplers:
                        samplers[trace.agent_id].sample_steps(trace.steps)
                self.store.write_json(rel, {"split": split, "trajectory": trajectory.model_dump(), "grade": grade_dict})
                logger.info(
                    "generated",
                    extra={
                        "variant": variant,
                        "split": split,
                        "index": i,
                        "problem_id": problem.problem_id,
                        "label": grade_dict["label"],
                    },
                )

    def score(self, variant: str) -> None:
        estimators = build_estimators()
        scorers = build_token_scorers(self.backend_factory)
        out_dir = self.scored_dir(variant)
        rows, step_rows = [], []
        for split, problems in self.problems().items():
            for problem in problems:
                name = safe_name(problem.problem_id)
                rel = f"{out_dir}/steps/{name}.json"
                if get_settings().experiment.resume and self.store.exists(rel):
                    record = self.store.read_json(rel)
                else:
                    raw = self.store.read_json(f"{self.variant_dir(variant)}/trajectories/{name}.json")
                    scored = score_trajectory(Trajectory.model_validate(raw["trajectory"]), estimators, scorers)
                    record = {**raw, "trajectory": scored.model_dump()}
                    self.store.write_json(rel, record)
                trajectory = Trajectory.model_validate(record["trajectory"])
                step_rows.extend(step_metric_rows(trajectory))
                rows.append(
                    {
                        "problem_id": problem.problem_id,
                        "split": split,
                        "features": extract_features(trajectory),
                        "label": record["grade"]["label"],
                        "parse_failure": record["grade"]["parse_failure"],
                        "tool_metadata": tool_metadata(trajectory),
                    }
                )
        self.store.write_json(f"{out_dir}/step_metrics.json", step_rows)
        self.store.write_json(f"{out_dir}/features.json", rows)
        logger.info("scored", extra={"variant": variant, "problems": len(rows)})

    def _matrix(self, variant: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        rows = [r for r in self.store.read_json(f"{self.scored_dir(variant)}/features.json") if r["split"] == split]
        order = get_settings().features.order
        Z = np.array([[r["features"][f] for f in order] for r in rows], dtype=float).reshape(len(rows), len(order))
        return Z, np.array([r["label"] for r in rows], dtype=int), [r["problem_id"] for r in rows]

    def model_path(self, variant: str, name: str) -> str:
        return f"{self.scored_dir(variant)}/models/{name}{risk_model_spec(name).suffix}"

    def train(self, variant: str) -> None:
        Z, y, _ = self._matrix(variant, "train")
        for name in get_settings().evaluation.risk_models:
            status_rel = f"{self.scored_dir(variant)}/models/{name}_status.json"
            try:
                model = risk_model_spec(name).build().fit(Z, y)
            except ValueError as e:
                logger.error("risk model training failed", extra={"variant": variant, "model": name, "error": str(e)})
                self.store.write_json(status_rel, {"status": "failed", "error": str(e)})
                continue
            model.save(self.store.path(self.model_path(variant, name)))
            status = {"status": "trained", "train_rows": len(y), "incorrect": int(y.sum()), **model.summary()}
            self.store.write_json(status_rel, status)
            logger.info("risk model trained", extra={"variant": variant, "model": name, **status})

    def evaluate(self, variant: str, all_variants: list[str]) -> None:
        """Per risk model: P(Y=1|Z) metrics, posterior-entropy ranking metrics (separately), predictions."""
        ev = get_settings().evaluation
        boot = (
            {
                "iterations": ev.bootstrap.iterations,
                "confidence": ev.bootstrap.confidence,
                "seed": get_settings().project.seed,
            }
            if ev.bootstrap.enabled
            else None
        )
        ranking = [m for m in ev.metrics if m in ("auroc", "auarc", "aurc", "auprc")]
        models = {}
        for name in ev.risk_models:
            path = self.store.path(self.model_path(variant, name))
            if path.exists():
                models[name] = risk_model_spec(name).load(path)
        result: dict[str, Any] = {"variant": variant, "run_id": self.run_id, "splits": {}}
        for split in ("validation", "test"):
            Z, y, ids = self._matrix(variant, split)
            if not len(y):
                continue
            split_res: dict[str, Any] = {"n": len(y), "accuracy": float(1 - y.mean()), "models": {}, "predictions": {}}
            groups = np.array(ids)
            for name, model in models.items():
                p = model.predict_error_proba(Z)
                h = binary_entropy(p)
                split_res["models"][name] = evaluate(y, p, ev.metrics, groups, boot, ev.ece_bins)
                split_res["models"][f"{name}_posterior_entropy"] = evaluate(y, h, ranking, groups, boot, ev.ece_bins)
                split_res["predictions"][name] = [
                    {"problem_id": i, "label": int(t), "p_incorrect": float(a), "posterior_entropy": float(b)}
                    for i, t, a, b in zip(ids, y, p, h, strict=True)
                ]
            result["splits"][split] = split_res
        if models and get_settings().experiment.spn_protocol == "cross_variant":
            result["cross_variant_test"] = {name: {} for name in models}
            for other in all_variants:
                if other == variant or not self.store.exists(f"{self.scored_dir(other)}/features.json"):
                    continue
                Z, y, ids = self._matrix(other, "test")
                if not len(y):
                    continue
                for name, model in models.items():
                    p = model.predict_error_proba(Z)
                    result["cross_variant_test"][name][other] = evaluate(
                        y, p, ev.metrics, np.array(ids), boot, ev.ece_bins
                    )
        self.store.write_json(f"{self.scored_dir(variant)}/evaluation.json", result)
        test = result["splits"].get("test", {}).get("models", {})
        logger.info(
            "evaluation",
            extra={"variant": variant, "test": {m: {k: v["value"] for k, v in r.items()} for m, r in test.items()}},
        )

    def summarize(self, variants: list[str]) -> None:
        summary = {}
        for v in variants:
            rel = f"{self.scored_dir(v)}/evaluation.json"
            if not self.store.exists(rel):
                continue
            test = self.store.read_json(rel)["splits"].get("test", {})
            summary[v] = {
                "accuracy": test.get("accuracy"),
                "n": test.get("n"),
                "models": {m: {k: r["value"] for k, r in res.items()} for m, res in test.get("models", {}).items()},
            }
        self.store.write_json("summary.json", summary)
        logger.info("summary", extra={"summary": summary})


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
