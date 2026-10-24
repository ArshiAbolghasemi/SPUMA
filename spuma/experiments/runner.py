"""Experiment execution: generate -> score -> train -> evaluate.

generate runs the agent workflow on every problem; uncertainty (MCSP, SE, SD, token length) is measured
inside the graph right after every response, on the generating engine
(`model.scoring_backend: same`). score turns the measured steps into features; train/evaluate fit and score
the risk models.

    artifacts/<experiment>/
      config.json                      sanitized config snapshot
      splits.json                      problem ids per split
      runs/<invocation>.json           reproducibility metadata of each invocation
      generation-<gen fp>/
        agents.json                    agent settings
        trajectories/<problem>.json    conversation, every step (prompt, tokens, log-probs, samples, seeds,
                                       UQ scores), rounds, termination reason, grade
        trajectories/<problem>.steps.jsonl
                                       the same steps, appended as they are measured (progress log)
        scored-<feature fp>/
          step_metrics.json            per-step metric rows
          steps.jsonl                  one full step per line, every problem
          features.json                Z per problem, label Y, parse failure
          models/<name>.pt             risk-model checkpoints (evaluation.risk_models)
          evaluation.json              risk models on validation/test
          chmm-<fp>/                   CHMM-weighted aggregation methods (`chmm.enabled`, spuma.chmm.experiment)
      summary.json                     test accuracy and risk-model metrics

Fingerprints in directory names (generation: model, agents, seeds, uncertainty settings; features:
aggregation and feature settings) mean a config change never reuses stale artifacts; with
`experiment.resume`, finished problems are kept and only missing or failed ones are generated again.
"""

import logging
import platform
import subprocess
import sys
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

import numpy as np

from spuma.agents.trajectory import Trajectory
from spuma.agents.workflow import build_workflow
from spuma.chmm.experiment import CHMMAggregation
from spuma.common.config import to_plain
from spuma.common.logs import progress, progress_counter
from spuma.common.utils import MainThreadCalls, safe_name
from spuma.config import PROJECT_ROOT, get_settings
from spuma.dataset.base import AnswerGrader, DatasetProvider, Problem
from spuma.dataset.registry import get_dataset, get_grader
from spuma.evaluation.metrics import binary_entropy, evaluate
from spuma.evaluation.risk import risk_model_spec
from spuma.experiments.artifacts import FileArtifactStore
from spuma.models.backend import ModelBackend
from spuma.models.batching import MicroBatcher
from spuma.models.config import ModelConfig
from spuma.models.engines import get_backend, release_backends
from spuma.models.validation import validate_backends
from spuma.uncertainty.estimators import StepUncertainty, step_metric_rows
from spuma.uncertainty.features import extract_features

logger = logging.getLogger(__name__)

PHASES = ("generate", "score", "train", "evaluate")
SPLITS = ("train", "validation", "test")


class ExperimentRunner:
    """Runs the experiment described by `get_settings()`."""

    def __init__(
        self,
        dataset: DatasetProvider | None = None,
        backend_factory: Callable[[ModelConfig, str], ModelBackend] = get_backend,
        grader: AnswerGrader | None = None,
        release: Callable[[], None] = release_backends,
    ) -> None:
        config = get_settings()
        self.release = release
        self.dataset = dataset or get_dataset()
        self.backend_factory = backend_factory
        self.grader = grader or get_grader()
        self.store = FileArtifactStore(PROJECT_ROOT / config.experiment.output_dir / config.experiment.name)
        self.run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"

    def generation_dir(self) -> str:
        return f"generation-{get_settings().generation_fingerprint()}"

    def scored_dir(self) -> str:
        return f"{self.generation_dir()}/scored-{get_settings().scoring_fingerprint()}"

    def run(self, phases: list[str] | None = None, limit: int | None = None) -> None:
        phases = phases or list(PHASES)
        # train and evaluate only read saved artifacts: no model engine to validate
        plan = validate_backends() if {"generate", "score"} & set(phases) else {}
        self.store.write_json("config.json", get_settings().sanitized())
        self.store.write_json(f"runs/{self.run_id}.json", self.metadata(phases, limit, plan))
        if "generate" in phases:
            self.write_splits(limit)
            logger.info("generate", extra={"dir": self.generation_dir()})
            self.generate()
            self.release()
        if "score" in phases:
            self.score()
        if "train" in phases:
            self.train()
        if "evaluate" in phases:
            self.evaluate()

    def metadata(self, phases: list[str], limit: int | None, plan: dict[str, Any]) -> dict[str, Any]:
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
            "math-verify",
            "uqlm",
        ]
        config = get_settings()
        se, mc = config.uncertainty.semantic_entropy, config.uncertainty.monte_carlo
        return {
            "run_id": self.run_id,
            "started": datetime.now(UTC).isoformat(),
            "argv": sys.argv,
            "phases": phases,
            "limit": limit,
            "base_seed": config.project.seed,
            "seed_derivation": "sha256(base_seed, problem_id, agent_id, round[, sample index])",
            "git_commit": commit,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: _version(p) for p in packages},
            "agents": {
                "workflow": config.agents.workflow,
                "max_rounds": config.agents.max_rounds,
                "completion_token": config.agents.completion_token,
            },
            "semantic_entropy_policy": {
                "n": se.n,
                "top_k": None,
                "temperature": se.temperature,
                "top_p": se.top_p,
                "samples": "the shared Monte Carlo samples",
                "probabilities": "tokenprob (white-box)",
                "clustering": "uqlm",
                "equivalence": se.equivalence,
                "nli_model": se.nli_model_name,
            },
            "monte_carlo_policy": {
                "num_generations": mc.num_generations,
                "temperature": mc.temperature,
                "top_p": mc.top_p,
                "top_k": mc.top_k,
                "length_normalize": mc.length_normalize,
                "seed_strategy": "per_sample",
                "uqlm_mode": "white_box",
                "uqlm_scorers": {"mcsp": "monte_carlo_probability", "sd": "semantic_density"},
                "density_nli_model": config.uncertainty.semantic_density.nli_model_name,
            },
            "metrics": config.uncertainty.metrics,
            "generation_dir": self.generation_dir(),
            "backends": {
                "models": {k: to_plain(m) for k, m in get_settings().all_models().items()},
                "plan": plan,
            },
        }

    def write_splits(self, limit: int | None) -> None:
        splits = self.dataset.load()
        if limit is not None:
            splits = {k: v[:limit] for k, v in splits.items()}
        self.store.write_json("splits.json", {k: [asdict(p) for p in v] for k, v in splits.items()})

    def problems(self) -> dict[str, list[Problem]]:
        if not self.store.exists("splits.json"):
            raise RuntimeError("no splits.json: run the generate phase first")
        return {k: [Problem(**p) for p in v] for k, v in self.store.read_json("splits.json").items()}

    def _all_problems(self) -> list[tuple[str, int, Problem]]:
        """(split, index, problem) over every split, in split order."""
        return [(split, i, p) for split, problems in self.problems().items() for i, p in enumerate(problems)]

    def _backends(self) -> dict[str, ModelBackend]:
        """Generation engine per agent (shared by agents on the same model)."""
        config = get_settings()
        return {
            a.id: self.backend_factory(config.model_for(a), config.model_for(a).backend) for a in config.agents.enabled
        }

    def trajectory_path(self, problem: Problem, suffix: str = ".json") -> str:
        return f"{self.generation_dir()}/trajectories/{safe_name(problem.problem_id)}{suffix}"

    def generate(self) -> None:
        """Run every problem not finished yet; uncertainty is measured inside the graph.

        `experiment.concurrent_problems` problems go at once in worker threads, each from a fresh graph state; with
        more than one, their model calls share batched engine calls. The main thread serves the NLI and
        math-verify parts of the uncertainty measurement (`MainThreadCalls`), grades, and saves.
        """
        config = get_settings()
        concurrency = config.experiment.concurrent_problems
        backends = self._backends()
        batchers: dict[int, MicroBatcher] = {}
        if concurrency > 1:
            backends = {a: batchers.setdefault(id(b), MicroBatcher(b)) for a, b in backends.items()}
        main = MainThreadCalls()
        uq = StepUncertainty(backends.__getitem__, main.call)
        self.store.write_json(
            f"{self.generation_dir()}/agents.json",
            [a.describe() for a in build_workflow(backends.__getitem__).agents],
        )

        pending, done = [], 0
        for split, i, problem in self._all_problems():
            rel = self.trajectory_path(problem)
            if config.experiment.resume and self.store.exists(rel):
                if self.store.read_json(rel)["trajectory"]["termination_reason"] != "error":
                    done += 1
                    continue
            pending.append((split, i, problem))
        logger.info("generate", extra={"pending": len(pending), "already_done": done, "concurrent": concurrency})

        def run_one(problem: Problem) -> Trajectory:
            log = self.trajectory_path(problem, ".steps.jsonl")
            self.store.remove(log)
            workflow = build_workflow(
                backends.__getitem__, uq, lambda step: self.store.append_jsonl(log, step.model_dump())
            )
            return workflow.run(problem.problem_id, problem.problem)

        pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="spuma-problem")
        futures = {pool.submit(run_one, p): (split, i, p) for split, i, p in pending}
        try:
            with progress_counter(len(futures), "generate") as advance:
                while futures:
                    main.serve(0.05)
                    for future in [f for f in futures if f.done()]:
                        split, i, problem = futures.pop(future)
                        self._save(split, i, problem, future.result())
                        advance()
        finally:
            main.close()
            pool.shutdown(wait=True, cancel_futures=True)
            for batcher in batchers.values():
                batcher.stop()

    def _save(self, split: str, i: int, problem: Problem, trajectory: Trajectory) -> None:
        failed = trajectory.termination_reason == "error"
        grade = None if failed else self.grader.grade(problem, trajectory.final_answer_text).to_dict()
        record = {
            "split": split,
            "trajectory": trajectory.model_dump(),
            "grade": grade,
        }
        self.store.write_json(self.trajectory_path(problem), record)
        logger.info(
            "generated",
            extra={
                "split": split,
                "index": i,
                "problem_id": problem.problem_id,
                "rounds": trajectory.rounds,
                "termination": trajectory.termination_reason,
                "label": None if failed else grade["label"],
            },
        )

    def score(self) -> None:
        """Features from the measured steps of every finished problem; failed or missing problems are reported,
        not filled in."""
        out_dir = self.scored_dir()
        rows, step_rows, step_records, skipped = [], [], [], []
        problems = self._all_problems()
        for split, _, problem in progress(problems, total=len(problems), desc="score"):
            rel = self.trajectory_path(problem)
            if not self.store.exists(rel):
                skipped.append({"problem_id": problem.problem_id, "reason": "missing"})
                continue
            record = self.store.read_json(rel)
            trajectory = Trajectory.model_validate(record["trajectory"])
            if trajectory.termination_reason == "error":
                skipped.append({"problem_id": problem.problem_id, "reason": trajectory.error})
                continue
            step_rows.extend(step_metric_rows(trajectory))
            step_records.extend(
                {"split": split, "role": a.role, **s.model_dump()} for a in trajectory.agents for s in a.steps
            )
            rows.append(
                {
                    "problem_id": problem.problem_id,
                    "split": split,
                    "features": extract_features(trajectory),
                    "label": record["grade"]["label"],
                    "parse_failure": record["grade"]["parse_failure"],
                    "rounds": trajectory.rounds,
                    "termination_reason": trajectory.termination_reason,
                }
            )
        if skipped:
            logger.warning("problems without features", extra={"count": len(skipped), "problems": skipped[:20]})
        self.store.write_json(f"{out_dir}/step_metrics.json", step_rows)
        self.store.write_jsonl(f"{out_dir}/steps.jsonl", step_records)
        self.store.write_json(f"{out_dir}/features.json", rows)
        self.store.write_json(f"{out_dir}/skipped.json", skipped)
        logger.info("scored", extra={"problems": len(rows), "skipped": len(skipped)})

    def _matrix(self, split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """One row per problem of `split`."""
        rows = [r for r in self.store.read_json(f"{self.scored_dir()}/features.json") if r["split"] == split]
        order = get_settings().features.order
        Z = np.array([[r["features"][f] for f in order] for r in rows], dtype=float).reshape(len(rows), len(order))
        return Z, np.array([r["label"] for r in rows], dtype=int), rows

    def model_path(self, name: str) -> str:
        return f"{self.scored_dir()}/models/{name}{risk_model_spec(name).suffix}"

    def train(self) -> None:
        Z, y, _ = self._matrix("train")
        for name in get_settings().evaluation.risk_models:
            status_rel = f"{self.scored_dir()}/models/{name}_status.json"
            try:
                model = risk_model_spec(name).build().fit(Z, y)
            except ValueError as e:
                logger.error("risk model training failed", extra={"model": name, "error": str(e)})
                self.store.write_json(status_rel, {"status": "failed", "error": str(e)})
                continue
            model.save(self.store.path(self.model_path(name)))
            status = {"status": "trained", "train_rows": len(y), "incorrect": int(y.sum()), **model.summary()}
            self.store.write_json(status_rel, status)
            logger.info("risk model trained", extra={"model": name, **status})
        if get_settings().chmm.enabled:
            CHMMAggregation(self).train()

    def evaluate(self) -> None:
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
            path = self.store.path(self.model_path(name))
            if path.exists():
                models[name] = risk_model_spec(name).load(path)
        result: dict[str, Any] = {"run_id": self.run_id, "splits": {}}
        for split in ("validation", "test"):
            Z, y, rows = self._matrix(split)
            if not len(y):
                continue
            split_res: dict[str, Any] = {"n": len(y), "accuracy": float(1 - y.mean()), "models": {}, "predictions": {}}
            groups = np.array([r["problem_id"] for r in rows])
            for name, model in models.items():
                p = model.predict_error_proba(Z)
                h = binary_entropy(p)
                split_res["models"][name] = evaluate(y, p, ev.metrics, groups, boot, ev.ece_bins)
                split_res["models"][f"{name}_posterior_entropy"] = evaluate(y, h, ranking, groups, boot, ev.ece_bins)
                split_res["predictions"][name] = [
                    {
                        "problem_id": r["problem_id"],
                        "label": int(t),
                        "p_incorrect": float(a),
                        "posterior_entropy": float(b),
                    }
                    for r, t, a, b in zip(rows, y, p, h, strict=True)
                ]
            result["splits"][split] = split_res
        self.store.write_json(f"{self.scored_dir()}/evaluation.json", result)
        test = result["splits"].get("test", {})
        summary = {
            "accuracy": test.get("accuracy"),
            "n": test.get("n"),
            "models": {m: {k: r["value"] for k, r in res.items()} for m, res in test.get("models", {}).items()},
        }
        if get_settings().chmm.enabled:
            chmm = CHMMAggregation(self).evaluate()
            summary["chmm"] = {m: {k: r["value"] for k, r in res.items()} for m, res in chmm["methods"].items()}
        self.store.write_json("summary.json", summary)
        logger.info("evaluation", extra={"test": summary})


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
