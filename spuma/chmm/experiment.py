"""CHMM-weighted aggregation of saved step uncertainties (SAUP-inspired), part of the experiment's train and
evaluate phases (`chmm.enabled`). Post-generation: it reads the saved trajectories and `features.json` (labels,
splits) and regenerates nothing.

    1. steps: every scored agent response of a problem in chronological order (the trajectory's `messages`;
       in a star the coordinator's responses are steps like any other), with its SE, MCSP, NLL and length.
    2. distances (`spuma.chmm.distances`): question drift and interaction gap per response, cached in
       `<scored>/chmm_distances-<fp>.json`.
    3. CHMM (`spuma.chmm.hmm`, hmmlearn `GaussianHMM`): scaler and HMM fitted on the training problems only,
       states ordered low/moderate/high deviation, then frozen; gamma_t(k) for every problem.
    4. weights: w_t = r_t / sum_j r_j, r_t = sum_k lambda_k(alpha) gamma_t(k), lambda(alpha) = [1, 1+a, 1+2a],
       fixed (never fitted to labels) and the same for every architecture and dataset.
    5. methods, all on the same problems, labels and splits:
         rms_se, rms_mcsp, last_se, last_mcsp       standalone scores (no SPN); MCSP is a confidence, so its
                                                    error score is -MCSP
         rms_spn                                    SPN on Z = RMS over all responses of [SE, MCSP, NLL, length]
         chmm_a<alpha>_spn                          SPN on Z = sum_t w_t [SE_t, MCSP_t, NLL_t, length_t]
       One joint SPN per SPN method (the `spn:` section, unchanged), trained on the training split.
       Standalone scores are ranked as they are; their Brier/ECE are reported only after a Platt (logistic)
       calibrator fitted on the training split.
    6. evaluation on the test split: AUROC, AUPRC, AUARC, AURC, Brier, ECE (wrong = 1) with paired
       problem-clustered bootstrap intervals (the same resampled problems for every method), and paired
       differences of the primary CHMM method against RMS SPN, RMS SE and RMS MCSP.

`train` does 1-5 and saves the CHMM, SPNs and calibrators; `evaluate` loads them (never fits) and does 6. An OOD
evaluation calls `evaluate(frozen_from=<the source experiment's chmm directory>)`.

Outputs, in `<scored>/chmm-<fp>/`: chmm.json (fitted CHMM, lambdas), models/, steps.jsonl (distances,
posteriors, weights per step), features.json (aggregated vectors per method), evaluation.json,
diagnostics.json, results.md, results_table.json, alpha.png.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from spuma.agents.trajectory import Trajectory
from spuma.chmm.distances import ESTIMATORS, situational_distances
from spuma.chmm.hmm import STATE_NAMES, SituationalHMM, state_coefficients, step_weights
from spuma.common.config import to_plain
from spuma.common.logs import progress
from spuma.common.utils import safe_name, stable_hash
from spuma.config import get_settings
from spuma.dataset.math.answers import extract_boxed
from spuma.evaluation.metrics import METRICS, expected_calibration_error
from spuma.spn.model import JointSPN

logger = logging.getLogger(__name__)

FEATURES = ("se", "mcsp", "nll", "length")
STANDALONE = ("rms_se", "rms_mcsp", "last_se", "last_mcsp")
METRIC_NAMES = ("auroc", "auprc", "auarc", "aurc", "brier", "ece")
CALIBRATION = ("brier", "ece")
LABELS = {
    "rms_se": "RMS SE",
    "rms_mcsp": "RMS MCSP",
    "last_se": "Last-step SE",
    "last_mcsp": "Last-step MCSP",
    "rms_spn": "RMS SPN",
}


def chmm_method(alpha: float) -> str:
    return f"chmm_a{alpha:g}_spn"


def _rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt((x**2).mean(0))


def chronological_steps(trajectory: Trajectory) -> list[dict[str, Any]]:
    """Every response in the order it was produced (`messages`), each matched to its agent's next step."""
    by_agent = {a.agent_id: a for a in trajectory.agents}
    seen: dict[str, int] = defaultdict(int)
    out = []
    for m in trajectory.messages:
        agent = by_agent[m["agent_id"]]
        step = agent.steps[seen[agent.agent_id]]
        seen[agent.agent_id] += 1
        out.append(
            {
                "agent_id": agent.agent_id,
                "role": agent.role,
                "round": step.round,
                "step_id": step.step_id,
                "text": step.content,
                "metrics": {f: float(step.metrics[f]) for f in FEATURES},
            }
        )
    if sum(seen.values()) != sum(len(a.steps) for a in trajectory.agents):
        raise ValueError(f"{trajectory.problem_id}: messages and steps do not match one to one")
    return out


class CHMMAggregation:
    """The CHMM experiment of one `ExperimentRunner` (its store, generation and scored directories)."""

    def __init__(self, runner: Any) -> None:
        self.runner, self.store = runner, runner.store
        self.config = get_settings().chmm
        self.out = f"{runner.scored_dir()}/chmm-{stable_hash(to_plain(self.config))}"

    @property
    def spn_methods(self) -> list[str]:
        return ["rms_spn", *(chmm_method(a) for a in self.config.alphas)]

    def lambdas(self) -> dict[str, list[float]]:
        return {f"{a:g}": state_coefficients(a).tolist() for a in self.config.alphas}

    def train(self) -> None:
        """Fit the CHMM, one SPN per SPN method and the standalone calibrators, on the training split only."""
        problems = [p for p in self._problems() if p["split"] == "train"]
        if not problems:
            raise RuntimeError("chmm: no training problems")
        hmm = SituationalHMM(self.config.hmm).fit([p["x"] for p in problems])
        hmm.save(self.store.path(f"{self.out}/models/chmm.pkl"))
        described = {**hmm.describe(), "lambdas": self.lambdas(), "train_problems": len(problems)}
        self.store.write_json(f"{self.out}/chmm.json", described)
        logger.info("chmm fitted", extra=described)
        self._weigh(problems, hmm)
        y = np.array([p["label"] for p in problems])
        spn = get_settings().spn
        for m in self.spn_methods:
            model = JointSPN.__new__(JointSPN)
            model._init(spn, list(FEATURES))
            model.fit(self._aggregate(problems, m), y)
            model.save(self.store.path(f"{self.out}/models/{m}.pt"))
            status = {"status": "trained", "train_rows": len(y), "incorrect": int(y.sum()), **model.summary()}
            self.store.write_json(f"{self.out}/models/{m}_status.json", status)
        calibrators = {}
        for m in STANDALONE:
            s = np.array([self._standalone(p, m) for p in problems])
            lr = LogisticRegression(C=1e6).fit(s[:, None], y)
            calibrators[m] = [float(lr.coef_[0, 0]), float(lr.intercept_[0])]
        self.store.write_json(f"{self.out}/models/calibrators.json", calibrators)
        logger.info("chmm models trained", extra={"methods": self.spn_methods, "output": self.out})

    def evaluate(self, frozen_from: Path | None = None) -> dict[str, Any]:
        """The nine methods on the test split, with the CHMM, SPNs and calibrators of `train` (or of
        `frozen_from`, another experiment's chmm directory) loaded and frozen."""
        src = Path(frozen_from) if frozen_from is not None else self.store.path(self.out)
        if not (src / "models" / "chmm.pkl").is_file():
            raise RuntimeError(f"no trained CHMM in {src}: run the train phase first")
        hmm = SituationalHMM.load(src / "models" / "chmm.pkl")
        trained = json.loads((src / "chmm.json").read_text())
        models = {m: JointSPN.load(src / "models" / f"{m}.pt") for m in self.spn_methods}
        calibrators = json.loads((src / "models" / "calibrators.json").read_text())
        if frozen_from is not None:
            self.store.write_json(f"{self.out}/chmm.json", {**trained, "frozen_from": str(src)})

        problems = self._problems()
        self._weigh(problems, hmm)
        checks = self._checks(problems, hmm, trained["lambdas"])
        idx = np.array([p["split"] == "test" for p in problems])
        if not idx.any():
            raise RuntimeError("chmm: no test problems")
        Z = {m: self._aggregate(problems, m) for m in self.spn_methods}
        standalone = {m: np.array([self._standalone(p, m) for p in problems]) for m in STANDALONE}
        y = np.array([p["label"] for p in problems])[idx]
        groups = np.array([p["problem_id"] for p in problems])[idx]
        scores: dict[str, dict[str, np.ndarray]] = {}
        for m in STANDALONE:
            s = standalone[m][idx]
            scores[m] = {"rank": s, "prob": _platt(calibrators[m], s)}
        for m in self.spn_methods:
            p_err = models[m].predict_error_proba(Z[m][idx])
            scores[m] = {"rank": p_err, "prob": p_err}

        evaluation = self._evaluate(y, groups, scores)
        evaluation |= {
            "run_id": self.runner.run_id,
            "frozen_from": None if frozen_from is None else str(src),
            "checks": checks,
            "lambdas": trained["lambdas"],
            "n_test": int(len(y)),
            "test_accuracy": float(1 - y.mean()),
        }
        self.store.write_json(f"{self.out}/evaluation.json", evaluation)
        self._write_records(problems, Z, standalone, scores, idx)
        diagnostics = self._diagnostics([p for p, t in zip(problems, idx, strict=True) if t])
        self.store.write_json(f"{self.out}/diagnostics.json", diagnostics)
        self._report(evaluation, diagnostics, trained)
        logger.info("chmm evaluation", extra={"output": self.out, "n_test": len(y)})
        return evaluation

    # -- data ------------------------------------------------------------------------------------------------

    def _problems(self) -> list[dict[str, Any]]:
        """Every scored problem (features.json order) with its chronological steps and distances."""
        runner = self.runner
        rows = self.store.read_json(f"{runner.scored_dir()}/features.json")
        problems = []
        for r in progress(rows, total=len(rows), desc="chmm load"):
            rel = f"{runner.generation_dir()}/trajectories/{safe_name(r['problem_id'])}.json"
            trajectory = Trajectory.model_validate(self.store.read_json(rel)["trajectory"])
            problems.append(
                {
                    "problem_id": r["problem_id"],
                    "split": r["split"],
                    "label": int(r["label"]),
                    "problem": trajectory.problem,
                    "steps": chronological_steps(trajectory),
                }
            )
        self._distances(problems)
        return problems

    def _distances(self, problems: list[dict[str, Any]]) -> None:
        """Distances per response, computed once per estimator setting and cached (resumable)."""
        rel = f"{self.runner.scored_dir()}/chmm_distances-{stable_hash(to_plain(self.config.distance))}.json"
        cache = self.store.read_json(rel) if self.store.exists(rel) else {}
        todo = [p for p in problems if p["problem_id"] not in cache]
        if todo:
            estimator = ESTIMATORS[self.config.distance.estimator](self.config.distance)
            for i, p in enumerate(progress(todo, total=len(todo), desc="chmm distances")):
                cache[p["problem_id"]] = situational_distances(
                    estimator, p["problem"], [s["agent_id"] for s in p["steps"]], [s["text"] for s in p["steps"]]
                )
                if (i + 1) % 100 == 0:
                    self.store.write_json(rel, cache)
            self.store.write_json(rel, cache)
        for p in problems:
            d = cache[p["problem_id"]]
            for s, ds in zip(p["steps"], d, strict=True):
                s["distance"] = ds
            p["x"] = np.array(
                [[ds["question_drift"], np.nan if ds["interaction_gap"] is None else ds["interaction_gap"]] for ds in d]
            )

    # -- CHMM ------------------------------------------------------------------------------------------------

    def _weigh(self, problems: list[dict[str, Any]], hmm: SituationalHMM) -> None:
        for p, gamma in zip(problems, hmm.posteriors([p["x"] for p in problems]), strict=True):
            p["gamma"] = gamma
            p["weights"] = {a: step_weights(gamma, a) for a in self.config.alphas}

    def _checks(self, problems: list[dict[str, Any]], hmm: SituationalHMM, trained: dict[str, list[float]]) -> dict:
        """Weights sum to 1 per problem; lambdas are the fixed formula and those the CHMM was trained with."""
        worst = max(abs(w.sum() - 1) for p in problems for w in p["weights"].values())
        if worst > 1e-9:
            raise AssertionError(f"CHMM weights do not sum to 1 (worst deviation {worst})")
        for a in self.config.alphas:
            if (
                self.lambdas()[f"{a:g}"] != [1.0, 1.0 + a, 1.0 + 2 * a]
                or trained.get(f"{a:g}") != self.lambdas()[f"{a:g}"]
            ):
                raise AssertionError(f"lambda({a}) is not [1, 1+a, 1+2a] or differs from training")
        if not np.all(np.diff(hmm.deviation) >= 0):
            raise AssertionError("CHMM states are not ordered by deviation")
        return {"max_weight_sum_deviation": float(worst), "lambdas_fixed": True, "states_ordered": True}

    # -- features --------------------------------------------------------------------------------------------

    def _aggregate(self, problems: list[dict[str, Any]], method: str) -> np.ndarray:
        out = []
        for p in problems:
            U = np.array([[s["metrics"][f] for f in FEATURES] for s in p["steps"]])
            if method == "rms_spn":
                out.append(_rms(U))
            else:
                alpha = next(a for a in self.config.alphas if chmm_method(a) == method)
                out.append(p["weights"][alpha] @ U)
        return np.array(out)

    @staticmethod
    def _standalone(p: dict[str, Any], method: str) -> float:
        """Error score, higher = more likely wrong (MCSP is a confidence: negated)."""
        kind, metric = method.split("_")
        values = np.array([s["metrics"][metric] for s in p["steps"]])
        v = float(_rms(values)) if kind == "rms" else float(values[-1])
        return -v if metric == "mcsp" else v

    # -- evaluation ------------------------------------------------------------------------------------------

    def _metric(self, name: str, y: np.ndarray, s: dict[str, np.ndarray]) -> float:
        if name == "ece":
            return expected_calibration_error(y, s["prob"], get_settings().evaluation.ece_bins)
        return METRICS[name](y, s["prob"] if name in CALIBRATION else s["rank"])

    def _evaluate(self, y: np.ndarray, groups: np.ndarray, scores: dict[str, dict[str, np.ndarray]]) -> dict:
        """Point values, paired problem-clustered bootstrap CIs and paired differences."""
        c = self.config
        methods = list(scores)
        point = {m: {k: self._metric(k, y, scores[m]) for k in METRIC_NAMES} for m in methods}
        uniq, inverse = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
        rng = np.random.default_rng(get_settings().project.seed)
        boot = np.full((c.bootstrap_iterations, len(methods), len(METRIC_NAMES)), np.nan)
        for b in progress(range(c.bootstrap_iterations), total=c.bootstrap_iterations, desc="bootstrap"):
            rows = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
            yb = y[rows]
            for i, m in enumerate(methods):
                sb = {k: v[rows] for k, v in scores[m].items()}
                for j, k in enumerate(METRIC_NAMES):
                    if k != "auarc":
                        boot[b, i, j] = self._metric(k, yb, sb)
        boot[:, :, METRIC_NAMES.index("auarc")] = 1 - boot[:, :, METRIC_NAMES.index("aurc")]
        a = (1 - c.confidence) / 2

        def interval(v: np.ndarray) -> list[float]:
            v = v[np.isfinite(v)]
            return [float(np.quantile(v, a)), float(np.quantile(v, 1 - a))] if len(v) else [np.nan, np.nan]

        results = {
            m: {k: {"value": point[m][k], "ci": interval(boot[:, i, j])} for j, k in enumerate(METRIC_NAMES)}
            for i, m in enumerate(methods)
        }
        primary = chmm_method(c.primary_alpha)
        comparisons = {}
        for other in ("rms_spn", "rms_se", "rms_mcsp"):
            i, o = methods.index(primary), methods.index(other)
            comparisons[f"{primary}-{other}"] = {
                k: {
                    "diff": point[primary][k] - point[other][k],
                    "ci": interval(boot[:, i, j] - boot[:, o, j]),
                    # share of resamples where the primary method is *not* better (one-sided bootstrap p)
                    "p_not_better": float(
                        np.mean(
                            (boot[:, i, j] - boot[:, o, j]) <= 0
                            if k in ("auroc", "auprc", "auarc")
                            else (boot[:, i, j] - boot[:, o, j]) >= 0
                        )
                    ),
                    "standalone_calibrated": other in STANDALONE and k in CALIBRATION,
                }
                for j, k in enumerate(METRIC_NAMES)
            }
        return {
            "split": "test",
            "bootstrap": {
                "iterations": c.bootstrap_iterations,
                "confidence": c.confidence,
                "paired": True,
                "cluster_by": "problem_id",
            },
            "methods": results,
            "comparisons": comparisons,
        }

    # -- outputs ---------------------------------------------------------------------------------------------

    def _write_records(self, problems, Z, standalone, scores, idx) -> None:
        steps, features = [], []
        test_pos = {i: k for k, i in enumerate(np.flatnonzero(idx))}
        for i, p in enumerate(problems):
            for t, s in enumerate(p["steps"]):
                steps.append(
                    {
                        "problem_id": p["problem_id"],
                        "split": p["split"],
                        "label": p["label"],
                        "t": t,
                        **{k: s[k] for k in ("agent_id", "role", "round", "step_id", "metrics")},
                        **s["distance"],
                        "gamma": dict(zip(STATE_NAMES, p["gamma"][t].tolist(), strict=True)),
                        "weights": {f"{a:g}": float(w[t]) for a, w in p["weights"].items()},
                    }
                )
            row = {
                "problem_id": p["problem_id"],
                "split": p["split"],
                "label": p["label"],
                "responses": len(p["steps"]),
                "features": {m: dict(zip(FEATURES, Z[m][i].tolist(), strict=True)) for m in Z},
                "standalone_scores": {m: float(standalone[m][i]) for m in STANDALONE},
            }
            if i in test_pos:
                row["p_incorrect"] = {m: float(s["prob"][test_pos[i]]) for m, s in scores.items()}
            features.append(row)
        self.store.write_jsonl(f"{self.out}/steps.jsonl", steps)
        self.store.write_json(f"{self.out}/features.json", features)

    def _diagnostics(self, problems: list[dict[str, Any]]) -> dict[str, Any]:
        """Where the primary-alpha weights go, as relative weight T * w_t (1 = uniform), by step kind and label.

        kinds: each role; first / final response; correction (the agent's boxed answer changed since its previous
        response); disagreement (interaction gap >= its median); uncertain (SE > 0).
        """
        alpha = self.config.primary_alpha
        rows = []
        for p in problems:
            T = len(p["steps"])
            last_box: dict[str, str | None] = {}
            for t, s in enumerate(p["steps"]):
                box = extract_boxed(s["text"])
                prev = last_box.get(s["agent_id"])
                rows.append(
                    {
                        "label": p["label"],
                        "role": s["role"],
                        "first": t == 0,
                        "final": t == T - 1,
                        "correction": box is not None and prev is not None and box != prev,
                        "gap": s["distance"]["interaction_gap"],
                        "drift": s["distance"]["question_drift"],
                        "se": s["metrics"]["se"],
                        "nll": s["metrics"]["nll"],
                        "rel": float(T * p["weights"][alpha][t]),
                        "state": int(np.argmax(p["gamma"][t])),
                        "T": T,
                    }
                )
                if box is not None:
                    last_box[s["agent_id"]] = box
        gaps = [r["gap"] for r in rows if r["gap"] is not None]
        median_gap = float(np.median(gaps)) if gaps else np.nan
        kinds = {f"role={r}": (lambda x, r=r: x["role"] == r) for r in sorted({x["role"] for x in rows})}
        kinds |= {
            "first": lambda x: x["first"],
            "final": lambda x: x["final"],
            "correction": lambda x: x["correction"],
            "disagreement": lambda x: x["gap"] is not None and x["gap"] >= median_gap,
            "uncertain (SE>0)": lambda x: x["se"] > 0,
        }
        multi = [r for r in rows if r["T"] > 1]

        def summary(sel: list[dict[str, Any]]) -> dict[str, Any]:
            out: dict[str, Any] = {"n": len(sel)}
            for name, lab in (("all", None), ("correct", 0), ("wrong", 1)):
                v = [r["rel"] for r in sel if lab is None or r["label"] == lab]
                out[f"rel_weight_{name}"] = float(np.mean(v)) if v else None
            out["state_share"] = {
                STATE_NAMES[k]: float(np.mean([r["state"] == k for r in sel])) if sel else None for k in range(3)
            }
            return out

        corr = {}
        for f in ("drift", "gap", "se", "nll"):
            pairs = [(r["rel"], r[f]) for r in multi if r[f] is not None]
            if len(pairs) > 2:
                rho = spearmanr(*zip(*pairs, strict=True)).statistic
                corr[f] = float(rho)
        return {
            "alpha": alpha,
            "responses": len(rows),
            "responses_in_multi_step_problems": len(multi),
            "median_interaction_gap": median_gap,
            "by_kind": {k: summary([r for r in multi if f(r)]) for k, f in kinds.items()},
            "spearman_rel_weight_vs": corr,
        }

    def _report(self, evaluation: dict[str, Any], diagnostics: dict[str, Any], hmm: dict[str, Any]) -> None:
        settings = get_settings()
        dataset, arch = settings.dataset.name.upper(), settings.agents.workflow
        res = evaluation["methods"]

        def cell(m: str, k: str) -> str:
            if m in STANDALONE and k in CALIBRATION:
                return "—"
            v = res[m][k]
            return f"{v['value']:.4f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"

        lines = [
            f"# CHMM situational weighting — {dataset}, {arch}",
            "",
            f"Test problems: {evaluation['n_test']} (accuracy {evaluation['test_accuracy']:.3f}). Values with "
            f"{self.config.confidence:.0%} paired problem-clustered bootstrap intervals "
            f"({self.config.bootstrap_iterations} resamples). Positive label: wrong = 1.",
            "",
            "| Dataset | Architecture | Method | Alpha | " + " | ".join(k.upper() for k in METRIC_NAMES) + " |",
            "|---|---|---|---|" + "---|" * len(METRIC_NAMES),
        ]
        for m in res:
            alpha = next((f"{a:g}" for a in self.config.alphas if chmm_method(a) == m), "—")
            name = LABELS.get(m, "CHMM + SPN")
            lines.append(
                f"| {dataset} | {arch} | {name} | {alpha} | " + " | ".join(cell(m, k) for k in METRIC_NAMES) + " |"
            )
        lines += [
            "",
            "Standalone scores, Platt-calibrated on the training split (Brier, ECE):",
            "",
            "| Method | Brier (calibrated) | ECE (calibrated) |",
            "|---|---|---|",
        ]
        for m in STANDALONE:
            lines.append(f"| {LABELS[m]} | {cell_cal(res, m, 'brier')} | {cell_cal(res, m, 'ece')} |")
        lines += [
            "",
            f"Paired differences, CHMM α={self.config.primary_alpha:g} + SPN minus baseline:",
            "",
            "| Baseline | " + " | ".join(k.upper() for k in METRIC_NAMES) + " |",
            "|---|" + "---|" * len(METRIC_NAMES),
        ]
        for key, comp in evaluation["comparisons"].items():
            other = key.split("-", 1)[1]
            cells = []
            for k in METRIC_NAMES:
                d = comp[k]
                mark = "†" if d["standalone_calibrated"] else ""
                cells.append(f"{d['diff']:+.4f}{mark} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] p={d['p_not_better']:.3f}")
            lines.append(f"| {LABELS[other]} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "† against the Platt-calibrated standalone score. p: share of resamples where the CHMM method "
            "is not better.",
            "",
            "## CHMM states (training fit, frozen; hmmlearn GaussianHMM, diagonal)",
            "",
            "| State | π | μ question drift | μ interaction gap | σ² drift | σ² gap | deviation |",
            "|---|---|---|---|---|---|---|",
        ]
        for k, s in enumerate(STATE_NAMES):
            mu, var = hmm["means"][k], hmm["variances"][k]
            lines.append(
                f"| {s} | {hmm['startprob'][k]:.3f} | {mu[0]:.3f} | {mu[1]:.3f} | {var[0]:.4f} | {var[1]:.4f} | "
                f"{hmm['deviation'][k]:.3f} |"
            )
        lines += [
            "",
            "Transition matrix A (rows: from low/moderate/high): "
            + "; ".join("[" + ", ".join(f"{v:.3f}" for v in row) + "]" for row in hmm["transmat"])
            + f". Missing interaction gaps filled with the training mean {hmm['missing_fill'][1]:.3f}; "
            f"converged: {hmm['converged']} after {hmm['iterations']} iterations.",
            "",
            f"## Where the α={diagnostics['alpha']:g} weights go (test, multi-response problems)",
            "",
            "Relative weight T·w_t (1 = uniform).",
            "",
            "| Responses | n | all | correct | wrong | share low / moderate / high |",
            "|---|---|---|---|---|---|",
        ]
        for kind, s in diagnostics["by_kind"].items():
            share = " / ".join(f"{v:.2f}" if v is not None else "—" for v in s["state_share"].values())
            vals = " | ".join(
                f"{s[f'rel_weight_{n}']:.3f}" if s[f"rel_weight_{n}"] is not None else "—"
                for n in ("all", "correct", "wrong")
            )
            lines.append(f"| {kind} | {s['n']} | {vals} | {share} |")
        lines += [
            "",
            "Spearman ρ of relative weight with: "
            + ", ".join(f"{k} {v:+.3f}" for k, v in diagnostics["spearman_rel_weight_vs"].items()),
            "",
            f"Checks: {evaluation['checks']}; λ: {evaluation['lambdas']}",
            "",
        ]
        self.store.write_json(f"{self.out}/results_table.json", _table_rows(evaluation, dataset, arch, self.config))
        self.store.path(f"{self.out}/results.md").write_text("\n".join(lines))
        self._plot(evaluation)

    def _plot(self, evaluation: dict[str, Any]) -> None:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        res, alphas = evaluation["methods"], self.config.alphas
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
        for ax, k in zip(axes, ("auroc", "auarc"), strict=True):
            v = np.array([res[chmm_method(a)][k]["value"] for a in alphas])
            ci = np.array([res[chmm_method(a)][k]["ci"] for a in alphas])
            ax.fill_between(alphas, ci[:, 0], ci[:, 1], color="#2a78d6", alpha=0.15, linewidth=0)
            ax.plot(alphas, v, color="#2a78d6", linewidth=2, marker="o", markersize=6, label="CHMM + SPN")
            base = res["rms_spn"][k]["value"]
            ax.axhline(base, color="#52514e", linewidth=1.5, linestyle="--", label="RMS SPN")
            ax.set_xscale("log", base=2)
            ax.set_xticks(alphas, [f"{a:g}" for a in alphas])
            ax.set_xlabel("α")
            ax.set_title(k.upper(), loc="left", fontsize=11)
            ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(colors="#52514e")
        axes[0].legend(frameon=False, fontsize=9)
        fig.savefig(self.store.path(f"{self.out}/alpha.png"), dpi=160)
        plt.close(fig)


def cell_cal(res: dict[str, Any], m: str, k: str) -> str:
    v = res[m][k]
    return f"{v['value']:.4f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"


def _table_rows(evaluation: dict[str, Any], dataset: str, arch: str, config: Any) -> list[dict[str, Any]]:
    """The consolidated table as rows (standalone Brier/ECE are the calibrated ones, flagged)."""
    out = []
    for m, r in evaluation["methods"].items():
        alpha = next((a for a in config.alphas if chmm_method(a) == m), None)
        out.append(
            {
                "dataset": dataset,
                "architecture": arch,
                "method": m,
                "alpha": alpha,
                "calibration_metrics_from_platt": m in STANDALONE,
                **{k: r[k]["value"] for k in METRIC_NAMES},
            }
        )
    return out


def _platt(coef: list[float], s: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(coef[0] * s + coef[1])))
