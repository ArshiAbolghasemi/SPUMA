"""Rebuild the feedback experiment report and figures from its saved joint-SPN evaluation."""

import hashlib
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spuma-report-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from spuma.evaluation.metrics import METRICS, risk_coverage
from spuma.spn.model import JointSPN

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / "artifacts/ma_qwen_math/generation-73c7092a6566/scored-fbc4ac53ebbc"
ASSETS = OUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
AGENTS = {"mathematician": "Mathematician", "enthusiast": "Enthusiast"}
COLORS = {"mathematician": "#2463a2", "enthusiast": "#b05b20", "correct": "#217e60", "incorrect": "#b83c4b"}
LABELS = {"mcsp": "MCSP confidence", "se": "SE (nats)", "nll": "Token NLL (nats/token)"}
rows = json.loads((SOURCE / "features.json").read_text())
steps = json.loads((SOURCE / "step_metrics.json").read_text())
evaluation = json.loads((SOURCE / "evaluation.json").read_text())
status = json.loads((SOURCE / "models/spn_status.json").read_text())
skipped = json.loads((SOURCE / "skipped.json").read_text())
config = json.loads((ROOT / "artifacts/ma_qwen_math/config.json").read_text())
splits = json.loads((ROOT / "artifacts/ma_qwen_math/splits.json").read_text())
model = JointSPN.load(SOURCE / "models/spn.pt")
assert evaluation["run_id"] == "20261017T190125-5cc95f"
assert config["model"]["name"] == "Qwen/Qwen2.5-7B-Instruct"
assert config["agents"]["workflow"] == "feedback"
assert Counter(r["split"] for r in rows) == {"train": 795, "validation": 200, "test": 500}
assert len(skipped) == 5 and status["train_rows"] == 795
assert all("maximum context length" in r["reason"] for r in skipped)
assert len({r["problem_id"] for r in rows}) == len(rows)
by_id = {r["problem_id"]: r for r in rows}
grouped = defaultdict(list)
for s in steps:
    assert s["problem_id"] in by_id
    assert all(np.isfinite(v) for v in s["metrics"].values())
    grouped[s["problem_id"], s["agent_id"]].append(s)
for r in rows:
    for a in AGENTS:
        records = sorted(grouped[r["problem_id"], a], key=lambda s: s["step_id"])
        assert len(records) == r["rounds"]
        assert [s["step_id"] for s in records] == list(range(len(records)))
        for m in ("mcsp", "se", "nll", "length"):
            values = np.array([s["metrics"][m] for s in records])
            actual = values.mean() if m == "length" else np.sqrt(np.mean(values**2))
            assert np.isclose(actual, r["features"][f"{a}_{m}"])
split_ids = {k: {p["problem_id"] for p in v} for k, v in splits.items()}
split_texts = {k: {re.sub(r"\s+", " ", p["problem"]).strip() for p in v} for k, v in splits.items()}
for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
    assert not split_ids[a] & split_ids[b]
    assert not split_texts[a] & split_texts[b]
assert all(r["problem_id"] in split_ids[r["split"]] for r in rows)
test = [r for r in rows if r["split"] == "test"]
test_ids = {r["problem_id"] for r in test}
test_steps = [s for s in steps if s["problem_id"] in test_ids]
predictions = {r["problem_id"]: r for r in evaluation["splits"]["test"]["predictions"]["spn"]}
y = np.array([r["label"] for r in test])
p = np.array([predictions[r["problem_id"]]["p_incorrect"] for r in test])
assert all(predictions[r["problem_id"]]["label"] == r["label"] for r in test)
Z = np.array([[r["features"][f] for f in model.feature_names] for r in test])
np.testing.assert_allclose(model.predict_error_proba(Z), p, atol=1e-10)
train_Z = np.array([[r["features"][f] for f in model.feature_names] for r in rows if r["split"] == "train"])
np.testing.assert_allclose(model.mean, train_Z.mean(0), atol=1e-10)
np.testing.assert_allclose(model.std, np.where(train_Z.std(0) > 0, train_Z.std(0), 1), atol=1e-10)
metrics = evaluation["splits"]["test"]["models"]["spn"]
for name in ("auroc", "auprc", "brier", "ece", "nll", "aurc", "auarc"):
    assert np.isclose(METRICS[name](y, p), metrics[name]["value"])
assert y.sum() == 125 and sum(r["parse_failure"] for r in test) == 1
round_counts = Counter(r["rounds"] for r in test)
assert round_counts == {1: 497, 2: 2, 4: 1}


def semantic_children(module):
    for child in module.children():
        if type(child).__name__ in ("Cat", "ModuleList"):
            yield from semantic_children(child)
        else:
            yield child


nodes, edges, seen = [], [], {}
variable_names = model.feature_names + ["Y_incorrect"]


def export_node(module):
    if id(module) in seen:
        return seen[id(module)]
    node_id = f"n{len(nodes):03}"
    seen[id(module)] = node_id
    scope = list(module.scope.query)
    node = {
        "id": node_id,
        "type": type(module).__name__,
        "scope": scope,
        "variables": [variable_names[i] for i in scope],
    }
    nodes.append(node)
    children = list(semantic_children(module))
    if node["type"] == "Sum":
        weights = module.weights.detach().numpy().reshape(-1)
        assert len(weights) == len(children) and np.isclose(weights.sum(), 1)
        node["weights"] = weights.tolist()
        assert all(set(c.scope.query) == set(scope) for c in children)
    elif node["type"] == "Product":
        scopes = [set(c.scope.query) for c in children]
        assert sum(map(len, scopes)) == len(set().union(*scopes)) == len(scope)
    elif node["type"] == "HistogramLeaf":
        node["edges_standardized"] = module.edges.detach().tolist()
        node["cell_probabilities"] = module.log_probs.detach().exp().tolist()
    elif node["type"] == "BinaryIndicatorLeaf":
        node["label_probabilities_0_1"] = module.log_weights.detach().exp().tolist()
    else:
        raise AssertionError(node["type"])
    for i, child in enumerate(children):
        edge = {"source": node_id, "target": export_node(child)}
        if "weights" in node:
            edge["weight"] = node["weights"][i]
        edges.append(edge)
    return node_id


root_id = export_node(model.root)
counts = Counter(n["type"] for n in nodes)
assert counts == {"Product": 72, "Sum": 46, "HistogramLeaf": 75, "BinaryIndicatorLeaf": 25}
root_children = list(semantic_children(model.root))
assert [list(c.scope.query) for c in root_children] == [[0, 1, 2, 3, 7, 8], [4, 6], [5]]
perturbed = Z[:20].copy()
perturbed[:, [4, 5, 6]] += [0.01, 0.1, 0.1]
np.testing.assert_allclose(model.predict_error_proba(perturbed), p[:20], atol=1e-10)
leaf_occurrences = Counter(i for n in nodes if n["type"].endswith("Leaf") for i in n["scope"])
structure = {
    "run_id": evaluation["run_id"],
    "checkpoint": str((SOURCE / "models/spn.pt").relative_to(ROOT)),
    "sha256": hashlib.sha256((SOURCE / "models/spn.pt").read_bytes()).hexdigest(),
    "root": root_id,
    "counts": dict(counts),
    "feature_order": model.feature_names,
    "mean": model.mean.tolist(),
    "std": model.std.tolist(),
    "nodes": nodes,
    "edges": edges,
}
(OUT / "trained_spn.json").write_text(json.dumps(structure, indent=2) + "\n")
dot = ["digraph trained_spn {", "rankdir=TB;"]
for n in nodes:
    label = f"{n['id']} {n['type']}\n" + ", ".join(f"F{i + 1}" if i < 8 else "Y" for i in n["scope"])
    dot.append(f"{n['id']} [label={json.dumps(label)}];")
for e in edges:
    weight = f' [label="{e["weight"]:.6f}"]' if "weight" in e else ""
    dot.append(f"{e['source']} -> {e['target']}{weight};")
(OUT / "trained_spn.dot").write_text("\n".join(dot + ["}"]) + "\n")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def save(fig, name):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(ASSETS / f"{name}.{suffix}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def ecdf(ax, values, label, color, linestyle="-"):
    values = np.sort(values)
    if len(values):
        ax.step(
            np.r_[values[0], values],
            np.r_[0, np.arange(1, len(values) + 1) / len(values)],
            where="post",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
        )
    ax.set_ylim(0, 1.03)
    ax.grid(alpha=0.18)


def box(ax, x, yy, w, h, title, body, color="#405264", fill="#f4f6f8"):
    ax.add_patch(
        FancyBboxPatch(
            (x, yy),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=.08",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.3,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2, yy + h - 0.28, title, ha="center", va="center", fontsize=11, weight="bold", color=color, zorder=4
    )
    ax.text(x + w / 2, yy + h / 2 - 0.18, body, ha="center", va="center", fontsize=9, linespacing=1.45, zorder=4)


def arrow(ax, points, dashed=False):
    if len(points) > 2:
        ax.plot(
            *zip(*points[:-1], strict=True), color="#536273", linewidth=1.2, linestyle="--" if dashed else "-", zorder=1
        )
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=12,
            color="#536273",
            linewidth=1.2,
            linestyle="--" if dashed else "-",
            zorder=1,
        )
    )


fig, ax = plt.subplots(figsize=(13.5, 7.6))
ax.set(xlim=(0, 16), ylim=(0, 9.7))
ax.axis("off")
arrow(ax, [(2.7, 7.9), (4.0, 7.9)])
arrow(ax, [(7.0, 7.9), (8.2, 7.9)])
arrow(ax, [(11.2, 7.9), (12.2, 7.9)])
arrow(ax, [(13.7, 8.5), (13.7, 9.15), (5.5, 9.15), (5.5, 8.5)])
arrow(ax, [(13.7, 7.3), (13.7, 5.8)])
arrow(ax, [(4.0, 7.6), (3.35, 7.6), (3.35, 2.4)], True)
arrow(ax, [(8.2, 7.6), (7.6, 7.6), (7.6, 3.5), (4.35, 3.5), (4.35, 2.4)], True)
arrow(ax, [(5.0, 1.65), (6.3, 1.65)])
arrow(ax, [(9.3, 1.65), (11.1, 1.65)])
box(ax, 0.4, 7.3, 2.3, 1.2, "MATH problem", "Question text only")
box(ax, 4.0, 7.3, 3.0, 1.2, "Mathematician", "Qwen2.5-7B-Instruct\nSolve / revise", COLORS["mathematician"], "#edf4fb")
box(
    ax,
    8.2,
    7.3,
    3.0,
    1.2,
    "Enthusiast",
    "Qwen2.5-7B-Instruct\nAsk / critique / accept",
    COLORS["enthusiast"],
    "#fff3e9",
)
box(ax, 12.2, 7.3, 3.0, 1.2, "Stop condition", "TASK_DONE or\nround limit (10)")
box(ax, 12.2, 4.4, 3.0, 1.4, "Answer to grade", "Last Mathematician response\nMath-Verify vs reference")
box(
    ax,
    1.0,
    0.9,
    4.0,
    1.5,
    "Step measurements",
    "10 frozen-context alternatives\nMCSP + weighted SE + token NLL\nMean generated length recorded",
)
box(ax, 6.3, 0.9, 3.0, 1.5, "Eight input features", "4 per agent\nRMS: MCSP / SE / NLL\nMean: token length")
box(
    ax,
    11.1,
    0.9,
    4.0,
    1.5,
    "Joint SPN",
    "Learn P(Z, Y) on training data\nOutput P(final answer incorrect | Z)\nEvaluate on held-out problems",
)
ax.text(9.7, 9.32, "Continue with full conversation when not done and round < 10", ha="center", fontsize=9)
ax.text(14.0, 6.4, "Stop", fontsize=9)
ax.text(
    4.0,
    4.5,
    "Frozen-context samples\nnever enter the dialogue.\nNo tool-execution loop.",
    fontsize=10,
    color="#405264",
)
ax.text(
    0.4, 0.28, "Solid arrows: conversation and stopping. Dashed arrows: measurement only; no UQ feedback.", fontsize=9
)
save(fig, "workflow")

fig, ax = plt.subplots(figsize=(12.8, 6.3))
ax.set(xlim=(0, 16), ylim=(0, 8))
ax.axis("off")
arrow(ax, [(6.8, 6.7), (6.8, 6.25), (3.8, 6.25), (3.8, 5.6)])
arrow(ax, [(8.0, 6.7), (8.0, 6.25), (10.15, 6.25), (10.15, 5.6)])
arrow(ax, [(9.2, 6.7), (9.2, 6.45), (13.65, 6.45), (13.65, 5.6)])
weights = root_children[0].weights.detach().numpy().reshape(-1)
arrow(ax, [(2.8, 3.7), (2.8, 2.5)])
arrow(ax, [(4.8, 3.7), (4.8, 3.05), (6.8, 3.05), (6.8, 2.5)])
box(ax, 6.0, 6.7, 4.0, 1.0, "Root product", "Joint density over F1 ... F8 and Y")
box(
    ax,
    0.6,
    3.7,
    6.4,
    1.9,
    "Sum: label-dependent block",
    "F1, F2, F3, F4, F8, Y\nMathematician metrics and length\n+ Enthusiast mean length",
    COLORS["mathematician"],
    "#edf4fb",
)
box(ax, 8.4, 3.7, 3.5, 1.9, "Sum: separate block", "F5, F7\nEnthusiast MCSP and NLL\nCancels in P(Y | Z)")
box(ax, 12.4, 3.7, 2.8, 1.9, "Histogram leaf", "F6: Enthusiast SE\nSeparate root factor\nCancels in P(Y | Z)")
box(
    ax,
    1.0,
    0.9,
    3.6,
    1.6,
    "Learned subtree A",
    f"Mixture weight {weights[0]:.4f}\nProducts / sums / repeated\nfeature and label leaves",
)
box(
    ax,
    5.0,
    0.9,
    3.6,
    1.6,
    "Learned subtree B",
    f"Mixture weight {weights[1]:.4f}\nProducts / sums / repeated\nfeature and label leaves",
)
ax.text(
    9.4,
    1.75,
    "Full checkpoint: 218 statistical modules\n46 sums + 72 products\n75 histogram + 25 binary label leaves",
    fontsize=10,
    linespacing=1.7,
)
ax.text(
    0.6,
    0.2,
    "Overview: deeper subtrees are collapsed; the full topology and learned parameters are exported separately.",
    fontsize=9,
)
save(fig, "trained_spn")

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
for label, name, color in ((0, "Correct", COLORS["correct"]), (1, "Incorrect", COLORS["incorrect"])):
    values = p[y == label]
    axes[0].hist(
        values,
        bins=np.linspace(0, 1, 16),
        weights=np.ones(len(values)) / len(values),
        histtype="step",
        linewidth=1.8,
        label=f"{name} (n={len(values)})",
        color=color,
    )
axes[0].set(
    xlabel="Predicted error probability", ylabel="Fraction within correctness group", title="Risk-score separation"
)
fpr, tpr, _ = roc_curve(y, p)
axes[1].plot(fpr, tpr, color=COLORS["mathematician"], label=f"Joint SPN: AUROC {roc_auc_score(y, p):.3f}")
se_baseline = np.array([r["features"]["mathematician_se"] for r in test])
se_auc = roc_auc_score(y, se_baseline)
se_ap = average_precision_score(y, se_baseline)
fpr_se, tpr_se, _ = roc_curve(y, se_baseline)
axes[1].plot(fpr_se, tpr_se, color=COLORS["enthusiast"], linestyle="-.", label=f"Mathematician SE: {se_auc:.3f}")
axes[1].plot([0, 1], [0, 1], "--", color="#777777", label="Random ranking")
axes[1].set(
    xlabel="False-positive rate",
    ylabel="True-positive rate",
    title="ROC: detecting incorrect answers",
    xlim=(0, 1),
    ylim=(0, 1.03),
)
precision, recall, _ = precision_recall_curve(y, p)
axes[2].step(
    recall,
    precision,
    where="post",
    color=COLORS["mathematician"],
    label=f"Joint SPN: AP {average_precision_score(y, p):.3f}",
)
precision_se, recall_se, _ = precision_recall_curve(y, se_baseline)
axes[2].step(
    recall_se,
    precision_se,
    where="post",
    color=COLORS["enthusiast"],
    linestyle="-.",
    label=f"Mathematician SE: AP {se_ap:.3f}",
)
axes[2].axhline(y.mean(), linestyle="--", color="#777777", label=f"Prevalence: {y.mean():.3f}")
axes[2].set(
    xlabel="Recall of incorrect answers", ylabel="Precision", title="Precision-recall", xlim=(0, 1), ylim=(0, 1.03)
)
for ax in axes:
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "discrimination")

fig, axes = plt.subplots(2, 3, figsize=(11.8, 7))
feature_stats, raw_scores = [], []
for i, (a, name) in enumerate(AGENTS.items()):
    for j, m in enumerate(LABELS):
        values = np.array([r["features"][f"{a}_{m}"] for r in test])
        for label, desc, color, ls in (
            (0, "Correct", COLORS["correct"], "-"),
            (1, "Incorrect", COLORS["incorrect"], "--"),
        ):
            ecdf(axes[i, j], values[y == label], desc, color, ls)
        axes[i, j].set(xlabel=LABELS[m], title=f"{name} | all-step RMS")
        if j == 0:
            axes[i, j].set_ylabel("Cumulative fraction within group")
        axes[i, j].legend(fontsize=8)
        score = 1 - values if m == "mcsp" else values
        raw_scores.append(
            [
                name,
                m.upper(),
                f"{values[y == 0].mean():.3f}",
                f"{values[y == 1].mean():.3f}",
                f"{roc_auc_score(y, score):.3f}",
            ]
        )
        feature_stats.append(
            [name, m.upper(), f"{values.mean():.3f}", f"{np.median(values):.3f}", f"{100 * np.mean(values == 0):.1f}"]
        )
fig.tight_layout()
save(fig, "feature_distributions")

fig, axes = plt.subplots(2, 3, figsize=(11.8, 7))
for j, m in enumerate(LABELS):
    for a, name in AGENTS.items():
        final = [max(grouped[r["problem_id"], a], key=lambda s: s["step_id"])["metrics"][m] for r in test]
        aggregate = [r["features"][f"{a}_{m}"] for r in test]
        ecdf(axes[0, j], final, name, COLORS[a])
        ecdf(axes[1, j], aggregate, name, COLORS[a])
    for i in (0, 1):
        axes[i, j].set(xlabel=LABELS[m], title="Final response" if i == 0 else "All-step RMS")
        axes[i, j].legend(fontsize=8)
        if j == 0:
            axes[i, j].set_ylabel("Cumulative fraction")
fig.tight_layout()
save(fig, "completed_trajectories")

active_rounds = sorted({s["round"] for s in test_steps})
fig, axes = plt.subplots(3, len(active_rounds), figsize=(12, 8.8), squeeze=False)
for i, m in enumerate(LABELS):
    limits = [s["metrics"][m] for s in test_steps]
    for j, round_num in enumerate(active_rounds):
        ax = axes[i, j]
        n = sum(r["rounds"] >= round_num for r in test)
        for a, name in AGENTS.items():
            values = [s["metrics"][m] for s in test_steps if s["round"] == round_num and s["agent_id"] == a]
            ecdf(ax, values, name, COLORS[a])
        cohort_label = "problem" if n == 1 else "problems"
        ax.set(
            xlabel=LABELS[m],
            title=f"Round {round_num} | {n} {cohort_label}",
            xlim=(min(limits) - 0.02, max(limits) + 0.03),
        )
        if j == 0:
            ax.set_ylabel("Cumulative fraction")
        if i == 0:
            ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "round_distributions")

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4))
idx = np.minimum((p * 15).astype(int), 14)
for b in sorted(set(idx)):
    mask = idx == b
    axes[0].scatter(p[mask].mean(), y[mask].mean(), s=20 + 3 * mask.sum(), color=COLORS["mathematician"], alpha=0.75)
axes[0].plot([0, 1], [0, 1], "--", color="#777777")
axes[0].set(
    xlabel="Mean predicted error probability",
    ylabel="Observed error fraction",
    title="Reliability: 15 fixed-width bins",
    xlim=(0, 1),
    ylim=(0, 1),
)
coverage, risk = risk_coverage(y, p)
axes[1].plot(coverage, risk, color=COLORS["mathematician"], label=f"Joint SPN: AURC {risk.mean():.3f}")
axes[1].axhline(y.mean(), linestyle="--", color="#777777", label="Full-test error prevalence")
axes[1].set(
    xlabel="Coverage: fraction of answers retained",
    ylabel="Error rate among retained answers",
    title="Retain lowest-risk answers first",
    xlim=(0, 1),
)
for ax in axes:
    ax.grid(alpha=0.18)
axes[1].legend(fontsize=8)
fig.tight_layout()
save(fig, "calibration_coverage")


def tex_escape(text):
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(c, c) for c in str(text))


class Report:
    """Write the same report blocks to Markdown, LaTeX, and plain text."""

    def __init__(self):
        self.md, self.tex, self.text = [], [], []
        self.figure_number = 0

    def title(self, text):
        self.md += [
            "# " + text,
            "",
            "[PDF](report.pdf) | [LaTeX](report.tex) | [Plain text](report.text) | [Workflow diagram](workflow.html)",
            "",
        ]
        self.tex += [
            r"\title{\textbf{" + tex_escape(text) + "}}",
            r"\author{SPUMA experiment}",
            r"\date{Evaluation: 17 October 2026; report: 18 October 2026}",
            r"\maketitle",
        ]
        self.text += [text.upper(), ""]

    def section(self, text):
        self.md += ["## " + text, ""]
        self.tex += [r"\section{" + tex_escape(text) + "}"]
        self.text += [text.upper(), ""]

    def paragraph(self, text):
        self.md += [text, ""]
        self.tex += [tex_escape(text), ""]
        self.text += [text, ""]

    def equation(self, latex, plain):
        self.md += ["$$", latex, "$$", ""]
        self.tex += [r"\begin{equation*}", latex, r"\end{equation*}"]
        self.text += [plain, ""]

    def code(self, text):
        self.md += ["```text", text, "```", ""]
        self.tex += [r"\begin{verbatim}", text, r"\end{verbatim}"]
        self.text += [text, ""]

    def table(self, headers, data, widths=None):
        data = [[str(v) for v in row] for row in data]
        self.md += ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        self.md += ["| " + " | ".join(row) + " |" for row in data] + [""]
        widths = widths or [0.92 / len(headers)] * len(headers)
        spec = "".join(r">{\raggedright\arraybackslash}p{" + f"{w:.3f}" + r"\linewidth}" for w in widths)
        self.tex += [
            r"\begin{center}\small\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{" + spec + "}",
            r"\toprule",
            " & ".join(map(tex_escape, headers)) + r" \\\midrule",
        ]
        self.tex += [" & ".join(map(tex_escape, row)) + r" \\" for row in data]
        self.tex += [r"\bottomrule\end{tabular}\end{center}"]
        self.text += [" | ".join(headers)] + [" | ".join(row) for row in data] + [""]

    def figure(self, name, caption):
        self.figure_number += 1
        title = f"Figure {self.figure_number}. {caption}"
        self.md += [f"![{html.escape(caption, quote=True)}](assets/{name}.png)", "", title, ""]
        self.tex += [
            r"\begin{figure}[p]\centering",
            r"\includegraphics[width=\linewidth]{assets/" + name + ".pdf}",
            r"\caption{" + tex_escape(caption) + "}",
            r"\end{figure}",
        ]
        self.text += [title, f"Image: assets/{name}.png", ""]

    def write(self):
        preamble = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype}
\usepackage[margin=23mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,array,graphicx}
\usepackage[dvipsnames]{xcolor}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\emergencystretch}{3em}
\widowpenalty=10000
\clubpenalty=10000
\begin{document}
"""
        (OUT / "report.md").write_text("\n".join(self.md) + "\n")
        (OUT / "report.tex").write_text(preamble + "\n".join(self.tex) + "\n\\end{document}\n")
        for suffix in ("text", "txt"):
            (OUT / f"report.{suffix}").write_text("\n".join(self.text) + "\n")


r = Report()
r.title("SPUMA: Qwen2.5-7B Mathematical Feedback Experiment")
r.section("Introduction")
r.paragraph(
    "This experiment asks whether uncertainty measured during a two-agent mathematical conversation predicts "
    "an incorrect final answer. A Master Mathematician proposes and revises a solution; a Curious Math Enthusiast "
    "asks questions, gives feedback, and decides when the discussion is complete. Both roles use "
    "Qwen/Qwen2.5-7B-Instruct. A joint sum-product network (SPN) estimates error risk after the conversation."
)
r.paragraph(
    "On 500 held-out test problems, final-answer accuracy is 75.0% (375 correct, 125 incorrect). "
    "The joint SPN achieves AUROC 0.9165 and average precision (reported as AUPRC) 0.7555. "
    "Although up to ten feedback rounds are allowed, 497/500 test problems stop after the first exchange. "
    "Consequently, this run primarily evaluates one Mathematician response followed by one Enthusiast response; "
    "it does not establish an accuracy benefit from repeated feedback."
)
r.paragraph(
    f"An exploratory single-signal comparison finds that Mathematician RMS semantic entropy alone "
    f"achieves AUROC {se_auc:.4f} and average precision {se_ap:.4f}, above the joint SPN on this test set. "
    "The present run therefore does not demonstrate improved error ranking from combining the eight "
    "inputs in the SPN. Raw SE is a ranking score, whereas the SPN also produces probability estimates."
)
r.paragraph(
    "The report covers only the joint SPN. All numeric results and plots are calculated from saved artifacts; "
    "no language-model generation, SPN retraining, hyperparameter search, or new bootstrap run is performed. "
    "Reported confidence intervals are those stored with the completed evaluation."
)

r.section("Dataset, usable samples, and target")
r.paragraph(
    "The dataset is EleutherAI/hendrycks_math on Hugging Face, covering algebra, counting and probability, "
    "geometry, intermediate algebra, number theory, prealgebra, and precalculus. Stable problem identifiers "
    "are sorted and shuffled with seed 42. Training and validation are disjoint draws from the official train "
    "split; test problems come from the official test split."
)
r.table(
    ["Partition", "Requested", "Usable", "Incorrect", "Accuracy"],
    [
        [
            sp.title(),
            len(splits[sp]),
            sum(x["split"] == sp for x in rows),
            sum(x["label"] for x in rows if x["split"] == sp),
            f"{100 * (1 - np.mean([x['label'] for x in rows if x['split'] == sp])):.2f}%",
        ]
        for sp in ("train", "validation", "test")
    ],
)
r.paragraph(
    "Five training trajectories failed when their prompts exceeded the 8,192-token context window. "
    "They were excluded, leaving 795 SPN training examples rather than the requested 800. "
    "All 200 validation and 500 test problems have usable features. Such exclusions can bias the training "
    "distribution toward shorter or easier conversations and must be disclosed."
)
r.equation(
    r"Y_i=\begin{cases}0,&\text{final Mathematician answer correct},\\"
    r"1,&\text{final Mathematician answer incorrect}.\end{cases}",
    "Y = 0 for a correct final Mathematician answer; Y = 1 otherwise.",
)
r.paragraph(
    "The last Mathematician response is graded, not the Enthusiast response. The evaluator extracts its last "
    "boxed answer and compares it with the reference through Math-Verify; absent boxed answers are incorrect. "
    "One test answer (0.2%) has a parse failure and remains in the main analysis. If the reference cannot be "
    "parsed, grading falls back to normalized string equality. This evaluates final answers, not every "
    "reasoning statement or proof step."
)

r.section("Multi-agent feedback workflow")
r.paragraph(
    "Each round has two responses. The Mathematician receives the original problem and all previous dialogue "
    "and is asked to solve accurately, explain clearly, respond to feedback, and use a boxed final answer. "
    "The Enthusiast sees the problem and latest solution, asks for clarification or points out concerns, "
    "and is instructed not to solve the problem independently. Each role sees its own previous messages "
    "as assistant messages and the other role's messages as user inputs."
)
r.paragraph(
    "The LangGraph workflow stops when the Enthusiast response contains TASK_DONE, or after ten rounds. "
    "Uncertainty is measured after each response but does not enter the dialogue or determine stopping. "
    "The Monte Carlo alternatives do not become additional conversation turns. This feedback run has no "
    "mathematical tool-execution loop, unlike the earlier Solver-Reviewer experiment."
)
r.figure(
    "workflow",
    "Feedback workflow and measurement path. Solid arrows carry the conversation and stopping decision; "
    "dashed arrows carry recorded responses to uncertainty measurement. "
    "Continued dialogue returns to the Mathematician. "
    "The final Mathematician response is graded separately against the reference. "
    "The lower path aggregates eight features "
    "and queries the joint SPN; neither reference answers nor predicted risk are supplied to the agents.",
)
r.paragraph(
    "All usable trajectories terminate with TASK_DONE. Test round counts are 497 at one round, two at two "
    "rounds, and one at four rounds. Mean test length is 1.010 rounds, or 2.020 agent responses. There are "
    "1,010 observed test responses and 3,022 responses across all usable partitions. A completion marker "
    "indicates the Enthusiast's stopping decision, not a guarantee of correctness."
)
r.table(
    ["Setting", "Saved value"],
    [
        ["Language model", "Qwen/Qwen2.5-7B-Instruct"],
        ["Generation and token scoring", "vLLM; scoring backend same"],
        ["Precision / tensor parallelism", "bfloat16 / one shard"],
        ["Sampling", "Temperature 0.7; top-p 1.0; no top-k truncation"],
        ["Maximum continuation / context", "2,048 / 8,192 tokens"],
        ["Concurrent problems", "8"],
        ["Monte Carlo alternatives", "10 per response, from its frozen context"],
        ["NLI fallback", "tasksource/ModernBERT-large-nli; batch size 32"],
        ["NLI limits", "8,000 characters per text; 2,048 tokens per pair"],
        ["Seed / Hub revisions", "42 / model and tokenizer revisions unpinned"],
    ],
    [0.34, 0.58],
)

r.section("Step-level confidence and uncertainty measures")
r.paragraph(
    "This run records Monte Carlo sequence probability (MCSP), likelihood-weighted semantic entropy (SE), "
    "token negative log-likelihood (NLL), and generated token length. It does not record full-vocabulary "
    "predictive entropy (PE). MCSP is a confidence score: larger means more probable responses. SE and NLL "
    "are uncertainty scores: larger means more disagreement or less probable realized tokens. "
    "All logarithms are natural and token scores use the raw model distribution."
)
r.paragraph(
    "For context c and response y of length L, the implementation defines a length-normalized sequence "
    "score q(y) by the geometric mean of token probabilities. This score lies between zero and one, "
    "but is not a normalized probability distribution over all possible response strings."
)
r.equation(
    r"q(y\mid c)=\exp\!\left(\frac{1}{L}\sum_{t=1}^{L}\log p_\theta(y_t\mid c,y_{<t})\right),\qquad "
    r"\mathrm{NLL}(y)=-\frac{1}{L}\sum_{t=1}^{L}\log p_\theta(y_t\mid c,y_{<t}).",
    "q(y|c) = exp(mean token log probability); NLL(y) = -mean token log probability.",
)
r.paragraph(
    "MCSP averages q over the actual response and its ten independently seeded alternatives. It "
    "summarizes sampled response confidence, rather than entropy over the next-token vocabulary."
)
r.equation(
    r"\mathrm{MCSP}=\frac{1}{11}\sum_{j=0}^{10}q(y^{(j)}\mid c).",
    "MCSP = mean q over the actual response and 10 alternatives.",
)
r.paragraph(
    "SE clusters these eleven responses by meaning. If both have boxed answers, normalized equality "
    "or Math-Verify decides equivalence; exactly one boxed answer means different responses; neither "
    "boxed answer triggers NLI entailment in either direction. No reference solution is involved. "
    "The new run uses UQLM's tokenprob entropy: cluster masses are sums of q, not unweighted cluster counts."
)
r.equation(
    r"w_k=\sum_{j\in C_k}q(y^{(j)}\mid c),\qquad "
    r"\pi_k=\frac{w_k}{\sum_\ell w_\ell},\qquad \mathrm{SE}=-\sum_k\pi_k\log\pi_k.",
    "Cluster weight = sum of member q scores; normalize cluster weights; SE = -sum pi*log(pi).",
)
r.paragraph(
    "UQLM compares responses with existing cluster representatives, and approximate comparisons may "
    "permit overlapping memberships; the denominator sums the resulting cluster weights. This is a "
    "sample-based estimate, not exact entropy over all mathematical solutions. Eleven responses give "
    "at most eleven clusters and an upper bound ln(11), approximately 2.398 nats. Zero SE means observed "
    "semantic agreement, which may still be agreement on an incorrect answer."
)
r.paragraph(
    "The completion marker is removed from scored text and rescored on the generating backend when "
    "other text remains. A marker-only response retains its generated tokens. Token length corresponds "
    "to this scoring view. This avoids treating the routine TASK_DONE string as substantive content "
    "when a nonempty response remains, but marker-only decisions can still affect the signals."
)

r.section("Exactly what enters the SPN")
r.paragraph(
    "Each problem produces one eight-dimensional vector Z: four aggregates from each agent. "
    "MCSP, SE, and token NLL use root mean square (RMS) across that agent's observed responses; "
    "token length uses an arithmetic mean. RMS gives high-valued episodes more influence than a mean. "
    "For MCSP this emphasizes high-confidence episodes, not high-uncertainty episodes."
)
r.equation(
    r"U_{ia}^{(m)}=\sqrt{\frac{1}{T_{ia}}\sum_{s=1}^{T_{ia}}(u_{ias}^{(m)})^2},\qquad "
    r"\overline L_{ia}=\frac{1}{T_{ia}}\sum_{s=1}^{T_{ia}}L_{ias}.",
    "RMS metric = sqrt(mean squared step scores); length feature = mean generated token length.",
)
r.table(
    ["Input", "Agent", "Measurement", "Aggregation"],
    [
        [
            f"F{i + 1}",
            AGENTS[name.split("_")[0]],
            {"mcsp": "MCSP confidence", "se": "Semantic entropy", "nll": "Token NLL", "length": "Token length"}[
                name.split("_")[-1]
            ],
            "Mean" if name.endswith("length") else "RMS",
        ]
        for i, name in enumerate(model.feature_names)
    ],
    [0.10, 0.25, 0.34, 0.23],
)
r.code(
    "Feature order in the saved checkpoint:\n"
    + "\n".join(f"F{i + 1}: {name}" for i, name in enumerate(model.feature_names))
)
r.paragraph(
    "Raw problem text, reference solutions, final-answer correctness, problem identifiers, split labels, "
    "round count, and termination reason are not SPN prediction inputs. Y is an additional observed "
    "variable during training only. The joint training table therefore has nine columns: eight features "
    "plus Y. At prediction time, the true label is unknown and the model evaluates both hypothetical "
    "label values to calculate error probability."
)
r.equation(
    r"U_{\mathrm{SPUMA}}(Z)=P(Y=1\mid Z)=\frac{P(Z,Y=1)}{P(Z,Y=0)+P(Z,Y=1)}.",
    "SPUMA error-risk score = P(Y=1|Z) = P(Z,Y=1) / [P(Z,Y=0)+P(Z,Y=1)].",
)
r.paragraph(
    "This final score expresses the model's estimated risk that the completed answer is wrong. "
    "It differs from response-level SE, and from posterior entropy H(Y|Z), which peaks at error "
    "probability 0.5. A confidently incorrect answer should have high error risk but can have low posterior entropy."
)

r.section("Trained joint SPN and why leaves outnumber inputs")
r.paragraph(
    "LearnSPN learns weighted sums representing mixtures and products representing factorized variable "
    "groups. The fitted checkpoint has 46 sum modules, 72 product modules, 75 histogram leaf modules, "
    "and 25 binary label leaf modules: 218 statistical modules and 217 edges. Tensor-concatenation "
    "and module-list wrappers are implementation details and are not counted as statistical nodes."
)
r.paragraph(
    "The eight inputs identify variables, while leaves identify local distributions within learned "
    "mixture branches. The same variable can have different distributions in many branches, so it appears "
    "in several leaf modules. A histogram leaf module can also model multiple variables independently. "
    "Thus leaf count is neither the number of inputs nor necessarily the number of scalar distributions. "
    "A binary label leaf module internally represents a weighted pair of Y=0/Y=1 indicators; those "
    "two conceptual indicators are not counted as two separate exported modules."
)
r.paragraph(
    "For example, a two-component mixture of histograms for one variable has one input "
    "but two histogram leaves, one distribution for each component. Learned SPN mixtures repeat "
    "this pattern across features and branches. In this checkpoint, the 75 histogram modules "
    "contain 101 scalar-variable scope occurrences; the 25 label modules add 25 occurrences of Y. "
    "None of these repeated local distributions introduces a new input variable."
)
r.table(
    ["Variable", "Leaf-module scope occurrences"],
    [[f"F{i + 1}: {model.feature_names[i]}" if i < 8 else "Y: incorrect label", leaf_occurrences[i]] for i in range(9)],
    [0.67, 0.25],
)
r.paragraph(
    "These occurrence counts describe topology, not feature importance. Repeated leaves do not "
    "create additional observations, additional input columns, or train/test duplication. The 100 leaf "
    "modules are the sum of 75 histogram modules and 25 label modules; histogram bins and tails are "
    "parameters inside a leaf, not extra leaves."
)
r.figure(
    "trained_spn",
    "Actual learned root factorization, with deeper subtrees collapsed for readability. "
    "F1-F8 correspond to the input table. The root product has a label-dependent sum block over F1,F2,F3,F4,F8,Y; "
    "a separate sum block over F5,F7; and a histogram factor over F6. The first block has two weighted subtrees "
    "with weights read from the checkpoint. Full node scopes, parameters, and edges are supplied in trained_spn.json "
    "and trained_spn.dot; the overview does not depict all 218 modules.",
)
r.paragraph(
    "A consequential structural result is that the Enthusiast's MCSP and NLL (F5,F7) form a factor "
    "separate from Y, and Enthusiast SE (F6) forms another separate factor. Both factors cancel in the "
    "posterior ratio P(Y=1|Z). In this fitted network, error-risk predictions therefore depend on the "
    "four Mathematician inputs plus Enthusiast mean length (F8), even though all eight features were "
    "offered to learning. This is a fitted independence assumption, not proof that the excluded "
    "signals cannot improve another model or another SPN structure."
)
r.table(
    ["SPN setting", "Value"],
    [
        ["Structure learner", "SPFlow LearnSPN; two-way clustering"],
        ["Minimum rows / feature slice", "50 / 2"],
        ["RDC dependence threshold", "0.3"],
        ["Continuous leaf", "10 uniform training-derived bins plus two exponential tails"],
        ["Histogram / label smoothing", "1.0 / 1.0"],
        ["Standardization", "Training means and standard deviations only"],
        ["Label dependence rule", "Avoid isolating a varying Y in its own independent product block"],
    ],
    [0.34, 0.58],
)

r.section("Observed uncertainty distributions")
r.paragraph(
    "The Mathematician has mean RMS SE 0.483 nats, with exactly zero SE on 56.0% of test "
    "trajectories. The Enthusiast has mean RMS SE 1.586 nats and no exact-zero RMS SE in this test set. "
    "Mean RMS MCSP is 0.925 versus 0.636, and mean RMS token NLL is 0.078 versus 0.478, "
    "for Mathematician versus Enthusiast respectively."
)
r.table(["Agent", "Measure", "RMS mean", "RMS median", "Exact zero (%)"], feature_stats, [0.24, 0.14, 0.18, 0.18, 0.18])
r.paragraph(
    "MCSP, SE, and NLL are different quantities and their numerical magnitudes are not directly "
    "comparable. Mathematician and Enthusiast also have different tasks: answer production versus "
    "open-ended feedback. Role differences cannot be interpreted as isolated effects of reasoning "
    "quality. In particular, the Enthusiast's high SE can reflect varied questions or feedback rather "
    "than disagreement about a boxed mathematical answer."
)
r.figure(
    "feature_distributions",
    "Discriminative distributions of the individual aggregated signals. Rows are "
    "Mathematician and Enthusiast; columns are RMS MCSP, SE, and token NLL. Green solid ECDFs represent 375 "
    "correct final answers and red dashed ECDFs represent 125 incorrect answers. Each curve is normalized "
    "within its group; its height is the fraction at or below the horizontal-axis score. Rightward shifts "
    "indicate larger values, but only SE and NLL have the default higher-means-more-uncertain direction. "
    "For MCSP, lower values indicate lower confidence. Overlap means no individual signal perfectly detects errors.",
)
r.table(["Agent", "Measure", "Correct mean", "Incorrect mean", "Raw AUROC"], raw_scores, [0.24, 0.14, 0.18, 0.18, 0.18])
r.paragraph(
    "Single-feature AUROC uses 1-MCSP for confidence and the original SE/NLL for uncertainty. "
    "These directions follow metric definitions; they were not selected to maximize test AUROC. "
    "An AUROC below 0.5 is retained rather than flipping the score after seeing test results. "
    "The single-feature analysis is descriptive and does not measure each input's contribution "
    "to the jointly fitted SPN; that would require training ablations."
)
r.figure(
    "completed_trajectories",
    "Final-response scores (top) versus all-step RMS inputs (bottom). "
    "Columns show MCSP confidence, SE in nats, and token NLL in nats/token. Blue is Mathematician; "
    "orange is Enthusiast. All curves use the same 500 test problems and are ECDFs. The bottom row "
    "contains the uncertainty/confidence inputs supplied to the SPN. The rows are very similar because "
    "99.4% of test conversations contain only one round, for which RMS equals the single observed score.",
)
r.figure(
    "round_distributions",
    "Per-round response distributions. Rows show MCSP, SE, and token NLL; "
    "columns are the observed rounds 1-4. Blue is Mathematician and orange is Enthusiast. Each panel "
    "states how many test problems reached that round; round 1 has 500, round 2 has 3, and rounds 3 "
    "and 4 each have 1. Within each metric the horizontal range is shared. These later-round samples "
    "are too small and too selected to support claims that additional feedback reduces uncertainty.",
)
r.paragraph(
    "An ECDF jump at zero identifies exactly zero scores, which is meaningful for semantic agreement. "
    "A curve farther left generally indicates smaller values, and an ECDF height of 0.8 at score x "
    "means 80% of that group's observations score at most x. Missing rounds are not imputed as zero. "
    "Repeated observations from the same problem are correlated, so evaluation intervals resample "
    "whole problems rather than treating every response as an independent test case."
)

r.section("Discrimination: detecting incorrect final answers")
r.paragraph(
    "The positive class is an incorrect final Mathematician answer. AUROC is the probability that "
    "a randomly selected incorrect answer receives higher risk than a randomly selected correct "
    "answer, counting ties as one half. Random ranking has AUROC 0.5. AUPRC is calculated as average "
    "precision, not trapezoidal PR area. Its prevalence reference is 0.25 on test and 0.22 on validation."
)
metric_rows = []
for name, title in (("auroc", "AUROC"), ("auprc", "AUPRC / average precision"), ("accuracy", "QA accuracy")):
    row = [title]
    for sp in ("validation", "test"):
        value = evaluation["splits"][sp]["models"]["spn"][name]
        row.append(f"{value['value']:.4f} [{value['ci_low']:.4f}, {value['ci_high']:.4f}]")
    metric_rows.append(row)
r.table(["Measure", "Validation (95% CI)", "Test (95% CI)"], metric_rows, [0.28, 0.32, 0.32])
r.table(
    ["Test ranking score", "AUROC", "Average precision"],
    [
        ["Joint SPN error probability", f"{roc_auc_score(y, p):.4f}", f"{average_precision_score(y, p):.4f}"],
        ["Mathematician RMS SE alone", f"{se_auc:.4f}", f"{se_ap:.4f}"],
    ],
    [0.50, 0.20, 0.22],
)
r.paragraph(
    "The Mathematician SE baseline requires no error-model fitting. Its better observed ranking "
    "shows that combining signals in this fitted SPN does not improve on the strongest individual "
    "signal in this test sample. This comparison is exploratory after inspecting individual-signal "
    "results; a paired significance test and a fresh confirmatory evaluation were not performed. "
    "Unscaled SE is not an error probability and is not used for Brier score or calibration comparisons."
)
r.paragraph(
    "Intervals use 1,000 percentile bootstrap resamples of problem identifiers with 95% confidence, "
    "as stored in evaluation.json. They describe uncertainty conditional on this run and test sample; "
    "they do not capture variation across independent model generations, training seeds, prompt "
    "choices, or datasets. No operating threshold is tuned on the test set in this report."
)
r.figure(
    "discrimination",
    "Joint-SPN discrimination on 500 test problems. Left: predicted-error histograms "
    "normalized separately for correct and incorrect answers; each class's bin heights sum to one, "
    "so height does not represent its population size. Middle: ROC curve, with false-positive rate "
    "on the horizontal axis and true-positive rate on the vertical axis; the dashed diagonal is random "
    "ranking. Orange dash-dot curves show the exploratory Mathematician RMS SE baseline. "
    "Right: precision versus recall for incorrect answers, with the dashed 0.25 prevalence "
    "reference. The curve label AP denotes average precision. These are descriptive curves from the "
    "saved predictions, with no threshold selected to maximize a test-set result.",
)
r.paragraph(
    "The joint SPN ranks incorrect answers usefully, but ranking quality does not establish "
    "probability calibration. Posterior entropy ranks test errors less well than direct error "
    "probability: AUROC 0.8910 and average precision 0.6258 versus 0.9165 and 0.7555. "
    "Confidently wrong answers can have low posterior entropy, so direct error risk is the "
    "appropriate primary score for identifying wrong answers in this evaluation."
)

r.section("Calibration and selective prediction")
cal_rows = []
for name, title in (
    ("brier", "Brier score"),
    ("ece", "ECE (15 bins)"),
    ("nll", "Binary risk NLL"),
    ("aurc", "Area under risk-coverage"),
    ("auarc", "Area under accuracy-coverage"),
):
    value = metrics[name]
    cal_rows.append([title, f"{value['value']:.4f}", f"[{value['ci_low']:.4f}, {value['ci_high']:.4f}]"])
r.table(["Test measure", "Value", "95% bootstrap CI"], cal_rows, [0.44, 0.16, 0.32])
r.paragraph(
    "Brier score averages squared probability errors. ECE summarizes the absolute difference "
    "between predicted and observed error probability over 15 equal-width bins. Binary risk NLL "
    "scores the correctness-label probability and differs from the response-token NLL used as an "
    "input feature. Lower values are better for these three measures; an ECE of 0.0779 does not "
    "justify claiming perfect calibration, and binning can hide local deviations."
)
r.figure(
    "calibration_coverage",
    "Reliability and selective prediction. Left: each occupied probability "
    "bin is a point at its mean predicted risk and observed error fraction; marker area grows with "
    "bin count. The dashed diagonal denotes perfect bin-level calibration; sparse bins have noisier "
    "observed fractions. Right: answers are retained from lowest predicted risk upward. Coverage "
    "is the retained fraction and risk is their error fraction; the dashed line is full-test error "
    "prevalence. Risk-coverage uses expected ordering within tied-score groups. This plots abstention "
    "behavior without selecting a deployment threshold.",
)
r.paragraph(
    "The reported AURC is the mean retained-set error rate over coverage k/n, k=1,...,n. "
    "AUARC is one minus AURC under the repository's convention. These metrics describe selective "
    "ranking on existing answers, not an intervention that improves the generated mathematical solutions."
)

r.section("Leakage checks and validity limits")
r.paragraph(
    "The saved splits contain no overlapping identifiers or whitespace-normalized exact problem "
    "texts across training, validation, and test. The runner trains the SPN only from train rows; "
    "standardization and histogram boundaries are fitted on those rows. The workflow receives "
    "problem text and conversation, not reference solutions. SE compares generated responses with "
    "each other rather than with references. The SPN feature vector excludes labels and reference answers. "
    "These checks found no direct split or answer-label leakage in the inspected pipeline."
)
r.paragraph(
    "The audit does not establish absence of near-duplicate problem variants, contamination of the "
    "language model's pretraining or instruction-tuning data, or test-guided choices made outside the "
    "saved code and artifacts. The checkpoint alone cannot establish the full history of prompt, "
    "feature, or hyperparameter selection. Bootstrap intervals do not correct such selection effects."
)
r.paragraph(
    "The earlier Qwen3-8B sequential Solver-Reviewer report used different agents, prompts, scoring "
    "engines, uncertainty definitions, sampling settings, and mathematical tool access. Its reported "
    "79.2% accuracy and 0.888 AUROC cannot serve as a controlled estimate of the feedback loop's causal "
    "effect. The current run has lower answer accuracy and higher error-ranking AUROC, but several "
    "factors changed simultaneously and error prevalence differs. The near-universal first-round "
    "termination further limits conclusions about iterative correction."
)
r.paragraph(
    "Other limits include one dataset, one model identity, one saved generation run, unpinned Hub "
    "revisions, five failed training conversations, finite Monte Carlo samples, automated answer "
    "grading without independent human regrading, and role-dependent text formats. Low SE can "
    "describe confident consensus on an incorrect answer. MCSP and NLL can reflect language fluency "
    "and response style as well as mathematical correctness."
)

r.section("Reproducing the experiment")
r.paragraph(
    "Run from the repository root after installing the project dependencies with Linux, "
    "an NVIDIA GPU, and vLLM for the saved generation backend. The registered experiment is math "
    "and its configuration is config/config-qwen-math.yaml. To evaluate only the joint SPN, "
    "set evaluation.risk_models to [spn] in that YAML file while preserving the other settings. "
    "The shared checked-in configuration can select additional risk models; this report uses only spn."
)
r.code(
    "scripts/run-experiments.sh --experiment math --limit 5\n"
    "scripts/run-experiments.sh --experiment math\n\n"
    "Reuse saved responses and features:\n"
    "scripts/run-experiments.sh --experiment math \\\n  --phase train --phase evaluate"
)
r.paragraph(
    "The smoke command limits each partition to five problems and cannot reproduce full-run "
    "metrics. Full phases are generate, score, train, and evaluate; generation/scoring use the model "
    "and NLI resources, while SPN fitting and evaluation run on CPU. Resume is enabled, with "
    "fingerprinted generation and scoring directories. Archive the exact artifact snapshot and "
    "pin model/tokenizer revisions for a future reproducibility study."
)

r.section("Reproducibility and delivered artifacts")
r.paragraph(
    "Saved evaluation run: " + evaluation["run_id"] + ". "
    "The report generator verifies feature aggregation, split disjointness, class counts, stored metrics, "
    "SPN node counts, sum-weight normalization, complete sum scopes, decomposable product scopes, and "
    "checkpoint predictions against all 500 saved test predictions. It does not regenerate answers or refit the SPN."
)
r.code("Source directory:\nartifacts/ma_qwen_math/\n  generation-73c7092a6566/scored-fbc4ac53ebbc/")
r.code(
    "Rebuild the report sources and scientific figures:\n"
    "uv run --no-sync python \\\n  docs/experiment/math_ma_qwen2_5_7b/generate_report.py\n\n"
    "Compile the generated LaTeX with a TeX installation:\n"
    "cd docs/experiment/math_ma_qwen2_5_7b\ntectonic report.tex"
)
r.paragraph(
    "Deliverables are report.tex (LaTeX source), report.pdf (compiled report), report.md "
    "(Markdown with figures), report.text and report.txt (plain text including equations, tables, "
    "and figure explanations), trained_spn.json (full learned topology and parameters), and "
    "trained_spn.dot (Graphviz topology). Figures are supplied as PNG for Markdown, vector PDF "
    "for LaTeX, and SVG for editing. The plain-text formats describe figures and list their image "
    "paths; they cannot embed graphical images."
)
r.write()

# A browser-readable copy of the workflow uses the same generated SVG, with no external assets.
svg = (ASSETS / "workflow.svg").read_text()
svg = svg[svg.index("<svg") :]
svg = svg.replace("<svg ", '<svg role="img" aria-labelledby="workflow-title workflow-desc" ', 1)
end_open = svg.index(">") + 1
svg = (
    svg[:end_open] + '<title id="workflow-title">Mathematical feedback workflow</title><desc id="workflow-desc">'
    "Two agents exchange solutions and feedback while a separate path measures uncertainty "
    "and predicts final-answer error.</desc>" + svg[end_open:]
)
(OUT / "workflow.html").write_text(
    '<!doctype html><html lang="en"><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1"><title>SPUMA feedback workflow</title>'
    "<style>body{margin:24px;font-family:system-ui;background:white;color:#243548}"
    ".diagram{width:100%;overflow-x:auto}svg{display:block;width:100%;min-width:960px;height:auto}"
    "@media print{.diagram{overflow:visible}svg{min-width:0}}</style><h1>SPUMA feedback workflow</h1>"
    '<div class="diagram">' + svg + "</div><p>Qwen2.5-7B-Instruct; up to 10 rounds. "
    "497 of 500 test problems stopped after the first exchange.</p></html>"
)
sys.stdout.write(
    f"Validated {len(rows)} usable problems, {len(steps)} responses, and {len(nodes)} SPN modules.\n"
    f"Created report sources and seven figures in {OUT}.\n"
)
