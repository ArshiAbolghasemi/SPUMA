# MATH: Qwen3-8B Solver–Reviewer experiment

This experiment measures whether uncertainty recorded during a two-agent mathematical
question-answering workflow predicts an incorrect final answer. A joint sum-product
network (SPN) learns the relationship between uncertainty features and final-answer
correctness. It scores completed trajectories; it does not revise the agents' answers.

The registered experiment is `math`, defined in
[`spuma/experiments/math.py`](../../spuma/experiments/math.py), with settings in
[`config/config-qwen-math.yaml`](../../config/config-qwen-math.yaml).
Its artifact directory is `artifacts/ma_qwen_math/`.

## Dataset and splits

The code loads `EleutherAI/hendrycks_math` from Hugging Face, covering algebra,
counting and probability, geometry, intermediate algebra, number theory,
prealgebra, and precalculus. The experiment uses 1,500 problems:

| Experimental split | Problems | Official source split |
|---|---:|---|
| Training | 800 | Train |
| Validation | 200 | Train |
| Test | 500 | Test |

Training and validation are disjoint. Selection sorts stable problem identifiers
before shuffling with seed 42. Training fits the error-prediction model; the language
model is not fine-tuned. Reference solutions are available only to the evaluator.

## Agents and tools

LangGraph runs the agents sequentially. Both use `Qwen/Qwen3-8B` with thinking mode
disabled. The Solver receives the problem and produces a proposed solution. The
Reviewer receives the problem and the Solver's final message, checks the reasoning,
and produces the answer used for grading. Both are instructed to put their final
answer in `\boxed{}`.

| Agent | Available tools |
|---|---|
| Solver | `sympy_solve`, `sympy_simplify`, `sympy_integrate`, `sympy_differentiate` |
| Reviewer | `sympy_solve`, `sympy_simplify`, `math_verify` |

Each agent has up to eight generation steps and five attempted tool calls. Tool
results and errors return to the requesting agent as observations. A step is one
model generation, rather than an individual sentence or mathematical operation.
Tool access does not imply tool use on every problem.

## Generation and uncertainty

| Setting | Value |
|---|---|
| Generation backend | vLLM |
| PE and token-NLL scoring backend | Transformers |
| Arithmetic | bfloat16 |
| Temperature / top-p | 0.7 / 0.9 |
| Top-k truncation | Disabled |
| Maximum new tokens | 2,048 per generation |
| Context limit | 8,192 tokens |
| Tensor-parallel shards | 1 |
| Concurrent problems | 8 |
| Seed | 42 |

Transformers teacher-forces the saved prompt and continuation token IDs to score
the actual generated response. PE and token NLL use the raw model distribution,
before temperature scaling or sampling truncation. All logarithms are natural.

- **Predictive entropy (PE):** mean full-vocabulary next-token entropy across
  generated tokens.
- **Token negative log-likelihood (NLL):** mean negative log probability of the
  actual generated tokens.
- **Semantic entropy (SE):** entropy of semantic clusters formed from the recorded
  response and ten additional responses sampled from the same frozen pre-step
  context. Tool requests in these alternatives are kept as text and never executed.
- **Token length:** number of generated tokens in the step.

SE uses UQLM with an answer-aware comparison. When both responses have boxed
answers, normalized equality or Math-Verify checks their equivalence. When only one
has a boxed answer, they differ. When neither has a boxed answer, comparison falls
back to `tasksource/ModernBERT-large-nli`, accepting entailment in either direction.
The NLI limits are 8,000 characters per text and 2,048 tokens per pair, with batch
size 32. Reference answers do not enter semantic clustering.

## Features, target, and joint SPN

For each agent, PE, SE, and token NLL are aggregated across its steps using root
mean square, `sqrt(mean(step_score ** 2))`. Token length uses an arithmetic mean.
The joint SPN receives these eight features in order:

```text
solver_pe, solver_se, solver_nll, solver_length,
reviewer_pe, reviewer_se, reviewer_nll, reviewer_length
```

The target is `Y = 1` for an incorrect Reviewer answer and `Y = 0` for a correct
answer. The evaluator extracts the last boxed answer and compares it with the
reference using Math-Verify. Missing boxed answers count as incorrect and are
flagged as parse failures. If the reference cannot be parsed, grading falls back
to normalized string equality.

The model learns a joint density over features `Z` and label `Y`, then predicts:

```text
P(Y = 1 | Z) = P(Z, Y = 1) / (P(Z, Y = 0) + P(Z, Y = 1))
```

SPFlow's LearnSPN uses two-way clustering, minimum row slice 50, minimum feature
slice 2, and RDC threshold 0.3. Continuous leaves use ten uniform bins derived from
training data, two exponential tails, and smoothing 1. Label leaves are binary
indicators with smoothing 1. A varying label cannot be split into an independent
product block by itself. Standardization uses training statistics only; validation
and test data are excluded from fitting.

## Running

Run commands from the repository root after completing the setup in the
[README](../../README.md#setup). Full generation requires Linux and an NVIDIA GPU
with vLLM installed. CPU and Apple Silicon support development and offline tests.

In `config/config-qwen-math.yaml`, set `evaluation.risk_models` to select only the
joint SPN for the commands in this document:

```yaml
evaluation:
  risk_models: [spn]
```

Keep the other evaluation settings. Set this list in the YAML file because
environment list overrides are merged with the configured list.

Run a smoke experiment or the complete experiment:

```bash
scripts/run-experiments.sh --experiment math --limit 5
scripts/run-experiments.sh --experiment math
```

`--limit 5` uses five problems per split and does not reproduce the reported results.
The phases are `generate`, `score`, `train`, and `evaluate`, in that order. Reuse
saved generations or fitted features by selecting phases:

```bash
scripts/run-experiments.sh --experiment math --phase score --phase train --phase evaluate
scripts/run-experiments.sh --experiment math --phase train --phase evaluate
```

The runner releases vLLM before loading Transformers for scoring. On a single GPU,
the phases can also run in separate processes:

```bash
scripts/run-experiments.sh --experiment math --phase generate
scripts/run-experiments.sh --experiment math --phase score --phase train --phase evaluate
```

Resume is enabled. Generation and scoring settings determine artifact fingerprints.
Outputs include split assignments, sanitized settings, trajectories, tool records,
per-step metrics, feature vectors, fitted models, and `evaluation.json`. Logs are
written to `logs/` and the console.

## Reported results

The report presents joint-SPN AUROC and AUPRC point estimates. An incorrect answer
is the positive class; larger scores indicate greater error risk.

| Split | AUROC | AUPRC |
|---|---:|---:|
| Validation | 0.864 | 0.709 |
| Test | 0.888 | 0.739 |

The Reviewer answers 396 of 500 test problems correctly: accuracy is 79.2% and error
prevalence is 20.8%. There are 22 test answers without an extractable boxed answer;
they remain in the analysis as incorrect. Validation accuracy is 75.0%.

For the joint SPN, ranking test answers by binary posterior entropy instead of
error probability gives AUROC 0.870 and AUPRC 0.566. Posterior entropy peaks near
error probability 0.5, so confident errors can receive low entropy.

The configured evaluator also supports calibration metrics and 1,000 bootstrap
iterations at 95% confidence; the report's tables present AUROC and AUPRC point
estimates only. Low response uncertainty does not establish mathematical correctness.
The model and tokenizer revisions are unpinned, so rerunning with later Hub revisions
may change outputs. These reported values describe the saved evaluation, not a new
run performed while writing this document.
