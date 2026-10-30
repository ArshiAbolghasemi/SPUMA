# SPUMA
Probabilistic Uncertainty Estimation in Multi-Agent LLM Systems via Sum-Product Networks

## Documentation

### Datasets

| Dataset | Task | Agents see | Answer format |
|---|---|---|---|
| [MATH](docs/datasets/math.md) | competition mathematics | the problem | `\boxed{...}` |
| [MMLU](docs/datasets/mmlu.md) | four-choice academic and professional questions | the question and choices A–D | `ANSWER: <A/B/C/D>` |
| [MuSiQue](docs/datasets/musique.md) | evidence-grounded multi-hop questions | the question and 20 numbered paragraphs | `EVIDENCE:` and `ANSWER:` |
| [MoreHopQA](docs/datasets/morehopqa.md) | closed-book multi-hop questions with a final reasoning step | the question | `ANSWER: <short value>` |

How every dataset is split and graded: [splits and grading](docs/datasets/splits-and-grading.md).

### Architectures

| Architecture | Agents | LLM calls per problem | Graded response |
|---|---|---|---|
| [Feedback loop](docs/architectures/feedbackloop.md) | solver ⇄ reviewer | 2 per round, up to 10 rounds | the solver's last response |
| [Star](docs/architectures/star.md) | Star coordinator → analyst → Star → verifier → Star | exactly 5 | the Star final synthesis |
| [Sequential](docs/architectures/sequential.md) | solver → verifier → finalizer | exactly 3 | the finalizer's response |

Model, sampling, seeds, per-response measurement and prompt composition: [shared mechanics](docs/architectures/common.md).

### Uncertainty quantification

| Page | Content |
|---|---|
| [Per-response metrics](docs/uncertainty/metrics.md) | semantic entropy, Monte Carlo sequence probability, negative log-likelihood, token length |
| [Scores per architecture](docs/uncertainty/methods.md) | every evaluated score and exactly which features it uses |
| [Sum-product network](docs/uncertainty/spn.md) | the joint SPN: P(wrong \| features), posterior entropy, structure learning, leaves, settings |
| [CHMM weighting](docs/uncertainty/chmm.md) | question drift, interaction gap, hidden states, weights (feedback loop) |
| [Training and evaluation](docs/uncertainty/evaluation.md) | SPN training, calibration, metrics, bootstrap, dataset-specific and frozen protocols |

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12 (uv installs it), and for real runs an NVIDIA
GPU (vLLM generation needs Linux). Apple Silicon and CPU are enough for development and tests.

```bash
git clone <repo-url> && cd SPUMA
scripts/install-dependencies.sh            # picks the torch build: cu126/cu128/cu130 from nvcc (else nvidia-smi), mps on macOS, else cpu
scripts/install-dependencies.sh --vllm     # Linux + NVIDIA: vLLM backend (brings its own pinned torch)
scripts/install-dependencies.sh --index https://mirror/simple   # everything, torch too, from a PyPI mirror
cp .env.example .env                       # then set HF_TOKEN (Hugging Face Hub access)
```

Downloads use a 30-minute read timeout (`UV_HTTP_TIMEOUT=1800`; set it to override), since torch and vLLM
wheels are GBs and a caching mirror fetches an uncached file before it sends a byte.

`.env` holds secrets only and is git-ignored; everything else is configuration (below).

## Experiments

Each experiment is run by name; the name maps to its settings file `config/<name>.yaml` (registered in
`spuma/experiments/registry.py`). OOD evaluations work the same way (`spuma/ood/registry.py`):

| experiment | module | settings | dataset | splits (train / validation / test) | what it runs |
|---|---|---|---|---|---|
| `math_ma_qwen2_5_7b_feedbackloop` | `spuma/experiments/math.py` | `config/math_ma_qwen2_5_7b_feedbackloop.yaml` | [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) (Hendrycks et al., 2021) | 800 / 200 from official train, 500 from official test | Qwen2.5-7B-Instruct Mathematician <-> Enthusiast feedback |

See the [MATH experiment report](docs/experiment/math_ma_qwen2_5_7b_feedbackloop/report.md) for results and analysis.

A settings file defines everything about its experiment: dataset and splits, model and backends
(`model.backend` for generation, `model.scoring_backend` for teacher-forced scoring), agents and their
prompts, uncertainty estimation, SPN, evaluation, and `experiment.name` (its artifact directory). Any value
can be overridden from the environment with the `SPUMA_` prefix and `__` for nesting, e.g.
`SPUMA_MODEL__BACKEND=transformers` or `SPUMA_DATASET__TEST_SAMPLES=50`.

### Running

```bash
scripts/run-experiments.sh                                  # every registered experiment, every phase
scripts/run-experiments.sh --experiment math_ma_qwen2_5_7b_feedbackloop           # one experiment by name (repeatable)
scripts/run-experiments.sh --experiment math_ma_qwen2_5_7b_feedbackloop --limit 5 # smoke run: 5 problems per split
scripts/run-experiments.sh --list                           # registered experiments
```

Phases run in order and can be selected with `--phase` (repeatable):

| phase | does | needs |
|---|---|---|
| `generate` | runs the agents on every problem; draws the semantic-entropy samples | GPU (vLLM or Transformers) |
| `score` | per-step PE/NLL (teacher forcing) and semantic entropy; writes the features | GPU + the NLI model |
| `train` | trains the risk models (joint SPN) | CPU |
| `evaluate` | AUROC, AUARC, ECE, ... with bootstrap intervals on validation/test | CPU |

Re-use saved work instead of recomputing it:

```bash
scripts/run-experiments.sh --experiment math_ma_qwen2_5_7b_feedbackloop --phase score --phase train --phase evaluate
scripts/run-experiments.sh --experiment math_ma_qwen2_5_7b_feedbackloop --phase train --phase evaluate
```

Artifacts go to `artifacts/<experiment.name>/`. Directory names carry fingerprints of the settings that
produced them (`generation-<fp>/`, inside it `scored-<fp>/`), so changing a setting never reuses stale
results, and with `experiment.resume: true` an interrupted run continues where it stopped:

```
artifacts/<experiment.name>/
  splits.json, config.json, summary.json, runs/<run_id>.json
  generation-<fp>/trajectories/<problem>.json            agent steps, semantic-entropy samples, grade
  generation-<fp>/scored-<fp>/steps/<problem>.json       + per-step metrics and token scores
  generation-<fp>/scored-<fp>/step_metrics.json          one row per step
  generation-<fp>/scored-<fp>/features.json              features Z, label Y, per problem
  generation-<fp>/scored-<fp>/models/, evaluation.json   risk models and their metrics
```

Logs go to `logs/` (JSON lines) and the console.

With vLLM generation and Transformers scoring, the runner releases the vLLM engine before scoring, so only
one Qwen2.5-7B-Instruct engine is on the GPU at a time. If GPU memory is not fully freed on a single 24 GB GPU, run
the two as separate invocations: `--phase generate`, then `--phase score --phase train --phase evaluate`.

### Backend benchmark

```bash
scripts/run-benchmark.sh --backend transformers --backend vllm --problems 8
```

## Out-of-distribution evaluation

Evaluates a trained SPN, frozen, on another dataset: the source experiment's agents, prompts, generation,
uncertainty and aggregation run on the OOD problems, and the checkpoint predicts P(incorrect | z) for each.
Nothing is trained, tuned or recalibrated; the checkpoint's SHA-256 is checked before and after.

```bash
scripts/run-ood.sh --ood ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop                    # generate, score, evaluate
scripts/run-ood.sh --ood ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop --limit 5          # smoke run
scripts/run-ood.sh --ood ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop --phase evaluate   # re-evaluate saved features
scripts/run-ood.sh --list                                                           # registered OOD evaluations
```

`ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop` (`config/ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop.yaml`) evaluates the MATH-trained SPN on 400 seeded questions from the human-verified
MoreHopQA test set (closed-book, pinned Hub revision). The source settings must reproduce the checkpoint's
`generation-<fp>/scored-<fp>/` fingerprints, so the OOD features are computed exactly as in training. Outputs go
to `artifacts/<ood.name>/`: the merged `settings.yaml`, trajectories, `steps.jsonl` and `step_metrics.json`
(per-step uncertainty), `features.json`, `ood_evaluation.json` (AUROC, AUPRC, AUARC, AURC, Brier, ECE, NLL,
accuracy with bootstrap intervals, per-problem predictions, comparison with the in-distribution test results
and feature means) and `ood_summary.json`.

## Tests

Tests live next to the code in `spuma/<package>/tests/`. Shared fixtures are in each package's
`tests/fixtures.py` and are loaded for every test through `pyproject.toml`.

```bash
scripts/run-tests.sh                        # offline suite: no GPU, no downloads (tiny in-process models)
scripts/run-tests.sh --cov                  # + coverage report
scripts/run-tests.sh spuma/spn -k histogram # any pytest arguments
scripts/run-tests.sh -m nli                 # real NLI model for semantic entropy (downloads it)
scripts/run-tests.sh -m vllm                # vLLM backend tests (Linux + NVIDIA GPU)
scripts/run-tests.sh --all                  # everything, including gpu/vllm/nli tests
```

Tests marked `gpu`, `vllm` or `nli` download models and are skipped unless selected as above.

## Lint and format

```bash
uv run --no-sync ruff check spuma           # lint (add --fix to apply safe fixes)
uv run --no-sync ruff format spuma          # format
uv run --no-sync ruff format --check spuma  # check formatting only (CI)
```

Use `--no-sync` with `uv run`: a plain `uv run` re-syncs without the hardware extra and removes torch.

## Data versioning (DVC)

`artifacts/` is versioned with [DVC](https://dvc.org/) (`artifacts.dvc` in git; the data in the `spuma`
S3-compatible remote configured in `.dvc/config`). Credentials stay out of git, in `.dvc/config.local`:

```bash
uv run --no-sync dvc remote modify --local spuma access_key_id <key-id>
uv run --no-sync dvc remote modify --local spuma secret_access_key <secret>
```

```bash
uv run --no-sync dvc pull                   # fetch the artifacts of the current commit
# ... run experiments ...
uv run --no-sync dvc add artifacts          # record the new artifacts (updates artifacts.dvc)
git add artifacts.dvc && git commit -m "feat: ..."
uv run --no-sync dvc push                   # upload the data
```

Checking out an older commit and running `dvc checkout` (or `dvc pull`) restores the artifacts that
commit's results came from.
