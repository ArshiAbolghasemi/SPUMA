# SPUMA
Probabilistic Uncertainty Estimation in Multi-Agent LLM Systems via Sum-Product Networks

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.14 (uv installs it), and for real runs an NVIDIA
GPU (vLLM generation needs Linux). Apple Silicon and CPU are enough for development and tests.

```bash
git clone <repo-url> && cd SPUMA
scripts/install-dependencies.sh            # picks the torch build: cu126/cu128/cu130 from nvcc, mps on macOS, else cpu
scripts/install-dependencies.sh --vllm     # Linux + NVIDIA: vLLM backend (brings its own pinned torch)
cp .env.example .env                       # then set HF_TOKEN (Hugging Face Hub access)
```

`.env` holds secrets only and is git-ignored; everything else is configuration (below).

## Experiments

Each experiment is a module in `spuma/experiments/` that names its settings file:

| experiment | module | settings | dataset | splits (train / validation / test) | what it runs |
|---|---|---|---|---|---|
| [math](docs/experiment/math_ma_qwen3_8b/report.md) | `spuma/experiments/math.py` | `config/config-qwen-math.yaml` | [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) (Hendrycks et al., 2021) | 800 / 200 from official train, 500 from official test | Qwen3-8B Solver -> Reviewer |

A settings file defines everything about its experiment: dataset and splits, model and backends
(`model.backend` for generation, `model.scoring_backend` for teacher-forced scoring), agents and their
tools, uncertainty estimation, SPN, evaluation, and `experiment.name` (its artifact directory). Any value
can be overridden from the environment with the `SPUMA_` prefix and `__` for nesting, e.g.
`SPUMA_MODEL__BACKEND=transformers` or `SPUMA_DATASET__TEST_SAMPLES=50`.

### Running

```bash
scripts/run-experiments.sh                                  # every registered experiment, every phase
scripts/run-experiments.sh --experiment math                # one experiment (flag is repeatable)
scripts/run-experiments.sh --experiment math --limit 5      # smoke run: 5 problems per split
scripts/run-experiments.sh --list                           # registered experiments
```

Phases run in order and can be selected with `--phase` (repeatable):

| phase | does | needs |
|---|---|---|
| `generate` | runs the agents on every problem; draws the semantic-entropy samples | GPU (vLLM or Transformers) |
| `score` | per-step PE/NLL (teacher forcing) and semantic entropy; writes the features | GPU + the NLI model |
| `train` | trains the risk models (joint SPN, class-conditional baseline) | CPU |
| `evaluate` | AUROC, AUARC, ECE, ... with bootstrap intervals on validation/test | CPU |

Re-use saved work instead of recomputing it:

```bash
scripts/run-experiments.sh --experiment math --phase score --phase train --phase evaluate  # keep generations
scripts/run-experiments.sh --experiment math --phase train --phase evaluate                # only SPN settings changed
```

Artifacts go to `artifacts/<experiment.name>/`. Directory names carry fingerprints of the settings that
produced them (`generation-<fp>/`, inside it `scored-<fp>/`), so changing a setting never reuses stale
results, and with `experiment.resume: true` an interrupted run continues where it stopped:

```
artifacts/<experiment.name>/
  splits.json, config.json, summary.json, runs/<run_id>.json
  generation-<fp>/trajectories/<problem>.json            agent steps, tool calls, semantic-entropy samples, grade
  generation-<fp>/scored-<fp>/steps/<problem>.json       + per-step metrics and token scores
  generation-<fp>/scored-<fp>/step_metrics.json          one row per step
  generation-<fp>/scored-<fp>/features.json              features Z, label Y, per problem
  generation-<fp>/scored-<fp>/models/, evaluation.json   risk models and their metrics
```

Logs go to `logs/` (JSON lines) and the console.

With vLLM generation and Transformers scoring, the runner releases the vLLM engine before scoring, so only
one Qwen3-8B engine is on the GPU at a time. If GPU memory is not fully freed on a single 24 GB GPU, run
the two as separate invocations: `--phase generate`, then `--phase score --phase train --phase evaluate`.

### Backend benchmark

```bash
scripts/run-benchmark.sh --backend transformers --backend vllm --problems 8
```

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
