# Training and evaluation procedure

## Use of the splits

| Split | Used for |
|---|---|
| train | fitting the CHMM (feedback loop), one SPN per SPN method, and the logistic calibrators of the standalone scores |
| validation | held out |
| test | every reported metric; nothing is ever fitted on it |

All methods of an experiment use the same trajectories, the same correctness labels (`wrong = 1`) and the same
splits. A problem's responses never cross splits. Unlabeled problems are left out of fitting and evaluation.

## The SPN

`JointSPN` (`spuma/spn/model.py`) is a single sum-product network over the features **and** the label:
- **Label:** Y (wrong = 1) is a binary variable with two indicator leaves, so P(Z, Y = c) and P(Z) are exact
  network queries and **P(wrong | Z) = P(Z, 1) / (P(Z, 0) + P(Z, 1))**.
- **Structure learning:** LearnSPN (SPFlow) on standardized features, with histogram leaves (10 uniform bins,
  Laplace smoothing 1), at least 50 instances per slice, 2 clusters per sum split, and an RDC independence threshold
  of 0.3.
- **Label dependence:** the label is never split off on its own while it still varies, so P(Y | Z) can depend on Z.
- **Missing features:** a feature given as NaN is integrated out exactly.

Every SPN method is trained with these same settings on its own feature vector. Its outputs are P(wrong) and the
[posterior entropy](methods.md#building-blocks). Full description: [sum-product network](spn.md).

## Standalone scores

RMS and last-step SE and MCSP are single numbers. They're ranked as they are. For Brier and ECE they're first
mapped to probabilities by a logistic (Platt) calibrator fitted on the training split, and these calibrated values
are reported separately.

## Metrics

Positive class: **wrong = 1**. Each score is used so that higher = more likely wrong.

| Metric | Definition | Better |
|---|---|---|
| AUROC | probability that a wrong answer is ranked riskier than a correct one (ties count half) | higher |
| AUPRC | average precision of detecting wrong answers | higher |
| AURC | area under the risk–coverage curve: keep the k lowest-risk answers (k = 1…n), risk(k) = error rate among them, AURC = mean_k risk(k); ties use the expected risk under a random order | lower |
| AUARC | 1 − AURC: the mean accuracy over coverage levels | higher |
| Brier | mean (p − y)² of a probability p | lower |
| ECE | Σ_b (n_b / n) · \|mean(y in bin b) − mean(p in bin b)\| over 15 equal-width bins of p (n_b = answers in bin b) | lower |

The posterior-entropy scores are rankings, so only AUROC, AUPRC, AURC and AUARC are reported for them.

## Bootstrap

- **Resampling:** problem-clustered, 10,000 resamples (`chmm.bootstrap_iterations`), 95% percentile intervals.
- **Paired:** every resample draws the same problems for all methods, so each **paired difference** has its own
  interval. Differences are reported against RMS SE and RMS MCSP for every architecture, and against RMS SPN for
  the feedback loop's primary CHMM α = 1.
- **One-sided p:** `p_not_better` = the share of resamples where the compared method is not better.

## Protocols

| Protocol | Models | Fitted on the evaluated dataset? |
|---|---|---|
| **Dataset-specific** | the CHMM, SPNs and calibrators trained on this dataset's training split | yes, train split only |
| **Frozen** | the models of the MATH experiment of the same architecture (`chmm.frozen_from`), applied unchanged | no |

**Frozen evaluation:**
- **Checks first:** the source and target must share the distance estimator, α values, feature layout and
  feature names.
- **Output:** `chmm-<fp>/frozen-<source experiment>/`.
- **Reuse:** it uses the target dataset's own saved trajectories, so nothing is regenerated.

## Outputs

Per experiment, in `artifacts/<experiment>/generation-<fp>/scored-<fp>/chmm-<fp>/`:

| File | Content |
|---|---|
| `models/` | the SPN of each method (`<method>.pt`), training status, calibrators, and the CHMM (`chmm.pkl`, feedback loop) |
| `evaluation.json` | every metric with its interval, paired differences, checks |
| `features.json` | each problem's feature vectors and standalone scores, and test-split predictions |
| `steps.jsonl` | each response's metrics (plus distances, posteriors and weights in the feedback loop) |
| `results.md`, `results_table.json` | the same results as a table |

`scripts/run-experiments.sh --compare` collects every experiment's dataset-specific and frozen tables into
`artifacts/comparison.md`.
