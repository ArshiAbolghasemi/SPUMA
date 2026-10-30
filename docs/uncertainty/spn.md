# Sum-product network (SPN)

SPUMA's learned uncertainty score is a **joint sum-product network** over the uncertainty features Z and the
correctness label Y (Y = 1: the final answer is wrong). It turns a feature vector into the probability that the
answer is wrong, **P(wrong | Z)**, and the posterior entropy of that prediction.

- **Implementation:** `spuma/spn/` (`model.py`: `JointSPN`; `leaves.py`: the label and histogram leaves;
  `config.py`: the `spn:` settings)
- **Library:** [SPFlow](https://github.com/SPFlow/SPFlow) (LearnSPN structure learning, log-space inference)

## What an SPN is

An SPN is a rooted graph that represents a probability distribution exactly:

| Node | Computes | Meaning |
|---|---|---|
| **Leaf** | a univariate distribution over one variable | a density of one feature, or the distribution of Y |
| **Product** | the product of its children, over disjoint sets of variables | those variable groups are independent here |
| **Sum** | a weighted sum of its children (weights sum to 1), over the same variables | a mixture: different clusters of problems |

Because products only combine disjoint variables and sums only mix children over the same variables, the network
is a valid distribution. Any joint or marginal probability is computed in one bottom-up pass.

## Joint model of features and label

`JointSPN` learns **one** network over the columns [Z, Y]:

```mermaid
flowchart TD
    R(("Σ")) --> P1(("×"))
    R --> P2(("×"))
    P1 --> L1["Y indicators<br/>θ₁"]
    P1 --> A1["solver_se<br/>histogram"]
    P1 --> B1(("×"))
    B1 --> C1["solver_mcsp<br/>histogram"]
    B1 --> D1["…"]
    P2 --> L2["Y indicators<br/>θ₂"]
    P2 --> A2["solver_se<br/>histogram"]
    P2 --> D2["…"]
```

*An illustrative structure. The actual one is learned from the training data and differs per method and dataset.*

Y is one binary variable with two indicator leaves, S(y) = θ · [Y = 1] + (1 − θ) · [Y = 0]. With that, every
query the method needs is exact:

```text
log P(Z, Y = c) = log S(Z, I(+y) = [c = 1], I(−y) = [c = 0])
log P(Z)        = log S(Z, I(+y) = 1, I(−y) = 1)                 (Y marginalized)
P(wrong | Z)    = P(Z, Y = 1) / ( P(Z, Y = 0) + P(Z, Y = 1) )
H(Y | Z)        = −p ln p − (1 − p) ln(1 − p),   p = P(wrong | Z)
```

- **P(wrong | Z)** is the method's risk score and probability (used for all six evaluation metrics).
- **H(Y | Z)**, the posterior entropy, is highest when the SPN cannot decide (p ≈ 0.5). It's reported as a
  separate ranking score.
- **Missing features:** a feature given as NaN is integrated out exactly, as for Y.

## Structure learning (LearnSPN)

Learned on the **training split only**, from the standardized features and the label:

```mermaid
flowchart TD
    D["rows × columns<br/>(slice of the training data)"] --> Q{"one column left?"}
    Q -- yes --> LEAF["leaf<br/>(histogram for a feature,<br/>indicators for Y)"]
    Q -- no --> IND{"columns split into<br/>independent groups?<br/>(RDC test)"}
    IND -- yes --> PROD["product node<br/>recurse on each column group"]
    IND -- no --> SMALL{"fewer than 50 rows?"}
    SMALL -- yes --> FACT["product of single-column leaves"]
    SMALL -- no --> SUM["sum node: cluster the rows (2 clusters)<br/>weight = share of rows; recurse on each cluster"]
```

- **Independence test:** the Randomized Dependence Coefficient (RDC). Columns are split into groups when their
  dependence is below the threshold 0.3.
- **Constant columns** (e.g. SE = 0 for every row of a slice) are split off as independent before the RDC test,
  which cannot handle zero variance.
- **The label stays connected:** while Y still varies within a slice, it is never split into a group of its own.
  It joins the feature it depends on most, so that P(Y | Z) can depend on Z everywhere in the network.
- **Deterministic:** structure learning is seeded (`spn.seed: 42`).

## Leaves

**Label leaf** (`BinaryIndicatorLeaf`): θ is the share of wrong answers among the training rows that reach the
leaf, Laplace-smoothed with pseudo-count 1 (`spn.prior_smoothing`).

**Feature leaves** (`HistogramLeaf`, `spn.leaf: histogram`): a piecewise-constant density per feature:
- **Bins:** 10 bins of equal width (`strategy: uniform`) between the feature's minimum and maximum over the full
  training data, after standardization.
- **Smoothing:** each bin's probability is Laplace-smoothed (pseudo-count 1), so no bin has density 0.
- **Tails:** two exponential tails beyond the training range, so a test value outside it gets a small, non-zero
  density instead of 0.

Gaussian (`normal`) and Laplace (`laplace`) leaves are also available.

## Inputs per architecture

The same SPN, with the same settings, is trained once per SPN method. Only its input differs (details in
[scores per architecture](methods.md)):

| Architecture | SPN methods | Features |
|---|---|---|
| Feedback loop | RMS SPN, CHMM α = 0.5 / 1 / 2 / 4 SPN | 8: solver and reviewer × (SE, MCSP, NLL, length), RMS- or CHMM-aggregated over each agent's rounds |
| Star | SPN | 20: each of the five responses × (SE, MCSP, NLL, length), raw |
| Sequential | SPN | 12: each of the three agents × (SE, MCSP, NLL, length), raw |

Before learning, each feature is standardized with the training mean and standard deviation, which are saved
with the model and reused on the test split.

## Settings (`spn:`)

| Setting | Value | Role |
|---|---|---|
| `backend` | `spflow` | structure learning and inference library |
| `target` | `incorrect` | Y = 1 means the final answer is wrong |
| `leaf` | `histogram` | feature leaves (`histogram`, `normal`, `laplace`) |
| `histogram.bins` / `strategy` / `smoothing` | 10 / `uniform` / 1.0 | histogram leaves |
| `min_instances_slice` | 50 | below this many rows a slice is fully factorized |
| `min_features_slice` | 2 | the smallest column set that is still split |
| `n_clusters` | 2 | clusters per sum node |
| `rdc_threshold` | 0.3 | independence threshold of the RDC test |
| `keep_label_dependence` | true | never split the varying label off on its own |
| `standardize` | true | standardize features with training statistics |
| `prior_smoothing` | 1.0 | Laplace pseudo-count of the label leaves |
| `seed` | 42 | structure-learning seed |

## Saved models

Each trained SPN is saved as `chmm-<fp>/models/<method>.pt`, holding the network, feature names, standardization
statistics and settings. Next to it, `<method>_status.json` records the training rows, the number of wrong answers,
the learned prior log P(Y) and the structure size (sums, products, leaves).

Loading a model never refits it. The [frozen protocol](evaluation.md#protocols) applies a source experiment's
saved SPNs to another dataset, after checking that the feature names match.
