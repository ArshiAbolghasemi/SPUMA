# Uncertainty scores per architecture

From the [four per-response metrics](metrics.md), each experiment builds two kinds of uncertainty scores:
- **Standalone scores:** one number per problem, no model.
- **SPN scores:** a joint sum-product network over a feature vector, which outputs P(wrong) and its posterior
  entropy.

The SPN methods of an architecture differ **only** in how the per-response values are turned into the feature
vector. Everything after that (the SPN, its settings, training and evaluation) is identical.

## Building blocks

| Operation | Definition |
|---|---|
| **RMS** over a set of responses | √( (1/n) Σ xₜ² ) |
| **Last step** | the value of the chronologically last response |
| **Raw** | a response's own value, without combining |
| **CHMM-weighted** over an agent's responses g | (T / n_g) · Σ_{t∈g} wₜ xₜ, with the [CHMM weights](chmm.md) wₜ normalized over all T responses of the trajectory. With equal weights this is the agent's plain mean |
| **P(wrong)** | the SPN's predicted probability that the final answer is wrong, P(Y = 1 \| features) |
| **Posterior entropy** | H = −p ln p − (1 − p) ln(1 − p) of p = P(wrong): highest when the SPN cannot decide. A ranking score only |

Standalone scores that use MCSP are negated (−MCSP), so that for every score higher = more likely wrong.

## Feedback loop

Agents: **solver** and **reviewer**, alternating for up to 10 rounds. Graded: the solver's last response.

| Score | Input | Features |
|---|---|---|
| RMS SE | RMS of SE over **all** solver and reviewer responses | 1 |
| RMS MCSP | RMS of MCSP over all responses, negated | 1 |
| Last-step SE | SE of the chronologically last response (the reviewer's closing turn) | 1 |
| Last-step MCSP | MCSP of that response, negated | 1 |
| RMS SPN → P(wrong), posterior entropy | `solver_se, solver_mcsp, solver_nll, solver_length, reviewer_se, reviewer_mcsp, reviewer_nll, reviewer_length`, each the **RMS** over that agent's own responses | 8 |
| CHMM α SPN → P(wrong), posterior entropy, for α ∈ {0.5, 1, 2, 4} | the same 8 features, each the **CHMM-weighted mean** over that agent's own responses | 8 each |

This is the only architecture where agents respond more than once, so it's the only one where the aggregation (RMS
vs CHMM weighting) is a choice. CHMM α = 1 is the primary CHMM configuration.

## Star

Responses: `star_delegate`, `analyst`, `star_request_verification`, `verifier`, `star_finalize` (always five).
Graded: `star_finalize`.

| Score | Input | Features |
|---|---|---|
| RMS SE | RMS of SE over the 5 responses | 1 |
| RMS MCSP | RMS of MCSP over the 5 responses, negated | 1 |
| Last-step SE | SE of `star_finalize` | 1 |
| Last-step MCSP | MCSP of `star_finalize`, negated | 1 |
| SPN → P(wrong), posterior entropy | the **raw** SE, MCSP, NLL and length of each response: `star_delegate_*`, `analyst_*`, `star_request_verification_*`, `verifier_*`, `star_finalize_*` | 20 |

Every call happens exactly once, so nothing is averaged: there is a single SPN and no RMS or CHMM variant.

## Sequential

Responses: `solver`, `verifier`, `finalizer` (always three). Graded: the finalizer.

| Score | Input | Features |
|---|---|---|
| RMS SE | RMS of SE over the 3 responses | 1 |
| RMS MCSP | RMS of MCSP over the 3 responses, negated | 1 |
| Last-step SE | SE of the finalizer | 1 |
| Last-step MCSP | MCSP of the finalizer, negated | 1 |
| SPN → P(wrong), posterior entropy | the **raw** SE, MCSP, NLL and length of each agent: `solver_*`, `verifier_*`, `finalizer_*` | 12 |

As in Star, each agent responds once, so there is a single SPN.

## Feature names across datasets

Feature names follow the architecture's roles or nodes, not a dataset's agent ids. For example, MATH's
mathematician and analyst are both `solver_*`. This makes models of the same architecture compatible across
datasets, which the [frozen protocol](evaluation.md#protocols) relies on.

## Not used as SPN inputs

CHMM state probabilities, the situational distances, agent-role indicators and reference answers are never SPN
features. NLL is used only as an SPN feature, not as a standalone score.

## Settings

`chmm.feature_layout` selects the layout:
- **`per_agent`** (default): as above.
- **`pooled`:** all responses of a trajectory aggregated into one [SE, MCSP, NLL, length] vector, with RMS and
  CHMM variants for every architecture. Kept to reproduce earlier runs.
