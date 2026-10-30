# Per-response uncertainty metrics

Every agent response is measured right after it is generated, inside the workflow, by the generating engine
(`spuma/uncertainty/estimators.py`). Every uncertainty score in SPUMA is built from these four numbers.

## Monte Carlo samples

For a response y⁰ generated from the context c, the engine draws **n = 10 samples** y¹ … y¹⁰ from exactly the same
prompt tokens:
- **Sampling:** temperature 0.7, top-p 1.0, each sample with its own deterministic seed.
- **Recorded per text:** its token log-probabilities under the raw model distribution.
- **Shared:** SE and MCSP use the same 11 texts (the response and its 10 samples); they are never drawn twice.

Before scoring, a `TASK_DONE` marker is removed from any text that contains it, and that text's log-probabilities
are recomputed by teacher forcing on the same engine. The marker therefore never enters a score.

The length-normalized sequence probability of a text y with L tokens is

```text
P(y) = exp( (1/L) Σ_t log p(y_t | c, y_<t) )
```

## Semantic entropy (SE)

*Do the response and its samples mean the same thing?*

1. Cluster the 11 texts by meaning. A text joins every cluster whose first member is equivalent to it.
2. Weight each cluster by probability: p_c = Σ_{y ∈ c} P(y) / Σ_y P(y).
3. **SE = −Σ_c p_c ln p_c** (nats).

| Value | Meaning |
|---|---|
| 0 | all 11 texts mean the same thing |
| up to ln 11 ≈ 2.4 | every text means something different |

Higher SE = more uncertain. Implemented with UQLM's semantic-entropy scorer (white-box, `tokenprob`).

**Equivalence is answer-aware** (`spuma/uncertainty/equivalence.py`):

| Both texts state an answer | Equivalent iff |
|---|---|
| MATH | their last `\boxed{}` answers are equal under Math-Verify (either direction) |
| MMLU | their last `ANSWER:` / `VERIFIED_ANSWER:` / `SUGGESTED_ANSWER:` lines give the same option letter |
| MuSiQue | those lines are equal after normalization (case, punctuation, articles, whitespace) |
| MoreHopQA | `\boxed{}` or those lines: the same number, the same date, or equal normalized text |

When only one of the two texts states an answer, they are not equivalent. When neither does (a delegation, a
critique, an unfinished derivation), an NLI model (`tasksource/ModernBERT-large-nli`) decides by entailment in
either direction.

## Monte Carlo sequence probability (MCSP)

*How probable does the model find its own outputs?*

```text
MCSP = (1/11) Σ_{i=0..10} P(yⁱ)
```

The mean length-normalized probability of the response and its 10 samples (UQLM's white-box
`monte_carlo_probability`). **Higher MCSP = more confident.** Wherever a risk score is needed (higher = more
likely wrong), SPUMA uses −MCSP.

## Negative log-likelihood (NLL)

*How surprised is the model by the response it actually gave?*

```text
NLL = −(1/L) Σ_t log p(y⁰_t | c, y⁰_<t)
```

The mean token negative log-likelihood of the response alone (no samples). Higher NLL = more uncertain.

## Token length

The number of generated tokens of the response (after removing `TASK_DONE`). Length has no fixed direction:
- **Long:** can mean an agent is struggling.
- **Short:** can mean an agent is guessing.
- **Per task:** some tasks simply need long answers. The SPN learns the relation for each architecture and dataset.

## Where the values are stored

- **Trajectories:** each step in `trajectories/<problem>.json` and `scored-<fp>/steps.jsonl` carries `metrics`
  (`se`, `mcsp`, `nll`, `length`) and `uq_details`: the samples, their seeds and log-probabilities, the clusters
  and the scored texts.
- **Flat rows:** `scored-<fp>/step_metrics.json` has one row per response.
