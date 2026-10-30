# MMLU

Four-choice questions from 57 academic and professional subjects (mathematics, law, medicine, history, …).

- **Paper:** Hendrycks et al., 2021, *Measuring Massive Multitask Language Understanding*
  ([arXiv:2009.03300](https://arxiv.org/abs/2009.03300))
- **Source:** [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu), config `all`, pinned revision
  `c30699e8356da336a370243923dbaf21066bb9fe`
- **Loader:** `spuma/dataset/mmlu/`
- **Settings:** `config/<architecture>/mmlu_ma_qwen2_5_7b_<architecture>.yaml`

## Splits

MMLU has no in-distribution training split. Each SPUMA split is drawn from its own official split, so no question
is reused:

| Split | Size | Drawn from |
|---|---|---|
| train | 800 | official **validation** (1,531) |
| validation | 200 | official **dev** (285; 5 per subject) |
| test | 500 | official **test** (14,042), never used for fitting |

`dataset.subsets: [all]` keeps all 57 subjects (or list subject names to restrict them). Problem ids:
`mmlu/<official split>/<subject>/<row>`.

## What the agents see

The question and the four choices, lettered A–D. The gold letter is private.

## Example

`mmlu/test/clinical_knowledge/661`, as the agents see it:

```text
Question:
The process of translation requires the presence of:

Choices:
A. mRNA, tRNA and ribosomes.
B. mRNA, ribosomes and RNA polymerase.
C. DNA, mRNA and RNA polymerase.
D. chromatin, DNA and amino acids.
```

Reference (private): `A`.

A final response the grader accepts:

```text
REASONING: Translation reads mRNA on ribosomes, with tRNA delivering amino acids.
ANSWER: A
```

## Grading

The last `ANSWER:` line of the graded response is read as an option letter (`B`, `(B)`, `B. text` and
`Option B` all give `B`). It is correct iff it equals the gold letter. No parsable letter means wrong, flagged as
a parse failure.

## Prompts

Composed from [`config/prompts/`](../../config/prompts) fragments: `datasets/mmlu.txt` (read all four choices,
return exactly one letter), `verification/mmlu.txt` (evaluate all four options) and `schemas/mmlu.txt`
(`REASONING:` / `ANSWER: <A/B/C/D>`). See [architectures](../architectures/common.md#prompts).
