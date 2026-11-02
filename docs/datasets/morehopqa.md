# MoreHopQA

Multi-hop questions that extend a HotpotQA-style two-hop question with a final reasoning step: arithmetic, a
date calculation, or a symbolic or string operation. SPUMA uses it closed-book: the agents see only the question.

- **Paper:** Schnitzler et al., 2024, *MoreHopQA: More Than Multi-hop Reasoning*
  ([arXiv:2406.13397](https://arxiv.org/abs/2406.13397))
- **Source:** [`alabnii/morehopqa`](https://huggingface.co/datasets/alabnii/morehopqa), pinned revision
  `e839079eb8eb686ecf42c6d334a87964f1b90eef` (the files are read directly; the repository's loading script is not run)
- **Loader:** `spuma/dataset/morehopqa/`
- **Settings:** `config/<architecture>/morehopqa_ma_qwen2_5_7b_<architecture>.yaml`

## Sources and splits

| Official file | Content | Used for |
|---|---|---|
| `verified/test-00000-of-00001.parquet` | 1,118 questions, answers verified by humans | **test** (500) |
| `data/without_human_verification.json` | 2,502 generated questions, answers **not** verified by humans | **train** (480) and **validation** (120) |

**Leakage filter.** No unverified question shares an id or question text with the verified set. But several MoreHopQA
questions can be built on the same first-hop HotpotQA question (`previous_question`). Every unverified question
whose first-hop question appears anywhere in the verified set is dropped; 616 remain, from which train and
validation are drawn.

Problem ids:
- **Verified test:** `morehopqa/test/<row>`
- **Unverified train/validation:** `morehopqa/train/<_id>`

Answer types: number, date, datetime, string, letter.

## What the agents see

The question only. The dataset's context paragraphs are not shown. Private: the answer, its type, the number of
hops, the reasoning type and whether it was human-verified.

## Example

`morehopqa/test/224` (date answer; reasoning: commonsense + arithmetic):

```text
What is the date 4 months before when Ting-Xing Ye's husband was born?
```

Reference (private): `1945-06-27`.

The shape of a final response the grader accepts (any parsable form of the same date matches):

```text
REASONING: <who the husband is, his birth date, and the date four months earlier>
ANSWER: June 27, 1945
```

## Grading

The answer is read from the last `\boxed{}` if present, else the last `ANSWER:` line. LaTeX text wrappers and
quotes are removed, then it's compared by answer type:
- **Number:** equal value (thousands separators ignored), else Math-Verify equivalence.
- **Date / datetime:** the same calendar date (and the same minute for datetime) after parsing both.
- **Otherwise:** exact match after normalization (case, punctuation, articles, whitespace).

## Prompts

Composed from [`config/prompts/`](../../config/prompts) fragments:
- **`datasets/morehopqa.txt`:** identify the intermediate facts from well-established knowledge, then apply the
  final operation step by step. Always give a best answer: never "cannot determine" or "unknown".
- **`verification/morehopqa.txt`:** check the facts and recompute the operation.
- **`schemas/morehopqa.txt`:** `REASONING:` / `ANSWER:` with a single short value.

The same dataset (verified set, MATH-style `\boxed{}` prompts) is also used by the out-of-distribution
evaluation `config/feedbackloop/ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop.yaml`.
