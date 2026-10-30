# MuSiQue

Multi-hop questions (2–4 hops) composed from single-hop questions, each with 20 context paragraphs: the
paragraphs that support the answer plus distractors.

- **Paper:** Trivedi et al., 2022, *MuSiQue: Multihop Questions via Single-hop Question Composition*
  ([arXiv:2108.00573](https://arxiv.org/abs/2108.00573))
- **Official release:** [github.com/StonyBrookNLP/musique](https://github.com/StonyBrookNLP/musique)
- **Source used:** [`dgslibisey/MuSiQue`](https://huggingface.co/datasets/dgslibisey/MuSiQue), a mirror of the official
  MuSiQue-Ans v1.0 files (19,938 train / 2,417 dev questions, official schema), pinned revision
  `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321`
- **Loader:** `spuma/dataset/musique/`
- **Settings:** `config/<architecture>/musique_ma_qwen2_5_7b_<architecture>.yaml`

## Variants

| Variant | Content | How to use |
|---|---|---|
| MuSiQue-Ans (default) | answerable questions | `dataset.variant: ans` |
| MuSiQue-Full | answerable and unanswerable contrast pairs | `dataset.variant: full`, `source: local`, `data_files` pointing at the official `musique_full_v1.0_{train,dev}.jsonl` |

## Splits

The official test files have no answers, so:

| Split | Size | Drawn from |
|---|---|---|
| train | 800 | official train |
| validation | 200 | official train, disjoint from train |
| test | 500 | official **dev** |

Problem ids: `musique_<variant>/<official split>/<id>`.

## What the agents see

The question and all 20 paragraphs, each as `[idx] title` followed by its text, supporting and distractor
paragraphs alike. Private: the answer and its aliases, answerability, the gold question decomposition and which
paragraphs are supporting.

## Example

`musique_ans/dev/4hop1__88342_75218_128008_92225` (4 hops), as the agents see it (shortened):

```text
Question:
When is the opening day of the league, that includes the team that played in the most games. of the event that
the MLB MVP award is handed out after?

Context paragraphs:

[0] Tim Scott (baseball)
Drafted by the Los Angeles Dodgers in the 2nd round of the 1984 MLB amateur draft, ...

[1] Exhibition game
Several MLB teams used to play regular exhibition games during the year against nearby teams ...

... (20 paragraphs in total)
```

Private grading data:
- **Answer:** `March 29, 2018`
- **Supporting paragraphs:** 3, 4, 5, 17
- **Decomposition:** *when do they give out the mlb mvp award* → *who played in the most #1 games* →
  *Which is the league of #2?* → *when is opening day of #3 season*

The shape of a final response the grader accepts:

```text
REASONING: <each hop, resolved from the cited paragraphs>
EVIDENCE: [3], [4], [5], [17]
ANSWER: March 29, 2018
```

## Grading

The last `ANSWER:` line, normalized (lower case, no punctuation or articles, collapsed whitespace), must equal
the normalized answer or one of its aliases. Answering `INSUFFICIENT_INFORMATION` to an answerable question is
wrong. For MuSiQue-Full's unanswerable questions, only `INSUFFICIENT_INFORMATION` is correct.

## Prompts

Composed from [`config/prompts/`](../../config/prompts) fragments:
- **`datasets/musique.txt`:** use the paragraphs, cite paragraph ids, ignore distractors, and report
  `INSUFFICIENT_INFORMATION` when the context does not support an answer.
- **`verification/musique.txt`:** validate each hop and each citation.
- **`schemas/musique.txt`:** the `REASONING:` / `EVIDENCE:` / `ANSWER:` format.

The question plus 20 paragraphs can reach about 5k tokens, which is why the engine's context length is 32,768.
