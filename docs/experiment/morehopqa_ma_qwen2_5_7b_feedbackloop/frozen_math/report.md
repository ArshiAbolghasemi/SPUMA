# MoreHopQA — Feedback — frozen MATH models

## Experimental setup

This experiment measures how well uncertainty scores identify incorrect final answers. The evaluated system answers **54 of 500 test problems correctly (10.80% accuracy)**. The positive class is an incorrect answer; its prevalence is **89.20%**.

MoreHopQA is evaluated closed-book: agents see the question without context paragraphs. Answers are checked by numeric, date, or normalized-text matching. Training and validation use unverified questions; testing uses human-verified questions. Training questions sharing a first-hop question with the verified pool are excluded.

The SPNs, preprocessing, standalone-score calibrators, and Feedback HMM where applicable are trained on MATH with the same architecture and applied unchanged. Target labels are used only for evaluation. The prompts and generated conversations belong to the target dataset.

All agents use Qwen2.5-7B-Instruct through vLLM with bfloat16 precision, temperature 0.7, top-p 1.0, up to 4,096 generated tokens, and a 32,768-token context window. The base seed is 42. Each observed response has ten additional samples from the same prompt. These samples measure uncertainty and do not enter the conversation.

## Architecture and uncertainty models

The solver and reviewer alternate for up to ten rounds. The final solver answer is graded. MATH stops when the reviewer emits TASK_DONE; the other datasets also require VERDICT: CORRECT. Each agent contributes four aggregated features, yielding eight SPN inputs.

![Architecture and measurement sequence](assets/architecture.svg)

Test conversations average **1.542 rounds**; 264 stop after one round and 0 reach ten rounds. A ten-round conversation does not by itself identify the stopping reason.

Each response contributes semantic entropy (SE), Monte Carlo sequence probability (MCSP), token negative log-likelihood (NLL), and length. SE measures disagreement among sampled meanings; MCSP averages length-normalized sequence probabilities. Standalone MCSP is negated so larger scores indicate greater error risk. RMS baselines pool all responses; last-step baselines use the chronologically last response.

RMS SPNs aggregate all four signals separately for each agent using root mean square. CHMM variants use a training-fitted three-state Gaussian HMM over question drift and interaction gap to weight responses. State coefficients are [1, 1+α, 1+2α], with α=1 primary and α∈{0.5,1,2,4} evaluated. The per-agent feature is (T/n_g)Σ w_t x_t. In this architecture, the last-step baseline measures the reviewer, while the final solver response is graded.

The joint SPN learns P(Z,Y) and estimates P(Y=1|Z). The class-conditional model learns separate feature densities for correct and incorrect answers and combines them with empirical training priors. Both use the same features. Forty-eight configurations vary leaf family, RDC threshold, minimum row slice, and leaf regularization; validation Brier score selects the model. Standardization uses training statistics. Frozen evaluations reuse the MATH-selected settings.

## Results

Higher AUROC, AUARC, and average precision are better; lower AURC, Brier, and ECE are better. **Bold** marks the best value and <u>underlining</u> marks the second-best distinct value within each table. Rankings use unrounded values and preserve ties. Brier and ECE use training-fitted logistic calibration for standalone scores and direct SPN probabilities. AUARC equals 1−AURC.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| RMS SE | **0.6448** | <u>0.1553</u> | **0.9406** | <u>0.8447</u> | 0.7120 | 0.7762 |
| RMS MCSP | 0.5654 | 0.1242 | 0.9073 | 0.8758 | 0.6407 | 0.7342 |
| Last-step SE | 0.4963 | 0.1072 | 0.8948 | 0.8928 | 0.7476 | 0.8070 |
| Last-step MCSP | 0.4639 | 0.0949 | 0.8867 | 0.9051 | 0.5111 | 0.6441 |
| RMS joint SPN | <u>0.6214</u> | 0.1549 | <u>0.9345</u> | 0.8451 | 0.3118 | 0.4132 |
| RMS class-conditional SPN | 0.5572 | 0.1179 | 0.9031 | 0.8821 | **0.2068** | **0.2031** |
| CHMM α=1 joint SPN | 0.5252 | 0.1175 | 0.9004 | 0.8825 | <u>0.2089</u> | 0.3064 |
| CHMM α=1 class-conditional SPN | 0.6086 | **0.1618** | 0.9085 | **0.8382** | 0.2577 | <u>0.3034</u> |

![Uncertainty-method comparison with confidence intervals](assets/metrics.svg)

Points show test estimates; bars show 95% percentile intervals from 10,000 paired bootstrap resamples of test problems. The dotted AUROC line marks 0.5; the dotted average-precision line marks error prevalence. Intervals describe test-sampling variation for fitted models and do not include variation across training seeds.

### Comparison with semantic entropy and answer accuracy

The joint SPN has AUROC **0.6214**, compared with **0.6448** for RMS SE, a difference of **-0.0234**. The class-conditional counterpart has AUROC **0.5572**, a difference of **-0.0642** relative to the joint model. These are point differences; a larger point estimate alone does not establish a reliable improvement.

For selective prediction, joint-SPN AURC is 0.8451, compared with 0.8447 for RMS SE. Brier scores are 0.3118 and 0.7120, respectively. Error ranking, selective risk, and probability calibration therefore need to be assessed separately.

Answer accuracy is 10.80%, leaving 54 correct and 446 incorrect test examples. Accuracy describes the generated answers; AUROC describes separation between correct and incorrect answers. This experiment alone does not establish that task difficulty causes a change in SPN performance. Average precision should be read against the 89.20% error prevalence.

At the primary CHMM setting, joint-SPN AUROC is 0.5252, a difference of -0.0962 from RMS aggregation. CHMM and RMS differ in their aggregation rule as well as in their response weights.

### CHMM sensitivity

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| RMS joint SPN | **0.6214** | <u>0.1549</u> | **0.9345** | <u>0.8451</u> | 0.3118 | 0.4132 |
| RMS class-conditional SPN | 0.5572 | 0.1179 | 0.9031 | 0.8821 | 0.2068 | 0.2031 |
| CHMM α=0.5 joint SPN | 0.5863 | 0.1454 | <u>0.9289</u> | 0.8546 | 0.2595 | 0.3261 |
| CHMM α=0.5 class-conditional SPN | 0.5548 | 0.1164 | 0.8976 | 0.8836 | **0.1314** | **0.1354** |
| CHMM α=1 joint SPN | 0.5252 | 0.1175 | 0.9004 | 0.8825 | 0.2089 | 0.3064 |
| CHMM α=1 class-conditional SPN | <u>0.6086</u> | **0.1618** | 0.9085 | **0.8382** | 0.2577 | 0.3034 |
| CHMM α=2 joint SPN | 0.5167 | 0.1117 | 0.8951 | 0.8883 | 0.1820 | 0.2329 |
| CHMM α=2 class-conditional SPN | 0.5181 | 0.1348 | 0.9035 | 0.8652 | 0.6482 | 0.7104 |
| CHMM α=4 joint SPN | 0.5860 | 0.1226 | 0.9131 | 0.8774 | 0.2086 | 0.2678 |
| CHMM α=4 class-conditional SPN | 0.5699 | 0.1301 | 0.9036 | 0.8699 | <u>0.1561</u> | <u>0.1619</u> |

![AUROC and AUARC at every evaluated CHMM alpha](assets/chmm_alpha.svg)

Alpha=1 remains the primary setting; the displayed test results are not used to select alpha.

### Frozen versus dataset-specific training

The following point differences are frozen minus dataset-specific results on the same target test problems. Positive AUROC/AUARC differences indicate an increase; positive Brier differences indicate larger probability error. These differences are descriptive, with no paired transfer intervals reported.

| Method | Δ AUROC | Δ AUARC | Δ Brier |
| --- | --- | --- | --- |
| RMS SE | +0.0000 | +0.0000 | +0.6168 |
| RMS MCSP | +0.0000 | +0.0000 | +0.5407 |
| Last-step SE | +0.0000 | +0.0000 | +0.6493 |
| Last-step MCSP | +0.0000 | +0.0000 | +0.4134 |
| RMS joint SPN | +0.0380 | +0.0196 | +0.2058 |
| RMS class-conditional SPN | -0.0828 | -0.0421 | +0.1003 |
| CHMM α=1 joint SPN | -0.1317 | -0.0448 | +0.1111 |
| CHMM α=1 class-conditional SPN | -0.0158 | +0.0138 | +0.1533 |

### Posterior-entropy scores

These scores use h(p)=−p ln(p)−(1−p)ln(1−p), which is largest near p=0.5. They rank uncertainty about the correctness label rather than error probability. Only ranking measures are reported.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ |
| --- | --- | --- | --- | --- |
| RMS joint SPN — entropy | **0.5287** | **0.1373** | 0.8975 | **0.8627** |
| CHMM α=0.5 joint SPN — entropy | 0.4778 | 0.1221 | 0.8913 | 0.8779 |
| CHMM α=1 joint SPN — entropy | 0.4939 | 0.1108 | 0.8931 | 0.8892 |
| CHMM α=2 joint SPN — entropy | 0.5089 | 0.1123 | <u>0.9032</u> | 0.8877 |
| CHMM α=4 joint SPN — entropy | 0.4314 | 0.0937 | 0.8763 | 0.9063 |
| RMS class-conditional SPN — entropy | 0.4214 | 0.0941 | 0.8710 | 0.9059 |
| CHMM α=0.5 class-conditional SPN — entropy | 0.4438 | 0.1035 | 0.8839 | 0.8965 |
| CHMM α=1 class-conditional SPN — entropy | <u>0.5240</u> | 0.1117 | **0.9065** | 0.8883 |
| CHMM α=2 class-conditional SPN — entropy | 0.4884 | <u>0.1313</u> | 0.8689 | <u>0.8687</u> |
| CHMM α=4 class-conditional SPN — entropy | 0.4257 | 0.0966 | 0.8652 | 0.9034 |

## Prompts used

System instructions and user-message templates are reproduced verbatim below. Braced placeholders are filled with the problem and the earlier responses available to the agent. Feedback also includes conversation history. Frozen models use these same target-task prompts.

### Solver

System prompt:

```text
GENERAL RULES:
- You are one agent in a multi-agent reasoning system. Act only in your own role.
- Base your answer on the question, any context given to you, and well-established
  knowledge. Do not invent facts.
- Never access reference answers, grading labels, or grading metadata.
- Follow the requested output format exactly.

You are answering a multi-hop question whose last step is
a reasoning operation.

Identify the entities and the intermediate facts the question
depends on. No context documents are provided: rely on
well-established knowledge.

Then apply the final operation the question asks for, such as
arithmetic, a date calculation, or a symbolic or string
manipulation. Carry it out step by step and check it.

Do not invent intermediate facts. If you are unsure of a fact,
say so.

Give the final answer in its simplest form: a number, a date,
a name, a word, or a letter.

Always give your best answer as a single value. Never answer
"cannot determine" or "unknown".

Never access reference answers or grading metadata.

You are the Solver in an iterative multi-agent reasoning
system.

Your task is to solve the original question and improve
your reasoning using feedback from the Reviewer.

INSTRUCTIONS:
1. Develop a supported answer to the question.
2. Show the essential reasoning steps.
3. Read and critically evaluate Reviewer feedback.
4. Correct identified errors when justified.
5. Reject unsupported corrections politely.
6. Preserve the original question's requirements.
7. Produce the current best answer in the required format.

Do not simulate the Reviewer.
Do not output TASK_DONE.

FINAL ANSWER FORMAT:
REASONING: <the intermediate facts and the final operation>
ANSWER: <a number, date, name, word, or letter only>
```

User-message template:

```text
{problem}
```

### Reviewer

System prompt:

```text
GENERAL RULES:
- You are one agent in a multi-agent reasoning system. Act only in your own role.
- Base your answer on the question, any context given to you, and well-established
  knowledge. Do not invent facts.
- Never access reference answers, grading labels, or grading metadata.
- Follow the requested output format exactly.

You are answering a multi-hop question whose last step is
a reasoning operation.

Identify the entities and the intermediate facts the question
depends on. No context documents are provided: rely on
well-established knowledge.

Then apply the final operation the question asks for, such as
arithmetic, a date calculation, or a symbolic or string
manipulation. Carry it out step by step and check it.

Do not invent intermediate facts. If you are unsure of a fact,
say so.

Give the final answer in its simplest form: a number, a date,
a name, a word, or a letter.

Always give your best answer as a single value. Never answer
"cannot determine" or "unknown".

Never access reference answers or grading metadata.

You are the Reviewer in an iterative multi-agent reasoning
system.

Your responsibility is to critically evaluate the Solver's
latest response.

INSTRUCTIONS:
1. Examine the original question and permitted context.
2. Independently evaluate the Solver's reasoning.
3. Identify factual, logical, or computational errors.
4. Check whether the proposed answer is sufficiently supported.
5. Explain precisely what needs correction.
6. Avoid unnecessary disagreement when the answer is valid.
7. Declare completion only when the answer is adequately
   justified according to the dataset-specific requirements.

OUTPUT:
VERDICT: CORRECT / INCORRECT / UNCERTAIN
FEEDBACK: <specific feedback>
SUGGESTED_ANSWER: <dataset-specific answer>

Append TASK_DONE only when the established Feedback Loop
completion criterion is satisfied: write TASK_DONE only
together with VERDICT: CORRECT, never with INCORRECT or
UNCERTAIN. When you find an error, end without TASK_DONE so
that the Solver can revise its answer.

DATASET-SPECIFIC VERIFICATION AND FINALIZATION (MoreHopQA):
- Verify each entity and intermediate fact on the reasoning chain.
- Recompute the final operation (arithmetic, date, or symbolic step) independently.
- Check that the answer has the type the question asks for (number, date, name, word, or letter).
- A reviewer must validate both the facts and the final operation.
- The final answer is a single short value, without explanation on the ANSWER line.
```

User-message template:

```text
{problem}

Latest response from the Solver:
{previous}

Review the response and give your feedback. End with TASK_DONE if and only if your verdict is CORRECT.
```
