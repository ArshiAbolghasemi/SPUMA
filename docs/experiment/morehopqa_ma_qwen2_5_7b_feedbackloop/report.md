# MoreHopQA — Feedback — dataset-specific training

## Experimental setup

This experiment measures how well uncertainty scores identify incorrect final answers. The evaluated system answers **54 of 500 test problems correctly (10.80% accuracy)**. The positive class is an incorrect answer; its prevalence is **89.20%**.

MoreHopQA is evaluated closed-book: agents see the question without context paragraphs. Answers are checked by numeric, date, or normalized-text matching. Training and validation use unverified questions; testing uses human-verified questions. Training questions sharing a first-hop question with the verified pool are excluded.

The uncertainty models use 480 training problems and 120 validation problems. Test problems are held out from fitting and model selection.

All agents use Qwen2.5-7B-Instruct through vLLM with bfloat16 precision, temperature 0.7, top-p 1.0, up to 4,096 generated tokens, and a 32,768-token context window. The base seed is 42. Each observed response has ten additional samples from the same prompt. These samples measure uncertainty and do not enter the conversation.

## Architecture and uncertainty models

The solver and reviewer alternate for up to ten rounds. The final solver answer is graded. MATH stops when the reviewer emits TASK_DONE; the other datasets also require VERDICT: CORRECT. Each agent contributes four aggregated features, yielding eight SPN inputs.

![Architecture and measurement sequence](assets/architecture.svg)

Test conversations average **1.542 rounds**; 264 stop after one round and 0 reach ten rounds. A ten-round conversation does not by itself identify the stopping reason.

Each response contributes semantic entropy (SE), Monte Carlo sequence probability (MCSP), token negative log-likelihood (NLL), and length. SE measures disagreement among sampled meanings; MCSP averages length-normalized sequence probabilities. Standalone MCSP is negated so larger scores indicate greater error risk. RMS baselines pool all responses; last-step baselines use the chronologically last response.

RMS SPNs aggregate all four signals separately for each agent using root mean square. CHMM variants use a training-fitted three-state Gaussian HMM over question drift and interaction gap to weight responses. State coefficients are [1, 1+α, 1+2α], with α=1 primary and α∈{0.5,1,2,4} evaluated. The per-agent feature is (T/n_g)Σ w_t x_t. In this architecture, the last-step baseline measures the reviewer, while the final solver response is graded.

The joint SPN learns P(Z,Y) and estimates P(Y=1|Z). The class-conditional model learns separate feature densities for correct and incorrect answers and combines them with empirical training priors. Both use the same features. Forty-eight configurations vary leaf family, RDC threshold, minimum row slice, and leaf regularization; validation Brier score selects the model. Standardization uses training statistics.

## Results

Higher AUROC, AUARC, and average precision are better; lower AURC, Brier, and ECE are better. **Bold** marks the best value and <u>underlining</u> marks the second-best distinct value within each table. Rankings use unrounded values and preserve ties. Brier and ECE use training-fitted logistic calibration for standalone scores and direct SPN probabilities. AUARC equals 1−AURC.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| RMS SE | <u>0.6448</u> | 0.1553 | **0.9406** | 0.8447 | **0.0952** | <u>0.0354</u> |
| RMS MCSP | 0.5654 | 0.1242 | 0.9073 | 0.8758 | 0.1000 | 0.0522 |
| Last-step SE | 0.4963 | 0.1072 | 0.8948 | 0.8928 | 0.0983 | 0.0461 |
| Last-step MCSP | 0.4639 | 0.0949 | 0.8867 | 0.9051 | <u>0.0976</u> | **0.0233** |
| RMS joint SPN | 0.5834 | 0.1353 | 0.9261 | 0.8647 | 0.1060 | 0.0729 |
| RMS class-conditional SPN | 0.6400 | <u>0.1600</u> | 0.9308 | <u>0.8400</u> | 0.1065 | 0.0759 |
| CHMM α=1 joint SPN | **0.6569** | **0.1624** | <u>0.9349</u> | **0.8376** | 0.0978 | 0.0549 |
| CHMM α=1 class-conditional SPN | 0.6244 | 0.1479 | 0.9257 | 0.8521 | 0.1044 | 0.0725 |

![Uncertainty-method comparison with confidence intervals](assets/metrics.svg)

Points show test estimates; bars show 95% percentile intervals from 10,000 paired bootstrap resamples of test problems. The dotted AUROC line marks 0.5; the dotted average-precision line marks error prevalence. Intervals describe test-sampling variation for fitted models and do not include variation across training seeds.

### Comparison with semantic entropy and answer accuracy

The joint SPN has AUROC **0.5834**, compared with **0.6448** for RMS SE, a difference of **-0.0614**. The class-conditional counterpart has AUROC **0.6400**, a difference of **+0.0566** relative to the joint model. These are point differences; a larger point estimate alone does not establish a reliable improvement.

For selective prediction, joint-SPN AURC is 0.8647, compared with 0.8447 for RMS SE. Brier scores are 0.1060 and 0.0952, respectively. Error ranking, selective risk, and probability calibration therefore need to be assessed separately.

Answer accuracy is 10.80%, leaving 54 correct and 446 incorrect test examples. Accuracy describes the generated answers; AUROC describes separation between correct and incorrect answers. This experiment alone does not establish that task difficulty causes a change in SPN performance. Average precision should be read against the 89.20% error prevalence.

At the primary CHMM setting, joint-SPN AUROC is 0.6569, a difference of +0.0735 from RMS aggregation. CHMM and RMS differ in their aggregation rule as well as in their response weights.

### CHMM sensitivity

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| RMS joint SPN | 0.5834 | 0.1353 | 0.9261 | 0.8647 | 0.1060 | 0.0729 |
| RMS class-conditional SPN | 0.6400 | 0.1600 | 0.9308 | 0.8400 | 0.1065 | 0.0759 |
| CHMM α=0.5 joint SPN | 0.5596 | 0.1363 | 0.9200 | 0.8637 | 0.1098 | 0.0885 |
| CHMM α=0.5 class-conditional SPN | 0.6550 | 0.1686 | 0.9279 | 0.8314 | <u>0.0969</u> | **0.0479** |
| CHMM α=1 joint SPN | 0.6569 | 0.1624 | **0.9349** | 0.8376 | 0.0978 | <u>0.0549</u> |
| CHMM α=1 class-conditional SPN | 0.6244 | 0.1479 | 0.9257 | 0.8521 | 0.1044 | 0.0725 |
| CHMM α=2 joint SPN | **0.6719** | 0.1691 | <u>0.9333</u> | 0.8309 | 0.0976 | 0.0572 |
| CHMM α=2 class-conditional SPN | 0.6494 | **0.1895** | 0.9273 | **0.8105** | 0.1067 | 0.0967 |
| CHMM α=4 joint SPN | <u>0.6680</u> | <u>0.1858</u> | 0.9290 | <u>0.8142</u> | **0.0955** | 0.0595 |
| CHMM α=4 class-conditional SPN | 0.5643 | 0.1388 | 0.8919 | 0.8612 | 0.1061 | 0.0934 |

![AUROC and AUARC at every evaluated CHMM alpha](assets/chmm_alpha.svg)

Alpha=1 remains the primary setting; the displayed test results are not used to select alpha.

### Posterior-entropy scores

These scores use h(p)=−p ln(p)−(1−p)ln(1−p), which is largest near p=0.5. They rank uncertainty about the correctness label rather than error probability. Only ranking measures are reported.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ |
| --- | --- | --- | --- | --- |
| RMS joint SPN — entropy | 0.4173 | 0.0755 | **0.8717** | 0.9245 |
| CHMM α=0.5 joint SPN — entropy | **0.4407** | 0.0813 | <u>0.8685</u> | 0.9187 |
| CHMM α=1 joint SPN — entropy | 0.3431 | 0.0667 | 0.8424 | 0.9333 |
| CHMM α=2 joint SPN — entropy | 0.3285 | 0.0682 | 0.8392 | 0.9318 |
| CHMM α=4 joint SPN — entropy | 0.3320 | 0.0723 | 0.8247 | 0.9277 |
| RMS class-conditional SPN — entropy | 0.3601 | 0.0706 | 0.8385 | 0.9294 |
| CHMM α=0.5 class-conditional SPN — entropy | 0.3455 | 0.0736 | 0.8391 | 0.9264 |
| CHMM α=1 class-conditional SPN — entropy | 0.3750 | 0.0755 | 0.8518 | 0.9245 |
| CHMM α=2 class-conditional SPN — entropy | 0.4076 | <u>0.0816</u> | 0.8675 | <u>0.9184</u> |
| CHMM α=4 class-conditional SPN — entropy | <u>0.4354</u> | **0.1105** | 0.8625 | **0.8895** |

## Prompts used

System instructions and user-message templates are reproduced verbatim below. Braced placeholders are filled with the problem and the earlier responses available to the agent. Feedback also includes conversation history.

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
