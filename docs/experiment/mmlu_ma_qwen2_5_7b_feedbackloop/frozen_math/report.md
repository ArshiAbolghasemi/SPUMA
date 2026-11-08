# MMLU — Feedback — frozen MATH models

## Experimental setup

This experiment measures how well uncertainty scores identify incorrect final answers. The evaluated system answers **373 of 500 test problems correctly (74.60% accuracy)**. The positive class is an incorrect answer; its prevalence is **25.40%**.

MMLU provides academic questions with four answer choices. The final answer letter is compared with the reference. Training uses official validation, validation uses official dev, and testing uses official test.

The SPNs, preprocessing, standalone-score calibrators, and Feedback HMM where applicable are trained on MATH with the same architecture and applied unchanged. Target labels are used only for evaluation. The prompts and generated conversations belong to the target dataset.

All agents use Qwen2.5-7B-Instruct through vLLM with bfloat16 precision, temperature 0.7, top-p 1.0, up to 4,096 generated tokens, and a 32,768-token context window. The base seed is 42. Each observed response has ten additional samples from the same prompt. These samples measure uncertainty and do not enter the conversation.

## Architecture and uncertainty models

The solver and reviewer alternate for up to ten rounds. The final solver answer is graded. MATH stops when the reviewer emits TASK_DONE; the other datasets also require VERDICT: CORRECT. Each agent contributes four aggregated features, yielding eight SPN inputs.

![Architecture and measurement sequence](assets/architecture.svg)

Test conversations average **1.210 rounds**; 432 stop after one round and 1 reach ten rounds. A ten-round conversation does not by itself identify the stopping reason.

Each response contributes semantic entropy (SE), Monte Carlo sequence probability (MCSP), token negative log-likelihood (NLL), and length. SE measures disagreement among sampled meanings; MCSP averages length-normalized sequence probabilities. Standalone MCSP is negated so larger scores indicate greater error risk. RMS baselines pool all responses; last-step baselines use the chronologically last response.

RMS SPNs aggregate all four signals separately for each agent using root mean square. CHMM variants use a training-fitted three-state Gaussian HMM over question drift and interaction gap to weight responses. State coefficients are [1, 1+α, 1+2α], with α=1 primary and α∈{0.5,1,2,4} evaluated. The per-agent feature is (T/n_g)Σ w_t x_t. In this architecture, the last-step baseline measures the reviewer, while the final solver response is graded.

The joint SPN learns P(Z,Y) and estimates P(Y=1|Z). The class-conditional model learns separate feature densities for correct and incorrect answers and combines them with empirical training priors. Both use the same features. Forty-eight configurations vary leaf family, RDC threshold, minimum row slice, and leaf regularization; validation Brier score selects the model. Standardization uses training statistics. Frozen evaluations reuse the MATH-selected settings.

## Results

Higher AUROC, AUARC, and average precision are better; lower AURC, Brier, and ECE are better. **Bold** marks the best value and <u>underlining</u> marks the second-best distinct value within each table. Rankings use unrounded values and preserve ties. Brier and ECE use training-fitted logistic calibration for standalone scores and direct SPN probabilities. AUARC equals 1−AURC.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| RMS SE | **0.8290** | **0.8992** | **0.6170** | **0.1008** | 0.2496 | 0.2503 |
| RMS MCSP | 0.5903 | 0.7768 | 0.3434 | 0.2232 | 0.2663 | 0.2570 |
| Last-step SE | 0.5290 | 0.7573 | 0.3027 | 0.2427 | 0.2164 | 0.1684 |
| Last-step MCSP | 0.5206 | 0.7093 | 0.3366 | 0.2907 | <u>0.1896</u> | **0.0118** |
| RMS joint SPN | <u>0.8083</u> | <u>0.8888</u> | <u>0.5765</u> | <u>0.1112</u> | **0.1834** | <u>0.1370</u> |
| RMS class-conditional SPN | 0.5690 | 0.8015 | 0.2911 | 0.1985 | 0.6467 | 0.6491 |
| CHMM α=1 joint SPN | 0.5956 | 0.7973 | 0.3800 | 0.2027 | 0.3153 | 0.3461 |
| CHMM α=1 class-conditional SPN | 0.5906 | 0.8151 | 0.3032 | 0.1849 | 0.6024 | 0.6092 |

![Uncertainty-method comparison with confidence intervals](assets/metrics.svg)

Points show test estimates; bars show 95% percentile intervals from 10,000 paired bootstrap resamples of test problems. The dotted AUROC line marks 0.5; the dotted average-precision line marks error prevalence. Intervals describe test-sampling variation for fitted models and do not include variation across training seeds.

### Comparison with semantic entropy and answer accuracy

The joint SPN has AUROC **0.8083**, compared with **0.8290** for RMS SE, a difference of **-0.0207**. The class-conditional counterpart has AUROC **0.5690**, a difference of **-0.2393** relative to the joint model. These are point differences; a larger point estimate alone does not establish a reliable improvement.

For selective prediction, joint-SPN AURC is 0.1112, compared with 0.1008 for RMS SE. Brier scores are 0.1834 and 0.2496, respectively. Error ranking, selective risk, and probability calibration therefore need to be assessed separately.

Answer accuracy is 74.60%, leaving 373 correct and 127 incorrect test examples. Accuracy describes the generated answers; AUROC describes separation between correct and incorrect answers. This experiment alone does not establish that task difficulty causes a change in SPN performance. Average precision should be read against the 25.40% error prevalence.

At the primary CHMM setting, joint-SPN AUROC is 0.5956, a difference of -0.2127 from RMS aggregation. CHMM and RMS differ in their aggregation rule as well as in their response weights.

### CHMM sensitivity

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| RMS joint SPN | **0.8083** | <u>0.8888</u> | **0.5765** | <u>0.1112</u> | <u>0.1834</u> | <u>0.1370</u> |
| RMS class-conditional SPN | 0.5690 | 0.8015 | 0.2911 | 0.1985 | 0.6467 | 0.6491 |
| CHMM α=0.5 joint SPN | <u>0.8031</u> | **0.8892** | <u>0.5516</u> | **0.1108** | **0.1695** | **0.1100** |
| CHMM α=0.5 class-conditional SPN | 0.5608 | 0.7688 | 0.2956 | 0.2312 | 0.7202 | 0.7302 |
| CHMM α=1 joint SPN | 0.5956 | 0.7973 | 0.3800 | 0.2027 | 0.3153 | 0.3461 |
| CHMM α=1 class-conditional SPN | 0.5906 | 0.8151 | 0.3032 | 0.1849 | 0.6024 | 0.6092 |
| CHMM α=2 joint SPN | 0.5313 | 0.7688 | 0.3108 | 0.2312 | 0.3835 | 0.4351 |
| CHMM α=2 class-conditional SPN | 0.6944 | 0.8494 | 0.3603 | 0.1506 | 0.2473 | 0.2467 |
| CHMM α=4 joint SPN | 0.6148 | 0.7943 | 0.3587 | 0.2057 | 0.3499 | 0.3803 |
| CHMM α=4 class-conditional SPN | 0.5770 | 0.8038 | 0.3050 | 0.1962 | 0.6407 | 0.6465 |

![AUROC and AUARC at every evaluated CHMM alpha](assets/chmm_alpha.svg)

Alpha=1 remains the primary setting; the displayed test results are not used to select alpha.

### Frozen versus dataset-specific training

The following point differences are frozen minus dataset-specific results on the same target test problems. Positive AUROC/AUARC differences indicate an increase; positive Brier differences indicate larger probability error. These differences are descriptive, with no paired transfer intervals reported.

| Method | Δ AUROC | Δ AUARC | Δ Brier |
| --- | --- | --- | --- |
| RMS SE | +0.0000 | +0.0000 | +0.1146 |
| RMS MCSP | +0.0000 | +0.0000 | +0.0773 |
| Last-step SE | +0.0000 | +0.0000 | +0.0295 |
| Last-step MCSP | +0.0000 | +0.0000 | -0.0017 |
| RMS joint SPN | -0.0251 | -0.0073 | +0.0503 |
| RMS class-conditional SPN | -0.2454 | -0.0922 | +0.5011 |
| CHMM α=1 joint SPN | -0.2149 | -0.0778 | +0.1736 |
| CHMM α=1 class-conditional SPN | -0.2144 | -0.0833 | +0.4560 |

### Posterior-entropy scores

These scores use h(p)=−p ln(p)−(1−p)ln(1−p), which is largest near p=0.5. They rank uncertainty about the correctness label rather than error probability. Only ranking measures are reported.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ |
| --- | --- | --- | --- | --- |
| RMS joint SPN — entropy | **0.8079** | <u>0.8887</u> | **0.5700** | <u>0.1113</u> |
| CHMM α=0.5 joint SPN — entropy | <u>0.8019</u> | **0.8889** | <u>0.5476</u> | **0.1111** |
| CHMM α=1 joint SPN — entropy | 0.4637 | 0.7310 | 0.2663 | 0.2690 |
| CHMM α=2 joint SPN — entropy | 0.5112 | 0.7591 | 0.2910 | 0.2409 |
| CHMM α=4 joint SPN — entropy | 0.4332 | 0.7221 | 0.2308 | 0.2779 |
| RMS class-conditional SPN — entropy | 0.4546 | 0.7250 | 0.2386 | 0.2750 |
| CHMM α=0.5 class-conditional SPN — entropy | 0.4518 | 0.7239 | 0.2409 | 0.2761 |
| CHMM α=1 class-conditional SPN — entropy | 0.4305 | 0.7131 | 0.2305 | 0.2869 |
| CHMM α=2 class-conditional SPN — entropy | 0.6944 | 0.8494 | 0.3603 | 0.1506 |
| CHMM α=4 class-conditional SPN — entropy | 0.4330 | 0.7117 | 0.2223 | 0.2883 |

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

You are answering an academic multiple-choice question.

Read the original question and all four choices carefully.
Identify the relevant facts, principles, and reasoning.

Evaluate the choices and select the best-supported option.
Do not assume that a confident explanation is correct.
Do not invent facts or modify the answer options.

Return exactly one option letter: A, B, C, or D.

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
REASONING: <concise explanation>
ANSWER: <A/B/C/D>
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

You are answering an academic multiple-choice question.

Read the original question and all four choices carefully.
Identify the relevant facts, principles, and reasoning.

Evaluate the choices and select the best-supported option.
Do not assume that a confident explanation is correct.
Do not invent facts or modify the answer options.

Return exactly one option letter: A, B, C, or D.

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

DATASET-SPECIFIC VERIFICATION AND FINALIZATION (MMLU):
- Independently evaluate all four options.
- Explain why the selected option is better supported.
- A reviewer must validate the option choice.
- A verified answer is one candidate option letter: A, B, C, or D.
- The final answer is exactly one option letter.
```

User-message template:

```text
{problem}

Latest response from the Solver:
{previous}

Review the response and give your feedback. End with TASK_DONE if and only if your verdict is CORRECT.
```
