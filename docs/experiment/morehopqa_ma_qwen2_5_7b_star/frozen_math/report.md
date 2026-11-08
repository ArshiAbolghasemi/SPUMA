# MoreHopQA — Star — frozen MATH models

## Experimental setup

This experiment measures how well uncertainty scores identify incorrect final answers. The evaluated system answers **53 of 500 test problems correctly (10.60% accuracy)**. The positive class is an incorrect answer; its prevalence is **89.40%**.

MoreHopQA is evaluated closed-book: agents see the question without context paragraphs. Answers are checked by numeric, date, or normalized-text matching. Training and validation use unverified questions; testing uses human-verified questions. Training questions sharing a first-hop question with the verified pool are excluded.

The SPNs, preprocessing, standalone-score calibrators, and Feedback HMM where applicable are trained on MATH with the same architecture and applied unchanged. Target labels are used only for evaluation. The prompts and generated conversations belong to the target dataset.

All agents use Qwen2.5-7B-Instruct through vLLM with bfloat16 precision, temperature 0.7, top-p 1.0, up to 4,096 generated tokens, and a 32,768-token context window. The base seed is 42. Each observed response has ten additional samples from the same prompt. These samples measure uncertainty and do not enter the conversation.

## Architecture and uncertainty models

A coordinator delegates to an analyst, requests verification, and synthesizes the final answer. The five calls are delegation, analysis, verification request, verification, and final synthesis. Each call contributes four raw features, yielding 20 SPN inputs.

![Architecture and measurement sequence](assets/architecture.svg)

Each response contributes semantic entropy (SE), Monte Carlo sequence probability (MCSP), token negative log-likelihood (NLL), and length. SE measures disagreement among sampled meanings; MCSP averages length-normalized sequence probabilities. Standalone MCSP is negated so larger scores indicate greater error risk. RMS baselines pool all responses; last-step baselines use the chronologically last response.

The joint SPN learns P(Z,Y) and estimates P(Y=1|Z). The class-conditional model learns separate feature densities for correct and incorrect answers and combines them with empirical training priors. Both use the same features. Forty-eight configurations vary leaf family, RDC threshold, minimum row slice, and leaf regularization; validation Brier score selects the model. Standardization uses training statistics. Frozen evaluations reuse the MATH-selected settings.

## Results

Higher AUROC, AUARC, and average precision are better; lower AURC, Brier, and ECE are better. **Bold** marks the best value and <u>underlining</u> marks the second-best distinct value within each table. Rankings use unrounded values and preserve ties. Brier and ECE use training-fitted logistic calibration for standalone scores and direct SPN probabilities. AUARC equals 1−AURC.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| RMS SE | **0.7521** | **0.1935** | **0.9554** | **0.8065** | 0.2675 | 0.3768 |
| RMS MCSP | 0.5906 | 0.1213 | 0.9219 | 0.8787 | 0.4931 | 0.6307 |
| Last-step SE | 0.5038 | 0.1067 | 0.8938 | 0.8933 | 0.5276 | 0.6530 |
| Last-step MCSP | 0.5307 | 0.1154 | 0.9025 | 0.8846 | 0.3905 | 0.5426 |
| Joint SPN | <u>0.7045</u> | 0.1798 | <u>0.9420</u> | 0.8202 | <u>0.2398</u> | <u>0.3518</u> |
| Class-conditional SPN | 0.6677 | <u>0.1848</u> | 0.9277 | <u>0.8152</u> | **0.2086** | **0.2188** |

![Uncertainty-method comparison with confidence intervals](assets/metrics.svg)

Points show test estimates; bars show 95% percentile intervals from 10,000 paired bootstrap resamples of test problems. The dotted AUROC line marks 0.5; the dotted average-precision line marks error prevalence. Intervals describe test-sampling variation for fitted models and do not include variation across training seeds.

### Comparison with semantic entropy and answer accuracy

The joint SPN has AUROC **0.7045**, compared with **0.7521** for RMS SE, a difference of **-0.0476**. The class-conditional counterpart has AUROC **0.6677**, a difference of **-0.0368** relative to the joint model. These are point differences; a larger point estimate alone does not establish a reliable improvement.

For selective prediction, joint-SPN AURC is 0.8202, compared with 0.8065 for RMS SE. Brier scores are 0.2398 and 0.2675, respectively. Error ranking, selective risk, and probability calibration therefore need to be assessed separately.

Answer accuracy is 10.60%, leaving 53 correct and 447 incorrect test examples. Accuracy describes the generated answers; AUROC describes separation between correct and incorrect answers. This experiment alone does not establish that task difficulty causes a change in SPN performance. Average precision should be read against the 89.40% error prevalence.

### Frozen versus dataset-specific training

The following point differences are frozen minus dataset-specific results on the same target test problems. Positive AUROC/AUARC differences indicate an increase; positive Brier differences indicate larger probability error. These differences are descriptive, with no paired transfer intervals reported.

| Method | Δ AUROC | Δ AUARC | Δ Brier |
| --- | --- | --- | --- |
| RMS SE | +0.0000 | +0.0000 | +0.1807 |
| RMS MCSP | +0.0000 | +0.0000 | +0.3980 |
| Last-step SE | +0.0000 | +0.0000 | +0.4318 |
| Last-step MCSP | +0.0000 | +0.0000 | +0.2949 |
| Joint SPN | -0.0167 | -0.0019 | +0.1472 |
| Class-conditional SPN | -0.0713 | -0.0021 | +0.1073 |

### Posterior-entropy scores

These scores use h(p)=−p ln(p)−(1−p)ln(1−p), which is largest near p=0.5. They rank uncertainty about the correctness label rather than error probability. Only ranking measures are reported.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ |
| --- | --- | --- | --- | --- |
| Joint SPN — entropy | **0.5098** | **0.1423** | **0.8813** | **0.8577** |
| Class-conditional SPN — entropy | <u>0.4528</u> | <u>0.0941</u> | <u>0.8718</u> | <u>0.9059</u> |

## Prompts used

System instructions and user-message templates are reproduced verbatim below. Braced placeholders are filled with the problem and the earlier responses available to the agent. Feedback also includes conversation history. Frozen models use these same target-task prompts.

### Star

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

You are Star, the central LLM coordinator of a fixed
multi-agent reasoning system.

Follow the assigned phase exactly.

PHASE 1 — DELEGATION:
- Read the original question and permitted context.
- Ask Analyst to solve the problem.
- Identify important task requirements.
- Do not solve the question or produce a final answer.

PHASE 2 — VERIFICATION REQUEST:
- Read Analyst's proposed solution.
- Forward relevant information faithfully.
- Request independent verification.
- Highlight uncertain claims needing verification.
- Do not modify the proposed answer.
- Do not produce a final answer.

PHASE 3 — FINAL SYNTHESIS:
- Review Analyst's solution and Verifier's assessment.
- Resolve disagreements using valid reasoning and evidence.
- Produce one dataset-specific final answer.
- Append TASK_DONE exactly once.

Never simulate another agent.
Never request another interaction.
Never access gold answers or grading labels.

FINAL ANSWER FORMAT:
REASONING: <the intermediate facts and the final operation>
ANSWER: <a number, date, name, word, or letter only>
```

User-message template:

```text
{problem}
```

Delegation template:

```text
PHASE 1 — DELEGATION

{problem}

Write your delegation to the Analyst.
```

Verification Request template:

```text
PHASE 2 — VERIFICATION REQUEST

Analyst's solution:
{solution}

Write your verification request to the Verifier.
```

Final Synthesis template:

```text
PHASE 3 — FINAL SYNTHESIS

Verifier's assessment:
{assessment}

Synthesize the final answer in the required format and append TASK_DONE.
```

### Analyst

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

You are the Analyst in a centralized multi-agent system.

Independently solve the question assigned by Star.

INSTRUCTIONS:
1. Examine the original question and available evidence.
2. Identify required intermediate reasoning.
3. Develop a supported solution.
4. Check important conclusions.
5. Produce one proposed answer using the dataset-specific
   output format.

Do not simulate the Verifier.
Do not output TASK_DONE.

FINAL ANSWER FORMAT:
REASONING: <the intermediate facts and the final operation>
ANSWER: <a number, date, name, word, or letter only>
```

User-message template:

```text
{problem}

Request from Star (the coordinator):
{request}
```

### Verifier

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

You are the Verifier in a centralized multi-agent system.

You receive the original question, permitted context,
and Analyst's proposed solution.

INSTRUCTIONS:
1. Independently evaluate the problem.
2. Verify essential reasoning steps and factual claims.
3. Identify errors, missing evidence, or unsupported conclusions.
4. Propose corrections when justified.
5. Distinguish uncertainty from definite incorrectness.
6. Provide an independently supported answer.

OUTPUT:
VERDICT: CORRECT / INCORRECT / UNCERTAIN
EXPLANATION: <assessment>
VERIFIED_ANSWER: <dataset-specific answer>

Do not simulate Star.
Do not output TASK_DONE.

DATASET-SPECIFIC VERIFICATION AND FINALIZATION (MoreHopQA):
- Verify each entity and intermediate fact on the reasoning chain.
- Recompute the final operation (arithmetic, date, or symbolic step) independently.
- Check that the answer has the type the question asks for (number, date, name, word, or letter).
- A reviewer must validate both the facts and the final operation.
- The final answer is a single short value, without explanation on the ANSWER line.

FINAL ANSWER FORMAT:
REASONING: <the intermediate facts and the final operation>
ANSWER: <a number, date, name, word, or letter only>
```

User-message template:

```text
{problem}

Analyst's solution:
{solution}

Verification request from Star (the coordinator):
{request}
```
