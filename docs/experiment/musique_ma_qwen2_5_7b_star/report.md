# MuSiQue — Star — dataset-specific training

## Experimental setup

This experiment measures how well uncertainty scores identify incorrect final answers. The evaluated system answers **118 of 500 test problems correctly (23.60% accuracy)**. The positive class is an incorrect answer; its prevalence is **76.40%**.

MuSiQue-Ans provides answerable multi-hop questions with context paragraphs. The normalized final answer must match the reference or an alias. Training and validation use disjoint subsets of official train; testing uses official dev.

The uncertainty models use 800 training problems and 200 validation problems. Test problems are held out from fitting and model selection.

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
| RMS SE | **0.8260** | **0.4279** | **0.9368** | **0.5721** | **0.1371** | **0.0517** |
| RMS MCSP | 0.7756 | 0.4084 | 0.9089 | 0.5916 | 0.1510 | <u>0.0569</u> |
| Last-step SE | 0.5419 | 0.2516 | 0.7838 | 0.7484 | 0.1819 | 0.0708 |
| Last-step MCSP | 0.7413 | 0.3696 | 0.8919 | 0.6304 | 0.1637 | 0.0623 |
| Joint SPN | <u>0.8196</u> | 0.4246 | 0.9316 | 0.5754 | <u>0.1413</u> | 0.0745 |
| Class-conditional SPN | 0.8191 | <u>0.4254</u> | <u>0.9335</u> | <u>0.5746</u> | 0.1516 | 0.1175 |

![Uncertainty-method comparison with confidence intervals](assets/metrics.svg)

Points show test estimates; bars show 95% percentile intervals from 10,000 paired bootstrap resamples of test problems. The dotted AUROC line marks 0.5; the dotted average-precision line marks error prevalence. Intervals describe test-sampling variation for fitted models and do not include variation across training seeds.

### Comparison with semantic entropy and answer accuracy

The joint SPN has AUROC **0.8196**, compared with **0.8260** for RMS SE, a difference of **-0.0065**. The class-conditional counterpart has AUROC **0.8191**, a difference of **-0.0005** relative to the joint model. These are point differences; a larger point estimate alone does not establish a reliable improvement.

For selective prediction, joint-SPN AURC is 0.5754, compared with 0.5721 for RMS SE. Brier scores are 0.1413 and 0.1371, respectively. Error ranking, selective risk, and probability calibration therefore need to be assessed separately.

Answer accuracy is 23.60%, leaving 118 correct and 382 incorrect test examples. Accuracy describes the generated answers; AUROC describes separation between correct and incorrect answers. This experiment alone does not establish that task difficulty causes a change in SPN performance. Average precision should be read against the 76.40% error prevalence.

### Posterior-entropy scores

These scores use h(p)=−p ln(p)−(1−p)ln(1−p), which is largest near p=0.5. They rank uncertainty about the correctness label rather than error probability. Only ranking measures are reported.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ |
| --- | --- | --- | --- | --- |
| Joint SPN — entropy | <u>0.1968</u> | <u>0.0900</u> | <u>0.6193</u> | <u>0.9100</u> |
| Class-conditional SPN — entropy | **0.2636** | **0.1060** | **0.6634** | **0.8940** |

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

You are answering a multi-hop question using supplied
context paragraphs.

Identify the intermediate facts needed to answer the question.
Find supporting evidence within the context.
Connect the facts in a logically valid reasoning chain.

Distinguish relevant passages from distractor paragraphs.
Do not invent intermediate facts or unsupported citations.

Use paragraph IDs when referencing evidence.

Every question can be answered from the supplied paragraphs.
Always give your best-supported answer, even when you are
uncertain.

Never access reference answers, gold supporting-fact labels,
question decomposition, or grading metadata.

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
REASONING: <brief multi-hop reasoning>
EVIDENCE: <supporting paragraph IDs>
ANSWER: <short answer>
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

You are answering a multi-hop question using supplied
context paragraphs.

Identify the intermediate facts needed to answer the question.
Find supporting evidence within the context.
Connect the facts in a logically valid reasoning chain.

Distinguish relevant passages from distractor paragraphs.
Do not invent intermediate facts or unsupported citations.

Use paragraph IDs when referencing evidence.

Every question can be answered from the supplied paragraphs.
Always give your best-supported answer, even when you are
uncertain.

Never access reference answers, gold supporting-fact labels,
question decomposition, or grading metadata.

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
REASONING: <brief multi-hop reasoning>
EVIDENCE: <supporting paragraph IDs>
ANSWER: <short answer>
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

You are answering a multi-hop question using supplied
context paragraphs.

Identify the intermediate facts needed to answer the question.
Find supporting evidence within the context.
Connect the facts in a logically valid reasoning chain.

Distinguish relevant passages from distractor paragraphs.
Do not invent intermediate facts or unsupported citations.

Use paragraph IDs when referencing evidence.

Every question can be answered from the supplied paragraphs.
Always give your best-supported answer, even when you are
uncertain.

Never access reference answers, gold supporting-fact labels,
question decomposition, or grading metadata.

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

DATASET-SPECIFIC VERIFICATION AND FINALIZATION (MuSiQue):
- Validate each required intermediate reasoning hop.
- Check citations against the supplied paragraph IDs.
- Reject answers based solely on distractor passages.
- A reviewer must validate all necessary supporting evidence.
- The final answer is a concise answer with its supporting paragraph IDs.
- Every question is answerable: a final answer is always required.

FINAL ANSWER FORMAT:
REASONING: <brief multi-hop reasoning>
EVIDENCE: <supporting paragraph IDs>
ANSWER: <short answer>
```

User-message template:

```text
{problem}

Analyst's solution:
{solution}

Verification request from Star (the coordinator):
{request}
```
