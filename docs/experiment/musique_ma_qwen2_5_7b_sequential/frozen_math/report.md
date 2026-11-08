# MuSiQue — Sequential — frozen MATH models

## Experimental setup

This experiment measures how well uncertainty scores identify incorrect final answers. The evaluated system answers **101 of 500 test problems correctly (20.20% accuracy)**. The positive class is an incorrect answer; its prevalence is **79.80%**.

MuSiQue-Ans provides answerable multi-hop questions with context paragraphs. The normalized final answer must match the reference or an alias. Training and validation use disjoint subsets of official train; testing uses official dev.

The SPNs, preprocessing, standalone-score calibrators, and Feedback HMM where applicable are trained on MATH with the same architecture and applied unchanged. Target labels are used only for evaluation. The prompts and generated conversations belong to the target dataset.

All agents use Qwen2.5-7B-Instruct through vLLM with bfloat16 precision, temperature 0.7, top-p 1.0, up to 4,096 generated tokens, and a 32,768-token context window. The base seed is 42. Each observed response has ten additional samples from the same prompt. These samples measure uncertainty and do not enter the conversation.

## Architecture and uncertainty models

A solver answers, a verifier checks the answer, and a finalizer produces the graded response. Exactly three calls contribute four raw features each, yielding 12 SPN inputs. There is no temporal aggregation within an agent.

![Architecture and measurement sequence](assets/architecture.svg)

Each response contributes semantic entropy (SE), Monte Carlo sequence probability (MCSP), token negative log-likelihood (NLL), and length. SE measures disagreement among sampled meanings; MCSP averages length-normalized sequence probabilities. Standalone MCSP is negated so larger scores indicate greater error risk. RMS baselines pool all responses; last-step baselines use the chronologically last response.

The joint SPN learns P(Z,Y) and estimates P(Y=1|Z). The class-conditional model learns separate feature densities for correct and incorrect answers and combines them with empirical training priors. Both use the same features. Forty-eight configurations vary leaf family, RDC threshold, minimum row slice, and leaf regularization; validation Brier score selects the model. Standardization uses training statistics. Frozen evaluations reuse the MATH-selected settings.

## Results

Higher AUROC, AUARC, and average precision are better; lower AURC, Brier, and ECE are better. **Bold** marks the best value and <u>underlining</u> marks the second-best distinct value within each table. Rankings use unrounded values and preserve ties. Brier and ECE use training-fitted logistic calibration for standalone scores and direct SPN probabilities. AUARC equals 1−AURC.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| RMS SE | **0.7824** | <u>0.3563</u> | **0.9365** | <u>0.6437</u> | <u>0.1587</u> | **0.1004** |
| RMS MCSP | 0.7439 | 0.3341 | 0.9138 | 0.6659 | 0.3716 | 0.4753 |
| Last-step SE | 0.6600 | 0.2644 | 0.8688 | 0.7356 | 0.3132 | 0.3462 |
| Last-step MCSP | 0.6873 | 0.2889 | 0.8960 | 0.7111 | 0.4519 | 0.5353 |
| Joint SPN | <u>0.7793</u> | **0.3757** | <u>0.9250</u> | **0.6243** | **0.1489** | <u>0.1233</u> |
| Class-conditional SPN | 0.6590 | 0.2558 | 0.8794 | 0.7442 | 0.2364 | 0.2323 |

![Uncertainty-method comparison with confidence intervals](assets/metrics.svg)

Points show test estimates; bars show 95% percentile intervals from 10,000 paired bootstrap resamples of test problems. The dotted AUROC line marks 0.5; the dotted average-precision line marks error prevalence. Intervals describe test-sampling variation for fitted models and do not include variation across training seeds.

### Comparison with semantic entropy and answer accuracy

The joint SPN has AUROC **0.7793**, compared with **0.7824** for RMS SE, a difference of **-0.0031**. The class-conditional counterpart has AUROC **0.6590**, a difference of **-0.1202** relative to the joint model. These are point differences; a larger point estimate alone does not establish a reliable improvement.

For selective prediction, joint-SPN AURC is 0.6243, compared with 0.6437 for RMS SE. Brier scores are 0.1489 and 0.1587, respectively. Error ranking, selective risk, and probability calibration therefore need to be assessed separately.

Answer accuracy is 20.20%, leaving 101 correct and 399 incorrect test examples. Accuracy describes the generated answers; AUROC describes separation between correct and incorrect answers. This experiment alone does not establish that task difficulty causes a change in SPN performance. Average precision should be read against the 79.80% error prevalence.

### Frozen versus dataset-specific training

The following point differences are frozen minus dataset-specific results on the same target test problems. Positive AUROC/AUARC differences indicate an increase; positive Brier differences indicate larger probability error. These differences are descriptive, with no paired transfer intervals reported.

| Method | Δ AUROC | Δ AUARC | Δ Brier |
| --- | --- | --- | --- |
| RMS SE | +0.0000 | +0.0000 | +0.0201 |
| RMS MCSP | +0.0000 | +0.0000 | +0.2237 |
| Last-step SE | +0.0000 | +0.0000 | +0.1524 |
| Last-step MCSP | +0.0000 | +0.0000 | +0.2906 |
| Joint SPN | -0.0196 | +0.0054 | +0.0096 |
| Class-conditional SPN | -0.1097 | -0.0813 | +0.0964 |

### Posterior-entropy scores

These scores use h(p)=−p ln(p)−(1−p)ln(1−p), which is largest near p=0.5. They rank uncertainty about the correctness label rather than error probability. Only ranking measures are reported.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ |
| --- | --- | --- | --- | --- |
| Joint SPN — entropy | **0.3411** | **0.1798** | <u>0.7133</u> | **0.8202** |
| Class-conditional SPN — entropy | <u>0.3124</u> | <u>0.1189</u> | **0.7142** | <u>0.8811</u> |

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

You are the Solver, the first agent in a sequential
multi-agent reasoning system.

Your responsibility is to independently solve the
provided question.

INSTRUCTIONS:
1. Read the original question and available context.
2. Identify the relevant domain-specific reasoning steps.
3. Construct a logically supported solution.
4. Check important intermediate conclusions.
5. Do not assume external agents will fix your mistakes.
6. Produce a proposed answer in the dataset-specific format.

Do not simulate the Verifier or Finalizer.
Do not access ground-truth answers.
Do not output TASK_DONE.

FINAL ANSWER FORMAT:
REASONING: <brief multi-hop reasoning>
EVIDENCE: <supporting paragraph IDs>
ANSWER: <short answer>
```

User-message template:

```text
{problem}
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

You are the Verifier, the second agent in a sequential
multi-agent reasoning system.

You receive:
- The original question and permitted context.
- The Solver's full response.

Your responsibility is to independently evaluate the
Solver's proposed answer.

INSTRUCTIONS:
1. Reconsider the original problem independently.
2. Check essential reasoning steps and factual claims.
3. Identify incorrect assumptions and unsupported inference.
4. Determine whether the proposed answer is justified.
5. If incorrect, explain the error and propose a correction.
6. If correct, explicitly confirm the supporting reasoning.
7. If evidence is insufficient, explain the uncertainty.
8. Do not automatically agree with the Solver.

Follow dataset-specific verification rules.

OUTPUT:
VERDICT: CORRECT / INCORRECT / UNCERTAIN
EXPLANATION: <verification reasoning>
VERIFIED_ANSWER: <dataset-specific answer>

Do not simulate the Finalizer.
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

Solver's response:
{solver}
```

### Finalizer

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

You are the Finalizer, the third and final agent in a
sequential multi-agent reasoning system.

You receive:
- The original question and permitted context.
- The Solver's proposed solution.
- The Verifier's assessment and suggested correction.

Your responsibility is to produce the best-supported
final answer.

INSTRUCTIONS:
1. Review both agents' reasoning.
2. Identify agreements and disagreements.
3. Assess which claims are supported by the question,
   supplied evidence, and valid reasoning.
4. Do not automatically trust either agent.
5. Resolve disagreements using available evidence.
6. If the Verifier is wrong, retain the Solver's justified
   answer instead of accepting an unsupported correction.
7. Produce one final response in the dataset-specific format.
8. Do not request further agent executions.
9. Append TASK_DONE exactly once.

The answer must be directly extractable by the dataset grader.

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

Solver's response:
{solver}

Verifier's assessment:
{verifier}

Give the final answer.
```
