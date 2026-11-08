# MuSiQue — Feedback — dataset-specific training

## Experimental setup

This experiment measures how well uncertainty scores identify incorrect final answers. The evaluated system answers **116 of 494 test problems correctly (23.48% accuracy)**. The positive class is an incorrect answer; its prevalence is **76.52%**.

MuSiQue-Ans provides answerable multi-hop questions with context paragraphs. The normalized final answer must match the reference or an alias. Training and validation use disjoint subsets of official train; testing uses official dev.

The uncertainty models use 783 training problems and 197 validation problems. Test problems are held out from fitting and model selection.

Context-length errors exclude 17 training, three validation, and six test conversations. The test results therefore use 494 problems, rather than the 500 problems used by the other MuSiQue architectures.

All agents use Qwen2.5-7B-Instruct through vLLM with bfloat16 precision, temperature 0.7, top-p 1.0, up to 4,096 generated tokens, and a 32,768-token context window. The base seed is 42. Each observed response has ten additional samples from the same prompt. These samples measure uncertainty and do not enter the conversation.

## Architecture and uncertainty models

The solver and reviewer alternate for up to ten rounds. The final solver answer is graded. MATH stops when the reviewer emits TASK_DONE; the other datasets also require VERDICT: CORRECT. Each agent contributes four aggregated features, yielding eight SPN inputs.

![Architecture and measurement sequence](assets/architecture.svg)

Test conversations average **1.907 rounds**; 290 stop after one round and 19 reach ten rounds. A ten-round conversation does not by itself identify the stopping reason.

Each response contributes semantic entropy (SE), Monte Carlo sequence probability (MCSP), token negative log-likelihood (NLL), and length. SE measures disagreement among sampled meanings; MCSP averages length-normalized sequence probabilities. Standalone MCSP is negated so larger scores indicate greater error risk. RMS baselines pool all responses; last-step baselines use the chronologically last response.

RMS SPNs aggregate all four signals separately for each agent using root mean square. CHMM variants use a training-fitted three-state Gaussian HMM over question drift and interaction gap to weight responses. State coefficients are [1, 1+α, 1+2α], with α=1 primary and α∈{0.5,1,2,4} evaluated. The per-agent feature is (T/n_g)Σ w_t x_t. In this architecture, the last-step baseline measures the reviewer, while the final solver response is graded.

The joint SPN learns P(Z,Y) and estimates P(Y=1|Z). The class-conditional model learns separate feature densities for correct and incorrect answers and combines them with empirical training priors. Both use the same features. Forty-eight configurations vary leaf family, RDC threshold, minimum row slice, and leaf regularization; validation Brier score selects the model. Standardization uses training statistics.

## Results

Higher AUROC, AUARC, and average precision are better; lower AURC, Brier, and ECE are better. **Bold** marks the best value and <u>underlining</u> marks the second-best distinct value within each table. Rankings use unrounded values and preserve ties. Brier and ECE use training-fitted logistic calibration for standalone scores and direct SPN probabilities. AUARC equals 1−AURC.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| RMS SE | 0.7404 | 0.3839 | 0.8909 | 0.6161 | <u>0.1594</u> | **0.0494** |
| RMS MCSP | 0.4608 | 0.2023 | 0.7386 | 0.7977 | 0.1857 | 0.0793 |
| Last-step SE | 0.3455 | 0.1688 | 0.7091 | 0.8312 | 0.1763 | 0.1033 |
| Last-step MCSP | 0.3737 | 0.1698 | 0.6873 | 0.8302 | 0.1784 | 0.0867 |
| RMS joint SPN | **0.7681** | **0.3963** | 0.8974 | **0.6037** | **0.1528** | <u>0.0548</u> |
| RMS class-conditional SPN | <u>0.7534</u> | <u>0.3906</u> | **0.9105** | <u>0.6094</u> | 0.1685 | 0.1026 |
| CHMM α=1 joint SPN | 0.7096 | 0.3458 | 0.8776 | 0.6542 | 0.1877 | 0.1583 |
| CHMM α=1 class-conditional SPN | 0.7521 | 0.3871 | <u>0.9051</u> | 0.6129 | 0.1654 | 0.1016 |

![Uncertainty-method comparison with confidence intervals](assets/metrics.svg)

Points show test estimates; bars show 95% percentile intervals from 10,000 paired bootstrap resamples of test problems. The dotted AUROC line marks 0.5; the dotted average-precision line marks error prevalence. Intervals describe test-sampling variation for fitted models and do not include variation across training seeds.

### Comparison with semantic entropy and answer accuracy

The joint SPN has AUROC **0.7681**, compared with **0.7404** for RMS SE, a difference of **+0.0277**. The class-conditional counterpart has AUROC **0.7534**, a difference of **-0.0146** relative to the joint model. These are point differences; a larger point estimate alone does not establish a reliable improvement.

For selective prediction, joint-SPN AURC is 0.6037, compared with 0.6161 for RMS SE. Brier scores are 0.1528 and 0.1594, respectively. Error ranking, selective risk, and probability calibration therefore need to be assessed separately.

Answer accuracy is 23.48%, leaving 116 correct and 378 incorrect test examples. Accuracy describes the generated answers; AUROC describes separation between correct and incorrect answers. This experiment alone does not establish that task difficulty causes a change in SPN performance. Average precision should be read against the 76.52% error prevalence.

At the primary CHMM setting, joint-SPN AUROC is 0.7096, a difference of -0.0585 from RMS aggregation. CHMM and RMS differ in their aggregation rule as well as in their response weights.

### CHMM sensitivity

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ | Brier ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| RMS joint SPN | **0.7681** | <u>0.3963</u> | 0.8974 | <u>0.6037</u> | <u>0.1528</u> | **0.0548** |
| RMS class-conditional SPN | 0.7534 | 0.3906 | **0.9105** | 0.6094 | 0.1685 | 0.1026 |
| CHMM α=0.5 joint SPN | <u>0.7608</u> | **0.4061** | 0.9049 | **0.5939** | **0.1522** | <u>0.0581</u> |
| CHMM α=0.5 class-conditional SPN | 0.7448 | 0.3826 | 0.9005 | 0.6174 | 0.1693 | 0.1115 |
| CHMM α=1 joint SPN | 0.7096 | 0.3458 | 0.8776 | 0.6542 | 0.1877 | 0.1583 |
| CHMM α=1 class-conditional SPN | 0.7521 | 0.3871 | <u>0.9051</u> | 0.6129 | 0.1654 | 0.1016 |
| CHMM α=2 joint SPN | 0.7101 | 0.3646 | 0.8760 | 0.6354 | 0.1825 | 0.1472 |
| CHMM α=2 class-conditional SPN | 0.7160 | 0.3739 | 0.8857 | 0.6261 | 0.1956 | 0.1705 |
| CHMM α=4 joint SPN | 0.7414 | 0.3847 | 0.8892 | 0.6153 | 0.1588 | 0.0716 |
| CHMM α=4 class-conditional SPN | 0.7283 | 0.3759 | 0.8904 | 0.6241 | 0.1704 | 0.1051 |

![AUROC and AUARC at every evaluated CHMM alpha](assets/chmm_alpha.svg)

Alpha=1 remains the primary setting; the displayed test results are not used to select alpha.

### Posterior-entropy scores

These scores use h(p)=−p ln(p)−(1−p)ln(1−p), which is largest near p=0.5. They rank uncertainty about the correctness label rather than error probability. Only ranking measures are reported.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ |
| --- | --- | --- | --- | --- |
| RMS joint SPN — entropy | 0.2578 | 0.1216 | 0.6503 | 0.8784 |
| CHMM α=0.5 joint SPN — entropy | 0.2775 | 0.1175 | 0.6564 | 0.8825 |
| CHMM α=1 joint SPN — entropy | 0.3368 | <u>0.1489</u> | 0.6829 | <u>0.8511</u> |
| CHMM α=2 joint SPN — entropy | <u>0.3484</u> | 0.1484 | <u>0.7024</u> | 0.8516 |
| CHMM α=4 joint SPN — entropy | 0.2895 | 0.1288 | 0.6678 | 0.8712 |
| RMS class-conditional SPN — entropy | 0.2898 | 0.1181 | 0.6632 | 0.8819 |
| CHMM α=0.5 class-conditional SPN — entropy | 0.3065 | 0.1259 | 0.6722 | 0.8741 |
| CHMM α=1 class-conditional SPN — entropy | 0.3329 | 0.1314 | 0.6982 | 0.8686 |
| CHMM α=2 class-conditional SPN — entropy | **0.4740** | **0.2643** | **0.7308** | **0.7357** |
| CHMM α=4 class-conditional SPN — entropy | 0.3160 | 0.1322 | 0.6764 | 0.8678 |

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
REASONING: <brief multi-hop reasoning>
EVIDENCE: <supporting paragraph IDs>
ANSWER: <short answer>
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

DATASET-SPECIFIC VERIFICATION AND FINALIZATION (MuSiQue):
- Validate each required intermediate reasoning hop.
- Check citations against the supplied paragraph IDs.
- Reject answers based solely on distractor passages.
- A reviewer must validate all necessary supporting evidence.
- The final answer is a concise answer with its supporting paragraph IDs.
- Every question is answerable: a final answer is always required.
```

User-message template:

```text
{problem}

Latest response from the Solver:
{previous}

Review the response and give your feedback. End with TASK_DONE if and only if your verdict is CORRECT.
```
