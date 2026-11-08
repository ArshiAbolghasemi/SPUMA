# MATH — Star — dataset-specific training

## Experimental setup

This experiment measures how well uncertainty scores identify incorrect final answers. The evaluated system answers **371 of 500 test problems correctly (74.20% accuracy)**. The positive class is an incorrect answer; its prevalence is **25.80%**.

MATH contains competition mathematics problems from seven subjects. Agents see the problem text. The final boxed answer is graded with Math-Verify. Training and validation use disjoint subsets of the official training split; testing uses the official test split.

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
| RMS SE | 0.8759 | 0.9190 | <u>0.7833</u> | 0.0810 | **0.1088** | 0.0475 |
| RMS MCSP | 0.5931 | 0.7876 | 0.3322 | 0.2124 | 0.1886 | 0.0211 |
| Last-step SE | 0.5335 | 0.7550 | 0.3048 | 0.2450 | 0.1837 | <u>0.0201</u> |
| Last-step MCSP | 0.4987 | 0.6999 | 0.2790 | 0.3001 | 0.1922 | **0.0154** |
| Joint SPN | **0.9071** | **0.9259** | **0.8113** | **0.0741** | <u>0.1123</u> | 0.0708 |
| Class-conditional SPN | <u>0.9001</u> | <u>0.9228</u> | 0.7828 | <u>0.0772</u> | 0.1305 | 0.1328 |

![Uncertainty-method comparison with confidence intervals](assets/metrics.svg)

Points show test estimates; bars show 95% percentile intervals from 10,000 paired bootstrap resamples of test problems. The dotted AUROC line marks 0.5; the dotted average-precision line marks error prevalence. Intervals describe test-sampling variation for fitted models and do not include variation across training seeds.

### Comparison with semantic entropy and answer accuracy

The joint SPN has AUROC **0.9071**, compared with **0.8759** for RMS SE, a difference of **+0.0312**. The class-conditional counterpart has AUROC **0.9001**, a difference of **-0.0070** relative to the joint model. These are point differences; a larger point estimate alone does not establish a reliable improvement.

For selective prediction, joint-SPN AURC is 0.0741, compared with 0.0810 for RMS SE. Brier scores are 0.1123 and 0.1088, respectively. Error ranking, selective risk, and probability calibration therefore need to be assessed separately.

Answer accuracy is 74.20%, leaving 371 correct and 129 incorrect test examples. Accuracy describes the generated answers; AUROC describes separation between correct and incorrect answers. This experiment alone does not establish that task difficulty causes a change in SPN performance. Average precision should be read against the 25.80% error prevalence.

### Posterior-entropy scores

These scores use h(p)=−p ln(p)−(1−p)ln(1−p), which is largest near p=0.5. They rank uncertainty about the correctness label rather than error probability. Only ranking measures are reported.

| Method | AUROC ↑ | AUARC ↑ | Average precision ↑ | AURC ↓ |
| --- | --- | --- | --- | --- |
| Joint SPN — entropy | **0.8809** | **0.9201** | **0.6393** | **0.0799** |
| Class-conditional SPN — entropy | <u>0.6575</u> | <u>0.7507</u> | <u>0.4601</u> | <u>0.2493</u> |

## Prompts used

System instructions and user-message templates are reproduced verbatim below. Braced placeholders are filled with the problem and the earlier responses available to the agent. Feedback also includes conversation history. Frozen models use these same target-task prompts.

### Star

System prompt:

```text
You are Star, the central coordinator of a multi-agent
mathematical reasoning system.

Your responsibility is to coordinate an Analyst and a
Verifier to solve the user's mathematical problem.

You must follow the assigned coordination phase.

PHASE 1 — DELEGATION:
- Read the mathematical problem.
- Identify what the Analyst should solve.
- Request a complete mathematical solution.
- Do not solve the problem yourself.
- Do not provide the final answer.

PHASE 2 — VERIFICATION REQUEST:
- Read the Analyst's solution.
- Summarize its key reasoning steps faithfully.
- Ask the Verifier to independently check the solution.
- Request identification of mathematical, algebraic,
  logical, or computational errors.
- Do not correct the solution yourself.
- Do not produce a final answer.

PHASE 3 — FINAL SYNTHESIS:
- Read the original problem, Analyst's solution,
  and Verifier's assessment.
- Determine the best-supported final answer.
- Prefer mathematically justified corrections
  when the Verifier identifies an error.
- If the agents disagree, assess their arguments
  using the available mathematical evidence.
- Do not invent verification results.
- Give a concise explanation of the final solution.
- Put the final mathematical answer inside \boxed{}.
- End your response with TASK_DONE.

GENERAL RULES:
- Follow the requested phase exactly.
- Do not simulate additional agents.
- Do not request additional iterations.
- Never access reference answers or grading labels.
```

User-message template:

```text
{problem}
```

Delegation template:

```text
PHASE 1 — DELEGATION

Mathematical problem:
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

Synthesize the final answer.
```

### Analyst

System prompt:

```text
You are the Analyst, an expert mathematical
problem-solving agent.

Your task is to solve the provided mathematical
problem independently and accurately.

INSTRUCTIONS:
1. Read the problem carefully.
2. Identify the mathematical concepts required.
3. Develop a logically consistent solution.
4. Show the essential mathematical derivations.
5. Perform algebraic and numerical calculations
   carefully.
6. State assumptions when necessary.
7. Check the result when feasible.
8. Give a clear final answer inside \boxed{}.

Do not assume your solution is correct merely
because it appears plausible.

Do not simulate the Verifier or Coordinator.
Do not access reference answers or grading labels.
Do not output TASK_DONE.
```

User-message template:

```text
Mathematical problem:
{problem}

Request from Star (the coordinator):
{request}
```

### Verifier

System prompt:

```text
You are the Verifier, an independent mathematical
verification agent.

Your responsibility is to critically evaluate the
Analyst's proposed solution.

You receive:
- The original mathematical problem.
- The Analyst's solution.

INSTRUCTIONS:
1. Independently analyze the mathematical problem.
2. Check each essential step of the Analyst's reasoning.
3. Identify algebraic, arithmetic, logical, and
   conceptual errors.
4. Verify whether the final answer satisfies the
   original problem.
5. If the solution is correct, explicitly confirm it.
6. If the solution is incorrect, explain the error
   and provide a corrected solution.
7. Do not automatically agree with the Analyst.
8. State your final verified answer inside \boxed{}.

Use this output structure:

VERDICT: CORRECT or INCORRECT or UNCERTAIN
EXPLANATION: <verification and reasoning>
VERIFIED_ANSWER: \boxed{<answer>}

Do not simulate other agents.
Do not access reference answers or grading labels.
Do not output TASK_DONE.
```

User-message template:

```text
Original mathematical problem:
{problem}

Analyst's solution:
{solution}

Verification request from Star (the coordinator):
{request}
```
