# SPUMA: Qwen2.5-7B Mathematical Feedback Experiment

## Introduction

This experiment asks whether uncertainty measured during a two-agent mathematical conversation predicts an incorrect final answer. A Master Mathematician proposes and revises a solution; a Curious Math Enthusiast asks questions, gives feedback, and decides when the discussion is complete. Both roles use Qwen/Qwen2.5-7B-Instruct. A joint sum-product network (SPN) estimates error risk after the conversation.

On 500 held-out test problems, final-answer accuracy is 75.0% (375 correct, 125 incorrect). The joint SPN achieves AUROC 0.9165 and average precision (reported as AUPRC) 0.7555. Although up to ten feedback rounds are allowed, 497/500 test problems stop after the first exchange. Consequently, this run primarily evaluates one Mathematician response followed by one Enthusiast response; it does not establish an accuracy benefit from repeated feedback.

An exploratory single-signal comparison finds that Mathematician RMS semantic entropy alone achieves AUROC 0.9449 and average precision 0.8356, above the joint SPN on this test set. The present run therefore does not demonstrate improved error ranking from combining the eight inputs in the SPN. Raw SE is a ranking score, whereas the SPN also produces probability estimates.

The report covers only the joint SPN. All numeric results and plots are calculated from saved artifacts; no language-model generation, SPN retraining, hyperparameter search, or new bootstrap run is performed. Reported confidence intervals are those stored with the completed evaluation.

## Dataset, usable samples, and target

The dataset is EleutherAI/hendrycks_math on Hugging Face, covering algebra, counting and probability, geometry, intermediate algebra, number theory, prealgebra, and precalculus. Stable problem identifiers are sorted and shuffled with seed 42. Training and validation are disjoint draws from the official train split; test problems come from the official test split.

| Partition | Requested | Usable | Incorrect | Accuracy |
| --- | --- | --- | --- | --- |
| Train | 800 | 795 | 187 | 76.48% |
| Validation | 200 | 200 | 44 | 78.00% |
| Test | 500 | 500 | 125 | 75.00% |

Five training trajectories failed when their prompts exceeded the 8,192-token context window. They were excluded, leaving 795 SPN training examples rather than the requested 800. All 200 validation and 500 test problems have usable features. Such exclusions can bias the training distribution toward shorter or easier conversations and must be disclosed.

$$
Y_i=\begin{cases}0,&\text{final Mathematician answer correct},\\1,&\text{final Mathematician answer incorrect}.\end{cases}
$$

The last Mathematician response is graded, not the Enthusiast response. The evaluator extracts its last boxed answer and compares it with the reference through Math-Verify; absent boxed answers are incorrect. One test answer (0.2%) has a parse failure and remains in the main analysis. If the reference cannot be parsed, grading falls back to normalized string equality. This evaluates final answers, not every reasoning statement or proof step.

## Multi-agent feedback workflow

Each round has two responses. The Mathematician receives the original problem and all previous dialogue and is asked to solve accurately, explain clearly, respond to feedback, and use a boxed final answer. The Enthusiast sees the problem and latest solution, asks for clarification or points out concerns, and is instructed not to solve the problem independently. Each role sees its own previous messages as assistant messages and the other role's messages as user inputs.

The following system prompts are copied from the saved run configuration; only line wrapping is added for readability.

Mathematician system prompt:

```text
You are the Master Mathematician, an expert in mathematical problem-solving. Solve
problems accurately, explain your reasoning clearly, and respond to the Curious Math
Enthusiast's questions and feedback. Correct mistakes when necessary, maintain
conversation context, and present final answers using \boxed{} when appropriate.
```

Enthusiast system prompt:

```text
You are the Curious Math Enthusiast, collaborating with the Master Mathematician.
Carefully examine mathematical solutions, ask relevant questions, request clarifications,
and provide constructive feedback. Do not solve the problem yourself. When the solution is
sufficiently complete and convincing, end your response with TASK_DONE. Otherwise,
continue the discussion naturally.
```

The LangGraph workflow stops when the Enthusiast response contains TASK_DONE, or after ten rounds. Uncertainty is measured after each response but does not enter the dialogue or determine stopping. The Monte Carlo alternatives do not become additional conversation turns. This feedback run has no mathematical tool-execution loop, unlike the earlier Solver-Reviewer experiment.

![Feedback workflow and measurement path. Solid arrows carry the conversation and stopping decision; dashed arrows carry recorded responses to uncertainty measurement. Continued dialogue returns to the Mathematician. The final Mathematician response is graded separately against the reference. The lower path aggregates eight features and queries the joint SPN; neither reference answers nor predicted risk are supplied to the agents.](assets/workflow.svg)

Figure 1. Feedback workflow and measurement path. Solid arrows carry the conversation and stopping decision; dashed arrows carry recorded responses to uncertainty measurement. Continued dialogue returns to the Mathematician. The final Mathematician response is graded separately against the reference. The lower path aggregates eight features and queries the joint SPN; neither reference answers nor predicted risk are supplied to the agents.

All usable trajectories terminate with TASK_DONE. Test round counts are 497 at one round, two at two rounds, and one at four rounds. Mean test length is 1.010 rounds, or 2.020 agent responses. There are 1,010 observed test responses and 3,022 responses across all usable partitions. A completion marker indicates the Enthusiast's stopping decision, not a guarantee of correctness.

| Setting | Saved value |
| --- | --- |
| Language model | Qwen/Qwen2.5-7B-Instruct |
| Generation and token scoring | vLLM; scoring backend same |
| Precision / tensor parallelism | bfloat16 / one shard |
| Sampling | Temperature 0.7; top-p 1.0; no top-k truncation |
| Maximum continuation / context | 2,048 / 8,192 tokens |
| Concurrent problems | 8 |
| Monte Carlo alternatives | 10 per response, from its frozen context |
| NLI fallback | tasksource/ModernBERT-large-nli; batch size 32 |
| NLI limits | 8,000 characters per text; 2,048 tokens per pair |
| Seed / Hub revisions | 42 / model and tokenizer revisions unpinned |

## Step-level confidence and uncertainty measures

This run records Monte Carlo sequence probability (MCSP), likelihood-weighted semantic entropy (SE), token negative log-likelihood (NLL), and generated token length. It does not record full-vocabulary predictive entropy (PE). MCSP is a confidence score: larger means more probable responses. SE and NLL are uncertainty scores: larger means more disagreement or less probable realized tokens. All logarithms are natural and token scores use the raw model distribution.

For context c and response y of length L, the implementation defines a length-normalized sequence score q(y) by the geometric mean of token probabilities. This score lies between zero and one, but is not a normalized probability distribution over all possible response strings.

$$
q(y\mid c)=\exp\!\left(\frac{1}{L}\sum_{t=1}^{L}\log p_\theta(y_t\mid c,y_{<t})\right),\qquad \mathrm{NLL}(y)=-\frac{1}{L}\sum_{t=1}^{L}\log p_\theta(y_t\mid c,y_{<t}).
$$

MCSP averages q over the actual response and its ten independently seeded alternatives. It summarizes sampled response confidence, rather than entropy over the next-token vocabulary.

$$
\mathrm{MCSP}=\frac{1}{11}\sum_{j=0}^{10}q(y^{(j)}\mid c).
$$

SE clusters these eleven responses by meaning. If both have boxed answers, normalized equality or Math-Verify decides equivalence; exactly one boxed answer means different responses; neither boxed answer triggers NLI entailment in either direction. No reference solution is involved. The new run uses UQLM's tokenprob entropy: cluster masses are sums of q, not unweighted cluster counts.

$$
w_k=\sum_{j\in C_k}q(y^{(j)}\mid c),\qquad \pi_k=\frac{w_k}{\sum_\ell w_\ell},\qquad \mathrm{SE}=-\sum_k\pi_k\log\pi_k.
$$

UQLM compares responses with existing cluster representatives, and approximate comparisons may permit overlapping memberships; the denominator sums the resulting cluster weights. This is a sample-based estimate, not exact entropy over all mathematical solutions. Eleven responses give at most eleven clusters and an upper bound ln(11), approximately 2.398 nats. Zero SE means observed semantic agreement, which may still be agreement on an incorrect answer.

The completion marker is removed from scored text and rescored on the generating backend when other text remains. A marker-only response retains its generated tokens. Token length corresponds to this scoring view. This avoids treating the routine TASK_DONE string as substantive content when a nonempty response remains, but marker-only decisions can still affect the signals.

## Exactly what enters the SPN

Each problem produces one eight-dimensional vector Z: four aggregates from each agent. MCSP, SE, and token NLL use root mean square (RMS) across that agent's observed responses; token length uses an arithmetic mean. RMS gives high-valued episodes more influence than a mean. For MCSP this emphasizes high-confidence episodes, not high-uncertainty episodes.

$$
U_{ia}^{(m)}=\sqrt{\frac{1}{T_{ia}}\sum_{s=1}^{T_{ia}}(u_{ias}^{(m)})^2},\qquad \overline L_{ia}=\frac{1}{T_{ia}}\sum_{s=1}^{T_{ia}}L_{ias}.
$$

| Input | Agent | Measurement | Aggregation |
| --- | --- | --- | --- |
| F1 | Mathematician | MCSP confidence | RMS |
| F2 | Mathematician | Semantic entropy | RMS |
| F3 | Mathematician | Token NLL | RMS |
| F4 | Mathematician | Token length | Mean |
| F5 | Enthusiast | MCSP confidence | RMS |
| F6 | Enthusiast | Semantic entropy | RMS |
| F7 | Enthusiast | Token NLL | RMS |
| F8 | Enthusiast | Token length | Mean |

```text
Feature order in the saved checkpoint:
F1: mathematician_mcsp
F2: mathematician_se
F3: mathematician_nll
F4: mathematician_length
F5: enthusiast_mcsp
F6: enthusiast_se
F7: enthusiast_nll
F8: enthusiast_length
```

Raw problem text, reference solutions, final-answer correctness, problem identifiers, split labels, round count, and termination reason are not SPN prediction inputs. Y is an additional observed variable during training only. The joint training table therefore has nine columns: eight features plus Y. At prediction time, the true label is unknown and the model evaluates both hypothetical label values to calculate error probability.

$$
U_{\mathrm{SPUMA}}(Z)=P(Y=1\mid Z)=\frac{P(Z,Y=1)}{P(Z,Y=0)+P(Z,Y=1)}.
$$

This final score expresses the model's estimated risk that the completed answer is wrong. It differs from response-level SE, and from posterior entropy H(Y|Z), which peaks at error probability 0.5. A confidently incorrect answer should have high error risk but can have low posterior entropy.

## Trained joint SPN and why leaves outnumber inputs

LearnSPN learns weighted sums representing mixtures and products representing factorized variable groups. The fitted checkpoint has 46 sum modules, 72 product modules, 75 histogram leaf modules, and 25 binary label leaf modules: 218 statistical modules and 217 edges. Tensor-concatenation and module-list wrappers are implementation details and are not counted as statistical nodes.

The eight inputs identify variables, while leaves identify local distributions within learned mixture branches. The same variable can have different distributions in many branches, so it appears in several leaf modules. A histogram leaf module can also model multiple variables independently. Thus leaf count is neither the number of inputs nor necessarily the number of scalar distributions. A binary label leaf module internally represents a weighted pair of Y=0/Y=1 indicators; those two conceptual indicators are not counted as two separate exported modules.

For example, a two-component mixture of histograms for one variable has one input but two histogram leaves, one distribution for each component. Learned SPN mixtures repeat this pattern across features and branches. In this checkpoint, the 75 histogram modules contain 101 scalar-variable scope occurrences; the 25 label modules add 25 occurrences of Y. None of these repeated local distributions introduces a new input variable.

| Variable | Leaf-module scope occurrences |
| --- | --- |
| F1: mathematician_mcsp | 23 |
| F2: mathematician_se | 25 |
| F3: mathematician_nll | 23 |
| F4: mathematician_length | 13 |
| F5: enthusiast_mcsp | 7 |
| F6: enthusiast_se | 1 |
| F7: enthusiast_nll | 7 |
| F8: enthusiast_length | 2 |
| Y: incorrect label | 25 |

These occurrence counts describe topology, not feature importance. Repeated leaves do not create additional observations, additional input columns, or train/test duplication. The 100 leaf modules are the sum of 75 histogram modules and 25 label modules; histogram bins and tails are parameters inside a leaf, not extra leaves.

![Actual learned root factorization, with deeper subtrees collapsed for readability. F1-F8 correspond to the input table. The root product has a label-dependent sum block over F1,F2,F3,F4,F8,Y; a separate sum block over F5,F7; and a histogram factor over F6. The first block has two weighted subtrees with weights read from the checkpoint. The overview collapses deeper branches and does not depict all 218 modules.](assets/trained_spn.svg)

Figure 2. Actual learned root factorization, with deeper subtrees collapsed for readability. F1-F8 correspond to the input table. The root product has a label-dependent sum block over F1,F2,F3,F4,F8,Y; a separate sum block over F5,F7; and a histogram factor over F6. The first block has two weighted subtrees with weights read from the checkpoint. The overview collapses deeper branches and does not depict all 218 modules.

A consequential structural result is that the Enthusiast's MCSP and NLL (F5,F7) form a factor separate from Y, and Enthusiast SE (F6) forms another separate factor. Both factors cancel in the posterior ratio P(Y=1|Z). In this fitted network, error-risk predictions therefore depend on the four Mathematician inputs plus Enthusiast mean length (F8), even though all eight features were offered to learning. This is a fitted independence assumption, not proof that the excluded signals cannot improve another model or another SPN structure.

| SPN setting | Value |
| --- | --- |
| Structure learner | SPFlow LearnSPN; two-way clustering |
| Minimum rows / feature slice | 50 / 2 |
| RDC dependence threshold | 0.3 |
| Continuous leaf | 10 uniform training-derived bins plus two exponential tails |
| Histogram / label smoothing | 1.0 / 1.0 |
| Standardization | Training means and standard deviations only |
| Label dependence rule | Avoid isolating a varying Y in its own independent product block |

## Observed uncertainty distributions

The Mathematician has mean RMS SE 0.483 nats, with exactly zero SE on 56.0% of test trajectories. The Enthusiast has mean RMS SE 1.586 nats and no exact-zero RMS SE in this test set. Mean RMS MCSP is 0.925 versus 0.636, and mean RMS token NLL is 0.078 versus 0.478, for Mathematician versus Enthusiast respectively.

| Agent | Measure | RMS mean | RMS median | Exact zero (%) |
| --- | --- | --- | --- | --- |
| Mathematician | MCSP | 0.925 | 0.931 | 0.0 |
| Mathematician | SE | 0.483 | 0.000 | 56.0 |
| Mathematician | NLL | 0.078 | 0.067 | 0.0 |
| Enthusiast | MCSP | 0.636 | 0.636 | 0.0 |
| Enthusiast | SE | 1.586 | 1.632 | 0.0 |
| Enthusiast | NLL | 0.478 | 0.427 | 0.0 |

MCSP, SE, and NLL are different quantities and their numerical magnitudes are not directly comparable. Mathematician and Enthusiast also have different tasks: answer production versus open-ended feedback. Role differences cannot be interpreted as isolated effects of reasoning quality. In particular, the Enthusiast's high SE can reflect varied questions or feedback rather than disagreement about a boxed mathematical answer.

![Discriminative distributions of the individual aggregated signals. Rows are Mathematician and Enthusiast; columns are RMS MCSP, SE, and token NLL. Green solid ECDFs represent 375 correct final answers and red dashed ECDFs represent 125 incorrect answers. Each curve is normalized within its group; its height is the fraction at or below the horizontal-axis score. Rightward shifts indicate larger values, but only SE and NLL have the default higher-means-more-uncertain direction. For MCSP, lower values indicate lower confidence. Overlap means no individual signal perfectly detects errors.](assets/feature_distributions.svg)

Figure 3. Discriminative distributions of the individual aggregated signals. Rows are Mathematician and Enthusiast; columns are RMS MCSP, SE, and token NLL. Green solid ECDFs represent 375 correct final answers and red dashed ECDFs represent 125 incorrect answers. Each curve is normalized within its group; its height is the fraction at or below the horizontal-axis score. Rightward shifts indicate larger values, but only SE and NLL have the default higher-means-more-uncertain direction. For MCSP, lower values indicate lower confidence. Overlap means no individual signal perfectly detects errors.

| Agent | Measure | Correct mean | Incorrect mean | Raw AUROC |
| --- | --- | --- | --- | --- |
| Mathematician | MCSP | 0.931 | 0.908 | 0.686 |
| Mathematician | SE | 0.189 | 1.362 | 0.945 |
| Mathematician | NLL | 0.071 | 0.101 | 0.674 |
| Enthusiast | MCSP | 0.634 | 0.643 | 0.476 |
| Enthusiast | SE | 1.565 | 1.648 | 0.574 |
| Enthusiast | NLL | 0.478 | 0.480 | 0.487 |

Single-feature AUROC uses 1-MCSP for confidence and the original SE/NLL for uncertainty. These directions follow metric definitions; they were not selected to maximize test AUROC. An AUROC below 0.5 is retained rather than flipping the score after seeing test results. The single-feature analysis is descriptive and does not measure each input's contribution to the jointly fitted SPN; that would require training ablations.

![Final-response scores (top) versus all-step RMS inputs (bottom). Columns show MCSP confidence, SE in nats, and token NLL in nats/token. Blue is Mathematician; orange is Enthusiast. All curves use the same 500 test problems and are ECDFs. The bottom row contains the uncertainty/confidence inputs supplied to the SPN. The rows are very similar because 99.4% of test conversations contain only one round, for which RMS equals the single observed score.](assets/completed_trajectories.svg)

Figure 4. Final-response scores (top) versus all-step RMS inputs (bottom). Columns show MCSP confidence, SE in nats, and token NLL in nats/token. Blue is Mathematician; orange is Enthusiast. All curves use the same 500 test problems and are ECDFs. The bottom row contains the uncertainty/confidence inputs supplied to the SPN. The rows are very similar because 99.4% of test conversations contain only one round, for which RMS equals the single observed score.

![Per-round response distributions. Rows show MCSP, SE, and token NLL; columns are the observed rounds 1-4. Blue is Mathematician and orange is Enthusiast. Each panel states how many test problems reached that round; round 1 has 500, round 2 has 3, and rounds 3 and 4 each have 1. Within each metric the horizontal range is shared. These later-round samples are too small and too selected to support claims that additional feedback reduces uncertainty.](assets/round_distributions.svg)

Figure 5. Per-round response distributions. Rows show MCSP, SE, and token NLL; columns are the observed rounds 1-4. Blue is Mathematician and orange is Enthusiast. Each panel states how many test problems reached that round; round 1 has 500, round 2 has 3, and rounds 3 and 4 each have 1. Within each metric the horizontal range is shared. These later-round samples are too small and too selected to support claims that additional feedback reduces uncertainty.

An ECDF jump at zero identifies exactly zero scores, which is meaningful for semantic agreement. A curve farther left generally indicates smaller values, and an ECDF height of 0.8 at score x means 80% of that group's observations score at most x. Missing rounds are not imputed as zero. Repeated observations from the same problem are correlated, so evaluation intervals resample whole problems rather than treating every response as an independent test case.

## Discrimination: detecting incorrect final answers

The positive class is an incorrect final Mathematician answer. AUROC is the probability that a randomly selected incorrect answer receives higher risk than a randomly selected correct answer, counting ties as one half. Random ranking has AUROC 0.5. AUPRC is calculated as average precision, not trapezoidal PR area. Its prevalence reference is 0.25 on test and 0.22 on validation.

| Measure | Validation (95% CI) | Test (95% CI) |
| --- | --- | --- |
| AUROC | 0.9242 [0.8754, 0.9661] | 0.9165 [0.8889, 0.9418] |
| AUPRC / average precision | 0.7626 [0.6292, 0.8901] | 0.7555 [0.6757, 0.8344] |
| AUARC | 0.9512 [0.9203, 0.9737] | 0.9413 [0.9228, 0.9574] |
| QA accuracy | 0.7800 [0.7200, 0.8301] | 0.7500 [0.7120, 0.7860] |

| Test ranking score | AUROC | Average precision |
| --- | --- | --- |
| Joint SPN error probability | 0.9165 | 0.7555 |
| Mathematician RMS SE alone | 0.9449 | 0.8356 |

The Mathematician SE baseline requires no error-model fitting. Its better observed ranking shows that combining signals in this fitted SPN does not improve on the strongest individual signal in this test sample. This comparison is exploratory after inspecting individual-signal results; a paired significance test and a fresh confirmatory evaluation were not performed. Unscaled SE is not an error probability and is not used for Brier score or calibration comparisons.

Intervals use 1,000 percentile bootstrap resamples of problem identifiers with 95% confidence, as stored in evaluation.json. They describe uncertainty conditional on this run and test sample; they do not capture variation across independent model generations, training seeds, prompt choices, or datasets. No operating threshold is tuned on the test set in this report.

![Joint-SPN discrimination on 500 test problems. Left: predicted-error histograms normalized separately for correct and incorrect answers; each class&#x27;s bin heights sum to one, so height does not represent its population size. Middle: ROC curve, with false-positive rate on the horizontal axis and true-positive rate on the vertical axis; the dashed diagonal is random ranking. Orange dash-dot curves show the exploratory Mathematician RMS SE baseline. Right: precision versus recall for incorrect answers, with the dashed 0.25 prevalence reference. The curve label AP denotes average precision. These are descriptive curves from the saved predictions, with no threshold selected to maximize a test-set result.](assets/discrimination.svg)

Figure 6. Joint-SPN discrimination on 500 test problems. Left: predicted-error histograms normalized separately for correct and incorrect answers; each class's bin heights sum to one, so height does not represent its population size. Middle: ROC curve, with false-positive rate on the horizontal axis and true-positive rate on the vertical axis; the dashed diagonal is random ranking. Orange dash-dot curves show the exploratory Mathematician RMS SE baseline. Right: precision versus recall for incorrect answers, with the dashed 0.25 prevalence reference. The curve label AP denotes average precision. These are descriptive curves from the saved predictions, with no threshold selected to maximize a test-set result.

The joint SPN ranks incorrect answers usefully, but ranking quality does not establish probability calibration. Posterior entropy ranks test errors less well than direct error probability: AUROC 0.8910 and average precision 0.6258 versus 0.9165 and 0.7555. Confidently wrong answers can have low posterior entropy, so direct error risk is the appropriate primary score for identifying wrong answers in this evaluation.

## Calibration and selective prediction

| Test measure | Value | 95% bootstrap CI |
| --- | --- | --- |
| Brier score | 0.1110 | [0.0946, 0.1287] |
| ECE (15 bins) | 0.0779 | [0.0625, 0.1112] |
| Binary risk NLL | 0.3569 | [0.3151, 0.4010] |
| Area under risk-coverage | 0.0587 | [0.0426, 0.0772] |
| AUARC (area under accuracy-coverage) | 0.9413 | [0.9228, 0.9574] |

Brier score averages squared probability errors. ECE summarizes the absolute difference between predicted and observed error probability over 15 equal-width bins. Binary risk NLL scores the correctness-label probability and differs from the response-token NLL used as an input feature. Lower values are better for these three measures; an ECE of 0.0779 does not justify claiming perfect calibration, and binning can hide local deviations.

AUARC (area under the accuracy-coverage curve) measures how accurately the system answers when it retains progressively larger sets of its lowest-risk answers. Sort answers by increasing predicted error probability, retain the first k of n answers, and calculate their accuracy at coverage k/n. Higher AUARC is better; ties use the expected accuracy under a uniformly random ordering within each tied-score group.

$$
\operatorname{AUARC}=\frac{1}{n}\sum_{k=1}^{n}\operatorname{Accuracy}(k/n)=1-\operatorname{AURC}.
$$

The joint SPN achieves validation AUARC **0.9512** and test AUARC **0.9413**. The test value summarizes retained-set accuracy across all coverage levels; it does not mean that full-test answer accuracy is 94.13%. Full coverage still gives the observed **75.0%** accuracy. AURC averages retained-set error over the same coverage grid. These metrics describe selective ranking on existing answers, not an intervention that improves the generated mathematical solutions.
