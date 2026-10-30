# MATH

Competition mathematics problems (algebra to precalculus) with step-by-step reference solutions.

- **Paper:** Hendrycks et al., 2021, *Measuring Mathematical Problem Solving With the MATH Dataset*
  ([arXiv:2103.03874](https://arxiv.org/abs/2103.03874))
- **Source:** [`EleutherAI/hendrycks_math`](https://huggingface.co/datasets/EleutherAI/hendrycks_math) on the Hugging Face Hub, one config per subject
- **Loader:** `spuma/dataset/math/`
- **Settings:** `config/feedbackloop/math_ma_qwen2_5_7b_feedbackloop.yaml`, `config/star/math_ma_qwen2_5_7b_star.yaml`,
  `config/sequential/math_ma_qwen2_5_7b_sequential.yaml`

## Splits

| Split | Size | Drawn from |
|---|---|---|
| train | 800 | official train (7,500 problems) |
| validation | 200 | official train, disjoint from train |
| test | 500 | official test (5,000 problems) |

All seven subjects are used: algebra, counting and probability, geometry, intermediate algebra, number theory,
prealgebra, precalculus. Problem ids: `math/<official split>/<subject>/<row>`.

## What the agents see

The problem text only. The reference solution and the subject/level metadata are private.

## Example

`math/test/precalculus/29` (Precalculus, Level 5)

```text
If $\frac{\sin (2A + B)}{\sin B} = 5,$ then find $\frac{\tan (A + B)}{\tan A}.$
```

Reference solution (private), ending in:

```text
\[\frac{\tan (A + B)}{\tan A} = \boxed{\frac{3}{2}}.\]
```

## Grading

The last `\boxed{...}` of the graded response is compared with the reference's last `\boxed{...}` using
[Math-Verify](https://github.com/huggingface/Math-Verify), so equivalent forms match (`\frac{3}{2}`, `1.5`,
`3/2`). No `\boxed{}` in the response means wrong, flagged as a parse failure.

## Prompts

The MATH experiments keep their own inline prompts in their settings files:
- **Feedback loop:** a Master Mathematician and a Curious Math Enthusiast.
- **Star:** a Star coordinator, an Analyst and a Verifier.
- **Sequential:** an Analyst, a Verifier and a Finalizer.

All ask for the final answer in `\boxed{}`.
