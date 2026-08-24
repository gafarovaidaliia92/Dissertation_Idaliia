# RQ3 — the bridge between RQ1 vulnerability and RQ2 reaction

**Question.** Do the two proxies meet? Does a bank the RQ1 model calls vulnerable
react differently to CBDC communications?

- **H3** — more vulnerable -> more negative CAR. Coefficients: **delta1** (level)
  and **b** on `S x vulnerability` (reaction).

Code: `rq3_link.py` -> `rq3_interaction.py` -> `rq3_measures.py`.

> **The interaction lives here now.** It used to sit in RQ2 labelled as the H2
> test. It is not one: it asks whether vulnerable banks react differently, which
> is this question. See `../_archive/` for the superseded version.

Three measures, all oriented **higher = more vulnerable**: RF score, OLS score,
`uninsured_share`.

## 1. Level link — delta1 (`rq3_link.txt`)

| Measure | delta1 (event FE) | p |
|---|---|---|
| RF score | +0.06011100 | 0.33223292 |
| OLS score | +0.01232232 | 0.78213952 |
| **`uninsured_share`** | **-0.01877493** | **0.00032066** |

The published RQ3 null is a statement about the **ML score**. It does not extend
to `uninsured_share`, which has a real level effect with the H3-predicted sign.

## 2. Interaction — b (`rq3_interaction.txt`)

`CAR = EventFE + BankFE + b*(S x vulnerability) + e`

| Measure | b | SE (event) | p (event) | p (two-way) |
|---|---|---|---|---|
| `uninsured_share` | -0.00049680 | 0.02134018 | 0.98188477 | 0.98153708 |
| RF score | -0.38507629 | 0.12471181 | 0.01148834 | 0.02208081 |

Verified numerically: S is absorbed by the event dummies
(R2 = 1.000000000000); the STATIC RF score is absorbed by the bank dummies;
**`uninsured_share` is NOT** (residual max 3.683e-01), because it is re-read from 8
characteristic quarters — so its main effect stays identified and is reported.

The event-clustered SE is the headline: the two-way covariance is not PSD at this
sample size and returns NaN SEs for many FE dummies.

## 3. Level vs reaction — the verdict (`rq3_measures.txt`)

`uninsured_share`: **level effect real** (p = 0.00032066), **reaction null** (p = 0.98188477). Vulnerable
banks sat at a lower abnormal return across these windows, but the gap does not
move with the safeguard content. That is a property of *which banks these are*
over 2019-2023, not a response to CBDC.

**THE H3 MECHANISM IS NOT SUPPORTED.**

The one significant interaction (RF score) is an **artefact**, and has the wrong
sign for the hypothesis anyway:

| Diagnostic | Result |
|---|---|
| Calendar placebo | corr(S, calendar rank) = **+0.60333196**; time x vuln gives -0.17749331; jointly neither survives |
| Custody banks | dropping 9 moves b to -0.13939961 (two-way p 0.06331919) — **63.8% of the magnitude** |
| Leave-one-event-out | sign never flips, so not one event — but that does not rescue it |

The OLS score does the same job (b = -0.26387427) and agrees
with the RF score at Spearman +0.68145928, so any credit
belongs to using a fitted composite, not to machine learning. Note RF vs
`uninsured_share` Spearman is only **+0.05754306**:
they rank banks almost independently.
