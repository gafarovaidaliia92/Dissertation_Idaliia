# Results — RQ1 → RQ2 → RQ3

The authoritative results document. Reads top to bottom in the order the argument
runs. Every figure is unrounded and traceable to a file under `results/`.

All numbers reconcile against the pre-refactor values: **220 of 220 keys, zero
mismatches** (`results/shared_inputs/reconciliation.txt`).

- [RQ1 — who is vulnerable](#rq1--who-is-vulnerable)
- [RQ2 — did the market react](#rq2--did-the-market-react)
- [RQ3 — do the two meet](#rq3--do-the-two-meet)
- [The through-line](#the-through-line)
- [What changed in the re-split](#what-changed-in-the-re-split)

---

# RQ1 — who is vulnerable

**Question.** Which balance-sheet characteristics predict the 2023Q1 deposit
outflow, and does ML identify vulnerable banks better than a linear baseline?

Two populations: **narrow** = 277 listed banks matched to CRSP (the ones RQ2/RQ3
can use, because they need stock prices); **wide** = 953 filers above $1bn with
`deposit_reliance ≥ 0.50`, of which 265 are listed.

## H1a — partially supported

| Variable | narrow coef | p | wide coef | p |
|---|---|---|---|---|
| `uninsured_share` | −0.23289077 | 0.00006061 | **−0.07713467** | **0.00006744** |
| `unrealised_losses` | +0.13258756 | 0.00535199 | +0.00118737 | 0.66298761 |
| `liquidity` | −0.07692121 | 0.42525326 | **−0.11036769** | **0.00003718** |
| `ROA` | +1.29383039 | 0.64753417 | +1.37071481 | 0.00756224 |

In-sample R²: narrow 0.1384, wide 0.0782.

- `uninsured_share` is the **only** predictor significant in both populations,
  with the H1a-consistent sign.
- `unrealised_losses` is narrow-only and vanishes in wide (p = 0.66).
- `liquidity` in wide is significant with the **wrong sign** — more liquidity,
  more outflow. It most likely picks up custodians and wholesale banks. Explain
  it in the text; do not bury it.

**Sign note that must be stated:** `unrealised_losses = (FV − cost)/equity`, so a
loss is a *negative* number and a *positive* coefficient is H1a-consistent.

## H1b — not robust

Single split (seed 42), wide: OLS RMSE 0.09195066 vs RF 0.09129361 — a 0.7146%
margin. That margin does not survive scrutiny.

| Population | Feature set | RF beats OLS | Mean margin |
|---|---|---|---|
| WIDE | full (10 features) | 26/30 repeats | +1.246% |
| WIDE | pruned (3 significant) | **9/30 repeats** | **−0.281%** |
| NARROW | full | 23/30 | +0.977% |
| NARROW | pruned (2) | 17/30 | +0.092% |

Under **nested tuning** (grid searched on inner folds only) RF loses to OLS in
all four combinations: wide full −1.146%, wide pruned −0.177%, narrow full
−5.367%, narrow pruned −2.704%.

> **The RF advantage comes from keeping the insignificant features**, which hurt
> OLS more than they hurt the forest. Prune them and it reverses; tune fairly and
> it disappears. Report H1b as **not robust**, not "formally supported".

> **A correction to the earlier write-up.** "In narrow all three models are worse
> than the mean" is an artefact of one unlucky seed. Averaged over 30 shuffles
> the narrow OOS R² is **positive**: OLS +0.01427315, RF +0.03352837.

## Robustness to the three failed banks

SVB, Signature and First Republic enter with `dep_growth = −1.0`, a censoring
convention rather than an observation. The worst *surviving* wide bank is
−0.76289449.

| Treatment | WIDE coef | p | NARROW coef | p |
|---|---|---|---|---|
| (a) censored at −1.0 | −0.07713467 | 0.0000674 | −0.23289077 | 0.0000606 |
| (b) three excluded | −0.03336049 | 0.0313480 | **−0.05945248** | **0.1045841** |
| (c) survivor-min | −0.06663649 | 0.0001793 | −0.08564723 | 0.0178311 |
| (d) winsorised at 1st pct | −0.03393842 | 0.0195633 | −0.12098568 | 0.0020124 |

> **Wide survives all four; narrow does not.** The narrow H1a headline is carried
> in large part by the three failures. **Lean on the wide result.**

## What RQ1 hands to RQ3

The leakage-free score: fitted on the 688 non-listed wide banks, applied to the
277 listed ones, oriented `score = −predicted growth` so **higher = more
vulnerable**. Transfer fit R² = 0.0302, SD 0.014 — compressed roughly 3.7× against
the in-population version. A weak measure, and RQ3 shows it behaving like one.

---

# RQ2 — did the market react

**Question.** Did the equity market react to Federal Reserve CBDC communications,
and does the average reaction depend on the *type* of communication?

## The signal

11 documents, 1338 sentences, 168 design sentences.
`S = (protective − expansive) / n_design`, an **event-level** variable with 11
values (10 distinct — two documents both score 1.00). Range −0.667 to +1.000;
4 events with S < 0, 7 with S ≥ 0.

Reliability against a human coder: Cohen's kappa **0.839** for whether a sentence
is about design, weighted kappa **0.847** for its stance.

> **Measurement caveat.** Design density runs from 2 to 65 sentences per
> document, and **both** documents scoring S = +1.00 rest on 3 and 4 sentences.
> The extremes of S are its noisiest points.

## The primary result: no reaction

Brown–Warner cross-correlation-robust test. Naive per-event cross-sectional
t-stats treat banks as independent within a date; they are not, and those
t-stats are inflated by roughly an order of magnitude.

| | Market model | Augmented (mkt + sector + rate) |
|---|---|---|
| Events with a robust mean-CAR reaction | **0 of 11** | 2 of 11 |
| Elevated volatility | 3 of 11 | 3 of 11 |
| **Suppressed** volatility | **7 of 11** | 7 of 11 |
| Elevated turnover | 3 of 11 | 3 of 11 |
| **Suppressed** turnover | **7 of 11** | 7 of 11 |

The two events reaching 5% under the augmented model (doc_id 3 and 6) carry
**opposite signs** (+0.028, −0.027), so they do not add up to a reaction.

> This is not "a null mean with violent trading underneath". On most of these
> dates bank stocks were **quieter than on an ordinary day**. Federal Reserve
> CBDC speeches were not events for bank equity.

## H2 — the average effect of communication type

| Test | gamma1 | SE | p |
|---|---|---|---|
| Event level: 11 event-mean CARs on 11 S | +0.00562863 | 0.01887397 | 0.77230408 |
| Pooled, no controls (cluster = event) | +0.00617690 | 0.01719368 | 0.72687493 |
| Pooled, with controls (cluster = event) | +0.00645428 | 0.01712870 | 0.71418685 |

Pearson corr(event-mean CAR, S) = **+0.09891973** (p 0.772);
Spearman = **+0.01366746** (p 0.968).
Protective minus expansive mean CAR = −0.00561 (Welch p 0.81) — point estimate
against H2, indistinguishable from zero.

**H2 is not supported.** The sign is in the predicted direction; nothing is close
to significant.

### Two structural points, enforced in code

1. **No event-FE specification exists for S.** S is constant within an event, so
   the event dummies absorb it exactly — verified numerically, R² = 1.000000000000,
   max residual ~2e-15. That is *collinearity*, not an effect of zero, and
   nothing is printed for it.
2. **SEs cluster on the EVENT** (11 clusters, t with 10 df), because that is the
   level at which S varies. Bank clustering shrinks the SE by a factor of ~14 and
   would turn this null into p < 0.0001; it is shown only to make the inflation
   visible.

### How tight is the null?

MDE for gamma1 = **2.371343 pp of CAR per one-SD move in S = 49.35% of a CAR
standard deviation.** That is **wide**.

> Write "H2 not supported, and a large average effect ruled out" — **not** "there
> is no effect". With 11 events a modest effect cannot be excluded.

---

# RQ3 — do the two meet

**Question.** Does a bank the RQ1 model calls vulnerable react differently to
CBDC communications?

Three measures, all oriented **higher = more vulnerable**: the RF score, an
OLS-predicted score built on the identical split, and `uninsured_share`.

They are not interchangeable — RF vs `uninsured_share` Spearman is only
**+0.05754306**, so they rank banks almost independently. RF vs OLS score is
**+0.68145928**.

## 1. The level link (delta1)

`CAR ~ vulnerability + controls + event FE`, bank-clustered. H3 predicts
delta1 < 0.

| Measure | delta1 | p |
|---|---|---|
| RF score | +0.06011100 | 0.33223292 |
| OLS score | +0.01232232 | 0.78213952 |
| **`uninsured_share`** | **−0.01877493** | **0.00032066** |

> **A correction to the earlier write-up.** The published RQ3 null is a statement
> about the **ML score**. It does not extend to `uninsured_share`, which has a
> real level effect with the H3-predicted sign.

## 2. The reaction (b) — the supervisor's two-way FE model

`CAR = EventFE + BankFE + b·(S × vulnerability) + e`

| Measure | b | SE (event) | p (event) | p (two-way) |
|---|---|---|---|---|
| `uninsured_share` | −0.00049680 | 0.02134018 | 0.98188477 | 0.98153708 |
| RF score | −0.38507629 | 0.12471181 | 0.01148834 | 0.02208081 |
| OLS score | −0.26387427 | 0.06728926 | 0.00285943 | 0.01370724 |

### What is absorbed, verified numerically

| Regressor on the fixed effects | R² | max residual | |
|---|---|---|---|
| S on event dummies | 1.000000000000 | 7.1e−14 | **absorbed** |
| RF score on bank dummies | 1.000000000000 | 1.8e−15 | **absorbed** (static) |
| `uninsured_share` on bank dummies | 0.913208 | 0.368 | **survives** |
| S × vulnerability on both | 0.455517 | 0.061 | survives → estimable |

> **A correction to the model's premise.** "The vulnerability main effect drops
> out under bank FE" holds only for a *static* measure. `uninsured_share` is
> re-read from the last Call Report before each event, and the 11 events draw on
> 8 distinct characteristic quarters, so its main effect stays identified. It is
> estimated and reported rather than suppressed.

**Inference.** The two-way (bank + event) clustered covariance is **not positive
semi-definite** at this sample size: many fixed-effect dummies come back with
negative variances and NaN SEs. The two-way SE for `b` is reported because it was
asked for, but the **event-clustered SE is the headline**, and the NaN count is
printed alongside.

## 3. Level versus reaction — the verdict

For `uninsured_share`:

- **LEVEL** delta1 = −0.01877493, p = 0.00032066 → real
- **REACTION** b = −0.00049680, p = 0.98188477 → null

Banks with more uninsured deposits did sit at a lower abnormal return across
these windows. But the gap **does not move with the safeguard content** of the
communication. That is a fact about *which banks these are* over 2019–2023 — a
period containing COVID and the 2022 rate cycle — not a response to what the Fed
said about CBDC design.

> ## THE H3 MECHANISM IS NOT SUPPORTED.

## 4. The one significant interaction is an artefact

The RF score's b = −0.38507629 (p 0.0115) is significant, and its sign is the
**opposite** of what the hypothesis predicts. Three diagnostics, run
symmetrically on both measures regardless of outcome:

| Diagnostic | Result |
|---|---|
| **Calendar placebo** | corr(S, calendar rank) = **+0.60333196** (p 0.049). Replacing S with calendar rank gives −0.17749331 — same sign, same order of magnitude. Entered jointly, **neither survives** |
| **Custody banks** | dropping the 9 trust/custody banks moves b to −0.13939961 (two-way p 0.06331919): **63.8% of the magnitude is custodians**, whose rate exposure is a competing explanation |
| **Leave-one-event-out** | b ∈ [−0.52063335, −0.30804490], sign never flips — so it is not one event, but that does not rescue it from the first two |

With 11 events the design cannot separate "more protective communication" from
"later in the sample".

## 5. Does machine learning add anything?

No — not as machine learning. The OLS-predicted score, same features and same
split, gives b = −0.26387427 (p 0.0137), and the two scores agree at Spearman
+0.68145928. Any credit belongs to using a **fitted balance-sheet composite**,
not to the algorithm. And in the level link the ordering reverses: the raw ratio
wins and both fitted scores are null.

---

# The through-line

| | Result | How tight is the null |
|---|---|---|
| RQ1 H1a | partial: `uninsured_share` only; narrow rests on 3 failures | — |
| RQ1 H1b | **not robust** — vanishes under pruning and fair tuning | — |
| RQ2 unconditional | **0 of 11 events**; volatility/turnover suppressed 7 of 11 | unambiguous |
| RQ2 H2 (gamma1) | not supported, correct sign, p = 0.714 | **wide** — 49.35% of a CAR SD |
| RQ3 delta1, ML score | null, wrong sign | tight — ~5% of a CAR SD |
| RQ3 delta1, `uninsured_share` | **significant**, correct sign | — |
| RQ3 b (reaction) | null for `uninsured_share`; artefact for the ML score | ~9.5% of a CAR SD |

**The causal chain.** The market did not react to these dates at all → there is
no average reaction for the communication type to modulate (gamma1 null) → and no
cross-section for vulnerability to explain (H3 mechanism unsupported). The three
nulls are one finding, not three independent ones.

**Two things that must travel with that sentence:**

1. The nulls are **not equally informative**. RQ3's coefficients are bounded at
   ~5–9.5% of a CAR SD; RQ2's gamma1 only at ~49%. Do not describe them all as
   equally tight.
2. `uninsured_share` **does** have a real level effect on CAR
   (−0.01877493, p = 0.00032066). It is not a reaction to CBDC content, but it is
   not nothing, and claiming a blanket null across the project would be wrong.

---

# What changed in the re-split

One conceptual change, no new analysis.

**The Safeguard × vulnerability interaction moved from RQ2 to RQ3.**

It had been presented as the **H2** test. It is not one. H2 asks whether the
*type of communication* moves the *average* reaction — a between-event question,
answered by `CAR ~ S`. The interaction asks whether *vulnerable banks react
differently* — a cross-sectional question about the bridge between the RQ1
profile and the market reaction, which is **H3**.

| | Before | Now |
|---|---|---|
| **RQ2** | unconditional test + `CAR ~ S` + the interaction as headline | unconditional test + `CAR ~ S` + MDE. Nothing else. |
| **RQ3** | `CAR ~ vulnerability_score` only | the level link for **three** measures + the two-way FE interaction + measures comparison + level-vs-reaction diagnostic + artefact checks |

**What each RQ now concludes:**

- **RQ2** — the market did not react to CBDC communications on average, and the
  type of communication does not explain what little variation there is.
- **RQ3** — the two vulnerability proxies do not meet. There is a level
  difference by uninsured funding, but no reaction to communication content, and
  the one significant interaction is an artefact of the calendar and of custody
  banks.

Nothing was re-estimated to achieve this: the coefficients are the same objects
as before, verified key by key in
`results/shared_inputs/reconciliation.txt`. Superseded versions, including the
interaction-as-H2 reports, are in `results/_archive/`.
