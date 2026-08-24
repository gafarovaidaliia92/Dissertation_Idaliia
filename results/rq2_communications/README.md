# RQ2 — market reaction to CBDC communications

**Question.** Did the equity market react to Federal Reserve CBDC
communications, and does the AVERAGE reaction depend on the type of
communication?

- **H2** — communications emphasising protective design (higher S) produce less
  negative CARs. Coefficient: **gamma1** on S. H2 predicts gamma1 > 0.

Code: `rq2_signal.py` -> `rq2_validation.py` -> `rq2_car.py` -> `rq2_reaction.py`
-> `rq2_avg_effect.py`.

> **What is NOT here.** The Safeguard x vulnerability interaction moved to RQ3.
> It asks whether VULNERABLE banks react differently, which is H3, not H2. The
> version that presented it as the H2 test is in `../_archive/`.

## The signal S

11 documents, 1338 sentences, 168 design sentences.
`S = (protective - expansive) / n_design`. Range -0.667 to +1.000; 4 events with
S < 0, 7 with S >= 0. Inter-coder reliability against a human coder: Cohen's
kappa 0.839 (is the sentence about design), weighted kappa 0.847 (its stance).

**Measurement caveat:** design density runs from 2 to 65 sentences per document,
and BOTH documents scoring S = +1.00 rest on 3 and 4 sentences.

## 1. Did the market react at all? (`rq2_reaction.txt`) — the primary result

| | Market model | Augmented (mkt+sector+rate) |
|---|---|---|
| Events with a robust mean-CAR reaction | **0 of 11** | 2 of 11 (opposite signs) |
| Elevated volatility / turnover | 3 of 11 | 3 of 11 |
| **Suppressed** volatility / turnover | **7 of 11** | 7 of 11 |

Brown-Warner cross-correlation-robust test. On most of these dates bank stocks
were *quieter than usual*. **Fed CBDC speeches were not events for bank equity.**

## 2. The average effect (`rq2_avg_effect.txt`)

| Test | gamma1 | SE | p |
|---|---|---|---|
| Event level, 11 event-mean CARs on 11 S | +0.00562863 | 0.01887397 | 0.77230408 |
| Pooled with controls (cluster = event) | +0.00645428 | 0.01712870 | 0.71418685 |

Pearson corr(event-mean CAR, S) = +0.09891973; Spearman = +0.01366746.

**H2 is not supported.** The sign is in the predicted direction; nothing is close
to significant.

### Structural points, enforced not described

- S varies ONLY across the 11 events, so **no event-FE spec is estimated for S**:
  the event dummies absorb it exactly (max residual 7.061e-14).
  Collinearity, not a zero.
- SEs cluster on the **EVENT** (11 clusters, t(10)). Bank clustering would treat
  276 banks as independent information about an 11-valued regressor and shrinks
  the SE by a factor of ~14; it is shown only to make that visible.

### How tight is the null?

MDE for gamma1 = 2.371343 pp of CAR per 1 SD of S = **49.35% of a CAR SD**. That is WIDE. Say "H2 not
supported and a large effect ruled out" — not "there is no effect".
