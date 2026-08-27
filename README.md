# CBDC design signals, bank vulnerability, and equity market reaction

Code: MIT (`LICENSE`). Data sources and what is and is not redistributed here: **[`DATA.md`](DATA.md)**.

MSc dissertation codebase. Three research questions, cleanly separated:

| RQ | Question | Verdict |
|---|---|---|
| **RQ1** | Which balance-sheet characteristics predict the 2023Q1 deposit outflow, and do trees beat OLS? | H1a partial; **H1b not robust** |
| **RQ2** | Did the equity market react to Fed CBDC communications, on average? | **No reaction**; H2 not supported |
| **RQ3** | Do the two vulnerability proxies meet — do vulnerable banks react differently? | **Mechanism not supported** |

---

## THE MAP

Everything in one table: hypothesis, variable, coefficient, script, output, result.

| RQ | Hypothesis | Key variable | Coefficient | Script | Output | Result |
|---|---|---|---|---|---|---|
| RQ1 | **H1a** balance-sheet fragility predicts outflow | `uninsured_share` | OLS coef | `rq1_model.py` | `rq1_vulnerability/comparison_narrow_vs_wide.txt` | wide **−0.07713467** (p 0.0000674) — the only sweep-through predictor |
| RQ1 | H1a | `unrealised_losses` | OLS coef | `rq1_model.py` | same | narrow +0.13258756 (p 0.0054); **vanishes in wide** (p 0.663) |
| RQ1 | H1a | `liquidity` | OLS coef | `rq1_model.py` | same | wide −0.11036769 (p 0.0000372) — **wrong sign**, likely custodians |
| RQ1 | **H1b** trees beat OLS out of sample | OOS RMSE | RF − OLS gap | `rq1_model.py` | `rq1_vulnerability/rq1_h1b_robustness.txt` | **not robust**: 26/30 repeats full features → **9/30** pruned; loses under nested tuning |
| RQ1 | robustness | 3 failed banks | OLS coef | `rq1_robustness.py` | `rq1_vulnerability/rq1_failed_bank_robustness.txt` | wide survives all 4 treatments; **narrow does not** (p 0.105 when excluded) |
| RQ1 | output for RQ3 | leakage-free score | — | `rq1_scores.py` | `rq1_vulnerability/vulnerability_scores_wide.csv` | transfer R² = 0.0302, SD 0.014 (weak, compressed ~3.7×) |
| RQ2 | signal construction | Safeguard **S** | — | `rq2_signal.py` | `rq2_communications/rq2_safeguard_scores.csv` | 11 events, S ∈ [−0.667, +1.000]; kappa 0.839 / 0.847 |
| RQ2 | — | CAR | — | `rq2_car.py` | `rq2_communications/rq2_car.csv` | 2956 bank-events, 276 banks, CAR SD 4.805 pp |
| RQ2 | **unconditional** did the market react? | mean CAR | Brown–Warner | `rq2_reaction.py` | `rq2_communications/rq2_reaction.txt` | **0 of 11 events**; volatility & turnover suppressed 7 of 11 |
| RQ2 | **H2** protective communications → less negative CAR | **S** | **gamma1** | `rq2_avg_effect.py` | `rq2_communications/rq2_avg_effect.txt` | **+0.00645428** (SE 0.01712870, p 0.714) — **not supported** |
| RQ2 | H2 bound | — | MDE(gamma1) | `rq2_reaction.py` | `rq2_communications/rq2_reaction.txt` | 2.371343 pp = **49.35% of a CAR SD** — a wide bound |
| RQ3 | **H3** level: vulnerable → lower CAR | `uninsured_share` | **delta1** | `rq3_link.py` | `rq3_bridge/rq3_link.txt` | **−0.01877493** (p 0.00032066) — **significant, correct sign** |
| RQ3 | H3 level | RF score | delta1 | `rq3_link.py` | same | +0.06011100 (p 0.332) — null, wrong sign |
| RQ3 | H3 level | OLS score | delta1 | `rq3_link.py` | same | +0.01232232 (p 0.782) — null |
| RQ3 | **H3** reaction: does the signal change it? | `S × uninsured_share` | **b** | `rq3_interaction.py` | `rq3_bridge/rq3_interaction.txt` | **−0.00049680** (p 0.982) — **null** |
| RQ3 | H3 reaction | `S × RF score` | b | `rq3_interaction.py` | same | −0.38507629 (p 0.0115) — significant but **artefact, wrong sign** |
| RQ3 | level vs reaction | — | — | `rq3_measures.py` | `rq3_bridge/rq3_measures.txt` | level real, reaction null → **H3 mechanism NOT supported** |
| RQ3 | artefact check | calendar placebo | — | `rq3_measures.py` | same | corr(S, calendar rank) = **+0.60333196**; jointly neither survives |
| RQ3 | artefact check | custody banks | — | `rq3_measures.py` | same | dropping 9 removes **63.8%** of the RF coefficient |
| RQ1 | H1b | tuned OOS RMSE / R² | — | `rq1_tuned_table.py` | `rq1_vulnerability/rq1_tuned_metrics.txt` | Table 5: OLS beats the tuned RF in **4 of 4** configurations |
| RQ2/3 | inference | event-clustered p | wild cluster bootstrap | `rq_wildboot.py` | `shared_inputs/rq_wildboot.txt` | 11 clusters → CRV1 unreliable; **S × RF score p 0.0115 → 0.0684**, no longer significant |
| RQ3 | H3 bound | `S × uninsured_share` | MDE(b) | `rq_wildboot.py` | same | 1.242988 pp = **25.87% of a CAR SD** — a wide bound |
| RQ1 | placebo | `uninsured_share × 1[2023Q1]` | pooled interaction | `rq1_placebo.py` | `rq1_vulnerability/rq1_placebo.md` | pre-crisis (2022Q2) coef **−0.0090** (p 0.508) vs stress **−0.0771**; interaction −0.0667, **p 0.0952** — suggestive, not significant at 5% |
| — | verification | all of the above | — | `run_all.py` | `shared_inputs/reconciliation.txt` | **300/300 numbers reconcile, 0 mismatches** |

---

## Quick start

```bash
python3 scripts/run_all.py --check-results   # START HERE — no data/ needed
python3 scripts/run_all.py            # rebuild everything (reuses frozen artefacts)
python3 scripts/run_all.py --verify   # recompute headline numbers + reconcile only
python3 scripts/collect_results.py    # rebuild the results/ showcase
```

**`--check-results` is the command to run in a fresh clone.** It recomputes
nothing: it reads the published numbers from `results/shared_inputs/` and
reconciles all 300 headline values against `config.FROZEN`, so it works with no
`data/` directory at all and prints `checked 300 of 300 keys; 0 mismatch(es)`.

`--verify` and `--force` both need `data/raw/` and `data/processed/`, which are
not in the repository: the Call Report archives, the CRSP extract and the NIC
and MDRM dictionaries are ~180 MB and the CRSP part is licensed. See
[Getting the raw data](#getting-the-raw-data) if you need to rebuild from source.

`run_all.py` is idempotent: each stage skips itself when its outputs exist. The
results are **final**, so re-running an estimator is a chance to lose them, not to
improve them. Pass `--force` to a stage only deliberately.

**After every `run_all.py` run, commit the contents of `results/`.** `data/` is
not under version control apart from `data/frozen/`, so the committed `results/`
snapshot is the only record an examiner can see. If it is not committed it drifts
out of step with the code that produced it, and the repository starts asserting
numbers it can no longer show.

**Grid-search ties are broken deterministically.** When two hyper-parameter
settings score within `GRID_TIE_TOL` (1e-9) of each other in the inner CV, the
difference is BLAS summation order rather than fit, and `GridSearchCV`'s argmax
would flip the reported winner between otherwise identical runs. The
lexicographically first grid point wins instead, so the nested-tuning report is
reproducible run to run (`_rq1_wide_core.deterministic_winner`).

## Manual steps

Three stages are outside `run_all.py` and have to be invoked deliberately. Each
is excluded for a reason, and each is needed only if you are rebuilding that part
from source — the pipeline otherwise consumes their frozen outputs.

```bash
# 1. The Safeguard signal. The classification step calls an LLM, so it is never
#    re-run automatically: the labels in data/frozen/ are what every RQ2 and RQ3
#    number is conditioned on. Needs ANTHROPIC_API_KEY.
python3 scripts/rq2_signal.py                # refuses to overwrite; reports and exits
python3 scripts/rq2_signal.py --force        # regenerate labels + rq2_safeguard_scores.csv

# 2. The event study. Treated as read-only because rq2_car.csv is the input to
#    every downstream RQ2/RQ3 regression. Needs data/raw/wrds/ and the Call
#    Report archives.
python3 scripts/rq2_car.py                   # -> data/processed/rq2_car.csv (2956 bank-events)

# 3. Validation of the labels against hand coding. Two stages with a human in
#    between: build a blind template, code it by hand, then score it.
python3 scripts/rq2_validation.py template   # -> data/frozen/rq2_validation_template.csv
#    ... the analyst fills in the human columns ...
python3 scripts/rq2_validation.py score      # -> rq2_validation_results.txt (Cohen's kappa)
```

A fourth is a one-off data fetch rather than a pipeline stage:
`python3 scripts/_fdic_class.py` refreshes the FDIC charter snapshot, and
`python3 scripts/_fetch_fulltext.py` re-downloads the body text of document 1.

## Layout

```
scripts/
  config.py              all paths, the seed, feature lists, thresholds, FROZEN numbers
  run_all.py             master entry point + reconciliation
  _stats.py              shared estimation helpers (clustering, absorption checks)

  rq1_build_panel.py     panels: wide 953 + narrow 277, charter flags, censoring
  rq1_model.py           H1a (OLS) + H1b (OOS, repeated CV, nested tuning)
  rq1_scores.py          leakage-free RF and OLS scores, higher = more vulnerable
  rq1_robustness.py      four treatments of the three failed banks

  rq2_signal.py          sentences -> classify -> Safeguard S    [FROZEN, LLM step]
  rq2_validation.py      human-vs-model kappa
  rq2_car.py             event study -> CAR                      [READ-ONLY]
  rq2_reaction.py        Brown-Warner test, abnormal vol/turnover, MDE for gamma1
  rq2_avg_effect.py      CAR ~ S  (gamma1), event-clustered

  rq3_link.py            CAR ~ vulnerability + controls (delta1), three measures
  rq3_interaction.py     CAR = EventFE + BankFE + b*(S x vulnerability)
  rq3_measures.py        measures comparison, level-vs-reaction, artefact checks
  rq3_custody_check.py   Appendix C: custody exclusions + S vs calendar time
  rq2_figures.py         Figure 2: the Safeguard score by event

  rq_wildboot.py         wild cluster bootstrap p-values for every
                         event-clustered estimate (11 clusters) + the MDE for
                         the uninsured x Safeguard interaction
  rq1_tuned_table.py     Table 5: the nested-tuning metrics behind H1b
  rq1_placebo.py         pre-crisis comparison period (2022Q2): is the
                         uninsured-deposit result stress-specific or a
                         persistent structural correlate?

  collect_results.py     rebuild results/ from data/processed/
  _*.py                  internal implementation modules, unchanged by the refactor

results/
  rq1_vulnerability/  rq2_communications/  rq3_bridge/  shared_inputs/  _archive/

docs/
  RESULTS.md                 the authoritative results write-up, RQ1 -> RQ2 -> RQ3
  rq2_codebook.md            the coding scheme behind the Safeguard signal
  rq1_pipeline_variants.md   narrow vs wide population, charter filter
```

Underscore-prefixed modules (`_rq1_core.py`, `_panel_wide.py`, …) hold the
original computation code. They were renamed but not rewritten, which is why the
refactor is provably behaviour-preserving.

## Invariants

These hold throughout and are asserted or verified numerically, not assumed:

1. **Score orientation.** Every vulnerability measure is oriented *higher = more
   vulnerable*; the scores are `−predicted deposit growth`, checked by an assert
   in `rq1_scores.py`.
2. **CAR is unchanged.** `rq2_car.py` was touched only to rename one import, and
   `rq2_car.csv` was verified byte-identical afterwards.
3. **H1b is not robust.** Reported as such, with the RF-beats-OLS share across 30
   seeds; the "narrow: everything worse than the mean" claim is a seed-42
   artefact and the repeated-CV means are reported instead.
4. **S is collinear with event dummies.** No event-FE specification is estimated
   for S; the absorption is verified numerically (R² = 1.000000000000, max
   residual ~7e-14) and reported as collinearity, not as a zero.
5. **The static ML score is absorbed by bank FE; `uninsured_share` is not**
   (R² = 0.9132 — it is re-read from 8 characteristic quarters). Its main effect
   is therefore estimated and reported, not suppressed.
6. **Two-way clustered covariance is not PSD here.** Some FE dummies get NaN SEs.
   The two-way SE for `b` is reported, but the **event-clustered SE is the
   headline**, and the NaN count is printed.
7. **Every figure is unrounded.** No specification is ever searched for
   significance; where a spec cannot be estimated the code raises `NotIdentified`
   and reports it rather than printing a pseudo-inverse.

## Reading the results

Start with **[`docs/RESULTS.md`](docs/RESULTS.md)** — it reads top to bottom as
RQ1 → RQ2 → RQ3.

## Data

Raw inputs are organised by source under `data/raw/`:

```
data/raw/
  call_*.zip        16 quarterly Call Report archives (2019Q3-2023Q2), kept flat
                    because rq2_car.py is read-only and maps quarters straight
                    to these filenames
  fed/              the 11-document corpus (xlsx) + the Money and Payments PDF/text
  fdic/             institution classes, failure list
  wrds/             CRSP daily parquet files + the stocknames extract
  nic/              NIC attribute/relationship files used to build the crosswalk
  mdrm/             Call Report field dictionary
  factors/          Ken French factors, FRED DGS2 (cached downloads)
```

Nothing under `data/` is in the repo (licensed / large). `.gitignore` excludes
`*.csv` globally, so CSVs in `results/` are not tracked either.

## Getting the raw data

`data/raw/` is not in the repository (~180 MB, part of it licensed). Only
`data/raw/fed/` — the hand-collected eleven-document corpus — is tracked,
because no script can rebuild it.

| Needed | Size | Where from |
|---|---|---|
| `call_*.zip` — 16 quarterly Call Reports, 2019Q3–2023Q2 | ~100 MB | FFIEC CDR Bulk Data Download: Call Reports → Single Period → Tab Delimited, one quarter at a time. Public |
| `wrds/{dsf,dsi,dsedelist}.parquet` (**not in this repo**) | 6.6 MB | **CRSP via WRDS — subscription required.** Daily returns for 440 listed banks, the value-weighted index and delisting returns, 2019-01 to 2024-12 |
| `nic/CSV_{RELATIONSHIPS,ATTRIBUTES_ACTIVE}.CSV` | 77 MB | FFIEC NIC. Public. Only needed to rebuild the crosswalk, which is already in `data/frozen/` |
| `mdrm/MDRM_CSV.csv` | 87 MB | FFIEC MDRM. Public |
| `fdic/` — failed-bank list, BKCLASS/SPECGRP | 364 KB | FDIC BankFind. Public |
| `factors/` — Fama–French daily, 49 industry portfolios | 4.4 MB | Kenneth French Data Library. Downloaded automatically and cached |

With those in place, `run_all.py --force` rebuilds everything except the two
manual stages below. Without them, use `--check-results`.