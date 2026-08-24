# Pipeline variants — narrow vs wide

RQ1 is estimated twice on the same variables with the same models, seed and
hyper-parameters. **The only difference is which banks the model trains on.**

| | **narrow** | **wide (all charters)** | **wide (charter-filtered)** |
|---|---|---|---|
| Population | 278 listed banks matched to CRSP | all 2022Q4 filers | all 2022Q4 filers, non-commercial types removed |
| Filters | RSSD→permno crosswalk must resolve | assets > $1bn, deposit_reliance ≥ 0.50 | + credit-card banks dropped, trust/custody flagged |
| Modelled N | 277 | 963 | 953 |
| Purpose | the sample RQ2/RQ3 can use (needs stock prices) | is the narrow result an artefact of N=277? | is the wide result driven by non-bank deposit-takers? |

The narrow sample is conditioned on being publicly traded *and* CRSP-matchable —
a selection on size and ownership structure. The wide variant drops that
condition; the charter filter then removes institutions that take deposits but
are not ordinary commercial/savings banks.

## Charter-type filter (the last RQ1 refinement)

Applied to the **wide training population only** (`_panel_wide.py`, after the
existing filters). Uses FDIC BankFind classification (`BKCLASS`, `SPECGRP`) fetched
by `scripts/_fdic_class.py`, merged by FDIC certificate, plus two Call Report
balance-sheet signals. Logic lives in `scripts/_charter_flags.py`.

**What FDIC can and cannot classify here:**
- Every one of the 963 filers is a commercial (`N`/`SM`/`NM`) or savings
  (`SB`/`SI`/`SL`) bank by `BKCLASS` — there are **no** trust-charter or
  industrial-loan classes left in the population, so `BKCLASS` drops nobody.
- `SPECGRP` cleanly isolates exactly one non-commercial deposit type:
  **credit-card banks** (`SPECGRP==3`). These 10 (Amex, Synchrony, Comenity ×2,
  Discover, Capital One, Barclays Delaware, Merrick, Credit One, Stride) are
  **hard-dropped** from training.
- The custody/trust houses (State Street, BNY Mellon, Bessemer, Deutsche Bank
  Trust, Sumitomo Mitsui Trust) all sit in the generic `SPECGRP==9`
  "All Other >$1B" bucket **alongside SVB and First Republic**. No single FDIC
  field separates a custodian from SVB.

**So trust/custody/servicer banks are FLAGGED, never silently dropped**
(`is_trust_or_specialized`), via a transparent composite rule (all three reported
in `flag_reason`):
- **A** `loans_to_assets < 0.40 AND uninsured_share > 0.55` → barely-lending,
  near-entirely-uninsured wholesale deposits (State Street, BNY Mellon, Sumitomo,
  Deutsche Bank Trust, Cenlar, Silvergate, …).
- **B** `fiduciary_ratio > 0.50` (RC-T managed+custody assets / total assets) →
  catches Northern Trust and BNY Mellon, which lend enough to pass A.
- **C** `SPECGRP == 3` (credit-card).

`is_trust_or_specialized = A or B or C`; `is_creditcard = C`. Only `is_creditcard`
hard-drops. **The three censored failures are verified NOT flagged** before any
drop (SVB/Signature/First Republic are real commercial lenders — loans/assets
0.67 / 0.68 / 1.57).

**Cenlar** (chartered as a savings institution but really a mortgage servicer) is
caught by signal A, flagged, and **kept** — exactly the belt-and-suspenders case:
FDIC's own class miscodes it, so it is flagged and reported rather than
hard-dropped on a bad label.

**Known limitation of the flag.** Signal B over-flags ~9 ordinary community banks
that happen to run large trust departments relative to their small balance sheet
(WSFS, Dacotah, Johnson Bank, Moody National, …) — harmless, since flagged banks
are kept, not dropped. Conversely a few brokerage/sweep banks that *are*
specialized are **not** flagged (Morgan Stanley Private Bank, Charles Schwab Bank,
Stifel Bank & Trust) because their deposits are insured sweeps (low uninsured) and
they lend enough to pass A. The flag is a documented heuristic, not an exhaustive
taxonomy.

## Scripts

| Script | Reads | Writes |
|---|---|---|
| `scripts/_panel_narrow.py` | `data/raw/`, `data/frozen/sample_banks.csv` | `panel_2022Q4.csv` |
| `scripts/_rq1_core.py` | `panel_2022Q4.csv` | `vulnerability_scores.csv`, `rq1_results.txt`, `rq1_shap_summary.png` |
| `scripts/_fdic_class.py` | `data/raw/call_12312022.zip`, FDIC BankFind API | `data/raw/fdic/institutions_class.csv` |
| `scripts/_charter_flags.py` | `data/raw/fdic/…`, Call Report schedules | *(module — no output)* |
| `scripts/_panel_wide.py` | `data/raw/`, `fdic_failures.csv`, `data/frozen/sample_banks.csv`, FDIC snapshot | `panel_2022Q4_wide.csv` |
| `scripts/_rq1_wide_core.py` | both panels + preserved all-charters panel | `*_wide` outputs, `*_narrow` copies, `comparison_narrow_vs_wide.txt` |

`_panel_wide.py` imports the extraction, filer-aware scope rule and variable
formulas from `_panel_narrow.py`; `_rq1_wide_core.py` imports `FEATURES`, `SEED`,
`N_SPLITS`, the RF/GB parameters and every modelling function from `_rq1_core.py`.
Neither narrow script is modified, and no pre-existing file in `data/processed/`
is overwritten.

```bash
# one-off fetch, run by hand when the FDIC snapshot is stale:
python scripts/_fdic_class.py               # -> data/raw/fdic/institutions_class.csv (STOPs on API failure)

# everything else goes through the entry points, not the modules:
python scripts/rq1_build_panel.py --force   # -> panel_2022Q4_wide.csv (953, charter-filtered)
python scripts/rq1_model.py --force         # -> *_wide outputs + comparison (WIDE-ALL folded into section 0)
```

The underscore-prefixed files above are implementation modules imported by
`rq1_build_panel.py` and `rq1_model.py`. They are not entry points, and calling
them directly is not part of the pipeline.

If `_fdic_class.py` cannot reach the FDIC API it exits non-zero and writes
nothing — the pipeline does **not** fall back to a hand-maintained name list.

## Files in `data/processed/`

**Narrow (pre-existing, untouched)**

| File | What it is |
|---|---|
| `panel_2022Q4.csv` | 278 listed banks, predictors at 2022Q4 + `dep_growth` |
| `vulnerability_scores.csv` | 277 scores from `cross_val_predict` within the listed sample |
| `rq1_results.txt` | CV metrics, OLS table, SHAP ranking, top-10 |
| `rq1_shap_summary.png` | SHAP beeswarm |

**Narrow copies** — byte-identical snapshots so the comparison stays reproducible
even if the narrow pipeline is re-run later: `panel_2022Q4_narrow.csv`,
`vulnerability_scores_narrow.csv`, `rq1_results_narrow.txt`,
`rq1_shap_summary_narrow.png`.

**Wide — charter-filtered (current `*_wide`)**

| File | What it is |
|---|---|
| `panel_2022Q4_wide.csv` | **953** banks (963 − 10 credit-card); adds `is_listed`, `BKCLASS`, `SPECGRP`, `loans_to_assets`, `fiduciary_ratio`, `is_creditcard`, `is_trust_or_specialized`, `flag_reason` |
| `rq1_results_wide.txt` | same report format, charter-filtered population |
| `rq1_shap_summary_wide.png` | SHAP beeswarm, filtered |
| `vulnerability_scores_wide.csv` | **RQ3 input** — see below; now carries `is_trust_or_specialized` |
| `comparison_narrow_vs_wide.txt` | narrow vs filtered-wide **plus** the charter-filter effect (section 0) |

**Wide — pre-filter population (`panel_2022Q4_wide_allcharters.csv`)** — the 963
banks that clear the size and deposit-reliance floors *before* the charter filter
removes the 10 credit-card banks. `_rq1_wide_core.py` reads this panel, runs the
WIDE-ALL variant on it, and writes the result into **section 0 of
`comparison_narrow_vs_wide.txt`** ("CHARTER FILTER EFFECT"), which reports N,
out-of-sample R² and the coefficient table for both populations side by side.

The variant no longer produces standalone files. `rq1_results_wide_allcharters.txt`,
`rq1_shap_summary_wide_allcharters.png`, `vulnerability_scores_wide_allcharters.csv`
and `comparison_narrow_vs_wide_allcharters.txt` were left behind by an earlier
code path that wrote them separately; nothing had regenerated them since 4 August
2026 and they were removed. The panel is the only `*_allcharters` artefact the
pipeline still needs.

**FDIC snapshot** — `data/raw/fdic/institutions_class.csv` (4707 certs; `CERT,
NAME, BKCLASS, SPECGRP, SPECGRPN, ACTIVE`; includes inactive banks so the three
failures are covered).

## `vulnerability_scores_wide.csv` — how it differs

It is **not** the wide-population CV scores. It is the leakage-free hand-off to RQ3:

```
train:  charter-filtered wide population MINUS every listed bank   (N=688)
apply:  the 277 listed banks                                       (never seen in training)
```

The narrow file uses `cross_val_predict`, where each bank is scored by a model
trained on 4/5 of the *same* listed sample. Here training and scoring sets share
no observations at all, which is a stricter hold-out — and it means the score is
not mechanically tied to the same banks whose stock returns RQ3 regresses it on.

Columns: `bank_IDRSSD, name, permno, pred_dep_growth, vulnerability_score,
is_trust_or_specialized`. `vulnerability_score = −pred_dep_growth`, so **higher =
more predicted outflow = more vulnerable**. **The 277 listed banks are never
dropped** — custodians (State Street, BNY Mellon, Northern Trust) stay, marked
`is_trust_or_specialized=True`, so the include/exclude decision can be made later
in RQ3.

## Headline findings

| | narrow (277) | wide-all (963) | wide-filtered (953) |
|---|---|---|---|
| OLS out-of-sample R² | −0.022 | +0.023 | +0.025 |
| RandomForest out-of-sample R² | −0.032 | +0.057 | +0.039 |
| H1b (trees beat linear OOS) | **not supported** | **SUPPORTED** | **SUPPORTED** |
| `uninsured_share` | −0.233, p<0.001 | −0.078, p<0.001 | −0.077, p<0.001 |
| `unrealised_losses` | +0.133, p=0.005 | +0.001, p=0.69 | +0.001, p=0.66 |
| SHAP #1 | `uninsured_share` | `liquidity` | `liquidity` |

**The charter filter does not change any RQ1 conclusion.** H1b stays supported
(at every size floor); `uninsured_share` stays significant with the same sign and
magnitude; `unrealised_losses` stays insignificant; SHAP #1 stays `liquidity`.
Dropping the 10 credit-card banks nudges RF out-of-sample R² down slightly
(+0.057 → +0.039) — expected, since credit-card banks are unusual points a tree
can exploit — but the qualitative story is identical.

**The top-10 most-vulnerable DOES change once trust cos are removed.** In the
filtered wide ranking, 4 of the top-10 are trust-flagged (BNY Mellon #1, Cenlar,
Deutsche Bank Trust, Sumitomo). Excluding the trust-flagged banks promotes **SVB
to #1** and brings in Goldman Sachs, JPMorgan and Cross River — a more
bank-like ranking. This is why the flag exists: the trust/custody banks are
mechanically vulnerable (near-100% uninsured) but represent a different business
model, and whether they belong in the headline is a judgement call left open by
the flag.

Both wide R² values are still small (≈0.02–0.06). The honest reading is "weak but
real signal", not "the model predicts deposit outflows".

## Known caveats

- **The flag is a heuristic, not a taxonomy** — see the charter-filter section:
  signal B over-flags ~9 community banks with big trust arms (kept, so harmless),
  and a few brokerage/sweep banks (Morgan Stanley Private Bank, Charles Schwab
  Bank, Stifel) are *not* flagged despite being specialized.
- **RMSE rises as the floor rises** ($0.091 → $0.146 → $0.187). The three censored
  failures stay in every subset while N shrinks, so they dominate the squared
  error more at higher floors. Compare R², not RMSE, across floors.
- **Two 2023 failures are excluded by the size floor**: Heartland Tri-State
  ($139m, failed July 2023) and Citizens Bank Sac City ($66m, November 2023).
  Both failed *after* the 2023Q1 measurement date and both filed a 2023Q1 report,
  so censoring them at −1.0 would be wrong for a 2022Q4→2023Q1 outcome. Only the
  three March–May failures are censored. **If `ASSET_FLOOR` is ever lowered below
  ~$1bn, restrict censoring to those three explicitly** (see the comment in
  `_panel_wide.py`).
- **The narrow and filtered-wide scores agree only moderately** (Spearman ρ = 0.60,
  5/10 top-10 overlap). RQ3 will give different answers depending on which file it
  consumes — run it both ways and report both.
