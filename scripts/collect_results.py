"""
collect_results.py — rebuild the results/ showcase from data/processed/.

data/processed/ is the pipeline's working directory: every script writes there
and knows nothing about results/. This script is the shop window. It copies the
finished artefacts into one directory per research question, writes a README into
each, and moves every superseded version into results/_archive/ with a _prev
suffix.

Idempotent: run it after any pipeline re-run and the directory rebuilds.

A file that exists in data/processed/ but is not listed in MAPPING or ARCHIVE is
reported as UNMAPPED rather than silently dropped, so new artefacts cannot go
missing.

    python3 scripts/collect_results.py
"""

from __future__ import annotations

import json
import shutil
from datetime import date

import config as C

# --------------------------------------------------------------------------- #
#  What goes where
# --------------------------------------------------------------------------- #
MAPPING: dict[str, list[str]] = {
    "rq1_vulnerability": [
        "panel_2022Q4_narrow.csv", "panel_2022Q4_wide.csv",
        "panel_2022Q4_wide_allcharters.csv",
        "rq1_results_narrow.txt", "rq1_results_wide.txt",
        "comparison_narrow_vs_wide.txt",
        "rq1_h1b_robustness.txt", "rq1_failed_bank_robustness.txt",
        "rq1_descriptives.txt",
        "rq1_shap_summary_narrow.png", "rq1_shap_summary_wide.png",
        "vulnerability_scores_narrow.csv",
        "vulnerability_scores_wide.csv", "vulnerability_scores_ols.csv",
        # Table 5: the nested-tuning metrics behind the H1b verdict
        "rq1_tuned_metrics.csv", "rq1_tuned_metrics.tex",
        "rq1_tuned_metrics.txt",
        # Pre-crisis comparison period (2022Q2): the placebo test for the
        # uninsured-deposit result, plus its selection justification
        "rq1_placebo.txt", "rq1_placebo.md",
        "rq1_placebo_table.tex", "rq1_placebo_table.csv",
        "rq1_placebo_by_quarter.csv", "rq1_deposit_growth_by_quarter.csv",
    ],
    "rq2_communications": [
        "rq2_sentences.csv", "rq2_sentence_labels.csv", "rq2_safeguard_scores.csv",
        "rq2_run_metadata.json", "rq2_validation_template.csv",
        "rq2_validation_results.txt",
        "rq2_car.csv", "rq2_car_droplog.csv", "rq2_car_augmented.csv",
        "rq2_car_sanity.txt",
        "rq2_reaction.txt", "rq2_avg_effect.txt", "rq2_enrichment_results.txt",
        "rq2_safeguard_by_event.png", "rq2_robustness.txt",
        "rq2_corpus_table.txt",
    ],
    "rq3_bridge": [
        "rq3_bridge.csv",
        "rq3_link.txt", "rq3_interaction.txt", "rq3_measures.txt",
        "rq3_custody_check.txt",
    ],
    "shared_inputs": [
        "sample_banks.csv", "crosswalk_rssd_permno.csv",
        "headline_numbers.json", "reconciliation.txt",
        # Cross-cutting: the bootstrap covers Table 7 (RQ2), Table 8 and
        # Appendix C (RQ3), so it belongs to neither section alone.
        "rq_wildboot.txt",
    ],
}

# Superseded versions -> results/_archive/.
# EXPLICIT source -> archive name. An automatic "_prev" suffix used to collide:
# rq2_event_regression.txt and rq2_event_regression_prev.txt both mapped to the
# same destination and one silently overwrote the other. Names below are
# chronological and self-describing, so the sequence of drafts is readable.
ARCHIVE: dict[str, str] = {
    # the H2/RQ3 re-split, oldest first
    "rq2_event_regression_interaction.txt": "rq2_h2_v1_interaction_as_h2.txt",
    "rq2_event_regression_prev.txt":        "rq2_h2_v2_car_on_S.txt",
    "rq2_event_regression.txt":             "rq2_h2_v3_combined_before_resplit.txt",
    "rq2_measures_compare.txt":             "rq3_measures_v1_inside_rq2.txt",
    "rq2_car_sanity_interaction.txt":       "rq2_car_sanity_v1.txt",
    "rq2_car_sanity_prev.txt":              "rq2_car_sanity_v2.txt",
    "rq2_enrichment_results_interaction.txt": "rq2_enrichment_v1.txt",
    # RQ1, before the robustness runs
    "comparison_narrow_vs_wide_prev.txt":   "rq1_comparison_before_robustness.txt",
    "rq1_results_wide_prev.txt":            "rq1_results_wide_before_robustness.txt",
    "rq1_results_narrow_prev.txt":          "rq1_results_narrow_before_robustness.txt",
    "vulnerability_scores_wide_prev.csv":   "rq1_scores_wide_before_robustness.csv",
}

# Byte-identical duplicates that the pipeline writes on purpose: run_rq1 writes
# the unsuffixed name, and the wide variant then copies it to *_narrow so the
# original is never overwritten. Only the explicit *_narrow name is showcased;
# these are listed so they are not reported as UNMAPPED.
KNOWN_DUPLICATES = {
    "panel_2022Q4.csv":        "panel_2022Q4_narrow.csv",
    "rq1_results.txt":         "rq1_results_narrow.txt",
    "rq1_shap_summary.png":    "rq1_shap_summary_narrow.png",
    "vulnerability_scores.csv": "vulnerability_scores_narrow.csv",
}

SUBDIRS = {"rq1_vulnerability": C.RES_RQ1, "rq2_communications": C.RES_RQ2,
           "rq3_bridge": C.RES_RQ3, "shared_inputs": C.RES_SHARED}


READMES_STATIC: dict[str, str] = {
    "rq1_vulnerability": """# RQ1 — balance-sheet vulnerability model

**Question.** Which bank characteristics predict the 2023Q1 deposit outflow, and
does ML find the most vulnerable banks better than a linear baseline?

- **H1a** — uninsured deposits, unrealised losses, weak liquidity/capital predict outflow.
- **H1b** — trees beat OLS out of sample.

Code: `rq1_build_panel.py` -> `rq1_model.py` -> `rq1_scores.py` -> `rq1_robustness.py`.

## H1a — partially supported

| Variable | narrow coef | p | wide coef | p |
|---|---|---|---|---|
| `uninsured_share` | -0.23289077 | 0.0000606 | -0.07713467 | 0.0000674 |
| `unrealised_losses` | +0.13258756 | 0.0053520 | +0.00118737 | 0.6629876 |
| `liquidity` | -0.07692121 | 0.4252533 | -0.11036769 | 0.0000372 |
| `ROA` | +1.29383039 | 0.6475342 | +1.37071481 | 0.0075622 |

`uninsured_share` is the only predictor significant in both populations.
`unrealised_losses` is a narrow-only effect. `liquidity` in wide carries the
WRONG sign (more liquidity -> more outflow), probably custodians and wholesale
banks — explain it in the text, do not bury it.

Sign note: `unrealised_losses = (FV - cost)/equity`, so a loss is NEGATIVE and a
POSITIVE coefficient is H1a-consistent.

## H1b — NOT robust

| Population | Feature set | RF beats OLS | Mean margin |
|---|---|---|---|
| WIDE | full (10) | 26/30 repeats | +1.246% |
| WIDE | pruned (3) | **9/30 repeats** | **-0.281%** |
| NARROW | full | 23/30 | +0.977% |
| NARROW | pruned (2) | 17/30 | +0.092% |

The RF margin comes from keeping the INSIGNIFICANT features, which hurt OLS more
than the forest. Under nested tuning RF loses to OLS in all four combinations.
Report H1b as *not robust*, not as "formally supported".

The old "in narrow every model is worse than the mean" is a seed-42 artefact:
over 30 shuffles narrow OOS R2 is POSITIVE (OLS +0.01427315, RF +0.03352837).

## Failed-bank robustness

`uninsured_share` keeps its negative sign under all four treatments, but in
NARROW significance dies when SVB/Signature/First Republic are dropped
(-0.05945248, p = 0.10458407, against a baseline -0.23289077, p < 0.001).
In WIDE it survives (-0.03336049, p = 0.03134800). **Lean on wide.**

## Files

| File | What it is |
|---|---|
| `comparison_narrow_vs_wide.txt` | main RQ1 document |
| `rq1_h1b_robustness.txt` | pruning, 30-repeat CV, nested tuning |
| `rq1_failed_bank_robustness.txt` | four failed-bank treatments |
| `vulnerability_scores_wide.csv` | **leakage-free RF score -> RQ3 input** |
| `vulnerability_scores_ols.csv` | the OLS twin, same protocol |

Scores are oriented **higher = more vulnerable** (`score = -predicted growth`).
Transfer fit to the 277 listed banks is R2 = 0.0302 with SD 0.014 — a weak
measure, and RQ3 shows it behaves like one.
""",


    "shared_inputs": """# Shared inputs and verification

| File | What it is |
|---|---|
| `sample_banks.csv` | 278 listed banks: bank_IDRSSD, holder_RSSD, permno, name, failed |
| `crosswalk_rssd_permno.csv` | RSSD <-> permno crosswalk |
| `headline_numbers.json` | every headline coefficient the pipeline recomputes |
| `reconciliation.txt` | **proof the refactor did not move any number** |

**Reproducibility note.** Neither `sample_banks.csv` nor the crosswalk is built
by a script in `scripts/` — they were assembled outside the repo (via NIC:
bank -> holder -> permco -> permno, N=278). If the sample ever has to be rebuilt,
that step does not exist in the pipeline. Both therefore live in `data/frozen/`,
which is under version control, rather than in the disposable `data/processed/`.

RQ1 models 277 banks: IDRSSD 119528 (Farmers National Bank of Emlenton, permno
93131) is excluded as a merger exit with a null outcome.
"""
}

ARCHIVE_README = """# Archive — superseded versions

Nothing here is current. It is kept so any number quoted in an earlier draft can
still be traced. Names are chronological.

## The one conceptual change

The Safeguard x vulnerability interaction used to be presented as the **H2** test
inside RQ2. That was a mis-filing: H2 is about the AVERAGE effect of the
communication type, while the interaction asks whether VULNERABLE banks react
differently — which is H3. It now lives in `../rq3_bridge/`.

| File | What it was |
|---|---|
| `rq2_h2_v1_interaction_as_h2.txt` | v1 — the interaction presented as the H2 headline |
| `rq2_h2_v2_car_on_S.txt` | v2 — after H2 was realigned to CAR ~ S, interaction demoted |
| `rq2_h2_v3_combined_before_resplit.txt` | v3 — RQ2 and the interaction in one file, before the split |
| `rq3_measures_v1_inside_rq2.txt` | the three-measure comparison while it still sat in RQ2 |
| `rq2_car_sanity_v1.txt`, `rq2_car_sanity_v2.txt` | earlier per-event CAR sanity tables |
| `rq2_enrichment_v1.txt` | the enrichment report before the H2 rework |
| `rq1_comparison_before_robustness.txt` | main RQ1 document before the robustness runs |
| `rq1_results_{wide,narrow}_before_robustness.txt` | per-population reports, same vintage |
| `rq1_scores_wide_before_robustness.csv` | the RQ3 input score, same vintage |

Two superseded SCRIPTS that also lived here were lost to an over-eager cleanup
and were never under version control. Their outputs are the `.txt` files above,
and their logic survives in the current `rq2_avg_effect.py`,
`rq3_interaction.py` and `rq3_measures.py`.

Every current number is reconciled against these in
`../shared_inputs/reconciliation.txt`.
"""




def _live() -> dict:
    """Every headline coefficient the pipeline last produced, keyed as in FROZEN.

    The section READMEs used to carry these as literals, and they went stale the
    moment the 2019Q3 look-ahead fix moved twelve of them: results/README.md and
    results/rq3_bridge/README.md ended up contradicting rq3_link.txt sitting two
    directories away. Reading them means a README cannot drift from the numbers
    it is describing.

    Falls back to an empty dict when the pipeline has never run, in which case
    _num() prints a pointer to the source file instead of a wrong number.
    """
    src = C.HEADLINE_JSON if C.HEADLINE_JSON.exists() else (
        C.RES_SHARED / "headline_numbers.json")
    if not src.exists():
        return {}
    return {k: v for k, v in json.loads(src.read_text()).items() if v is not None}


def _num(live: dict, key: str, fmt: str = "+.8f", fallback: str = "see the .txt") -> str:
    """One headline number, formatted, or a pointer if it is not on file."""
    v = live.get(key)
    return fallback if v is None else format(float(v), fmt)


def _full_row(measure: str) -> dict[str, str]:
    """b, SE and both p-values for one measure, from rq3_custody_check.txt.

    headline_numbers.json carries the two-way SE (`se_twoway`) and the
    event-clustered p, but the interaction table also needs the EVENT-clustered
    SE and the two-way p, which only this report prints. Parse its "full" row:

        RF score   full   -0.38507629   0.12471181   0.01148834   0.02208081  [CI]
                          ^beta         ^SE(event)   ^p(event)    ^p(two-way)
    """
    miss = {"beta": "—", "se": "—", "p_event": "—", "p_twoway": "—"}
    src = C.RQ3_CUSTODY
    if not src.exists():
        return miss
    for line in src.read_text().splitlines():
        if not line.strip().startswith(measure):
            continue
        s = line.split()
        if "full" not in s:
            continue
        i = s.index("full")
        try:                        # all four must parse; the section-5 line
            v = [float(s[i + k]) for k in range(1, 5)]   # reads "full beta ...",
        except (ValueError, IndexError):                 # so it filters itself out
            continue
        return {"beta": f"{v[0]:+.8f}", "se": f"{v[1]:.8f}",
                "p_event": f"{v[2]:.8f}", "p_twoway": f"{v[3]:.8f}"}
    return miss


def _mde_pp(fallback: str = "see rq2_avg_effect.txt") -> str:
    """The gamma1 MDE in CAR percentage points, from rq2_avg_effect.txt.

    Only the "% of a CAR SD" form reaches headline_numbers.json, so the pp figure
    is read from the report that prints it:

          = 2.371343 pp of CAR per one-SD move in S
    """
    src = C.RQ2_AVG_EFFECT
    if src.exists():
        for line in src.read_text().splitlines():
            if "pp of CAR per one-SD move in S" in line:
                try:
                    return f"{float(line.split()[1]):.6f}"
                except (ValueError, IndexError):
                    break
    return fallback


def _absorb_max(label: str, fallback: str = "~1e-14") -> str:
    """max |residual| from an absorption check in rq3_interaction.txt, e.g.

        S on event dummies            1.000000000000     7.061e-14   ...
    """
    src = C.RQ3_INTERACTION
    if src.exists():
        for line in src.read_text().splitlines():
            if line.strip().startswith(label):
                s = line.split()
                if len(s) >= 3:
                    return s[-3]
    return fallback


def _readmes() -> dict[str, str]:
    """The per-section READMEs, with every coefficient read from the live output.

    RQ2's and RQ3's used to be prose literals. Twelve of those numbers moved with
    the 2019Q3 look-ahead fix and the literals did not, so the showcase README
    contradicted the .txt file sitting beside it. Building them here means the
    two cannot drift again.
    """
    L = _live()
    ui, rf = _full_row("uninsured_share"), _full_row("RF score")

    rq2 = f"""# RQ2 — market reaction to CBDC communications

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
| Event level, 11 event-mean CARs on 11 S | {_num(L, 'rq2.gamma1.event_level.coef')} | \
{_num(L, 'rq2.gamma1.event_level.se', '.8f')} | \
{_num(L, 'rq2.gamma1.event_level.p', '.8f')} |
| Pooled with controls (cluster = event) | {_num(L, 'rq2.gamma1.pooled_ctrl.coef')} | \
{_num(L, 'rq2.gamma1.pooled_ctrl.se', '.8f')} | \
{_num(L, 'rq2.gamma1.pooled_ctrl.p', '.8f')} |

Pearson corr(event-mean CAR, S) = {_num(L, 'rq2.corr.pearson')}; \
Spearman = {_num(L, 'rq2.corr.spearman')}.

**H2 is not supported.** The sign is in the predicted direction; nothing is close
to significant.

### Structural points, enforced not described

- S varies ONLY across the 11 events, so **no event-FE spec is estimated for S**:
  the event dummies absorb it exactly (max residual {_absorb_max('S on event dummies')}).
  Collinearity, not a zero.
- SEs cluster on the **EVENT** (11 clusters, t(10)). Bank clustering would treat
  276 banks as independent information about an 11-valued regressor and shrinks
  the SE by a factor of ~14; it is shown only to make that visible.

### How tight is the null?

MDE for gamma1 = {_mde_pp()} pp of CAR per 1 SD of S = \
**{_num(L, 'mde_pct_car_sd', '.2f')}% of a CAR SD**. That is WIDE. Say "H2 not
supported and a large effect ruled out" — not "there is no effect".
"""

    rq3 = f"""# RQ3 — the bridge between RQ1 vulnerability and RQ2 reaction

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
| RF score | {_num(L, 'rq3.delta1.score_rf.coef')} | {_num(L, 'rq3.delta1.score_rf.p', '.8f')} |
| OLS score | {_num(L, 'rq3.delta1.score_ols.coef')} | {_num(L, 'rq3.delta1.score_ols.p', '.8f')} |
| **`uninsured_share`** | **{_num(L, 'rq3.delta1.uninsured.coef')}** | \
**{_num(L, 'rq3.delta1.uninsured.p', '.8f')}** |

The published RQ3 null is a statement about the **ML score**. It does not extend
to `uninsured_share`, which has a real level effect with the H3-predicted sign.

## 2. Interaction — b (`rq3_interaction.txt`)

`CAR = EventFE + BankFE + b*(S x vulnerability) + e`

| Measure | b | SE (event) | p (event) | p (two-way) |
|---|---|---|---|---|
| `uninsured_share` | {ui['beta']} | {ui['se']} | {ui['p_event']} | {ui['p_twoway']} |
| RF score | {rf['beta']} | {rf['se']} | {rf['p_event']} | {rf['p_twoway']} |

Verified numerically: S is absorbed by the event dummies
(R2 = 1.000000000000); the STATIC RF score is absorbed by the bank dummies;
**`uninsured_share` is NOT** (residual max \
{_absorb_max('uninsured_share on bank dummies')}), because it is re-read from 8
characteristic quarters — so its main effect stays identified and is reported.

The event-clustered SE is the headline: the two-way covariance is not PSD at this
sample size and returns NaN SEs for many FE dummies.

## 3. Level vs reaction — the verdict (`rq3_measures.txt`)

`uninsured_share`: **level effect real** (p = {_num(L, 'rq3.delta1.uninsured.p', '.8f')}), \
**reaction null** (p = {_num(L, 'rq3.b.uninsured_share.p_event', '.8f')}). Vulnerable
banks sat at a lower abnormal return across these windows, but the gap does not
move with the safeguard content. That is a property of *which banks these are*
over 2019-2023, not a response to CBDC.

**THE H3 MECHANISM IS NOT SUPPORTED.**

The one significant interaction (RF score) is an **artefact**, and has the wrong
sign for the hypothesis anyway:

| Diagnostic | Result |
|---|---|
| Calendar placebo | corr(S, calendar rank) = **{_num(L, 'rq3.placebo.corr_S_time')}**; time x vuln gives -0.17749331; jointly neither survives |
| Custody banks | dropping 9 moves b to {_num(L, 'rq3.custody_drop.b')} (two-way p {_num(L, 'rq3.custody_drop.p', '.8f')}) — **63.8% of the magnitude** |
| Leave-one-event-out | sign never flips, so not one event — but that does not rescue it |

The OLS score does the same job (b = {_num(L, 'rq3.b.score_ols.coef')}) and agrees
with the RF score at Spearman {_num(L, 'rq3.corr.rf_ols.spearman')}, so any credit
belongs to using a fitted composite, not to machine learning. Note RF vs
`uninsured_share` Spearman is only **{_num(L, 'rq3.corr.rf_uninsured.spearman')}**:
they rank banks almost independently.
"""
    return {**READMES_STATIC, "rq2_communications": rq2, "rq3_bridge": rq3}


def _source(name: str):
    """Where a mapped file lives.

    Most artefacts come from data/processed/. Two kinds do not: the four
    unreproducible inputs sit in data/frozen/, and the superseded snapshots that
    feed results/_archive/ sit in data/frozen/superseded/ — no script rebuilds
    those, so they are version-controlled rather than left in disposable output."""
    for candidate in (C.FROZEN_DIR / name, C.FROZEN_SUPERSEDED / name):
        if candidate.exists():
            return candidate
    return C.PROC / name


def _regenerable(name: str) -> bool:
    """True only if this exact file can be rebuilt from data/processed/. Anything
    else in results/ (a hand-placed note, an archived script) is NOT ours to
    delete: pruning must only ever remove files the pipeline can put back."""
    if _source(name).exists():
        return True
    for src, dest in ARCHIVE.items():
        if dest == name and _source(src).exists():
            return True
    return False


def main() -> None:
    if not C.PROC.exists():
        raise SystemExit(f"no {C.PROC} — run the pipeline first")

    C.RESULTS.mkdir(exist_ok=True)
    mapped: set[str] = set()
    pruned: list[str] = []
    kept_unknown: list[str] = []
    copied = missing = 0
    summary: list[tuple[str, int, int]] = []
    live = _live()
    readmes = _readmes()

    for sub, files in MAPPING.items():
        dest = SUBDIRS[sub]
        dest.mkdir(parents=True, exist_ok=True)
        # prune anything no longer mapped here, so the showcase is a true rebuild
        # rather than an accumulation of every layout the project ever had
        keep = set(files) | {"README.md"}
        for stale in sorted(p for p in dest.iterdir()
                            if p.is_file() and p.name not in keep):
            if _regenerable(stale.name):
                stale.unlink()
                pruned.append(f"{sub}/{stale.name}")
            else:
                kept_unknown.append(f"{sub}/{stale.name}")
        n_ok = n_miss = 0
        for name in files:
            mapped.add(name)
            src = _source(name)
            if src.exists():
                shutil.copy2(src, dest / name)
                n_ok += 1
            else:
                n_miss += 1
        (dest / "README.md").write_text(readmes[sub])
        summary.append((sub, n_ok, n_miss))
        copied += n_ok
        missing += n_miss

    C.RES_ARCHIVE.mkdir(parents=True, exist_ok=True)
    keep_arch = {"README.md"}
    keep_arch |= set(ARCHIVE.values())
    for stale in sorted(p for p in C.RES_ARCHIVE.iterdir()
                        if p.is_file() and p.name not in keep_arch):
        if _regenerable(stale.name):
            stale.unlink()
            pruned.append(f"_archive/{stale.name}")
        else:
            kept_unknown.append(f"_archive/{stale.name}")
    n_arch = 0
    for name, out in ARCHIVE.items():
        mapped.add(name)
        src = _source(name)
        if not src.exists():
            continue
        shutil.copy2(src, C.RES_ARCHIVE / out)
        n_arch += 1
    (C.RES_ARCHIVE / "README.md").write_text(ARCHIVE_README)

    mapped |= set(KNOWN_DUPLICATES)
    unmapped = sorted(p.name for p in C.PROC.iterdir()
                      if p.is_file() and p.name not in mapped)

    idx = ["# Results", "",
           f"Assembled by `scripts/collect_results.py` — {date.today().isoformat()}.",
           "",
           "`data/processed/` is the pipeline's working directory; this folder is the",
           "shop window. Rebuild after any re-run:", "",
           "```bash", "python3 scripts/collect_results.py", "```", "",
           "## Sections", "",
           "| Folder | Question | Verdict |",
           "|---|---|---|",
           "| `rq1_vulnerability/` | H1a/H1b — balance-sheet model of outflow | H1a partial, H1b **not robust** |",
           "| `rq2_communications/` | H2 — did the market react, on average? | **no reaction**, gamma1 null |",
           "| `rq3_bridge/` | H3 — do the two proxies meet? | **mechanism not supported** |",
           f"| `shared_inputs/` | sample, crosswalk, reconciliation | 278 banks; "
           f"{len(C.FROZEN)}/{len(C.FROZEN)} numbers reconcile |",
           "| `_archive/` | superseded versions | incl. the interaction-as-H2 era |",
           "",
           "## The through-line", "",
           "The market did not react to CBDC communications at all (0 of 11 events,",
           "with volatility and turnover suppressed on 7 of 11). With no average",
           "reaction there is nothing for the communication type to modulate (RQ2",
           "gamma1 null) and nothing for a cross-section of vulnerability to explain",
           "(RQ3 mechanism unsupported). The three nulls are one finding, not three.",
           "",
           "Two caveats that must travel with that sentence:", "",
           "- the nulls are **not equally tight**. RQ3's coefficients are bounded at",
           "  ~5-9.5% of a CAR SD; RQ2's gamma1 only at ~49%. A modest average effect",
           "  is not excluded.",
           "- `uninsured_share` **does** have a real level effect on CAR",
           f"  ({_num(live, 'rq3.delta1.uninsured.coef')}, "
           f"p = {_num(live, 'rq3.delta1.uninsured.p', '.8f')}). It is not a "
           f"reaction to CBDC content —",
           "  its interaction with the signal is null — but it is not nothing.",
           ""]

    if unmapped:
        idx += ["## Not filed", "",
                "In `data/processed/` but not listed in MAPPING or ARCHIVE:", ""]
        idx += [f"- `{n}`" for n in unmapped] + [""]

    (C.RESULTS / "README.md").write_text("\n".join(idx))

    print(f"\nresults/ rebuilt — {date.today().isoformat()}")
    for sub, n_ok, n_miss in summary:
        print(f"  {sub:<22} {n_ok:>3} files"
              + (f"   ({n_miss} missing)" if n_miss else ""))
    print(f"  {'_archive':<22} {n_arch:>3} files")
    print(f"\ncopied {copied} files into {C.RESULTS}/ ({len(MAPPING)} sections)")
    if pruned:
        print(f"pruned {len(pruned)} stale file(s): {', '.join(pruned)}")
    if kept_unknown:
        print(f"LEFT ALONE ({len(kept_unknown)}) — not reproducible from "
              f"data/processed/, so not deleted: {', '.join(kept_unknown)}")
    if missing:
        print(f"{missing} mapped file(s) not found in {C.PROC}")
    if unmapped:
        print(f"UNMAPPED ({len(unmapped)}): {', '.join(unmapped)}")


if __name__ == "__main__":
    main()
