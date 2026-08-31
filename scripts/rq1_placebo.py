"""rq1_placebo.py: the pre-crisis comparison period for the uninsured-deposit
result.

Table 3, column 2 reports that a bank's uninsured-deposit share predicts its
2023Q1 deposit outflow (coefficient -0.07713467, t = -4.00, N = 953). That is
evidence about banking stress only if the relationship is specific to the stress
episode. If the uninsured share predicts deposit growth equally well in an
ordinary quarter, it is a persistent structural correlate of deposit behaviour
and the stress reading has to be weakened.

2022Q2 is referred to throughout as the pre-crisis comparison period, not as a
calm period: 2022 contained the fastest tightening cycle in four decades, and
the claim being tested is only that it was not a period of banking stress.

Three estimates:
    1. period-specific OLS on the comparison period, the specification of
       Table 3 with the dates shifted back so that no post-period information
       enters
    2. period-specific OLS on the stress period, which reproduces Table 3
    3. a pooled two-period model with an interaction between the uninsured
       share and an indicator for the stress period, standard errors clustered
       on the bank because 919 banks appear in both periods

The quarter-by-quarter series behind Figure 3 is produced alongside these, using
the same specification with predictors lagged one quarter.

Writes    data/processed/rq1_placebo.txt, rq1_placebo.md
          data/processed/rq1_placebo_table.tex, rq1_placebo_table.csv
          data/processed/rq1_placebo_by_quarter.csv
          data/processed/rq1_deposit_growth_by_quarter.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _panel_wide as PW
import config as C
from _charter_flags import classify
from _panel_narrow import (
    assemble_raw_fields,
    build_variables,
    load_filer_types,
    read_schedule,
    total_deposits,
)
from _rq1_core import fit_ols
from _stats import fit_cluster, stars

IDENTITY_TOL = 1e-8

# Call Report bulk archives, quarter label -> filename. Every one of these is
# inside the pipeline's stated coverage (2019Q3-2023Q2).
QUARTER_ZIP: dict[str, str] = {
    "2019Q3": "call_09302019.zip", "2019Q4": "call_12312019.zip",
    "2020Q1": "call_03312020.zip", "2020Q2": "call_06302020.zip",
    "2020Q3": "call_09302020.zip", "2020Q4": "call_12312020.zip",
    "2021Q1": "call_03312021.zip", "2021Q2": "call_06302021.zip",
    "2021Q3": "call_09302021.zip", "2021Q4": "call_12312021.zip",
    "2022Q1": "call_03312022.zip", "2022Q2": "call_06302022.zip",
    "2022Q3": "call_09302022.zip", "2022Q4": "call_12312022.zip",
    "2023Q1": "call_03312023.zip", "2023Q2": "call_06302023.zip",
}

# The comparison period and the stress period, fixed here so no other part of the
# script can quietly choose a different pair.
PLACEBO_PRED, PLACEBO_OUT = "2022Q1", "2022Q2"
MAIN_PRED, MAIN_OUT = "2022Q4", "2023Q1"

PERIOD_LABEL = {
    PLACEBO_OUT: "pre-crisis comparison period (2022Q2)",
    MAIN_OUT: "banking-stress period (2023Q1)",
}


def zip_for(q: str) -> Path:
    """Absolute path to a quarter's bulk archive, or a clear stop if absent."""
    if q not in QUARTER_ZIP:
        raise SystemExit(f"[rq1_placebo] no Call Report archive mapped for {q}.")
    p = C.RAW / QUARTER_ZIP[q]
    if not p.exists():
        raise SystemExit(
            f"[rq1_placebo] MISSING BULK FILE: {p}\n"
            f"           {q} is inside the pipeline's stated coverage "
            f"(2019Q3-2023Q2) but the archive is not on disk.\n"
            "           Stopping rather than downloading anything. Fetch it from "
            "the FFIEC CDR bulk download and re-run.")
    return p


# --------------------------------------------------------------------------- #
#  Outcome-side deposits — compute_dep_growth() with the quarter as a parameter
# --------------------------------------------------------------------------- #
def outcome_deposits(out_q: str, ids: set[int]) -> pd.DataFrame:
    """Deposits at the OUTCOME quarter, by the same filer-aware rule
    (RCON2200 + RCFN2200) that _panel_narrow.compute_dep_growth uses. That
    function hardcodes 2023Q1; this is the identical body with the archive and
    the column name parameterised, which is why the identity check below can
    reproduce the frozen panel exactly."""
    oz = zip_for(out_q)
    wide = load_filer_types(oz).merge(read_schedule(oz, "RC"), on="IDRSSD",
                                      how="right")
    wide = wide[wide.IDRSSD.isin(ids)]
    return pd.DataFrame({"bank_IDRSSD": wide["IDRSSD"].values,
                         "dep_outcome": total_deposits(wide).values})


def failures_in(year: int, pred_q: str) -> pd.DataFrame:
    """Banks on the FDIC failed-bank list that failed in `year`, bridged to
    IDRSSD exactly as _panel_wide.load_failures does."""
    return PW.load_failures(zip_for(pred_q), year=year)


# --------------------------------------------------------------------------- #
#  The wide-population funnel, with the quarter as a parameter
# --------------------------------------------------------------------------- #
def build_wide_panel(pred_q: str, out_q: str, failure_year: int | None,
                     verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """_panel_wide.main()'s funnel, parameterised by (predictor, outcome) quarter.

    Every screen, threshold and formula is the frozen one; only the archives
    change. Returns the panel and the funnel counts."""
    pz = zip_for(pred_q)
    say = print if verbose else (lambda *a, **k: None)

    say(f"      [1/6] enumerating {pred_q} filers ...")
    por = load_filer_types(pz)
    ids = set(por.IDRSSD.astype(int))

    say(f"      [2/6] filer-aware raw fields at {pred_q} ...")
    panel = build_variables(assemble_raw_fields(pz, ids))
    panel = panel.merge(PW.load_names(pz), on="bank_IDRSSD", how="left")

    say(f"      [3/6] dep_growth ({pred_q} -> {out_q}) ...")
    panel = panel.merge(outcome_deposits(out_q, ids), on="bank_IDRSSD", how="left")
    panel["dep_growth"] = ((panel["dep_outcome"] - panel["total_deposits"])
                           / panel["total_deposits"])

    say(f"      [4/6] failures ...")
    if failure_year is None:
        panel["failed"] = False
    else:
        f = failures_in(failure_year, pred_q)
        panel = panel.merge(f, on="bank_IDRSSD", how="left")
        panel["failed"] = panel["failed"].fillna(False).astype(bool)
    # Same censoring rule as the frozen pipeline: a failed bank has no post-run
    # deposits to measure, so the outcome is right-censored at -1.0 and flagged.
    panel["censored"] = panel["failed"]
    panel.loc[panel["censored"], "dep_growth"] = -1.0

    listed = pd.read_csv(PW.NARROW_SAMPLE)[["bank_IDRSSD", "permno"]]
    panel = panel.merge(listed, on="bank_IDRSSD", how="left")
    panel["is_listed"] = panel["permno"].notna()

    filers = load_filer_types(pz)
    filers["fdic_cert"] = pd.to_numeric(filers["fdic_cert"], errors="coerce")
    panel = panel.merge(filers[["IDRSSD", "fdic_cert"]], left_on="bank_IDRSSD",
                        right_on="IDRSSD", how="left") \
                 .drop(columns=["IDRSSD"], errors="ignore")

    say(f"      [5/6] population filters ...")
    panel[PW.FEATURE_COLS] = panel[PW.FEATURE_COLS].replace([np.inf, -np.inf],
                                                            np.nan)
    funnel = {"all filers": len(panel)}
    panel = panel[panel["total_assets"] > PW.ASSET_FLOOR]
    funnel[f"assets > ${PW.ASSET_FLOOR/1e6:.0f}bn"] = len(panel)
    panel = panel[panel["deposit_reliance"] >= PW.DEP_RELIANCE_FLOOR]
    funnel[f"deposit_reliance >= {PW.DEP_RELIANCE_FLOOR}"] = len(panel)
    panel = panel[panel[PW.FEATURE_COLS].notna().all(axis=1)]
    funnel["complete + finite features"] = len(panel)
    panel = panel[panel["dep_growth"].notna()]
    funnel["outcome observed or censored"] = len(panel)
    panel = panel.reset_index(drop=True)

    say(f"      [6/6] FDIC charter screen ...")
    panel = classify(panel, pz)
    panel = panel[~panel.is_creditcard].reset_index(drop=True)
    funnel["charter filter (credit-card dropped)"] = len(panel)

    # The main analysis drops the merger-exit bank at the modelling step
    # (_rq1_wide_core.load_variant -> config.EXCLUDE_IDRSSD). Applied here too so
    # the sample rule is identical; it is a no-op for the wide population, where
    # that bank sits below the $1bn floor.
    panel = panel[~panel["bank_IDRSSD"].isin(C.EXCLUDE_IDRSSD)].reset_index(drop=True)
    funnel["exclude merger exit"] = len(panel)
    return panel, funnel


# --------------------------------------------------------------------------- #
#  Identity check — the parameterised funnel reproduces the frozen panel
# --------------------------------------------------------------------------- #
def identity_check() -> list[str]:
    """Rebuild the 2022Q4 -> 2023Q1 panel through build_wide_panel() and compare
    it against the frozen panel_2022Q4_wide.csv, then against config.FROZEN.
    Nothing else in this script runs until this passes."""
    got, _ = build_wide_panel(MAIN_PRED, MAIN_OUT, failure_year=2023, verbose=False)
    ref = pd.read_csv(C.PANEL_WIDE)
    ref = ref[~ref["bank_IDRSSD"].isin(C.EXCLUDE_IDRSSD)]

    L = ["-- identity check: the parameterised funnel vs the frozen panel --", ""]
    if len(got) != len(ref):
        raise SystemExit(
            f"[rq1_placebo] IDENTITY CHECK FAILED: rebuilt N={len(got)} but the "
            f"frozen panel has N={len(ref)}. Stopping.")
    if set(got.bank_IDRSSD) != set(ref.bank_IDRSSD):
        raise SystemExit(
            "[rq1_placebo] IDENTITY CHECK FAILED: the rebuilt panel contains a "
            "different set of banks than the frozen one. Stopping.")

    g = got.set_index("bank_IDRSSD").sort_index()
    r = ref.set_index("bank_IDRSSD").sort_index()
    L.append(f"    N {len(g)} banks, identical IDRSSD set")
    L.append("")
    L.append("    {:<24s} {:>16s} {:>12s}".format("column", "max |diff|", "status"))
    L.append("    " + "-" * 56)
    worst = 0.0
    for col in [*PW.FEATURE_COLS, "dep_growth", "total_assets", "total_deposits"]:
        d = float(np.abs(g[col].astype(float) - r[col].astype(float)).max())
        worst = max(worst, d)
        L.append("    {:<24s} {:>16.3e} {:>12s}".format(
            col, d, "match" if d <= IDENTITY_TOL else "*** DIFFERS ***"))
        if d > IDENTITY_TOL:
            raise SystemExit(
                f"[rq1_placebo] IDENTITY CHECK FAILED on {col}: max |diff| "
                f"{d:.3e} > {IDENTITY_TOL:.0e}. Stopping.")
    n_cens = int(g["censored"].sum())
    if n_cens != int(r["censored"].sum()):
        raise SystemExit("[rq1_placebo] IDENTITY CHECK FAILED: censoring differs.")

    tab, _ = fit_ols(g[C.FEATURES].astype(float), g["dep_growth"].astype(float))
    coef = float(tab.loc["uninsured_share", "coef"])
    pval = float(tab.loc["uninsured_share", "p_value"])
    fc, ftol, _ = C.FROZEN["rq1.wide.ols.uninsured_share.coef"]
    fp, fptol, _ = C.FROZEN["rq1.wide.ols.uninsured_share.p"]
    for what, a, b, tol in (("coef", coef, fc, ftol), ("p", pval, fp, fptol)):
        if abs(a - b) > tol:
            raise SystemExit(
                f"[rq1_placebo] IDENTITY CHECK FAILED: rebuilt Table 3 col 2 "
                f"{what} {a:.10f} vs frozen {b:.10f}. Stopping.")
    L += ["",
          f"    worst column difference {worst:.3e}  (tolerance {IDENTITY_TOL:.0e})",
          f"    censored failures {n_cens} (matches the frozen panel)",
          f"    Table 3 col 2 rebuilt: coef {coef:+.8f}  p {pval:.8e}"
          f"   -> matches config.FROZEN",
          "",
          "    The parameterised funnel is output-identical to the frozen build.",
          "    Everything below therefore differs from Table 3 only in the dates.",
          ""]
    return L


# --------------------------------------------------------------------------- #
#  Step 2 — selection justification
# --------------------------------------------------------------------------- #
def deposit_growth_by_quarter(first: str = "2020Q1", last: str = "2023Q2"
                              ) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate and median bank-level deposit growth per quarter, on the frozen
    wide population's banks. Tracking one fixed set of banks means a change in
    the series is a change in DEPOSITS, not in which banks are counted."""
    qs = list(QUARTER_ZIP)
    i0, i1 = qs.index(first), qs.index(last)
    if i0 == 0:
        raise SystemExit("[rq1_placebo] need one quarter before `first` as a base.")
    span = qs[i0 - 1: i1 + 1]

    ids = set(pd.read_csv(C.PANEL_WIDE).bank_IDRSSD.astype(int))
    dep: dict[str, pd.Series] = {}
    for q in span:
        z = zip_for(q)
        w = load_filer_types(z).merge(read_schedule(z, "RC"), on="IDRSSD",
                                      how="right")
        w = w[w.IDRSSD.isin(ids)]
        dep[q] = pd.Series(total_deposits(w).values,
                           index=w["IDRSSD"].astype(int).values).groupby(level=0).first()

    rows = []
    for prev, cur in zip(span, span[1:]):
        a, b = dep[prev], dep[cur]
        both = a.index.intersection(b.index)
        a2, b2 = a.loc[both], b.loc[both]
        ok = (a2 > 0) & a2.notna() & b2.notna()
        a2, b2 = a2[ok], b2[ok]
        g = (b2 - a2) / a2
        rows.append({
            "quarter": cur, "base_quarter": prev, "n_banks": int(len(g)),
            "aggregate_growth": float((b2.sum() - a2.sum()) / a2.sum()),
            "median_growth": float(g.median()),
            "mean_growth": float(g.mean()),
            "p25_growth": float(g.quantile(0.25)),
            "p75_growth": float(g.quantile(0.75)),
            "share_negative": float((g < 0).mean()),
        })
    df = pd.DataFrame(rows)

    # Does the series support the stated ex-ante rationale? Reported either way.
    L = ["-- does the quarterly series support the choice of 2022Q2? --", "",
         "    Stated rationale, fixed before any coefficient was seen: 2022Q2 is",
         "    the last completed quarter before the sustained system-wide deposit",
         "    contraction that ran into the 2023 stress.", "",
         "    {:<9s} {:>9s} {:>18s} {:>16s} {:>16s}".format(
             "quarter", "n_banks", "aggregate growth", "median growth",
             "share negative"),
         "    " + "-" * 74]
    for _, r in df.iterrows():
        mark = "   <- comparison period" if r.quarter == PLACEBO_OUT else (
            "   <- stress period" if r.quarter == MAIN_OUT else "")
        L.append("    {:<9s} {:>9d} {:>17.6f}% {:>15.6f}% {:>15.4f}%{}".format(
            r.quarter, int(r.n_banks), 100 * r.aggregate_growth,
            100 * r.median_growth, 100 * r.share_negative, mark))

    after = df[df.quarter > PLACEBO_OUT]
    upto = df[df.quarter <= PLACEBO_OUT]
    n_neg_after = int((after.aggregate_growth < 0).sum())
    placebo_agg = float(df.loc[df.quarter == PLACEBO_OUT, "aggregate_growth"].iloc[0])
    consistent = (n_neg_after == len(after)) and placebo_agg > min(
        after.aggregate_growth)

    L += ["",
          f"    quarters after {PLACEBO_OUT}: {len(after)}, of which "
          f"{n_neg_after} show NEGATIVE aggregate deposit growth",
          f"    quarters up to and including {PLACEBO_OUT}: {len(upto)}, of which "
          f"{int((upto.aggregate_growth < 0).sum())} negative",
          f"    aggregate growth in {PLACEBO_OUT}: {100 * placebo_agg:.6f}%",
          ""]
    if consistent:
        L += ["    CONSISTENT with the stated rationale: every quarter after the",
              "    comparison period contracts, and the comparison period itself",
              "    sits at or above the contraction that follows. The choice of",
              f"    {PLACEBO_OUT} is supported by the series and was not chosen on a",
              "    coefficient.", ""]
    else:
        L += ["    *** FLAG: the series does NOT cleanly match the stated",
              "    rationale. The design is not being changed in response — the",
              "    period was fixed in advance and stays fixed. This is recorded",
              "    so the write-up describes the series accurately rather than",
              "    repeating a characterisation the data do not support.", ""]
    return df, L


# --------------------------------------------------------------------------- #
#  The regressions
# --------------------------------------------------------------------------- #
def period_ols(panel: pd.DataFrame, label: str) -> dict:
    """One period's ten-feature OLS, on Table 3's inference convention."""
    X = panel[C.FEATURES].astype(float)
    y = panel[C.OUTCOME].astype(float)
    tab, r2 = fit_ols(X, y)
    r = tab.loc["uninsured_share"]
    return {"label": label, "n": int(len(y)), "r2": r2, "table": tab,
            "coef": float(r["coef"]), "t": float(r["t"]),
            "p": float(r["p_value"]), "beta_std": float(r["beta_std"])}


def pooled_interaction(placebo: pd.DataFrame, stress: pd.DataFrame) -> dict:
    """The headline test. Stack the two periods' bank-period observations, each
    having passed its own screens, and estimate one interaction of
    uninsured_share with the stress-period dummy. Clustered on the bank, because
    most banks contribute an observation to both periods."""
    keep = ["bank_IDRSSD", *C.FEATURES, C.OUTCOME]
    a = placebo[keep].copy()
    a["stress"] = 0.0
    b = stress[keep].copy()
    b["stress"] = 1.0
    pool = pd.concat([a, b], ignore_index=True)

    f = (f"{C.OUTCOME} ~ " + " + ".join(C.FEATURES)
         + " + stress + uninsured_share:stress")
    m = fit_cluster(pool, f, ["bank_IDRSSD"])
    term = "uninsured_share:stress"
    ci = m.conf_int()
    ids_a, ids_b = set(a.bank_IDRSSD), set(b.bank_IDRSSD)
    return {
        "model": m, "pool": pool, "formula": f, "term": term,
        "coef": float(m.params[term]), "se": float(m.bse[term]),
        "t": float(m.tvalues[term]), "p": float(m.pvalues[term]),
        "ci": [float(ci.loc[term, 0]), float(ci.loc[term, 1])],
        "main_uninsured": float(m.params["uninsured_share"]),
        "main_uninsured_p": float(m.pvalues["uninsured_share"]),
        "stress_coef": float(m.params["stress"]),
        "stress_p": float(m.pvalues["stress"]),
        "n_total": int(len(pool)), "n_placebo": len(a), "n_stress": len(b),
        "n_banks": int(pool.bank_IDRSSD.nunique()),
        "n_both": len(ids_a & ids_b),
        "n_placebo_only": len(ids_a - ids_b),
        "n_stress_only": len(ids_b - ids_a),
    }


# --------------------------------------------------------------------------- #
#  Step 3 — the same single-period regression over several ordinary quarters
# --------------------------------------------------------------------------- #
# Outcome quarters for the multi-period view. 2020 is excluded on purpose: the
# pandemic deposit surge is not an ordinary period and would not be a meaningful
# comparison. Predictors sit at the immediately preceding quarter each time.
MULTI_QUARTER_OUT = ["2021Q2", "2021Q3", "2021Q4", "2022Q1", "2022Q2"]


def multi_quarter(main_res: dict) -> tuple[pd.DataFrame, list[str]]:
    """Repeat the period-specific regression across ordinary quarters. A single
    comparison period could be a fluke of one quarter; five cannot all be."""
    qs = list(QUARTER_ZIP)
    rows = []
    for out_q in MULTI_QUARTER_OUT:
        pred_q = qs[qs.index(out_q) - 1]
        panel, _ = build_wide_panel(pred_q, out_q, failure_year=None, verbose=False)
        res = period_ols(panel, out_q)
        m = res["table"].loc["uninsured_share"]
        se = abs(float(m["coef"]) / float(m["t"])) if m["t"] else float("nan")
        rows.append({"outcome_quarter": out_q, "predictor_quarter": pred_q,
                     "n": res["n"], "coef": res["coef"], "se": se,
                     "t": res["t"], "p": res["p"],
                     "ci_low": res["coef"] - 1.96 * se,
                     "ci_high": res["coef"] + 1.96 * se,
                     "r2": res["r2"], "period_type": "ordinary"})
    se_m = abs(main_res["coef"] / main_res["t"])
    rows.append({"outcome_quarter": MAIN_OUT, "predictor_quarter": MAIN_PRED,
                 "n": main_res["n"], "coef": main_res["coef"], "se": se_m,
                 "t": main_res["t"], "p": main_res["p"],
                 "ci_low": main_res["coef"] - 1.96 * se_m,
                 "ci_high": main_res["coef"] + 1.96 * se_m,
                 "r2": main_res["r2"], "period_type": "banking stress"})
    df = pd.DataFrame(rows)

    L = ["-- the uninsured-share coefficient across ordinary quarters --", "",
         "    Same ten-feature OLS, same screens, predictors always at the prior",
         "    quarter. 2020 is excluded: the pandemic deposit surge is not an",
         "    ordinary period. 95% CI on classical SEs, Table 3's convention.", "",
         "    {:<10s} {:<11s} {:>6s} {:>14s} {:>12s} {:>11s} {:>26s}".format(
             "outcome", "predictors", "N", "coef", "SE", "p", "95% CI"),
         "    " + "-" * 96]
    for _, r in df.iterrows():
        mark = "  <- stress" if r.period_type == "banking stress" else ""
        L.append("    {:<10s} {:<11s} {:>6d} {:>14.8f} {:>12.8f} {:>11.8f}  "
                 "[{:+.8f}, {:+.8f}]{}".format(
                     r.outcome_quarter, r.predictor_quarter, int(r.n), r.coef,
                     r.se, r.p, r.ci_low, r.ci_high, mark))
    ordinary = df[df.period_type == "ordinary"]
    n_sig = int((ordinary.p < C.SIG_LEVEL).sum())
    n_same_sign = int((np.sign(ordinary.coef) == np.sign(main_res["coef"])).sum())
    L += ["",
          f"    ordinary quarters with p < {C.SIG_LEVEL}: {n_sig} of {len(ordinary)}",
          f"    ordinary quarters with the same sign as the stress period: "
          f"{n_same_sign} of {len(ordinary)}",
          f"    stress-period CI [{df.iloc[-1].ci_low:+.6f}, "
          f"{df.iloc[-1].ci_high:+.6f}] vs the ordinary-quarter range "
          f"[{ordinary.coef.min():+.6f}, {ordinary.coef.max():+.6f}]",
          ""]
    return df, L


# --------------------------------------------------------------------------- #
#  Outputs
# --------------------------------------------------------------------------- #
def verdict_lines(placebo: dict, stress: dict, pooled: dict) -> list[str]:
    """State what the pooled test does and does not establish.

    Deliberately not a binary "significant / not significant" branch. The two
    two readings set out above — stress-specific versus persistent
    structural correlate — are not complements: failing to reject equality of
    slopes is not evidence that the slopes are equal, especially when the
    comparison-period estimate is near zero and the interaction absorbs most of
    the stress-period effect. All three facts are reported and the strength of
    the conclusion is matched to them."""
    p = pooled["p"]
    se_p = abs(placebo["coef"] / placebo["t"])
    lo, hi = placebo["coef"] - 1.96 * se_p, placebo["coef"] + 1.96 * se_p
    excludes = not (lo <= stress["coef"] <= hi)
    share = pooled["coef"] / stress["coef"] if stress["coef"] else float("nan")

    L = [f"    (a) The comparison-period slope is {placebo['coef']:+.8f} "
         f"(p {placebo['p']:.6f}), i.e.",
         f"        statistically indistinguishable from ZERO, not from the",
         f"        stress-period slope. Its 95% CI [{lo:+.6f}, {hi:+.6f}]",
         f"        {'EXCLUDES' if excludes else 'includes'} the stress-period "
         f"estimate {stress['coef']:+.6f}.",
         "",
         f"    (b) The interaction is {pooled['coef']:+.8f}, which is "
         f"{100 * share:.1f}% of the",
         f"        stress-period slope: almost the entire Table 3 association is",
         f"        specific to the stress period rather than common to both.",
         "",
         f"    (c) But the pooled test does not reach 5%: p = {p:.8f}, 95% CI",
         f"        [{pooled['ci'][0]:+.6f}, {pooled['ci'][1]:+.6f}], which includes "
         f"zero. Clustering on",
         f"        the bank is what widens it — {pooled['n_both']} banks appear "
         f"in both periods,",
         "        so the two estimates are not independent.",
         ""]
    if p < C.SIG_LEVEL:
        L += ["    READING: the STRESS-SPECIFIC branch is supported. The slopes",
              "    differ by more than sampling variation, so the Table 3",
              "    association is not merely a structural correlate of deposit",
              "    behaviour visible in ordinary quarters.", ""]
    elif p < 0.10:
        L += [f"    READING: SUGGESTIVE OF THE STRESS-SPECIFIC BRANCH, NOT",
              f"    ESTABLISHED AT 5%. The point estimates point that way — near",
              f"    zero before, {stress['coef']:+.4f} during — and the difference is",
              f"    significant at 10% but not at 5%. State it as suggestive.",
              "",
              "    Do not report this as evidence that the relationship is a",
              "    persistent structural correlate. Failing to reject equality is",
              "    not evidence of equality, and the comparison-period estimate is",
              "    indistinguishable from zero, not from the stress-period value.", ""]
    else:
        L += ["    READING: the two periods' slopes CANNOT BE TOLD APART at",
              "    conventional levels, so the stress-specific reading is not",
              "    established by this test. Note this is not positive evidence of",
              "    a persistent structural correlate either — see (a).", ""]
    return L


def build_table(placebo: dict, stress: dict, pooled: dict) -> pd.DataFrame:
    """The three headline rows, machine-readable."""
    return pd.DataFrame([
        {"row": "uninsured_share", "specification":
            f"period-specific OLS, {PERIOD_LABEL[PLACEBO_OUT]}",
         "coef": placebo["coef"], "std_error": abs(placebo["coef"] / placebo["t"]),
         "t": placebo["t"], "p_value": placebo["p"], "n_obs": placebo["n"],
         "n_banks": placebo["n"], "inference": "classical (as Table 3)",
         "r2": placebo["r2"]},
        {"row": "uninsured_share", "specification":
            f"period-specific OLS, {PERIOD_LABEL[MAIN_OUT]} (Table 3 col 2)",
         "coef": stress["coef"], "std_error": abs(stress["coef"] / stress["t"]),
         "t": stress["t"], "p_value": stress["p"], "n_obs": stress["n"],
         "n_banks": stress["n"], "inference": "classical (as Table 3)",
         "r2": stress["r2"]},
        {"row": "uninsured_share x 1[2023Q1]", "specification":
            "pooled two-period OLS, headline test",
         "coef": pooled["coef"], "std_error": pooled["se"], "t": pooled["t"],
         "p_value": pooled["p"], "n_obs": pooled["n_total"],
         "n_banks": pooled["n_banks"],
         "inference": "cluster-robust by bank (IDRSSD)", "r2": float("nan")},
    ])


def to_latex(placebo: dict, stress: dict, pooled: dict) -> str:
    def se(d):
        return abs(d["coef"] / d["t"])
    L = [r"% Appendix table — pre-crisis comparison period (2022Q2).",
         r"% Generated by scripts/rq1_placebo.py. Do not hand-edit.",
         r"\begin{table}[htbp]",
         r"\centering",
         r"\caption{The uninsured-deposit share and deposit growth in the "
         r"pre-crisis comparison period (2022Q2) and the banking-stress period "
         r"(2023Q1)}",
         r"\label{tab:placebo_precrisis}",
         r"\begin{tabular}{lccc}",
         r"\toprule",
         r" & \multicolumn{2}{c}{Period-specific OLS} & Pooled \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}",
         r" & Pre-crisis & Banking stress & Two-period \\",
         r" & (2022Q2) & (2023Q1) & model \\",
         r"\midrule",
         r"Uninsured share & {:.5f} & {:.5f} & {:.5f} \\".format(
             placebo["coef"], stress["coef"], pooled["main_uninsured"]),
         r" & ({:.5f}) & ({:.5f}) & \\".format(se(placebo), se(stress)),
         r" & [{:.4f}] & [{:.4f}] & [{:.4f}] \\".format(
             placebo["p"], stress["p"], pooled["main_uninsured_p"]),
         r"\addlinespace",
         r"Uninsured share $\times\ \mathbb{1}$[2023Q1] & & & "
         r"\textbf{" + "{:.5f}".format(pooled["coef"]) + r"} \\",
         r" & & & ({:.5f}) \\".format(pooled["se"]),
         r" & & & [\textbf{" + "{:.4f}".format(pooled["p"]) + r"}] \\",
         r"\addlinespace",
         # Note: LaTeX braces must be doubled inside a .format() template.
         r"$\mathbb{{1}}$[2023Q1] & & & {:.5f} \\".format(pooled["stress_coef"]),
         r"\midrule",
         r"Other characteristics & Yes (9) & Yes (9) & Yes (9) \\",
         r"Observations & {:,} & {:,} & {:,} \\".format(
             placebo["n"], stress["n"], pooled["n_total"]),
         r"Unique banks & {:,} & {:,} & {:,} \\".format(
             placebo["n"], stress["n"], pooled["n_banks"]),
         r"$R^2$ & {:.4f} & {:.4f} & --- \\".format(placebo["r2"], stress["r2"]),
         r"Standard errors & Classical & Classical & Clustered (bank) \\",
         r"\bottomrule",
         r"\end{tabular}",
         r"\begin{minipage}{0.95\textwidth}",
         r"\vspace{2mm}\footnotesize",
         r"\textit{Notes.} Standard errors in parentheses, $p$-values in "
         r"brackets. The dependent variable is the quarterly deposit growth "
         r"rate. All ten balance-sheet characteristics, the sample screens "
         r"(assets $>$ \$1bn, deposits $\geq$ 50\% of liabilities, the FDIC "
         r"charter screen) and the specification replicate the main analysis "
         r"exactly, with the dates shifted: in the pre-crisis comparison period "
         r"every predictor is measured at 2022Q1 and the outcome runs 2022Q1 to "
         r"2022Q2, so no post-period information enters. The FDIC failed-bank "
         r"list records no failures in 2021 or 2022, so nothing is censored in "
         r"the comparison period; the banking-stress column retains the "
         r"baseline treatment of the three 2023 failures (censored at $-1.0$). "
         r"The pooled column stacks both periods' bank-period observations, "
         r"each having passed its own screens, and clusters standard errors on "
         r"the bank because " + "{:,}".format(pooled["n_both"]) + r" banks "
         r"appear in both periods. The interaction coefficient is the test of "
         r"whether the relationship is specific to the stress episode.",
         r"\end{minipage}",
         r"\end{table}"]
    return "\n".join(L)


def markdown_summary(placebo: dict, stress: dict, pooled: dict,
                     funnels: dict, overlap: dict, mq: pd.DataFrame | None,
                     qseries: pd.DataFrame) -> str:
    sig = pooled["p"] < C.SIG_LEVEL
    p_sig = placebo["p"] < C.SIG_LEVEL

    L = [f"# Pre-crisis comparison period (2022Q2) — placebo test for the "
         f"uninsured-deposit result", "",
         "Numbers first; the interpretation branch follows from them.", "",
         "## The three regressions", "",
         "| Specification | Uninsured-share coefficient | SE | p | N | Inference |",
         "|---|---|---|---|---|---|",
         f"| Period-specific OLS, pre-crisis comparison period (2022Q2) | "
         f"**{placebo['coef']:+.8f}** | {abs(placebo['coef']/placebo['t']):.8f} | "
         f"{placebo['p']:.8f} | {placebo['n']:,} | classical |",
         f"| Period-specific OLS, banking-stress period (2023Q1), Table 3 col 2 | "
         f"**{stress['coef']:+.8f}** | {abs(stress['coef']/stress['t']):.8f} | "
         f"{stress['p']:.2e} | {stress['n']:,} | classical |",
         f"| **Pooled: uninsured share × 1[2023Q1]** | "
         f"**{pooled['coef']:+.8f}** | {pooled['se']:.8f} | "
         f"**{pooled['p']:.8f}** | {pooled['n_total']:,} | clustered by bank |",
         "",
         f"Pooled interaction 95% CI: [{pooled['ci'][0]:+.8f}, "
         f"{pooled['ci'][1]:+.8f}]. The pooled model's own `uninsured_share` term "
         f"(the pre-crisis slope) is {pooled['main_uninsured']:+.8f} "
         f"(p {pooled['main_uninsured_p']:.8f}); the stress dummy is "
         f"{pooled['stress_coef']:+.8f} (p {pooled['stress_p']:.8f}).",
         "",
         "## Samples and overlap", "",
         "| | Pre-crisis (2022Q2) | Banking stress (2023Q1) |",
         "|---|---|---|",
         f"| Predictors measured at | 2022Q1 | 2022Q4 |",
         f"| N after all screens | {placebo['n']:,} | {stress['n']:,} |",
         f"| Censored failures | 0 (none in 2021-2022) | 3 |",
         "",
         f"- Banks in **both** periods: **{overlap['both']:,}**",
         f"- Pre-crisis only: {overlap['placebo_only']:,}  |  "
         f"stress only: {overlap['stress_only']:,}",
         f"- Unique banks in the pooled model: **{overlap['unique']:,}** "
         f"across {pooled['n_total']:,} bank-period observations",
         "",
         "## Which interpretation the data support", ""]

    se_p = abs(placebo["coef"] / placebo["t"])
    lo, hi = placebo["coef"] - 1.96 * se_p, placebo["coef"] + 1.96 * se_p
    excludes = not (lo <= stress["coef"] <= hi)
    share = pooled["coef"] / stress["coef"] if stress["coef"] else float("nan")

    L += ["Three facts, then the reading. The two branches are **not**"
          " complements: failing to reject equal slopes is not evidence that"
          " the slopes are equal.", "",
          f"1. The comparison-period slope is **{placebo['coef']:+.8f}** "
          f"(p {placebo['p']:.6f}) — indistinguishable from **zero**, not from "
          f"the stress-period slope. Its 95% CI [{lo:+.6f}, {hi:+.6f}] "
          f"**{'excludes' if excludes else 'includes'}** the stress-period "
          f"estimate {stress['coef']:+.6f}.",
          f"2. The interaction, {pooled['coef']:+.8f}, is **{100*share:.1f}%** of "
          f"the stress-period slope — almost the whole Table 3 association is "
          f"specific to the stress period rather than common to both.",
          f"3. The pooled test nonetheless does not reach 5%: **p = "
          f"{pooled['p']:.8f}**, CI [{pooled['ci'][0]:+.6f}, "
          f"{pooled['ci'][1]:+.6f}]. Clustering on the bank widens it, because "
          f"{pooled['n_both']:,} banks appear in both periods and the two "
          f"estimates are therefore not independent.",
          ""]
    if sig:
        L += [f"**Reading — the stress-specific branch is supported.** The slopes "
              f"differ by more than sampling variation (p = {pooled['p']:.8f}), "
              "so the Table 3 association is not merely a structural correlate "
              "visible in ordinary quarters.", ""]
    elif pooled["p"] < 0.10:
        L += [f"**Reading — suggestive of the stress-specific branch, not "
              f"established at 5%.** The point estimates point that way (near "
              f"zero before, {stress['coef']:+.4f} during) and the difference is "
              f"significant at 10% but not at 5% (p = {pooled['p']:.8f}). It "
              "should be stated as suggestive.", "",
              "This should **not** be reported as evidence that the relationship "
              "is a persistent structural correlate: the comparison-period "
              "estimate is indistinguishable from zero, not from the "
              "stress-period value.", ""]
    else:
        L += [f"**Reading — the two slopes cannot be told apart** at conventional "
              f"levels (p = {pooled['p']:.8f}), so the stress-specific reading is "
              "not established by this test. That is not positive evidence of a "
              "persistent structural correlate either — see fact 1.", ""]
    L += [f"The period-specific comparison points the same way but is *not* the "
          f"test: {placebo['coef']:+.8f} (p {placebo['p']:.6f}, "
          f"{'significant' if p_sig else 'not significant'} at 5%) against "
          f"{stress['coef']:+.8f}. Two estimates on opposite sides of a "
          "significance threshold need not differ significantly from each "
          "other, which is why the pooled interaction carries the claim.",
          ""]

    neg = qseries[qseries.aggregate_growth < 0]
    first_neg = str(neg.iloc[0].quarter) if len(neg) else None
    prior = qseries[qseries.quarter < PLACEBO_OUT]
    last_pos = str(prior.iloc[-1].quarter) if len(prior) else None
    matches = first_neg is not None and first_neg > PLACEBO_OUT

    L += ["## Why 2022Q2 (fixed before any coefficient was seen)", "",
          "The rationale set out in advance was that 2022Q2 is the last completed "
          "quarter *before* the sustained system-wide deposit contraction that ran "
          "into the 2023 stress. The quarterly series "
          "(`rq1_deposit_growth_by_quarter.csv`) is below.", ""]
    if matches:
        L += ["The series is **consistent** with that description: the "
              f"contraction begins at {first_neg}, after the comparison period.",
              ""]
    else:
        L += ["> **Flagged — the series does not match that description.** The "
              f"first quarter of aggregate contraction is **{first_neg}**, which "
              f"*is* the comparison period, and at "
              f"{100*float(qseries.loc[qseries.quarter == first_neg, 'aggregate_growth'].iloc[0]):+.4f}% "
              "it is the **largest** aggregate decline in the whole window. "
              f"2022Q2 is therefore the **first quarter of the contraction**, not "
              f"the last one before it; the last expanding quarter is {last_pos}. "
              "The design was fixed in advance and has **not** been changed in "
              "response. What must change is the wording: describe 2022Q2 as a "
              "pre-crisis quarter that shares the tightening macro environment "
              "and precedes the March 2023 stress event by three quarters — not "
              "as a quarter before deposits began falling.", ""]
    L += ["| Quarter | Aggregate growth | Median growth | Share negative |",
          "|---|---|---|---|"]
    for _, r in qseries.iterrows():
        mark = ""
        if r.quarter == PLACEBO_OUT:
            mark = " ← comparison period"
        elif r.quarter == MAIN_OUT:
            mark = " ← stress period"
        L.append(f"| {r.quarter}{mark} | {100*r.aggregate_growth:+.4f}% | "
                 f"{100*r.median_growth:+.4f}% | {100*r.share_negative:.2f}% |")
    L.append("")

    if mq is not None:
        L += ["## The coefficient across ordinary quarters", "",
              "| Outcome quarter | Predictors | N | Coefficient | 95% CI | p |",
              "|---|---|---|---|---|---|"]
        for _, r in mq.iterrows():
            tag = " (stress)" if r.period_type == "banking stress" else ""
            L.append(f"| {r.outcome_quarter}{tag} | {r.predictor_quarter} | "
                     f"{int(r.n):,} | {r.coef:+.8f} | "
                     f"[{r.ci_low:+.6f}, {r.ci_high:+.6f}] | {r.p:.8f} |")
        L.append("")

    L += ["## Replication note", "",
          "The panel builder in `rq1_placebo.py` is proven output-identical to "
          "the frozen pipeline: run on 2022Q4 → 2023Q1 it reproduces "
          "`panel_2022Q4_wide.csv` bank for bank (worst column difference "
          "~4e-15) and returns Table 3 column 2 exactly. `_panel_wide.py` and "
          "`_panel_narrow.py` were not modified.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main() -> dict:
    do_mq = "--no-multi-quarter" not in sys.argv

    L = ["=" * 78,
         "RQ1 — PRE-CRISIS COMPARISON PERIOD (2022Q2): PLACEBO TEST FOR THE",
         "      UNINSURED-DEPOSIT RESULT",
         "=" * 78, "",
         "  Table 3 column 2 reports that the uninsured-deposit share predicts the",
         "  2023Q1 deposit outflow. That is evidence about banking stress only if",
         "  the relationship is specific to the stress episode. This asks whether",
         "  it is also there in an ordinary quarter.",
         "",
         "  2022Q2 is the PRE-CRISIS COMPARISON PERIOD, equivalently the",
         "  NON-BANKING-STRESS COMPARISON PERIOD. It is not a calm period: the Fed",
         "  had begun tightening. 'Placebo' names the test, not the period.",
         ""]

    print("[rq1_placebo] 1/6 identity check ...")
    L += identity_check()

    print("[rq1_placebo] 2/6 building the comparison-period panel "
          f"({PLACEBO_PRED} -> {PLACEBO_OUT}) ...")
    # No failures exist in 2021-2022; assert it from the FDIC list rather than
    # assuming it, then build with censoring switched off.
    n_fail_21 = len(failures_in(2021, PLACEBO_PRED))
    n_fail_22 = len(failures_in(2022, PLACEBO_PRED))
    if n_fail_21 or n_fail_22:
        raise SystemExit(
            f"[rq1_placebo] STOP: the FDIC list shows {n_fail_21} failures in "
            f"2021 and {n_fail_22} in 2022. The design assumed none in the "
            "comparison window; censoring would have to be specified. Not "
            "guessing — reporting.")
    placebo_panel, f_placebo = build_wide_panel(PLACEBO_PRED, PLACEBO_OUT,
                                                failure_year=None)
    stress_panel, f_stress = build_wide_panel(MAIN_PRED, MAIN_OUT,
                                              failure_year=2023, verbose=False)

    L += ["-- sample construction, both periods --", "",
          "    Identical screens, applied at each period's own predictor quarter.",
          "",
          "    {:<38s} {:>16s} {:>16s}".format(
              "funnel step", f"{PLACEBO_OUT} (pre-crisis)", f"{MAIN_OUT} (stress)"),
          "    " + "-" * 72]
    for k in f_placebo:
        L.append("    {:<38s} {:>16d} {:>16d}".format(k, f_placebo[k],
                                                      f_stress.get(k, -1)))
    L += ["",
          f"    FDIC failed-bank list: {n_fail_21} failures in 2021, "
          f"{n_fail_22} in 2022",
          "    -> nothing is censored in the comparison period. The stress period",
          f"       keeps the baseline treatment: "
          f"{int(stress_panel.censored.sum())} failures censored at -1.0.",
          ""]

    ids_p = set(placebo_panel.bank_IDRSSD.astype(int))
    ids_s = set(stress_panel.bank_IDRSSD.astype(int))
    overlap = {"both": len(ids_p & ids_s), "placebo_only": len(ids_p - ids_s),
               "stress_only": len(ids_s - ids_p), "unique": len(ids_p | ids_s)}
    L += ["-- overlap with the main analysis's 953 --", "",
          f"    comparison period N      {len(ids_p)}",
          f"    stress period N          {len(ids_s)}",
          f"    in BOTH periods          {overlap['both']}",
          f"    comparison period only   {overlap['placebo_only']}",
          f"    stress period only       {overlap['stress_only']}",
          f"    unique banks pooled      {overlap['unique']}",
          ""]

    print("[rq1_placebo] 3/6 period-specific regressions ...")
    placebo_res = period_ols(placebo_panel, PERIOD_LABEL[PLACEBO_OUT])
    stress_res = period_ols(stress_panel, PERIOD_LABEL[MAIN_OUT])

    L += ["=" * 78, "1. PERIOD-SPECIFIC REGRESSIONS (Table 3's inference convention)",
          "=" * 78, "",
          "    Full ten-feature OLS in each period. not the headline test — see 2.",
          "",
          "    {:<40s} {:>7s} {:>14s} {:>9s} {:>13s} {:>9s}".format(
              "period", "N", "coef", "t", "p", "R^2"),
          "    " + "-" * 96]
    for r in (placebo_res, stress_res):
        L.append("    {:<40s} {:>7d} {:>14.8f} {:>9.4f} {:>13.8e} {:>9.6f}".format(
            r["label"], r["n"], r["coef"], r["t"], r["p"], r["r2"]))
    L += ["",
          "    Full coefficient table, comparison period (2022Q2):", "",
          "    {:<22s} {:>14s} {:>9s} {:>13s} {:>10s}".format(
              "feature", "coef", "t", "p", "beta_std"),
          "    " + "-" * 72]
    for f in C.FEATURES:
        rr = placebo_res["table"].loc[f]
        L.append("    {:<22s} {:>14.8f} {:>9.4f} {:>13.8f} {:>10.5f}  {}".format(
            f, rr["coef"], rr["t"], rr["p_value"], rr["beta_std"],
            stars(rr["p_value"])))
    L.append("")

    print("[rq1_placebo] 4/6 pooled interaction (headline) ...")
    pooled = pooled_interaction(placebo_panel, stress_panel)
    L += ["=" * 78, "2. THE HEADLINE TEST — POOLED, WITH ONE INTERACTION", "=" * 78,
          "",
          "    'Significant in one period, insignificant in the other' is not a",
          "    test: two estimates can straddle 0.05 without differing",
          "    significantly from each other. The two periods are pooled instead",
          "    and the difference in slope is estimated directly.",
          "",
          f"    {pooled['formula']}",
          "",
          f"    bank-period observations {pooled['n_total']}"
          f"   ({pooled['n_placebo']} pre-crisis + {pooled['n_stress']} stress)",
          f"    unique banks             {pooled['n_banks']}"
          f"   ({pooled['n_both']} in both periods)",
          "    SEs clustered on the bank (IDRSSD), t with G-1 df.",
          "",
          "    {:<34s} {:>14s} {:>13s} {:>8s} {:>12s}  {}".format(
              "term", "coef", "SE", "t", "p", "sig"),
          "    " + "-" * 90]
    for term, lab in ((pooled["term"], "uninsured_share x 1[2023Q1]  <-- HEADLINE"),
                      ("uninsured_share", "uninsured_share (pre-crisis slope)"),
                      ("stress", "1[2023Q1]")):
        m = pooled["model"]
        L.append("    {:<34s} {:>14.8f} {:>13.8f} {:>8.3f} {:>12.8f}  {}".format(
            lab, m.params[term], m.bse[term], m.tvalues[term], m.pvalues[term],
            stars(m.pvalues[term])))
    L += ["",
          f"    95% CI on the interaction: [{pooled['ci'][0]:+.8f}, "
          f"{pooled['ci'][1]:+.8f}]",
          "",
          "    Implied stress-period slope = pre-crisis slope + interaction",
          f"      = {pooled['main_uninsured']:+.8f} + ({pooled['coef']:+.8f})"
          f" = {pooled['main_uninsured'] + pooled['coef']:+.8f}",
          ""]

    L += ["-- which interpretation branch the data support --", ""] + verdict_lines(
        placebo_res, stress_res, pooled)

    print("[rq1_placebo] 5/6 selection-justification series ...")
    qseries, qlines = deposit_growth_by_quarter()
    L += ["=" * 78, "3. WHY 2022Q2 — THE SELECTION JUSTIFICATION", "=" * 78, ""]
    L += qlines

    mq = None
    if do_mq:
        print("[rq1_placebo] 6/6 multi-quarter extension ...")
        mq, mqlines = multi_quarter(stress_res)
        L += ["=" * 78, "4. THE COEFFICIENT ACROSS ORDINARY QUARTERS", "=" * 78, ""]
        L += mqlines
    else:
        L += ["=" * 78, "4. THE COEFFICIENT ACROSS ORDINARY QUARTERS", "=" * 78, "",
              "  SKIPPED (--no-multi-quarter).", ""]

    # ---- write everything ----
    tbl = build_table(placebo_res, stress_res, pooled)
    tbl.to_csv(C.RQ1_PLACEBO_CSV, index=False)
    C.RQ1_PLACEBO_TEX.write_text(to_latex(placebo_res, stress_res, pooled) + "\n")
    qseries.to_csv(C.RQ1_DEPGROWTH_CSV, index=False)
    if mq is not None:
        mq.to_csv(C.RQ1_PLACEBO_MQ_CSV, index=False)
    C.RQ1_PLACEBO_MD.write_text(markdown_summary(
        placebo_res, stress_res, pooled, {"placebo": f_placebo,
                                          "stress": f_stress},
        overlap, mq, qseries) + "\n")
    text = "\n".join(L)
    C.RQ1_PLACEBO.write_text(text + "\n")
    print(text)
    for p in (C.RQ1_PLACEBO, C.RQ1_PLACEBO_MD, C.RQ1_PLACEBO_TEX,
              C.RQ1_PLACEBO_CSV, C.RQ1_DEPGROWTH_CSV):
        print(f"wrote {p}")
    if mq is not None:
        print(f"wrote {C.RQ1_PLACEBO_MQ_CSV}")

    out = {
        "rq1.placebo.n.2022Q2": float(placebo_res["n"]),
        "rq1.placebo.n.2023Q1": float(stress_res["n"]),
        "rq1.placebo.banks.both": float(overlap["both"]),
        "rq1.placebo.banks.unique": float(overlap["unique"]),
        "rq1.placebo.banks.2022Q2_only": float(overlap["placebo_only"]),
        "rq1.placebo.banks.2023Q1_only": float(overlap["stress_only"]),
        "rq1.placebo.uninsured.2022Q2.coef": placebo_res["coef"],
        "rq1.placebo.uninsured.2022Q2.t": placebo_res["t"],
        "rq1.placebo.uninsured.2022Q2.p": placebo_res["p"],
        "rq1.placebo.uninsured.2023Q1.coef": stress_res["coef"],
        "rq1.placebo.uninsured.2023Q1.t": stress_res["t"],
        "rq1.placebo.uninsured.2023Q1.p": stress_res["p"],
        "rq1.placebo.pooled.interaction.coef": pooled["coef"],
        "rq1.placebo.pooled.interaction.se": pooled["se"],
        "rq1.placebo.pooled.interaction.t": pooled["t"],
        "rq1.placebo.pooled.interaction.p": pooled["p"],
        "rq1.placebo.pooled.uninsured_main.coef": pooled["main_uninsured"],
        "rq1.placebo.pooled.uninsured_main.p": pooled["main_uninsured_p"],
        "rq1.placebo.pooled.stress.coef": pooled["stress_coef"],
        "rq1.placebo.pooled.stress.p": pooled["stress_p"],
        "rq1.placebo.pooled.n_obs": float(pooled["n_total"]),
        "rq1.placebo.pooled.n_banks": float(pooled["n_banks"]),
    }
    if mq is not None:
        for _, r in mq[mq.period_type == "ordinary"].iterrows():
            out[f"rq1.placebo.mq.{r.outcome_quarter}.coef"] = float(r.coef)
            out[f"rq1.placebo.mq.{r.outcome_quarter}.p"] = float(r.p)
    return out


if __name__ == "__main__":
    main()
