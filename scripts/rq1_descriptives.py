"""
rq1_descriptives.py — Tables 1 and 2 of the dissertation: summary statistics and
the correlation matrix for the listed sample.

Reads  : data/processed/panel_2022Q4_narrow.csv
Writes : data/processed/rq1_descriptives.txt
Returns: the headline values, keyed for config.FROZEN

WHY THIS EXISTS. Both tables were previously computed outside the pipeline. The
audit could not reproduce them from the 2022Q4 panel their caption names: every
minimum, maximum and standard deviation instead matched the 2021Q4
characteristics carried in the CAR panel, i.e. the quarter read for the January
and February 2022 events, not the quarter the study measures its predictors at.
This script computes them where the caption says they come from.

SAMPLE. Identical to RQ1's, via the same two filters as _rq1_core.load_panel():
the merger-exit bank (config.EXCLUDE_IDRSSD) is dropped and rows with a null
outcome are dropped. On this panel both remove the same single bank, so N = 277
out of the 278 matched to CRSP. No other screen is applied, and the three
censored failures stay in, as everywhere else in RQ1.

VARIABLES. The six characteristics that appear in Tables 1 and 2. They are a
subset of config.FEATURES: the RQ2/RQ3 control set plus uninsured_share and
deposit_reliance. unrealised_losses, liquidity, capital and int_inc_ratio are
RQ1 predictors but are not in these two tables.

    python3 scripts/rq1_descriptives.py
"""

from __future__ import annotations

import pandas as pd

import config as C

VARS = ["uninsured_share", "size", "ROA", "NPL_ratio", "equity_ratio",
        "deposit_reliance"]

NICE = {
    "uninsured_share": "Uninsured-deposit share",
    "size": "Size (log assets)",
    "ROA": "Return on assets",
    "NPL_ratio": "NPL ratio",
    "equity_ratio": "Equity ratio",
    "deposit_reliance": "Deposit reliance",
}

# |r| at or above this is flagged in the report: the text claims the
# correlations are low enough that the linear model is untroubled by
# collinearity, and that claim should fail loudly rather than quietly.
COLLINEARITY_FLAG = 0.7


def load_sample() -> pd.DataFrame:
    """The RQ1 listed sample: the narrow panel minus the merger exit and minus
    any null outcome. Deliberately the same two filters as _rq1_core.load_panel(),
    so Table 1 describes exactly the banks Table 3 is estimated on."""
    df = pd.read_csv(C.PANEL_NARROW.with_name("panel_2022Q4_narrow.csv"))
    n0 = len(df)
    df = df[~df["bank_IDRSSD"].isin(C.EXCLUDE_IDRSSD)]
    n1 = len(df)
    df = df[df[C.OUTCOME].notna()].reset_index(drop=True)
    n2 = len(df)
    missing = df[VARS].isna().any(axis=1).sum()
    if missing:
        raise SystemExit(f"{missing} row(s) have a null among {VARS}")
    return df, {"matched": n0, "after_exclude": n1, "final": n2}


def summary(df: pd.DataFrame) -> pd.DataFrame:
    s = df[VARS].agg(["mean", "std", "min", "max"]).T
    s.columns = ["mean", "sd", "min", "max"]
    return s


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    return df[VARS].corr(method="pearson")


def build_report(df: pd.DataFrame, counts: dict, s: pd.DataFrame,
                 corr: pd.DataFrame) -> str:
    L = ["=" * 88,
         "TABLES 1 AND 2 — DESCRIPTIVE STATISTICS, LISTED SAMPLE",
         "=" * 88, "",
         f"  matched to CRSP                       : {counts['matched']}",
         f"  less the merger exit "
         f"({', '.join(str(i) for i in C.EXCLUDE_IDRSSD)})       : {counts['after_exclude']}",
         f"  less null outcome                     : {counts['final']}",
         f"  N used in both tables                 : {counts['final']}",
         "",
         "  Characteristics measured at 2022Q4, the quarter before the run, on the",
         "  same sample RQ1 estimates Table 3 on. The three 2023 failures are",
         "  retained (censored at dep_growth = -1.0), as everywhere else in RQ1.",
         "",
         "-- Table 1. Summary statistics --", "",
         "    {:<26s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
             "Variable", "Mean", "SD", "Min", "Max"),
         "    " + "-" * 66]
    for v in VARS:
        r = s.loc[v]
        L.append("    {:<26s} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            NICE[v], r["mean"], r["sd"], r["min"], r["max"]))
    L += ["",
          "    (unrounded, for reconciliation)",
          "    {:<26s} {:>13s} {:>13s} {:>13s} {:>13s}".format(
              "Variable", "mean", "sd", "min", "max"),
          "    " + "-" * 82]
    for v in VARS:
        r = s.loc[v]
        L.append("    {:<26s} {:>13.8f} {:>13.8f} {:>13.8f} {:>13.8f}".format(
            v, r["mean"], r["sd"], r["min"], r["max"]))

    L += ["", "-- Table 2. Pearson correlations --", "",
          "    " + " " * 26 + "".join(f"{f'({i+1})':>9s}" for i in range(len(VARS))),
          "    " + "-" * (26 + 9 * len(VARS))]
    for i, v in enumerate(VARS):
        cells = "".join(f"{corr.loc[v, w]:>9.2f}" if j <= i else " " * 9
                        for j, w in enumerate(VARS))
        L.append(f"    ({i+1}) {NICE[v]:<22s}{cells}")

    off = [(a, b, corr.loc[a, b]) for i, a in enumerate(VARS)
           for b in VARS[i + 1:]]
    off.sort(key=lambda t: -abs(t[2]))
    L += ["", "    strongest pairs by |r|:"]
    for a, b, r in off[:3]:
        L.append(f"      {a} x {b}: {r:+.8f}")
    flagged = [(a, b, r) for a, b, r in off if abs(r) >= COLLINEARITY_FLAG]
    L += ["",
          f"    pairs with |r| >= {COLLINEARITY_FLAG}: "
          + (", ".join(f"{a} x {b} ({r:+.3f})" for a, b, r in flagged)
             if flagged else "none")]
    L += ["", "=" * 88, ""]
    return "\n".join(L)


def main() -> dict:
    df, counts = load_sample()
    s = summary(df)
    corr = correlations(df)
    report = build_report(df, counts, s, corr)
    C.RQ1_DESCRIPTIVES.write_text(report)
    print(report)
    print(f"wrote {C.RQ1_DESCRIPTIVES}")

    out = {"rq1.desc.N": float(counts["final"])}
    for v in VARS:
        for stat in ("mean", "sd", "min", "max"):
            out[f"rq1.desc.{v}.{stat}"] = float(s.loc[v, stat])
    for i, a in enumerate(VARS):
        for b in VARS[i + 1:]:
            out[f"rq1.corr.{a}.{b}"] = float(corr.loc[a, b])
    return out


if __name__ == "__main__":
    main()
