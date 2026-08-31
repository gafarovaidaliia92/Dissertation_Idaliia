"""rq1_tuned_table.py: Table 5, the nested-tuning metrics behind the H1b verdict.

The write-up states that the tuned random forest loses to OLS in all four
configurations, but reports only the four RMSE gaps (-1.146%, -0.177%, -5.367%,
-2.704%). The tuned RMSE and R-squared those gaps are computed from appear
nowhere. This script tabulates them.

The numbers are already computed and already persisted:
_rq1_wide_core.nested_cv_tuned() produces them and run3_report() writes them into
section 3.4 of rq1_h1b_robustness.txt, once per population. This script parses
that file rather than re-estimating, because the results are final and re-running
a twenty-minute estimator risks losing them.

Writes    data/processed/rq1_tuned_metrics.txt, .csv, .tex
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C

VERIFY_TOL = 1e-8

# population tag in the report  ->  (label for the table, key used in frozen)
POPULATIONS = {"WIDE": ("wide", "wide"), "NARROW": ("listed", "listed")}
MODELS = ("OLS", "RandomForest", "GradientBoosting")

_MODEL_ROW = re.compile(
    r"^\s+(OLS|RandomForest|GradientBoosting)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
_FEATSET = re.compile(r"^\s+\[(full|pruned)(?: \((\d+)\))? feature set\]\s*$")
# searched against a whole population block, so it needs MULTILINE for "^"
_N = re.compile(r"^\s+N = (\d+)\s+full feature set = (\d+)\s+"
                r"significant-only set = (\d+)", re.M)


def _split_populations(txt: str) -> dict[str, str]:
    """The report concatenates the two populations; split on the RUN 3 banner."""
    parts = re.split(r"RUN 3 — .*?\[(WIDE|NARROW), N=\d+\]", txt)
    # re.split with one capture group -> [before, tag, body, tag, body, ...]
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[parts[i]] = parts[i + 1]
    missing = set(POPULATIONS) - set(out)
    if missing:
        raise SystemExit(
            f"[rq1_tuned_table] {C.RQ1_H1B_ROBUST.name} has no RUN 3 block for "
            f"{sorted(missing)} — cannot build Table 5.")
    return out


def _parse_population(body: str, tag: str) -> list[dict]:
    """Pull section 3.4's two feature-set blocks out of one population's body."""
    m = _N.search(body)
    if not m:
        raise SystemExit(f"[rq1_tuned_table] no 'N = ...' header in the {tag} block.")
    n_obs, n_full, n_sig = (int(g) for g in m.groups())

    # section 3.4 only: everything after the 3.4 banner. 3.2 and 3.3 carry rows
    # in the same shape (single-split and repeated CV), and picking those up
    # instead of the tuned ones is exactly the error this table exists to avoid.
    i = body.find("-- 3.4 TUNED hyper-parameters")
    if i < 0:
        raise SystemExit(f"[rq1_tuned_table] no section 3.4 in the {tag} block.")
    section = body[i:]

    rows, feat, n_feat = [], None, None
    for line in section.splitlines():
        fm = _FEATSET.match(line)
        if fm:
            feat = "full" if fm.group(1) == "full" else "significant"
            n_feat = n_full if feat == "full" else (
                int(fm.group(2)) if fm.group(2) else n_sig)
            continue
        rm = _MODEL_ROW.match(line)
        if rm and feat is not None:
            rows.append({
                "population": POPULATIONS[tag][0],
                "n_banks": n_obs,
                "feature_set": feat,
                "n_features": n_feat,
                "model": rm.group(1),
                "oos_rmse": float(rm.group(2)),
                "oos_r2": float(rm.group(3)),
            })
    got = {(r["feature_set"], r["model"]) for r in rows}
    want = {(f, m) for f in ("full", "significant") for m in MODELS}
    if got != want:
        raise SystemExit(
            f"[rq1_tuned_table] {tag} section 3.4 parsed {len(rows)} rows; "
            f"missing {sorted(want - got)}. Refusing to emit a partial table.")
    return rows


def parse() -> pd.DataFrame:
    if not C.RQ1_H1B_ROBUST.exists():
        raise SystemExit(
            f"[rq1_tuned_table] {C.RQ1_H1B_ROBUST} not found. Run rq1_model.py "
            "(~20 min) or restore data/processed/ from results/.")
    bodies = _split_populations(C.RQ1_H1B_ROBUST.read_text())
    rows = []
    for tag in POPULATIONS:
        rows += _parse_population(bodies[tag], tag)
    df = pd.DataFrame(rows)
    order = {"wide": 0, "listed": 1}
    forder = {"full": 0, "significant": 1}
    morder = {m: i for i, m in enumerate(MODELS)}
    return (df.assign(_p=df.population.map(order), _f=df.feature_set.map(forder),
                      _m=df.model.map(morder))
              .sort_values(["_p", "_f", "_m"])
              .drop(columns=["_p", "_f", "_m"])
              .reset_index(drop=True))


# --------------------------------------------------------------------------- #
#  optional: re-run the tuning and check the parse against it
# --------------------------------------------------------------------------- #
def verify(df: pd.DataFrame) -> list[str]:
    """Re-run nested_cv_tuned() on both populations and both feature sets, and
    assert the recomputed metrics equal the parsed ones. Expensive."""
    import _rq1_wide_core as core

    L = ["-- verification: nested_cv_tuned() re-run and compared --", ""]
    for tag, path in (("WIDE", C.PANEL_WIDE), ("NARROW", C.PANEL_NARROW)):
        meta, X, y = core.load_variant(path)
        ols_table = core.run_variant(POPULATIONS[tag][0], meta, X, y)["ols"]
        sig = core.significant_features(ols_table)
        for feat, Xs in (("full", X), ("significant", X[sig])):
            nested = core.nested_cv_tuned(Xs, y)
            for model in MODELS:
                row = df[(df.population == POPULATIONS[tag][0])
                         & (df.feature_set == feat) & (df.model == model)].iloc[0]
                for col, got in (("oos_rmse", nested[model]["oos_RMSE"]),
                                 ("oos_r2", nested[model]["oos_R2"])):
                    d = abs(got - row[col])
                    status = "match" if d <= VERIFY_TOL else "*** MISMATCH ***"
                    L.append(f"    {POPULATIONS[tag][0]:<7s} {feat:<12s} "
                             f"{model:<18s} {col:<9s} parsed {row[col]:.8f}  "
                             f"recomputed {got:.8f}  {status}")
                    if d > VERIFY_TOL:
                        raise SystemExit(
                            f"[rq1_tuned_table] {tag}/{feat}/{model}/{col} moved: "
                            f"{row[col]:.8f} -> {got:.8f}. Stop.")
    L.append("")
    return L


# --------------------------------------------------------------------------- #
#  LaTeX
# --------------------------------------------------------------------------- #
def to_latex(df: pd.DataFrame) -> str:
    """A booktabs fragment: one row per configuration, OLS beside the tuned RF.

    The claim the table supports is about RF versus OLS, so those are the two
    columns. GradientBoosting is in the CSV as well, and it loses to OLS in all
    four configurations too."""
    nice_pop = {"wide": "Wide", "listed": "Listed"}
    nice_feat = {"full": "All", "significant": "OLS-significant"}

    L = [r"% Table 5 — nested-tuning out-of-sample metrics.",
         r"% Generated by scripts/rq1_tuned_table.py from",
         r"% data/processed/rq1_h1b_robustness.txt section 3.4. Do not hand-edit.",
         r"\begin{tabular}{llrrrrr}",
         r"\toprule",
         r" & & \multicolumn{2}{c}{OLS} & \multicolumn{2}{c}{Tuned RF} & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Population & Features & RMSE & $R^2$ & RMSE & $R^2$ "
         r"& $\Delta$RMSE (\%) \\",
         r"\midrule"]
    # groupby(sort=False) so the fragment keeps df's order — wide before listed,
    # full before significant — and matches the console table row for row.
    for (pop, feat), g in df.groupby(["population", "feature_set"], sort=False):
        n_banks = int(g.n_banks.iloc[0])
        n_feat = int(g.n_features.iloc[0])
        ols = g[g.model == "OLS"].iloc[0]
        rf = g[g.model == "RandomForest"].iloc[0]
        r_ols, q_ols = float(ols.oos_rmse), float(ols.oos_r2)
        r_rf, q_rf = float(rf.oos_rmse), float(rf.oos_r2)
        gap = 100 * (r_ols - r_rf) / r_ols
        L.append(f"{nice_pop[pop]} ($N={n_banks}$) & {nice_feat[feat]} "
                 f"({n_feat}) & {r_ols:.5f} & {q_ols:+.5f} & {r_rf:.5f} & "
                 f"{q_rf:+.5f} & {gap:+.3f} \\\\")
    L += [r"\bottomrule",
          r"\end{tabular}"]
    return "\n".join(L)


def caption() -> list[str]:
    return [
        "  CAPTION (the CV design, so the table cannot be misread):",
        "",
        "    Out-of-sample RMSE and R^2 under nested cross-validation, in which the",
        f"    hyper-parameters are chosen by a grid search on the inner folds of each",
        f"    outer training fold, so no configuration is ever selected on the data",
        f"    that scores it. Outer CV: KFold({C.N_SPLITS}, shuffle=True, "
        f"random_state={C.SEED}).",
        f"    Inner CV: KFold({C.INNER_SPLITS}, shuffle=True, "
        f"random_state={C.SEED}), scoring =",
        "    neg_root_mean_squared_error. Grid ties within 1e-9 are broken",
        "    lexicographically (deterministic_winner) so the table is reproducible.",
        "",
        "    This is a SINGLE outer 5-fold split. It is not the 30-shuffle repeated",
        f"    cross-validation reported separately in section 3.3 "
        f"(config.N_REPEATS = {C.N_REPEATS});",
        "    do not describe this table as averaging over 30 shuffles.",
        "",
        "    OLS has nothing to tune and is scored on the SAME outer folds, so the",
        "    comparison is like for like.",
        "",
    ]


def main() -> dict:
    force = "--force" in sys.argv
    df = parse()

    L = ["=" * 78,
         "TABLE 5 — OUT-OF-SAMPLE METRICS UNDER NESTED TUNING", "=" * 78, "",
         "  The text reports that the tuned random forest loses to OLS in all four",
         "  configurations but never gives the tuned RMSE and R^2 the claim rests",
         "  on. Here they are.",
         "",
         f"  source: {C.RQ1_H1B_ROBUST.name}, section 3.4 (parsed, not re-estimated)",
         ""] + caption()

    L += ["  {:<9s} {:>7s} {:<16s} {:>6s} {:<18s} {:>14s} {:>14s}".format(
              "population", "N", "features", "k", "model", "OOS RMSE", "OOS R^2"),
          "  " + "-" * 92]
    for _, r in df.iterrows():
        L.append("  {:<9s} {:>7d} {:<16s} {:>6d} {:<18s} {:>14.8f} {:>14.8f}".format(
            r.population, r.n_banks, r.feature_set, r.n_features, r.model,
            r.oos_rmse, r.oos_r2))
    L.append("")

    # the four gaps, which is the sentence the text actually makes
    L += ["-- the claim: does the tuned RF beat OLS? --", "",
          "  {:<9s} {:<16s} {:>14s} {:>14s} {:>14s} {:>12s}  {}".format(
              "population", "features", "OLS RMSE", "RF RMSE", "OLS - RF",
              "% of OLS", "verdict"),
          "  " + "-" * 96]
    out: dict = {}
    for (pop, feat), g in df.groupby(["population", "feature_set"], sort=False):
        r_ols = float(g[g.model == "OLS"].oos_rmse.iloc[0])
        r_rf = float(g[g.model == "RandomForest"].oos_rmse.iloc[0])
        q_ols = float(g[g.model == "OLS"].oos_r2.iloc[0])
        q_rf = float(g[g.model == "RandomForest"].oos_r2.iloc[0])
        gap = r_ols - r_rf
        L.append("  {:<9s} {:<16s} {:>14.8f} {:>14.8f} {:>14.8f} {:>12.6f}  {}".format(
            pop, feat, r_ols, r_rf, gap, 100 * gap / r_ols,
            "RF wins" if gap > 0 else "OLS wins"))
        slug = f"rq1.tuned.{pop}.{'full' if feat == 'full' else 'sig'}"
        out[f"{slug}.OLS.rmse"] = r_ols
        out[f"{slug}.OLS.r2"] = q_ols
        out[f"{slug}.RF.rmse"] = r_rf
        out[f"{slug}.RF.r2"] = q_rf
    n_ols_wins = sum(1 for (_, _), g in df.groupby(["population", "feature_set"])
                     if float(g[g.model == "OLS"].oos_rmse.iloc[0])
                     <= float(g[g.model == "RandomForest"].oos_rmse.iloc[0]))
    L += ["",
          f"  OLS wins {n_ols_wins} of 4 configurations. The text's 'in all four",
          f"  configurations' is {'CORRECT' if n_ols_wins == 4 else 'NOT SUPPORTED'}.",
          "",
          "  Note the R^2 column: under fair tuning the RF's out-of-sample R^2 is",
          "  near zero or negative in three of the four configurations, while OLS",
          "  stays positive in three. The forest is not merely losing narrowly.",
          ""]

    if force:
        L += verify(df)

    df.to_csv(C.RQ1_TUNED_CSV, index=False)
    tex = to_latex(df)
    C.RQ1_TUNED_TEX.write_text(tex + "\n")
    L += ["-- LaTeX fragment (also written to "
          f"{C.RQ1_TUNED_TEX.name}) --", "", tex, ""]

    text = "\n".join(L)
    C.RQ1_TUNED_CSV.with_suffix(".txt").write_text(text + "\n")
    print(text)
    print(f"wrote {C.RQ1_TUNED_CSV}")
    print(f"wrote {C.RQ1_TUNED_TEX}")
    return out


if __name__ == "__main__":
    main()
