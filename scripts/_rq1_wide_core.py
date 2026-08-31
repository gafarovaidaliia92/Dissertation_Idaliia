"""
run_rq1_wide.py — WIDE variant of Step 2 (RQ1), plus the narrow-vs-wide comparison.

The narrow pipeline trains on 277 listed banks. That population is selected on
being publicly traded and CRSP-matchable, which is a selection on size and
ownership, and it is small enough that the tree models have nothing to learn from
(out-of-sample R^2 came out negative for all three models). This variant trains
the SAME models with the SAME hyper-parameters on the full 2022Q4 filer
population above a size floor, so we can tell whether the negative RQ1 result is
a property of the phenomenon or an artefact of N=277.

Everything except the training population is imported from run_rq1.py — FEATURES,
SEED, N_SPLITS, RF/GB parameters, the OLS/CV/SHAP/scoring functions. Nothing in
run_rq1.py or build_panel.py is modified, and no existing file in
data/processed/ is overwritten: the four narrow outputs are first copied to
*_narrow names, and everything this script produces is written as *_wide.

RQ3 hand-off (leakage-free): the score a listed bank receives must not come from
a model that trained on that bank. So for RQ3 the model is fitted on the wide
population EXCLUDING all 278 listed banks and then applied to the listed banks.
That is a cleaner hold-out than the narrow pipeline's cross_val_predict, because
the training and scoring sets share no observations at all.

Outputs (all new files):
    data/processed/panel_2022Q4_narrow.csv        copy of the narrow panel
    data/processed/vulnerability_scores_narrow.csv  copy
    data/processed/rq1_results_narrow.txt         copy
    data/processed/rq1_shap_summary_narrow.png    copy
    data/processed/rq1_results_wide.txt
    data/processed/rq1_shap_summary_wide.png
    data/processed/vulnerability_scores_wide.csv  RQ3 input, leakage-free
    data/processed/comparison_narrow_vs_wide.txt
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict

import _rq1_core as run_rq1
from _panel_narrow import ZIP_2022Q4, load_filer_types
from _panel_wide import ASSET_FLOOR, SENSITIVITY_FLOORS
from _charter_flags import classify
from _rq1_core import (
    EXCLUDE_IDRSSD,
    FEATURES,
    GB_PARAMS,
    N_SPLITS,
    OUTCOME,
    RF_PARAMS,
    SEED,
    compute_shap,
    cross_validate_models,
    fit_ols,
    make_vulnerability_scores,
    model_factory,
    pick_better_tree,
    run_diagnostics,
)

PROC = Path("data/processed")

# --------------------------------------------------------------------------- #
#  RUN 3 / RUN 4 configuration (additional robustness runs)
# --------------------------------------------------------------------------- #
OUT_H1B_ROBUST = PROC / "rq1_h1b_robustness.txt"
OUT_FAILED_ROBUST = PROC / "rq1_failed_bank_robustness.txt"

N_REPEATS = 30          # repeated 5-fold CV: 30 independent shuffles
SIG_LEVEL = 0.05        # what counts as an OLS-significant predictor

# Tuning grids searched INSIDE the cross-validation (nested CV), so the winner is
# never chosen on data used to score it.
RF_GRID = {
    "n_estimators": [200],
    "max_depth": [3, 4, 6, None],
    "min_samples_leaf": [1, 5, 10],
    "max_features": [0.4, 0.6, 1.0],
}
GB_GRID = {
    "n_estimators": [200, 500],
    "learning_rate": [0.03, 0.10],
    "max_depth": [2, 3],
    "min_samples_leaf": [5, 10],
}
INNER_SPLITS = 3

# The failed banks are NOT listed here. They are read off the panel's `failed`
# column, which _panel_wide.load_failures() derives from the FDIC failed-bank
# list by certificate. Repeating the three RSSDs as a literal would have been a
# second, silently divergent definition of "failed".


def failed_banks(panel: pd.DataFrame) -> dict[int, str]:
    """{bank_IDRSSD: name} for the banks this panel flags as 2023 failures,
    ordered by IDRSSD so the report is stable across panels."""
    f = panel[panel["failed"].astype(bool)]
    return {int(r.bank_IDRSSD): str(r["name"])
            for _, r in f.sort_values("bank_IDRSSD").iterrows()}

NARROW_PANEL = PROC / "panel_2022Q4.csv"
WIDE_PANEL = PROC / "panel_2022Q4_wide.csv"                 # charter-filtered (953)
WIDE_ALLCHARTERS_PANEL = PROC / "panel_2022Q4_wide_allcharters.csv"  # pre-filter (963)

# narrow originals -> *_narrow copies (originals are never rewritten)
NARROW_COPIES = {
    PROC / "panel_2022Q4.csv":        PROC / "panel_2022Q4_narrow.csv",
    PROC / "vulnerability_scores.csv": PROC / "vulnerability_scores_narrow.csv",
    PROC / "rq1_results.txt":          PROC / "rq1_results_narrow.txt",
    PROC / "rq1_shap_summary.png":     PROC / "rq1_shap_summary_narrow.png",
}

OUT_RESULTS_WIDE = PROC / "rq1_results_wide.txt"
OUT_SHAP_WIDE = PROC / "rq1_shap_summary_wide.png"
OUT_SCORES_WIDE = PROC / "vulnerability_scores_wide.csv"
OUT_COMPARISON = PROC / "comparison_narrow_vs_wide.txt"


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_variant(path: Path, subset: pd.Series | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Mirror of run_rq1.load_panel() but for an arbitrary panel file: drop the
    merger-exit bank and any null outcome, return (meta, X, y). Censored failures
    are retained on purpose. `subset` is an optional boolean mask applied first.
    """
    df = pd.read_csv(path)
    if subset is not None:
        df = df[subset.reindex(df.index, fill_value=False)]
    df = df[~df["bank_IDRSSD"].isin(EXCLUDE_IDRSSD)].copy()
    df = df[df[OUTCOME].notna()].reset_index(drop=True)
    X = df[FEATURES].astype(float)
    y = df[OUTCOME].astype(float)
    assert not X.isna().any().any(), f"{path.name}: features contain nulls"
    return df, X, y


def shap_to(path: Path):
    """
    compute_shap() writes to the module-level run_rq1.OUT_SHAP_PNG. Redirect it
    rather than duplicating the function, so the narrow and wide figures are
    produced by identical code.
    """
    run_rq1.OUT_SHAP_PNG = path


# --------------------------------------------------------------------------- #
#  One full RQ1 run on a given population
# --------------------------------------------------------------------------- #
def run_variant(label: str, meta: pd.DataFrame, X: pd.DataFrame, y: pd.Series,
                shap_png: Path) -> dict:
    """OLS + 5-fold CV of the three models + SHAP on the better tree + OOS
    vulnerability scores. Identical settings for every population."""
    print(f"\n=== {label}: N={len(y)} ===")
    ols_table, ols_r2 = fit_ols(X, y)
    metrics, oos = cross_validate_models(X, y)
    better = pick_better_tree(metrics)

    chosen = model_factory()[better]
    chosen.fit(X, y)
    shap_to(shap_png)
    shap_imp = compute_shap(chosen, X)

    scores = make_vulnerability_scores(meta, oos[better])
    report = run_diagnostics(metrics, ols_table, ols_r2, shap_imp, scores, better)
    print(f"    better tree = {better}  |  "
          f"OOS R2: OLS={metrics.loc['OLS','oos_R2']:.4f} "
          f"{better}={metrics.loc[better,'oos_R2']:.4f}")
    return {"label": label, "N": len(y), "metrics": metrics, "ols": ols_table,
            "ols_r2": ols_r2, "shap": shap_imp, "scores": scores,
            "better": better, "report": report}


# --------------------------------------------------------------------------- #
#  Size-floor sensitivity
# --------------------------------------------------------------------------- #
def sensitivity_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Re-run the CV comparison at each size floor. Only the floor changes."""
    rows = []
    for floor in SENSITIVITY_FLOORS:
        sub = wide[wide["total_assets"] > floor]
        X = sub[FEATURES].astype(float)
        y = sub[OUTCOME].astype(float)
        metrics, _ = cross_validate_models(X, y)
        better = pick_better_tree(metrics)
        rows.append({
            "floor": f"${floor/1e6:.0f}bn",
            "N": len(y),
            "listed": int(sub["is_listed"].sum()),
            "OLS_oos_RMSE": metrics.loc["OLS", "oos_RMSE"],
            "OLS_oos_R2": metrics.loc["OLS", "oos_R2"],
            "tree": better,
            "tree_oos_RMSE": metrics.loc[better, "oos_RMSE"],
            "tree_oos_R2": metrics.loc[better, "oos_R2"],
            "H1b": "SUPPORTED" if metrics.loc[better, "oos_RMSE"]
                   < metrics.loc["OLS", "oos_RMSE"] else "not supported",
        })
        print(f"    floor {rows[-1]['floor']:>6}: N={len(y):>4}  "
              f"H1b {rows[-1]['H1b']}")
    return pd.DataFrame(rows).set_index("floor")


# --------------------------------------------------------------------------- #
#  Charter flag for an arbitrary meta frame (e.g. the 277 scoring set)
# --------------------------------------------------------------------------- #
def attach_flags(meta: pd.DataFrame) -> pd.DataFrame:
    """Add is_trust_or_specialized / flag_reason to a frame that has
    bank_IDRSSD + uninsured_share, joining the FDIC certificate first. Used so the
    RQ3 scoring set carries the same flag as the wide training panel — without
    dropping anyone."""
    filers = load_filer_types(ZIP_2022Q4)
    filers["fdic_cert"] = pd.to_numeric(filers["fdic_cert"], errors="coerce")
    m = meta.merge(filers[["IDRSSD", "fdic_cert"]], left_on="bank_IDRSSD",
                   right_on="IDRSSD", how="left").drop(columns=["IDRSSD"],
                                                       errors="ignore")
    return classify(m, ZIP_2022Q4)


# --------------------------------------------------------------------------- #
#  RQ3 hand-off: train wide-excluding-listed, apply to the listed banks
# --------------------------------------------------------------------------- #
def rq3_scores(wide: pd.DataFrame, narrow_meta: pd.DataFrame,
               narrow_X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Fit on the wide population with every listed bank removed, then score the
    listed banks. No listed bank contributes to the model that scores it, so the
    predictions are out-of-sample by construction — no cross_val_predict needed.
    """
    train = wide[~wide["is_listed"].astype(bool)]
    Xtr = train[FEATURES].astype(float)
    ytr = train[OUTCOME].astype(float)
    print(f"    training population (wide minus listed): N={len(ytr)}")

    metrics, _ = cross_validate_models(Xtr, ytr)
    better = pick_better_tree(metrics)
    model = model_factory()[better]
    model.fit(Xtr, ytr)

    pred = model.predict(narrow_X)
    scores = pd.DataFrame({
        "bank_IDRSSD": narrow_meta["bank_IDRSSD"].values,
        "name": narrow_meta["name"].values,
        "permno": narrow_meta["permno"].values,
        "pred_dep_growth": pred,
        "vulnerability_score": -pred,       # oriented: higher = more vulnerable
    })
    # Attach the charter flag to the SCORING set. The 277 listed banks are NEVER
    # dropped here (custodians State Street / BNY Mellon / Northern Trust stay);
    # the flag just marks which are custody/trust-type so that decision can be
    # made later, downstream in RQ3.
    flags = attach_flags(narrow_meta)[["bank_IDRSSD", "is_trust_or_specialized",
                                       "flag_reason"]]
    scores = scores.merge(flags, on="bank_IDRSSD", how="left")
    scores["is_trust_or_specialized"] = scores["is_trust_or_specialized"].fillna(False)
    scores = scores.sort_values("vulnerability_score",
                                ascending=False).reset_index(drop=True)

    # how well the transferred model does on the listed banks it never saw
    y_listed = narrow_meta[OUTCOME].astype(float).values
    transfer = {"RMSE": float(np.sqrt(mean_squared_error(y_listed, pred))),
                "R2": float(r2_score(y_listed, pred))}
    print(f"    transfer to {len(scores)} listed banks: "
          f"RMSE={transfer['RMSE']:.5f}  R2={transfer['R2']:.4f}")
    return scores, metrics, better, transfer


# --------------------------------------------------------------------------- #
#  RUN 3 — H1b under feature pruning, repeated CV and in-CV tuning
# --------------------------------------------------------------------------- #
def significant_features(ols_table: pd.DataFrame, level: float = SIG_LEVEL
                         ) -> list[str]:
    """The OLS-significant predictors, read off the fitted table rather than
    hard-coded, so the set cannot silently drift from the numbers reported."""
    return [f for f in FEATURES if ols_table.loc[f, "p_value"] < level]


def repeated_cv(X: pd.DataFrame, y: pd.Series, n_repeats: int = N_REPEATS
                ) -> pd.DataFrame:
    """`n_repeats` independent 5-fold splits. Both the fold shuffle AND the
    estimators' own random_state vary with the repeat, so the spread reflects
    the full sampling variation of the comparison, not just the fold draw.
    One row per (repeat, model)."""
    rows = []
    for r in range(n_repeats):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=r)
        models = {
            "OLS": LinearRegression(),
            "RandomForest": RandomForestRegressor(**{**RF_PARAMS, "random_state": r}),
            "GradientBoosting": GradientBoostingRegressor(
                **{**GB_PARAMS, "random_state": r}),
        }
        for name, model in models.items():
            pred = cross_val_predict(model, X, y, cv=kf)
            rows.append({"repeat": r, "model": name,
                         "oos_RMSE": float(np.sqrt(mean_squared_error(y, pred))),
                         "oos_R2": float(r2_score(y, pred))})
    return pd.DataFrame(rows)


def repeated_summary(rep: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mean/SD per model plus the paired RF-vs-OLS comparison. The paired form
    is the honest one: RF and OLS see identical folds within a repeat, so the
    per-repeat difference removes the fold-draw noise they share."""
    summ = rep.groupby("model").agg(
        mean_oos_RMSE=("oos_RMSE", "mean"), sd_oos_RMSE=("oos_RMSE", "std"),
        min_oos_RMSE=("oos_RMSE", "min"), max_oos_RMSE=("oos_RMSE", "max"),
        mean_oos_R2=("oos_R2", "mean"), sd_oos_R2=("oos_R2", "std"))
    w = rep.pivot(index="repeat", columns="model", values="oos_RMSE")
    diff = w["OLS"] - w["RandomForest"]          # >0 means RF wins
    diff_gb = w["OLS"] - w["GradientBoosting"]
    paired = {
        "n": len(w),
        "rf_wins": int((diff > 0).sum()),
        "rf_win_share": float((diff > 0).mean()),
        "mean_diff": float(diff.mean()),
        "sd_diff": float(diff.std()),
        "mean_diff_pct": float((diff / w["OLS"]).mean() * 100),
        "gb_wins": int((diff_gb > 0).sum()),
        "gb_win_share": float((diff_gb > 0).mean()),
    }
    return summ, paired


# Two grid points whose inner-CV scores differ by less than this are a TIE, not a
# ranking. The difference is BLAS summation order, which is not stable run to run:
# GridSearchCV's argmax then flips the reported winner between otherwise identical
# runs (observed on the narrow population, outer fold 5, max_features 0.4 vs 0.6).
# On a tie we take the lexicographically first grid point — sorted by parameter
# name, then by the string form of the value — so the choice depends only on the
# grid, never on floating-point noise.
GRID_TIE_TOL = 1e-9


def deterministic_winner(gs: GridSearchCV) -> dict:
    """The winning grid point, broken deterministically on ties (GRID_TIE_TOL)."""
    res = gs.cv_results_
    scores = np.asarray(res["mean_test_score"], dtype=float)
    best = np.nanmax(scores)
    tied = [i for i, s in enumerate(scores) if s >= best - GRID_TIE_TOL]

    def sort_key(i: int) -> tuple:
        return tuple(sorted((k, str(v)) for k, v in res["params"][i].items()))

    return dict(res["params"][min(tied, key=sort_key)])


def nested_cv_tuned(X: pd.DataFrame, y: pd.Series) -> dict:
    """Nested CV: the hyper-parameters are chosen by an inner grid search on the
    TRAINING part of each outer fold, so no configuration is ever selected on the
    data it is scored on. Returns OOS metrics plus the winning grid point of each
    outer fold (they need not agree — that is informative in itself).

    The winner is picked by deterministic_winner(), not by GridSearchCV's own
    best_params_, and the fold's model is refitted with it."""
    outer = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    out = {}
    specs = {
        "RandomForest": (RandomForestRegressor(random_state=SEED, n_jobs=-1), RF_GRID),
        "GradientBoosting": (GradientBoostingRegressor(random_state=SEED,
                                                       subsample=0.8), GB_GRID),
    }
    for name, (est, grid) in specs.items():
        preds = np.full(len(y), np.nan)
        winners = []
        for tr, te in outer.split(X):
            gs = GridSearchCV(est, grid, cv=KFold(n_splits=INNER_SPLITS,
                                                  shuffle=True, random_state=SEED),
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
            gs.fit(X.iloc[tr], y.iloc[tr])
            params = deterministic_winner(gs)
            best = clone(est).set_params(**params).fit(X.iloc[tr], y.iloc[tr])
            preds[te] = best.predict(X.iloc[te])
            winners.append(params)
        out[name] = {
            "oos_RMSE": float(np.sqrt(mean_squared_error(y, preds))),
            "oos_R2": float(r2_score(y, preds)),
            "winners": winners,
        }
    # OLS has nothing to tune; scored on the same outer folds for comparability
    preds = cross_val_predict(LinearRegression(), X, y, cv=outer)
    out["OLS"] = {"oos_RMSE": float(np.sqrt(mean_squared_error(y, preds))),
                  "oos_R2": float(r2_score(y, preds)), "winners": []}
    return out


def run3_report(X_full: pd.DataFrame, y: pd.Series, ols_table: pd.DataFrame,
                label: str) -> str:
    sig = significant_features(ols_table)
    X_sig = X_full[sig]
    L = ["=" * 90,
         f"RUN 3 — H1b UNDER FEATURE PRUNING, REPEATED CV AND IN-CV TUNING  [{label}]",
         "=" * 90, "",
         f"  N = {len(y)}   full feature set = {len(FEATURES)}   "
         f"significant-only set = {len(sig)}",
         "",
         "  HOW THE MODELS ARE CODED (so the comparison is reproducible):",
         f"    library            : scikit-learn {sklearn.__version__}",
         "    OLS                : sklearn.linear_model.LinearRegression (no penalty,",
         "                         no intercept-free trick; identical design matrix",
         "                         to the statsmodels OLS used for the H1a table)",
         f"    RandomForest       : sklearn.ensemble.RandomForestRegressor({_kw(RF_PARAMS)})",
         f"    GradientBoosting   : sklearn.ensemble.GradientBoostingRegressor("
         f"{_kw(GB_PARAMS)})",
         f"    cross-validation   : sklearn.model_selection.KFold("
         f"n_splits={N_SPLITS}, shuffle=True)",
         "    scoring            : pooled cross_val_predict, then RMSE / R2 against y",
         "",
         "-- 3.1 which predictors are OLS-significant at 5% in this population? --", "",
         "    {:<22s} {:>14s} {:>12s} {:>10s}  {}".format(
             "feature", "coef", "p_value", "beta_std", "kept?"),
         "    " + "-" * 72]
    for f in FEATURES:
        r = ols_table.loc[f]
        L.append("    {:<22s} {:>14.8f} {:>12.8f} {:>10.5f}  {}".format(
            f, r["coef"], r["p_value"], r["beta_std"],
            "KEPT" if f in sig else ""))
    L += ["", f"    significant-only feature set: {sig}", ""]

    # ---- single-split comparison, both feature sets ----
    L += ["-- 3.2 the original single 5-fold split (seed 42), full vs pruned --", "", ]
    tab = []
    for tag, Xs in (("full (10 features)", X_full), (f"pruned ({len(sig)})", X_sig)):
        m, _ = cross_validate_models(Xs, y)
        tab.append((tag, m))
    L += ["    {:<22s} {:<18s} {:>14s} {:>14s}".format(
              "feature set", "model", "oos_RMSE", "oos_R2"),
          "    " + "-" * 72]
    for tag, m in tab:
        for name in ("OLS", "RandomForest", "GradientBoosting"):
            L.append("    {:<22s} {:<18s} {:>14.8f} {:>14.8f}".format(
                tag, name, m.loc[name, "oos_RMSE"], m.loc[name, "oos_R2"]))
        gap = m.loc["OLS", "oos_RMSE"] - m.loc["RandomForest", "oos_RMSE"]
        L.append("    {:<22s} {:<18s} {:>14.8f}   ({:+.4f}% of OLS RMSE)".format(
            tag, "OLS - RF gap", gap, 100 * gap / m.loc["OLS", "oos_RMSE"]))
        L.append("")

    # ---- repeated CV ----
    L += ["-- 3.3 REPEATED cross-validation: is the RF edge stable or noise? --", "",
          f"    {N_REPEATS} independent 5-fold splits (fold shuffle AND estimator",
          "    random_state both vary with the repeat).", ""]
    store = {}
    for tag, Xs in (("full", X_full), ("pruned", X_sig)):
        rep = repeated_cv(Xs, y)
        summ, paired = repeated_summary(rep)
        store[tag] = (summ, paired)
        L += [f"    [{tag} feature set]", "",
              "    {:<18s} {:>14s} {:>13s} {:>14s} {:>14s} {:>13s}".format(
                  "model", "mean RMSE", "SD RMSE", "min RMSE", "max RMSE", "mean R2"),
              "    " + "-" * 92]
        for name in ("OLS", "RandomForest", "GradientBoosting"):
            r = summ.loc[name]
            L.append("    {:<18s} {:>14.8f} {:>13.8f} {:>14.8f} {:>14.8f} {:>13.8f}".format(
                name, r.mean_oos_RMSE, r.sd_oos_RMSE, r.min_oos_RMSE,
                r.max_oos_RMSE, r.mean_oos_R2))
        L += ["",
              f"    paired RF-vs-OLS over the same {paired['n']} splits:",
              f"      RF beats OLS in {paired['rf_wins']} of {paired['n']} repeats "
              f"({paired['rf_win_share'] * 100:.4f}%)",
              f"      mean RMSE advantage (OLS - RF) = {paired['mean_diff']:+.8f}"
              f"   SD {paired['sd_diff']:.8f}",
              f"      i.e. {paired['mean_diff_pct']:+.6f}% of the OLS RMSE on average",
              f"      GB beats OLS in {paired['gb_wins']} of {paired['n']} repeats "
              f"({paired['gb_win_share'] * 100:.4f}%)",
              ""]

    # ---- nested tuning ----
    L += ["-- 3.4 TUNED hyper-parameters, chosen inside the CV (nested) --", "",
          "    Grids searched on the inner folds only:",
          f"      RandomForest     : {RF_GRID}",
          f"      GradientBoosting : {GB_GRID}",
          f"      inner CV         : KFold({INNER_SPLITS}, shuffle=True), scoring = "
          "neg_root_mean_squared_error",
          "", ]
    for tag, Xs in (("full", X_full), (f"pruned ({len(sig)})", X_sig)):
        nested = nested_cv_tuned(Xs, y)
        L += [f"    [{tag} feature set]", "",
              "    {:<18s} {:>16s} {:>16s}".format("model", "oos_RMSE", "oos_R2"),
              "    " + "-" * 54]
        for name in ("OLS", "RandomForest", "GradientBoosting"):
            L.append("    {:<18s} {:>16.8f} {:>16.8f}".format(
                name, nested[name]["oos_RMSE"], nested[name]["oos_R2"]))
        gap = nested["OLS"]["oos_RMSE"] - nested["RandomForest"]["oos_RMSE"]
        L += ["",
              f"      tuned OLS - RF gap = {gap:+.8f}"
              f"  ({100 * gap / nested['OLS']['oos_RMSE']:+.6f}% of OLS RMSE)  "
              f"H1b {'SUPPORTED' if gap > 0 else 'NOT supported'}",
              "      winning grid point per outer fold:"]
        for nm in ("RandomForest", "GradientBoosting"):
            for i, wgp in enumerate(nested[nm]["winners"]):
                L.append(f"        {nm:<18s} fold {i + 1}: {wgp}")
        L.append("")
    return "\n".join(L), store


def _kw(d: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in d.items())


MARKER = "7. SUPERVISOR ROBUSTNESS RUNS (RUN 3, RUN 4) — POINTERS"


def append_robustness_pointer() -> None:
    """Append a section 7 to comparison_narrow_vs_wide.txt pointing at the two
    robustness reports, with the headline numbers inlined so the main document
    is not silently contradicted by a file next to it. Idempotent: any previously
    appended section 7 is stripped first."""
    base = OUT_COMPARISON.read_text()
    cut = base.find("-" * 100 + "\n" + MARKER)
    if cut > 0:
        base = base[:cut].rstrip("\n") + "\n"

    def grab(path: Path, pat: str, n: int = 1) -> list[str]:
        if not path.exists():
            return []
        return [l.strip() for l in path.read_text().splitlines() if pat in l][:n]

    def _split_narrow(path: Path) -> tuple[str, str]:
        """(wide block, narrow block) of a robustness report."""
        txt = path.read_text() if path.exists() else ""
        parts = txt.split("NARROW, N=277")
        return parts[0], parts[-1]

    def narrow_mean_r2() -> tuple[float, float]:
        """Mean OOS R2 over the 30 narrow repeats, full feature set — read from
        the RUN 3 report rather than restated, so this pointer cannot drift away
        from the document it points at."""
        _, narrow = _split_narrow(OUT_H1B_ROBUST)
        out: dict[str, float] = {}
        for line in narrow.splitlines():
            s = line.split()
            # repeated-CV summary row: model, meanRMSE, sdRMSE, minRMSE, maxRMSE, meanR2
            if len(s) == 6 and s[0] in ("OLS", "RandomForest"):
                out.setdefault(s[0], float(s[5]))
        return out.get("OLS", float("nan")), out.get("RandomForest", float("nan"))

    def failed_pair(block: str) -> tuple[float, float, float, float]:
        """(baseline coef, baseline p, excluded coef, excluded p) for
        uninsured_share in one population of the RUN 4 report."""
        vals = [float("nan")] * 4
        if "uninsured_share:" not in block:
            return tuple(vals)                                    # type: ignore[return-value]
        sub = block.split("uninsured_share:")[1].split("unrealised_losses:")[0]
        for line in sub.splitlines():
            s = line.split()
            if len(s) < 5:
                continue
            if "(a) baseline" in line:
                vals[0], vals[1] = float(s[-4]), float(s[-3])
            elif "(b) failed banks excluded" in line:
                vals[2], vals[3] = float(s[-4]), float(s[-3])
        return tuple(vals)                                        # type: ignore[return-value]

    L = ["-" * 100, MARKER, "-" * 100,
         "  Both are reported in full in:",
         f"    {OUT_H1B_ROBUST.name}        — RUN 3: pruned features, repeated CV, "
         "in-CV tuning",
         f"    {OUT_FAILED_ROBUST.name}  — RUN 4: alternative failed-bank treatments",
         "",
         "  RUN 3 headline (WIDE). The RF-over-OLS margin in section 1 above is not",
         "  robust to how the comparison is run:"]
    for l in grab(OUT_H1B_ROBUST, "RF beats OLS in", 2):
        L.append(f"    {l}")
    L += ["    (first line = full 10 features, second = OLS-significant-only set)",
          "    Under NESTED tuning the trees lose to OLS in every feature set and",
          "    both populations — see section 3.4 of the RUN 3 report.",
          "",
          "  RUN 3 also corrects a single-split artefact in the NARROW numbers above:",
          "  section 1 reports negative oos_R2 for every narrow model, but that is one",
          "  unlucky seed. Averaged over 30 splits the narrow oos_R2 is POSITIVE",
          "  (OLS {:+.4f}, RandomForest {:+.4f}).".format(*narrow_mean_r2()),
          "",
          "  RUN 4 headline. uninsured_share keeps its negative sign under all four",
          "  failed-bank treatments in BOTH populations, but its significance does not",
          "  survive in the narrow sample once the three failed banks are dropped:",
          "    WIDE   baseline {:.8f} (p {:.8f})  ->  excluded {:.8f} "
          "(p {:.8f})".format(*failed_pair(_split_narrow(OUT_FAILED_ROBUST)[0])),
          "    NARROW baseline {:.8f} (p {:.8f})  ->  excluded {:.8f} "
          "(p {:.8f})".format(*failed_pair(_split_narrow(OUT_FAILED_ROBUST)[1])),
          "  The narrow H1a headline is therefore carried in large part by SVB,",
          "  Signature and First Republic. The wide result is not.",
          "",
          "=" * 100, ""]
    OUT_COMPARISON.write_text(base + "\n".join(L))


# --------------------------------------------------------------------------- #
#  RUN 4 — alternative treatments of the three failed banks
# --------------------------------------------------------------------------- #
def failed_treatments(panel: pd.DataFrame) -> dict:
    """(a) baseline, (b) drop the three, (c) censor at the empirical minimum
    among surviving banks, (d) winsorise the outflow tail at the 1st percentile.

    (c) is the defensible middle option: it keeps the three banks in the sample
    and keeps them ranked as the worst outflows observed, but stops -1.0 from
    acting as a 4x outlier relative to every other bank. (d) is added because it
    treats the tail as a whole rather than singling out three observations."""
    base = panel[panel[OUTCOME].notna()].copy()
    is_failed = base["failed"].astype(bool)
    surv_min = float(base.loc[~is_failed, OUTCOME].min())
    p01 = float(base[OUTCOME].quantile(0.01))

    a = base.copy()
    b = base[~is_failed].copy()
    c = base.copy()
    c.loc[is_failed, OUTCOME] = surv_min
    d = base.copy()
    d[OUTCOME] = d[OUTCOME].clip(lower=p01)

    return {
        "(a) baseline: censored at -1.0": (a, {"n_failed": int(is_failed.sum())}),
        "(b) failed banks excluded": (b, {"n_failed": 0}),
        f"(c) censored at survivor min ({surv_min:.8f})": (c, {"n_failed": int(is_failed.sum())}),
        f"(d) winsorised at 1st pct ({p01:.8f})": (d, {"n_failed": int(is_failed.sum())}),
    }, {"surv_min": surv_min, "p01": p01, "n_failed": int(is_failed.sum())}


def run4_report(panel: pd.DataFrame, label: str) -> str:
    treatments, info = failed_treatments(panel)
    L = ["=" * 90,
         f"RUN 4 — ALTERNATIVE TREATMENTS OF THE THREE FAILED BANKS  [{label}]",
         "=" * 90, "",
         "  SVB, Signature and First Republic enter the baseline panel with",
         "  dep_growth = -1.0 (censored total loss of deposits). That value is not an",
         "  observation, it is a coding choice, and it sits far outside the rest of",
         f"  the distribution: the worst SURVIVING bank is {info['surv_min']:.8f} and the",
         f"  1st percentile is {info['p01']:.8f}. Four treatments are compared.",
         "",
         f"  banks flagged as failed in this population: {info['n_failed']}",
         "    " + ", ".join(f"{k} ({v})"
                              for k, v in failed_banks(panel).items()),
         "",
         "-- 4.1 H1a: OLS coefficients under each treatment --", "",
         "    {:<44s} {:>7s} {:>14s} {:>12s} {:>10s}".format(
             "treatment", "N", "coef", "p_value", "beta_std"),
         "    " + "-" * 92]
    keep = ["uninsured_share", "unrealised_losses", "liquidity", "ROA"]
    tables = {}
    for name, (df, meta) in treatments.items():
        X = df[FEATURES].astype(float)
        y = df[OUTCOME].astype(float)
        tbl, r2 = fit_ols(X, y)
        tables[name] = (tbl, r2, X, y)
    for var in keep:
        L.append(f"    {var}:")
        for name, (tbl, r2, X, y) in tables.items():
            r = tbl.loc[var]
            L.append("    {:<44s} {:>7d} {:>14.8f} {:>12.8f} {:>10.5f}  {}".format(
                "  " + name, len(y), r["coef"], r["p_value"], r["beta_std"],
                "***" if r["p_value"] < 0.01 else "**" if r["p_value"] < 0.05
                else "*" if r["p_value"] < 0.10 else "(n.s.)"))
        L.append("")
    L += ["    in-sample OLS R^2 per treatment:"]
    for name, (tbl, r2, X, y) in tables.items():
        L.append(f"      {name:<46s} {r2:.8f}")
    L += ["",
          "-- 4.2 H1b: out-of-sample comparison under each treatment --", "",
          "    {:<44s} {:<18s} {:>14s} {:>14s}".format(
              "treatment", "model", "oos_RMSE", "oos_R2"),
          "    " + "-" * 96]
    verdicts = {}
    for name, (tbl, r2, X, y) in tables.items():
        m, _ = cross_validate_models(X, y)
        for mod in ("OLS", "RandomForest", "GradientBoosting"):
            L.append("    {:<44s} {:<18s} {:>14.8f} {:>14.8f}".format(
                name if mod == "OLS" else "", mod,
                m.loc[mod, "oos_RMSE"], m.loc[mod, "oos_R2"]))
        better = pick_better_tree(m)
        gap = m.loc["OLS", "oos_RMSE"] - m.loc[better, "oos_RMSE"]
        h1b = gap > 0
        verdicts[name] = (better, gap, h1b, m)
        L += ["    {:<44s} {:<18s} {:>14.8f}   H1b {}".format(
            "", f"OLS - {better} gap", gap,
            "SUPPORTED" if h1b else "NOT supported"), ""]
    L += ["-- 4.3 verdict --", ""]
    signs = {n: tables[n][0].loc["uninsured_share", "coef"] for n in tables}
    ps = {n: tables[n][0].loc["uninsured_share", "p_value"] for n in tables}
    all_sig = all(p < 0.05 for p in ps.values())
    all_neg = all(c < 0 for c in signs.values())
    L += [f"    uninsured_share keeps a NEGATIVE sign in all four treatments: "
          f"{'YES' if all_neg else 'NO'}",
          f"    uninsured_share significant at 5% in all four: "
          f"{'YES' if all_sig else 'NO'}",
          f"    coefficient range across treatments: "
          f"[{min(signs.values()):.8f}, {max(signs.values()):.8f}]",
          "",
          f"    H1b verdict per treatment: "
          + "; ".join(f"{n.split(':')[0]} -> {'SUPPORTED' if v[2] else 'not supported'}"
                      for n, v in verdicts.items()),
          ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  Comparison document
# --------------------------------------------------------------------------- #
def side_by_side(left: list[str], right: list[str], lw: int = 46) -> list[str]:
    n = max(len(left), len(right))
    left += [""] * (n - len(left))
    right += [""] * (n - len(right))
    return [f"{a:<{lw}}  |  {b}" for a, b in zip(left, right)]


def build_comparison(narrow: dict, wide: dict, sens: pd.DataFrame,
                     rq3_metrics: pd.DataFrame, rq3_better: str,
                     rq3_transfer: dict, rq3_scores_df: pd.DataFrame,
                     wide_panel: pd.DataFrame, allcharters: dict,
                     wide_scores_flagged: pd.DataFrame,
                     wide_trust_excluded: pd.DataFrame) -> str:
    L, add = [], None
    add = L.append
    add("=" * 100)
    add("RQ1 — NARROW vs WIDE TRAINING POPULATION")
    add("=" * 100)
    add(f"seed={SEED}  |  {N_SPLITS}-fold CV  |  features={len(FEATURES)}  "
        f"|  identical hyper-parameters in all variants")
    add("")

    # ---- charter filter effect (section 0) ----
    add("-" * 100)
    add("0. CHARTER FILTER EFFECT  —  pre-filter wide (all charters) vs filtered wide")
    add("-" * 100)
    add(f"  pre-filter WIDE-ALL : N={allcharters['N']}  (all 2022Q4 filers "
        f">${ASSET_FLOOR/1e6:.0f}bn, deposit_reliance>=0.50)")
    add(f"  filtered  WIDE      : N={wide['N']}  "
        f"(credit-card banks HARD-DROPPED; trust/custody FLAGGED & kept)")
    add(f"  removed by the filter: {allcharters['N'] - wide['N']} credit-card banks "
        f"(FDIC SPECGRP==3)")
    add(f"  trust/custody/servicer flagged & kept: "
        f"{int(wide_panel.is_trust_or_specialized.sum())}")
    add("")
    for tag, v in (("WIDE-ALL", allcharters), ("WIDE(filtered)", wide)):
        m, b = v["metrics"], v["better"]
        h1b = m.loc[b, "oos_RMSE"] < m.loc["OLS", "oos_RMSE"]
        add(f"  {tag:<16} better={b:<15} "
            f"OLS oosR2={m.loc['OLS','oos_R2']:+.4f}  "
            f"{b} oosR2={m.loc[b,'oos_R2']:+.4f}  "
            f"H1b {'SUPPORTED' if h1b else 'NOT supported'}")
    add("")
    add("  OLS signs on the two headline predictors (does the filter move them?):")
    add(f"    {'variable':<20}{'WIDE-ALL coef':>15}{'  p':>10}   "
        f"{'FILTERED coef':>15}{'  p':>10}")
    for var in ("uninsured_share", "unrealised_losses"):
        ra = allcharters["ols"].loc[var]
        rf = wide["ols"].loc[var]
        add(f"    {var:<20}{ra['coef']:>15.5f}{ra['p_value']:>10.5f}   "
            f"{rf['coef']:>15.5f}{rf['p_value']:>10.5f}")
    add("")
    add("  SHAP top-5:")
    la = ["   WIDE-ALL"] + allcharters["shap"].head(5).round(6).to_string(
        index=False).split("\n")
    rf = ["   FILTERED"] + wide["shap"].head(5).round(6).to_string(
        index=False).split("\n")
    L.extend(side_by_side(la, rf, lw=42))
    add("")
    add("  TOP-10 vulnerable — does removing trust cos change it?")
    def t10(df, flagcol=None):
        out = []
        for i, (_, r) in enumerate(df.head(10).iterrows()):
            mark = ""
            if flagcol is not None and r.get(flagcol, False):
                mark = " *trust"
            out.append(f"{i+1:>2}. {r['name'][:30]:<30} {r.vulnerability_score:>7.4f}{mark}")
        return out
    la = ["   FILTERED wide, all kept (* = trust-flagged)"] + \
        t10(wide_scores_flagged, "is_trust_or_specialized")
    rf = ["   FILTERED wide, trust-flagged EXCLUDED"] + t10(wide_trust_excluded)
    L.extend(side_by_side(la, rf, lw=52))
    add("")
    add("NARROW = 278 listed banks matched to CRSP (277 modelled after the merger exit).")
    add(f"WIDE   = all 2022Q4 Call Report filers, assets > ${ASSET_FLOOR/1e6:.0f}bn, "
        f"deposit_reliance >= 0.50 ({wide['N']} modelled; "
        f"{int(wide_panel.is_listed.sum())} of them are listed).")
    add("")

    # ---- out-of-sample metrics ----
    add("-" * 100)
    add("1. OUT-OF-SAMPLE PERFORMANCE (H1b)")
    add("-" * 100)
    ln = [f"NARROW  (N={narrow['N']})", ""]
    ln += narrow["metrics"][["oos_RMSE", "oos_R2"]].round(5).to_string().split("\n")
    rn = [f"WIDE  (N={wide['N']})", ""]
    rn += wide["metrics"][["oos_RMSE", "oos_R2"]].round(5).to_string().split("\n")
    L.extend(side_by_side(ln, rn))
    add("")
    for v in (narrow, wide):
        m, b = v["metrics"], v["better"]
        h1b = m.loc[b, "oos_RMSE"] < m.loc["OLS", "oos_RMSE"]
        add(f"  {v['label']:<7} better tree = {b:<17} "
            f"H1b {'SUPPORTED' if h1b else 'NOT supported'}  "
            f"({b} oos_RMSE {m.loc[b,'oos_RMSE']:.5f} "
            f"{'<' if h1b else '>='} OLS {m.loc['OLS','oos_RMSE']:.5f})")
    add("")

    # ---- OLS signs on the two headline predictors ----
    add("-" * 100)
    add("2. OLS — the two predictors that carried H1a in the narrow sample")
    add("-" * 100)
    add(f"{'variable':<20}{'variant':<9}{'coef':>11}{'t':>9}{'p':>11}"
        f"{'beta_std':>11}  sign")
    for var in ("uninsured_share", "unrealised_losses"):
        for v in (narrow, wide):
            r = v["ols"].loc[var]
            add(f"{var:<20}{v['label']:<9}{r['coef']:>11.5f}{r['t']:>9.3f}"
                f"{r['p_value']:>11.6f}{r['beta_std']:>11.5f}  {r['sign']}")
        add("")
    add(f"  in-sample OLS R^2:  narrow={narrow['ols_r2']:.4f}   "
        f"wide={wide['ols_r2']:.4f}")
    add("")

    # ---- SHAP ----
    add("-" * 100)
    add("3. SHAP top-5 (chosen tree, fitted on the full respective population)")
    add("-" * 100)
    ln = [f"NARROW — {narrow['better']}", ""]
    ln += narrow["shap"].head(5).round(6).to_string(index=False).split("\n")
    rn = [f"WIDE — {wide['better']}", ""]
    rn += wide["shap"].head(5).round(6).to_string(index=False).split("\n")
    L.extend(side_by_side(ln, rn))
    add("")

    # ---- top-10 vulnerable ----
    add("-" * 100)
    add("4. TOP-10 BY VULNERABILITY SCORE")
    add("-" * 100)
    def top10(df):
        return [f"{i+1:>2}. {r['name'][:34]:<34} {r.vulnerability_score:>7.4f}"
                for i, (_, r) in enumerate(df.head(10).iterrows())]
    ln = ["NARROW (cross_val_predict, listed only)", ""] + top10(narrow["scores"])
    rn = [f"WIDE (cross_val_predict, all {wide['N']})", ""] + top10(wide["scores"])
    L.extend(side_by_side(ln, rn, lw=52))
    add("")

    # ---- sensitivity ----
    add("-" * 100)
    add("5. SIZE-FLOOR SENSITIVITY (wide population; only the floor changes)")
    add("-" * 100)
    add(sens.round(5).to_string())
    add("")

    # ---- RQ3 ----
    add("-" * 100)
    add("6. RQ3 HAND-OFF — leakage-free scores for the listed banks")
    add("-" * 100)
    add(f"  trained on the wide population EXCLUDING all listed banks "
        f"(N={len(wide_panel) - int(wide_panel.is_listed.sum())}), "
        f"model = {rq3_better}")
    add(f"  its own {N_SPLITS}-fold CV on that training set:  "
        f"oos_RMSE={rq3_metrics.loc[rq3_better,'oos_RMSE']:.5f}  "
        f"oos_R2={rq3_metrics.loc[rq3_better,'oos_R2']:.4f}")
    add(f"  transferred to {len(rq3_scores_df)} listed banks: "
        f"RMSE={rq3_transfer['RMSE']:.5f}  R2={rq3_transfer['R2']:.4f}")
    add("")
    n_flagged = int(rq3_scores_df.is_trust_or_specialized.sum())
    add(f"  ({n_flagged} of the {len(rq3_scores_df)} listed scored banks carry "
        f"is_trust_or_specialized=True — kept, not dropped; '* trust' below)")
    add("  top-10 listed banks by leakage-free vulnerability_score:")
    for i, (_, r) in enumerate(rq3_scores_df.head(10).iterrows()):
        mark = " * trust" if r.get("is_trust_or_specialized", False) else ""
        add(f"    {i+1:>2}. {r['name'][:40]:<40} permno={int(r.permno):<6} "
            f"{r.vulnerability_score:>7.4f}{mark}")
    add("")
    add("  rank agreement with the narrow scores (same 277 banks):")
    j = rq3_scores_df.merge(narrow["scores"], on="bank_IDRSSD",
                            suffixes=("_wide", "_narrow"))
    add(f"    Spearman rho = "
        f"{j.vulnerability_score_wide.corr(j.vulnerability_score_narrow, method='spearman'):.4f}"
        f"   Pearson r = "
        f"{j.vulnerability_score_wide.corr(j.vulnerability_score_narrow):.4f}")
    ov = len(set(rq3_scores_df.head(10).bank_IDRSSD)
             & set(narrow["scores"].head(10).bank_IDRSSD))
    add(f"    top-10 overlap: {ov}/10 banks")
    add("")
    add("=" * 100)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    np.random.seed(SEED)

    print("[1/6] preserving the narrow outputs as *_narrow copies ...")
    for src, dst in NARROW_COPIES.items():
        if not src.exists():
            raise FileNotFoundError(f"missing narrow output {src} — run run_rq1.py first")
        shutil.copy2(src, dst)
        print(f"      {src.name} -> {dst.name}")

    print("\n[2/6] re-running the narrow variant (same code, for comparison) ...")
    n_meta, n_X, n_y = load_variant(NARROW_PANEL)
    tmp_png = Path(tempfile.gettempdir()) / "rq1_shap_narrow_recompute.png"
    narrow = run_variant("NARROW", n_meta, n_X, n_y, tmp_png)

    print("\n[3/7] running the wide variant (charter-filtered, 953) ...")
    w_meta, w_X, w_y = load_variant(WIDE_PANEL)
    wide = run_variant("WIDE", w_meta, w_X, w_y, OUT_SHAP_WIDE)
    OUT_RESULTS_WIDE.write_text(wide["report"] + "\n")
    print(f"      wrote {OUT_RESULTS_WIDE}")
    print(f"      wrote {OUT_SHAP_WIDE}")

    print("\n[4/7] re-running the PRE-FILTER wide (all charters, 963) for the "
          "charter-effect comparison ...")
    a_meta, a_X, a_y = load_variant(WIDE_ALLCHARTERS_PANEL)
    tmp_png_all = Path(tempfile.gettempdir()) / "rq1_shap_wide_allcharters.png"
    allcharters = run_variant("WIDE-ALL", a_meta, a_X, a_y, tmp_png_all)

    # trust-excluded top-10: same filtered wide scores, but rank with the flagged
    # trust/custody banks removed — answers "does the top-10 change once trust cos
    # are removed" WITHOUT re-training (they stay in the model; only the ranking
    # view drops them). w_meta carries is_trust_or_specialized from the panel.
    wide_scores_flagged = wide["scores"].merge(
        w_meta[["bank_IDRSSD", "is_trust_or_specialized"]],
        on="bank_IDRSSD", how="left")
    wide_scores_flagged["is_trust_or_specialized"] = \
        wide_scores_flagged["is_trust_or_specialized"].fillna(False)
    wide_trust_excluded = wide_scores_flagged[
        ~wide_scores_flagged.is_trust_or_specialized].reset_index(drop=True)

    print("\n[5/7] size-floor sensitivity ...")
    sens = sensitivity_table(w_meta)

    print("\n[6/7] RQ3 leakage-free scores (train wide-minus-listed) ...")
    scores_w, rq3_m, rq3_b, rq3_t = rq3_scores(w_meta, n_meta, n_X)
    scores_w.to_csv(OUT_SCORES_WIDE, index=False)
    print(f"      wrote {OUT_SCORES_WIDE}")

    print("\n[7/9] comparison document ...")
    text = build_comparison(narrow, wide, sens, rq3_m, rq3_b, rq3_t, scores_w,
                            w_meta, allcharters, wide_scores_flagged,
                            wide_trust_excluded)
    OUT_COMPARISON.write_text(text + "\n")
    print(f"      wrote {OUT_COMPARISON}\n")

    print("[8/9] RUN 3 — feature pruning, repeated CV, in-CV tuning ...")
    r3_wide, store_w = run3_report(w_X, w_y, wide["ols"], "WIDE, N=953")
    r3_narrow, store_n = run3_report(n_X, n_y, narrow["ols"], "NARROW, N=277")
    OUT_H1B_ROBUST.write_text(r3_wide + "\n\n" + r3_narrow + "\n")
    print(f"      wrote {OUT_H1B_ROBUST}")

    print("\n[9/9] RUN 4 — alternative treatments of the three failed banks ...")
    wide_panel_raw = pd.read_csv(WIDE_PANEL)
    wide_panel_raw = wide_panel_raw[
        ~wide_panel_raw.bank_IDRSSD.isin(EXCLUDE_IDRSSD)]
    narrow_panel_raw = pd.read_csv(NARROW_PANEL)
    narrow_panel_raw = narrow_panel_raw[
        ~narrow_panel_raw.bank_IDRSSD.isin(EXCLUDE_IDRSSD)]
    r4 = (run4_report(wide_panel_raw, "WIDE, N=953") + "\n\n"
          + run4_report(narrow_panel_raw, "NARROW, N=277"))
    OUT_FAILED_ROBUST.write_text(r4 + "\n")
    print(f"      wrote {OUT_FAILED_ROBUST}")

    append_robustness_pointer()
    print(f"      appended RUN 3 / RUN 4 pointers to {OUT_COMPARISON.name}")


if __name__ == "__main__":
    main()
