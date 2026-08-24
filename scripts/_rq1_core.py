"""
run_rq1.py — Model 1 (RQ1): supervised ML predicting the 2023 deposit outflow.

RQ1: which bank characteristics predict sensitivity to fast, CBDC-style deposit
outflows, and can ML identify the most exposed banks?
  H1a — reliance on uninsured deposits, unrealised securities losses, thin
        liquidity/capital predict the 2023 outflow (linear baseline + SHAP signs).
  H1b — tree-based ML predicts outflow sensitivity better than a linear baseline
        out-of-sample (Section 3.4).

Reads ONLY data/processed/panel_2022Q4.csv; writes ONLY to data/processed/.
Outputs: vulnerability_scores.csv, rq1_shap_summary.png, rq1_results.txt.

Design notes:
  * The 3 censored failures (SVB, Signature, First Republic; dep_growth = -1.0)
    STAY in training — dropping them reintroduces survivorship bias (Section 3.2).
  * The one merger-exit bank (IDRSSD 119528, null dep_growth) is excluded: it left
    by acquisition, not by run. Final N = 277.
  * Out-of-sample predictions use cross_val_predict so each bank is scored by a fold
    that did not train on it (no leakage) — this feeds the RQ3 vulnerability score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: only saving PNGs, no display
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import shap

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
SEED = 42
PROC = Path("data/processed")
PANEL_CSV = PROC / "panel_2022Q4.csv"
OUT_SCORES = PROC / "vulnerability_scores.csv"
OUT_SHAP_PNG = PROC / "rq1_shap_summary.png"
OUT_RESULTS = PROC / "rq1_results.txt"

OUTCOME = "dep_growth"
# All bank characteristics enter as features (Section 3.3). NB: `capital` is the
# tier-1 leverage ratio in PERCENT units (8.30 = 8.30%); the rest are fractions.
# Scale differences are irrelevant to trees and to OLS fit/R2; standardised betas
# below make the linear magnitudes comparable.
FEATURES = [
    "uninsured_share", "unrealised_losses", "deposit_reliance", "liquidity",
    "capital", "size", "ROA", "NPL_ratio", "equity_ratio", "int_inc_ratio",
]
EXCLUDE_IDRSSD = [119528]  # Farmers National Bank of Emlenton — merger exit, null y

N_SPLITS = 5  # 5-fold CV, fixed seed (Section 3.4 out-of-sample comparison)

# Regularised tree settings appropriate for a small N=277 sample (shallow trees,
# min-samples floors, bagging) to limit overfitting.
RF_PARAMS = dict(n_estimators=500, max_depth=4, min_samples_leaf=5,
                 max_features=0.6, random_state=SEED, n_jobs=-1)
GB_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=2,
                 min_samples_leaf=8, subsample=0.8, random_state=SEED)


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the panel, drop the merger-exit bank, return (meta, X, y). The 3
    censored failures are retained on purpose (Section 3.2)."""
    df = pd.read_csv(PANEL_CSV)
    df = df[~df["bank_IDRSSD"].isin(EXCLUDE_IDRSSD)].copy()
    df = df[df[OUTCOME].notna()].reset_index(drop=True)
    X = df[FEATURES].astype(float)
    y = df[OUTCOME].astype(float)
    assert not X.isna().any().any(), "features contain nulls — check panel build"
    return df, X, y


# --------------------------------------------------------------------------- #
#  1. Linear baseline (H1a)
# --------------------------------------------------------------------------- #
def fit_ols(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, float]:
    """OLS of dep_growth on all features. Returns a coefficient table (coef,
    t, p, standardised beta) and R^2. Betas come from a parallel fit on
    z-scored X and y so magnitudes are comparable across features (H1a)."""
    Xc = sm.add_constant(X)
    res = sm.OLS(y, Xc).fit()

    Xz = (X - X.mean()) / X.std(ddof=0)
    yz = (y - y.mean()) / y.std(ddof=0)
    res_std = sm.OLS(yz, sm.add_constant(Xz)).fit()

    table = pd.DataFrame({
        "coef": res.params,
        "t": res.tvalues,
        "p_value": res.pvalues,
        "beta_std": res_std.params,
    }).drop(index="const")
    table["sign"] = np.sign(table["coef"]).map({1.0: "+", -1.0: "-", 0.0: "0"})
    table = table.reindex(FEATURES)
    return table, float(res.rsquared)


# --------------------------------------------------------------------------- #
#  2. Tree-based models (H1b)
# --------------------------------------------------------------------------- #
def model_factory() -> dict:
    """Fresh unfitted estimators keyed by name (cloned per CV fold)."""
    return {
        "OLS": LinearRegression(),
        "RandomForest": RandomForestRegressor(**RF_PARAMS),
        "GradientBoosting": GradientBoostingRegressor(**GB_PARAMS),
    }


# --------------------------------------------------------------------------- #
#  3. Out-of-sample comparison
# --------------------------------------------------------------------------- #
def cross_validate_models(X: pd.DataFrame, y: pd.Series
                          ) -> tuple[pd.DataFrame, dict]:
    """5-fold CV (fixed seed). For each model report out-of-sample RMSE/R^2
    (pooled cross_val_predict) and in-sample RMSE/R^2 (fit on all) so the
    overfitting gap is visible (Section 3.4). Returns metrics table and the
    dict of pooled OOS predictions."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows, oos = [], {}
    for name, model in model_factory().items():
        oos_pred = cross_val_predict(model, X, y, cv=kf)  # each bank scored by held-out fold
        oos[name] = oos_pred
        oos_rmse = np.sqrt(mean_squared_error(y, oos_pred))
        oos_r2 = r2_score(y, oos_pred)
        model.fit(X, y)                                    # in-sample reference
        in_pred = model.predict(X)
        rows.append({"model": name,
                     "oos_RMSE": oos_rmse, "oos_R2": oos_r2,
                     "in_RMSE": np.sqrt(mean_squared_error(y, in_pred)),
                     "in_R2": r2_score(y, in_pred)})
    metrics = pd.DataFrame(rows).set_index("model")
    return metrics, oos


def pick_better_tree(metrics: pd.DataFrame) -> str:
    """The tree model with the lower out-of-sample RMSE (H1b winner)."""
    trees = metrics.loc[["RandomForest", "GradientBoosting"], "oos_RMSE"]
    return trees.idxmin()


# --------------------------------------------------------------------------- #
#  4. SHAP importances (H1a inside the ML model)
# --------------------------------------------------------------------------- #
def compute_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| importance per feature from the chosen tree, ranked, plus a
    saved beeswarm summary plot. Model must already be fitted on all data."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    imp = (pd.DataFrame({"feature": FEATURES,
                         "mean_abs_shap": np.abs(sv).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    plt.figure()
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    plt.savefig(OUT_SHAP_PNG, dpi=150, bbox_inches="tight")
    plt.close("all")
    return imp


# --------------------------------------------------------------------------- #
#  5. Vulnerability score for RQ3
# --------------------------------------------------------------------------- #
def make_vulnerability_scores(meta: pd.DataFrame, oos_pred: np.ndarray
                              ) -> pd.DataFrame:
    """Out-of-sample predicted outflow per bank (no leakage) and the oriented
    vulnerability score. Orientation, Section 3.6: dep_growth is negative for an
    outflow (delta1 < 0), so vulnerability_score = -pred_dep_growth makes HIGHER
    = more predicted outflow / more vulnerable."""
    out = pd.DataFrame({
        "bank_IDRSSD": meta["bank_IDRSSD"].values,
        "name": meta["name"].values,
        "permno": meta["permno"].values,
        "pred_dep_growth": oos_pred,                 # raw OOS prediction
        "vulnerability_score": -oos_pred,            # oriented: higher = more vulnerable
    }).sort_values("vulnerability_score", ascending=False).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
#  6. Diagnostics
# --------------------------------------------------------------------------- #
def run_diagnostics(metrics: pd.DataFrame, ols_table: pd.DataFrame, ols_r2: float,
                    shap_imp: pd.DataFrame, scores: pd.DataFrame,
                    better_tree: str) -> str:
    """Assemble the human-readable results block: CV metrics, OLS table, SHAP
    ranking, H1b verdict, top-10 vulnerable banks, overfitting flag."""
    lines = []
    add = lines.append
    add("=" * 74)
    add("RQ1 — MODEL 1 RESULTS")
    add("=" * 74)
    add(f"N = {len(scores)}  |  features = {len(FEATURES)}  |  seed = {SEED}"
        f"  |  {N_SPLITS}-fold CV")
    add(f"censored failures kept in training: SVB / Signature / First Republic")

    add("\n--- Out-of-sample comparison (H1b, Section 3.4) ---")
    add(metrics.round(5).to_string())
    ols_rmse = metrics.loc["OLS", "oos_RMSE"]
    best_rmse = metrics.loc[better_tree, "oos_RMSE"]
    h1b = best_rmse < ols_rmse
    add(f"\nbetter tree model: {better_tree} (oos_RMSE={best_rmse:.5f})")
    add(f"H1b {'SUPPORTED' if h1b else 'NOT supported'}: "
        f"{better_tree} oos_RMSE {best_rmse:.5f} "
        f"{'<' if h1b else '>='} OLS oos_RMSE {ols_rmse:.5f}")

    add("\n--- OLS coefficients (H1a) ---")
    add(f"R^2 (in-sample) = {ols_r2:.4f}")
    add(ols_table.round(5).to_string())

    add("\n--- SHAP mean|importance| ranking (chosen tree) ---")
    add(shap_imp.round(6).to_string(index=False))

    add("\n--- Top 10 banks by vulnerability_score (face validity) ---")
    add(scores.head(10)[["bank_IDRSSD", "name", "pred_dep_growth",
                         "vulnerability_score"]].round(4).to_string(index=False))

    # overfitting flag
    add("\n--- Overfitting check (in-sample vs OOS RMSE gap) ---")
    for m in metrics.index:
        gap = metrics.loc[m, "oos_RMSE"] - metrics.loc[m, "in_RMSE"]
        ratio = metrics.loc[m, "oos_RMSE"] / max(metrics.loc[m, "in_RMSE"], 1e-9)
        flag = "  <-- large gap" if ratio > 1.6 else ""
        add(f"  {m:18s} in={metrics.loc[m,'in_RMSE']:.5f}  oos={metrics.loc[m,'oos_RMSE']:.5f}"
            f"  ratio={ratio:.2f}{flag}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    np.random.seed(SEED)
    print("[1/7] loading panel ...")
    meta, X, y = load_panel()
    print(f"      N={len(y)} banks, {len(FEATURES)} features; "
          f"censored kept = {int(meta['censored'].sum())}")

    print("[2/7] OLS baseline (H1a) ...")
    ols_table, ols_r2 = fit_ols(X, y)

    print("[3/7] cross-validating OLS / RandomForest / GradientBoosting (H1b) ...")
    metrics, oos = cross_validate_models(X, y)
    better_tree = pick_better_tree(metrics)
    print(f"      better tree model: {better_tree}")

    print("[4/7] SHAP importances on chosen tree ...")
    chosen = model_factory()[better_tree]
    chosen.fit(X, y)
    shap_imp = compute_shap(chosen, X)

    print("[5/7] out-of-sample vulnerability scores (cross_val_predict) ...")
    scores = make_vulnerability_scores(meta, oos[better_tree])
    scores.to_csv(OUT_SCORES, index=False)
    print(f"      wrote {OUT_SCORES}")

    print("[6/7] diagnostics ...")
    report = run_diagnostics(metrics, ols_table, ols_r2, shap_imp, scores, better_tree)
    print(report)

    print("[7/7] saving results text ...")
    OUT_RESULTS.write_text(report + "\n")
    print(f"      wrote {OUT_RESULTS}")
    print(f"      wrote {OUT_SHAP_PNG}")


if __name__ == "__main__":
    main()
