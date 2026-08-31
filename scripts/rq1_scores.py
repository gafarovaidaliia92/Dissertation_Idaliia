"""rq1_scores.py: builds the leakage-free vulnerability scores that RQ3 consumes.

A listed bank's score must not come from a model that trained on that bank, so
the model is fitted on the wide population with every listed bank removed (688
banks) and then applied to the 277 listed ones. Training and scoring sets share
no observations.

Orientation, which every downstream consumer assumes:

    score = -1 x predicted 2023Q1 deposit growth

so a predicted outflow gives a positive score and higher means more vulnerable.
An assertion at the end of the script checks it.

Two estimators run under the identical protocol, which separates the effect of
the algorithm from the effect of using a fitted composite at all:

    random forest      -> vulnerability_scores_wide.csv (vulnerability_score)
    linear regression  -> vulnerability_scores_ols.csv  (score_ols)

The random forest file also carries is_trust_or_specialized, used by the
custody-bank diagnostic in rq3_measures.py.

The transferred forest explains R-squared = 0.0302 of the listed banks' outflow
and its scores have a standard deviation of 0.014, about 3.7 times narrower than
the in-population version. That is a weak measure, and the RQ3 results should be
read with it in mind.

Outputs   data/processed/vulnerability_scores_wide.csv
          data/processed/vulnerability_scores_ols.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import config as C
from _rq1_core import EXCLUDE_IDRSSD, FEATURES, OUTCOME


def _populations() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(C.PANEL_WIDE)
    wide = wide[~wide.bank_IDRSSD.isin(EXCLUDE_IDRSSD)]
    wide = wide[wide[OUTCOME].notna()]
    narrow = pd.read_csv(C.PANEL_NARROW)
    narrow = narrow[~narrow.bank_IDRSSD.isin(EXCLUDE_IDRSSD)]
    narrow = narrow[narrow[OUTCOME].notna()].reset_index(drop=True)
    return wide, narrow


def build_ols_score() -> dict:
    """The OLS twin of the RF score: same features, same 688/277 split."""
    wide, narrow = _populations()
    train = wide[~wide.is_listed.astype(bool)]
    model = LinearRegression().fit(train[FEATURES].astype(float),
                                   train[OUTCOME].astype(float))
    pred = model.predict(narrow[FEATURES].astype(float))
    pd.DataFrame({
        "bank_IDRSSD": narrow.bank_IDRSSD.values,
        "permno": narrow.permno.values,
        "pred_dep_growth_ols": pred,
        "score_ols": -pred,                 # higher = more vulnerable
    }).to_csv(C.SCORES_OLS, index=False)
    y = narrow[OUTCOME].astype(float).values
    return {"n_train": len(train), "n_apply": len(pred),
            "transfer_RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "transfer_R2": float(r2_score(y, pred))}


def main() -> dict:
    force = "--force" in sys.argv
    if not C.SCORES_WIDE.exists() or force:
        print("[rq1_scores] building the RF leakage-free score ...")
        import _rq1_wide_core as core
        _, n_X, _ = core.load_variant(C.PANEL_NARROW)
        w_meta, _, _ = core.load_variant(C.PANEL_WIDE)
        n_meta, _, _ = core.load_variant(C.PANEL_NARROW)
        scores, _, better, transfer = core.rq3_scores(w_meta, n_meta, n_X)
        scores.to_csv(C.SCORES_WIDE, index=False)
        print(f"[rq1_scores] RF model = {better}, transfer R2 = {transfer['R2']:.8f}")
    else:
        print(f"[rq1_scores] present, not rebuilt: {C.SCORES_WIDE.name}")

    info = build_ols_score()
    print(f"[rq1_scores] OLS score: trained on {info['n_train']}, applied to "
          f"{info['n_apply']}, transfer R2 = {info['transfer_R2']:.8f}")

    rf = pd.read_csv(C.SCORES_WIDE)
    assert (rf.vulnerability_score == -rf.pred_dep_growth).all(), \
        "RF score orientation broken: score must equal -pred_dep_growth"
    ols = pd.read_csv(C.SCORES_OLS)
    assert (ols.score_ols == -ols.pred_dep_growth_ols).all(), \
        "OLS score orientation broken"
    print("[rq1_scores] orientation check passed: higher = more vulnerable")
    return info


if __name__ == "__main__":
    main()
