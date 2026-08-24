"""
rq1_model.py — RQ1, step 2: the vulnerability model itself.

HYPOTHESES
  H1a  Uninsured deposits, unrealised losses and weak liquidity/capital predict
       the 2023Q1 deposit outflow.
       -> OLS coefficient table, narrow and wide, unrounded.
  H1b  Tree models beat OLS out of sample.
       -> OLS / RandomForest / GradientBoosting compared on OOS RMSE and R2.

H1b IS NOT ROBUST, and this script is built to show that rather than hide it.
Three passes, each stricter than the last:

  1. the original single 5-fold split (seed 42);
  2. REPEATED cross-validation — 30 independent shuffles, with the fold draw AND
     the estimators' random_state both varying, reporting the mean, the SD and
     the share of repeats in which RF actually beats OLS;
  3. NESTED tuning — hyper-parameters chosen by an inner grid search on the
     training part of each outer fold, so no configuration is ever selected on
     the data that scores it.

Each pass is run on two feature sets: all 10 features, and the OLS-significant
subset (read off the fitted table, never hard-coded).

What comes out of it: the RF-over-OLS margin depends on keeping the
INSIGNIFICANT features, which hurt OLS more than they hurt the forest. Prune
them and the advantage disappears (26/30 repeats -> 9/30 in the wide
population); tune fairly and the trees lose in every combination. Separately,
the old claim that every narrow model is worse than the mean turns out to be a
seed-42 artefact: averaged over 30 shuffles the narrow OOS R2 is slightly
POSITIVE (OLS +0.0143, RF +0.0335).

Outputs   data/processed/rq1_results.txt / _narrow.txt / _wide.txt
          data/processed/comparison_narrow_vs_wide.txt   (main RQ1 document)
          data/processed/rq1_h1b_robustness.txt          (passes 2 and 3)
          data/processed/rq1_shap_summary*.png
          data/processed/vulnerability_scores_wide.csv   (also written here;
                                                          see rq1_scores.py)

IDEMPOTENT and EXPENSIVE (~20 minutes: repeated CV plus nested tuning on two
populations). Existing outputs are not recomputed unless --force is passed. The
estimation code is unchanged and lives in _rq1_core.py and _rq1_wide_core.py.
"""

from __future__ import annotations

import sys

import config as C


def main() -> None:
    force = "--force" in sys.argv
    outputs = (C.RQ1_COMPARISON, C.RQ1_RESULTS_WIDE, C.RQ1_H1B_ROBUST)
    if all(p.exists() for p in outputs) and not force:
        for p in outputs:
            print(f"[rq1_model] present, not rebuilt: {p.name}")
        print("[rq1_model] pass --force to re-estimate (~20 min)")
        return
    import _rq1_wide_core as core
    core.main()


if __name__ == "__main__":
    main()
