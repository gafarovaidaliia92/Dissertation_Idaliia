"""rq1_model.py: estimates the RQ1 models and tests both hypotheses.

    H1a  uninsured deposits, unrealised losses and weak liquidity or capital
         predict the 2023Q1 outflow. Tested with the OLS coefficient table on
         both populations.
    H1b  tree models predict the outflow better than OLS out of sample. Tested
         with OLS, random forest and gradient boosting on out-of-sample RMSE
         and R-squared.

H1b is examined in three passes of increasing strictness, because a single split
is not enough to establish a difference this small:

    1. one 5-fold split at seed 42, the specification reported in Table 4;
    2. repeated cross-validation over 30 shuffles, varying both the fold draw
       and the estimators' random_state, reporting the mean, the standard
       deviation and the share of shuffles in which the forest wins;
    3. nested tuning, with hyper-parameters chosen by an inner grid search on
       the training part of each outer fold.

Each pass runs on two feature sets: all ten features, and the subset significant
in the OLS table, read off the fitted table rather than fixed in advance.

Outputs   data/processed/rq1_results.txt, _narrow.txt, _wide.txt
          data/processed/comparison_narrow_vs_wide.txt   main RQ1 document
          data/processed/rq1_h1b_robustness.txt          passes 2 and 3
          data/processed/rq1_shap_summary*.png
          data/processed/vulnerability_scores_wide.csv

Repeated cross-validation and nested tuning on two populations take roughly
twenty minutes, so existing outputs are not recomputed unless --force is passed.
The estimation code is in _rq1_core.py and _rq1_wide_core.py.
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
