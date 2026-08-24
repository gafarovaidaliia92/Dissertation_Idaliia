"""
rq1_robustness.py — RQ1, step 4: do the RQ1 findings depend on three banks?

SVB, Signature and First Republic enter the panel with dep_growth = -1.0. That
is a censoring convention, not a measurement, and it sits far outside the rest of
the distribution: in the wide population the worst SURVIVING bank is -0.7629 and
the 1st percentile is -0.1808. Three observations coded at an extreme value can
carry a coefficient on their own, so H1a and H1b are re-estimated under four
treatments:

    (a) baseline            censored at -1.0 (as published)
    (b) excluded            the three banks dropped entirely
    (c) survivor-min        recoded to the worst outflow actually observed among
                            surviving banks — keeps them in, keeps them ranked
                            worst, stops -1.0 acting as a 4x outlier
    (d) winsorised          the whole outflow tail clipped at the 1st percentile

WHAT IT FINDS, and it changes a conclusion:
  * WIDE — uninsured_share keeps its negative sign and stays significant under
    all four. The wide H1a result does NOT depend on the three failures.
  * NARROW — significance does not survive exclusion (-0.0594, p = 0.105, against
    a baseline of -0.2329, p < 0.001). The narrow H1a headline is carried in
    large part by SVB, Signature and First Republic. Lean on the wide result.

Outputs   data/processed/rq1_failed_bank_robustness.txt
Estimation code unchanged, in _rq1_wide_core.py.
"""

from __future__ import annotations

import sys

import pandas as pd

import config as C


def main() -> None:
    force = "--force" in sys.argv
    if C.RQ1_FAILED_ROBUST.exists() and not force:
        print(f"[rq1_robustness] present, not rebuilt: {C.RQ1_FAILED_ROBUST.name}")
        return
    import _rq1_wide_core as core
    wide = pd.read_csv(C.PANEL_WIDE)
    wide = wide[~wide.bank_IDRSSD.isin(C.EXCLUDE_IDRSSD)]
    narrow = pd.read_csv(C.PANEL_NARROW)
    narrow = narrow[~narrow.bank_IDRSSD.isin(C.EXCLUDE_IDRSSD)]
    text = (core.run4_report(wide, "WIDE, N=953") + "\n\n"
            + core.run4_report(narrow, "NARROW, N=277"))
    C.RQ1_FAILED_ROBUST.write_text(text + "\n")
    print(text)
    print(f"wrote {C.RQ1_FAILED_ROBUST}")


if __name__ == "__main__":
    main()
