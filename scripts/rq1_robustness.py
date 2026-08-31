"""rq1_robustness.py: re-estimates RQ1 under four treatments of the three banks
that failed in 2023.

Silicon Valley Bank, Signature and First Republic enter the panel with
dep_growth = -1.0. That is a censoring convention rather than a measurement, and
it sits well outside the rest of the distribution: in the wide population the
worst surviving bank is -0.7629 and the first percentile is -0.1808. Three
observations at an extreme value can carry a coefficient on their own, so H1a is
re-estimated four ways:

    (a) baseline       censored at -1.0, as published
    (b) excluded       the three banks dropped
    (c) survivor-min   recoded to the worst outflow observed among surviving
                       banks, which keeps them ranked worst without letting
                       -1.0 act as an outlier four times the size
    (d) winsorised     the outflow tail clipped at the first percentile

In the wide population uninsured_share keeps its sign and its significance under
all four. In the narrow population significance does not survive exclusion
(-0.0594, p = 0.105, against a baseline of -0.2329, p < 0.001), which is why the
wide result carries the headline.

Output    data/processed/rq1_failed_bank_robustness.txt
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
