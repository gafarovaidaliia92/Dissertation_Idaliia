"""
rq1_build_panel.py — RQ1, step 1: construct the bank-characteristics panels.

Builds both estimation populations from the raw Call Report ZIPs:

    NARROW  277 listed banks matched to CRSP (278 minus one merger exit). This is
            the population RQ2/RQ3 can use, because it needs stock prices.
    WIDE    953 filers with assets > $1bn and deposit_reliance >= 0.50, after the
            charter filter drops 10 credit-card banks and flags 23 trust/custody
            institutions (kept, not dropped). 265 of them are listed.

Also handles: FDIC charter classification (BKCLASS / SPECGRP), the charter flags,
and the failed-bank censoring — SVB, Signature and First Republic enter with
dep_growth = -1.0. That censoring is a coding choice, not an observation, and
rq1_robustness.py re-runs everything under four alternative treatments of it.

Outputs   data/processed/panel_2022Q4.csv                 (narrow, 277)
          data/processed/panel_2022Q4_wide.csv            (wide, 953)
          data/processed/panel_2022Q4_wide_allcharters.csv (pre-filter, 963)

IDEMPOTENT. The panels are inputs to every frozen result in the project, so this
script does not rebuild them when they already exist. Pass --force to rebuild
from the raw ZIPs.

The extraction logic itself lives in the internal modules _panel_narrow.py,
_panel_wide.py, _charter_flags.py and _fdic_class.py, which are unchanged.
"""

from __future__ import annotations

import sys

import config as C


def main() -> None:
    force = "--force" in sys.argv
    have = all(p.exists() for p in
               (C.PANEL_NARROW, C.PANEL_WIDE, C.PANEL_WIDE_ALL))
    if have and not force:
        for p in (C.PANEL_NARROW, C.PANEL_WIDE, C.PANEL_WIDE_ALL):
            print(f"[rq1_build_panel] present, not rebuilt: {p.name}")
        print("[rq1_build_panel] pass --force to rebuild from the raw Call Reports")
        return

    import _panel_narrow
    import _panel_wide
    print("[rq1_build_panel] building the narrow panel ...")
    _panel_narrow.main()
    print("[rq1_build_panel] building the wide panels (charter filter included) ...")
    _panel_wide.main()
    print("[rq1_build_panel] done")


if __name__ == "__main__":
    main()
