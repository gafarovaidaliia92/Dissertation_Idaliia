"""rq1_build_panel.py: builds the two estimation samples from the Call Report
archives.

    narrow   277 listed banks matched to CRSP, the sample RQ2 and RQ3 use
    wide     953 filers with assets above $1bn and deposits of at least half of
             liabilities, after the FDIC charter screen drops 10 credit-card
             banks; 265 of them are listed

Also applies the FDIC charter classification and the failed-bank censoring:
Silicon Valley Bank, Signature and First Republic enter with dep_growth = -1.0.
That censoring is a coding choice rather than an observation, and
rq1_robustness.py re-estimates the model under four alternative treatments.

Outputs   data/processed/panel_2022Q4.csv                  narrow, 277
          data/processed/panel_2022Q4_wide.csv             wide, 953
          data/processed/panel_2022Q4_wide_allcharters.csv pre-screen, 963

The panels feed every result in the study, so the script does not rebuild them
when they already exist. Pass --force to rebuild from the raw archives. The
extraction itself is in _panel_narrow.py, _panel_wide.py, _charter_flags.py and
_fdic_class.py.
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
