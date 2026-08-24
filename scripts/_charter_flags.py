"""
charter_flags.py — classify each bank as an ordinary commercial/savings bank vs a
trust/custody/credit-card/servicer specialist, using FDIC classification plus two
balance-sheet signals from the Call Report.

Why a composite rule and not a single FDIC field:
  * FDIC BKCLASS puts EVERY 2022Q4 filer in a commercial (N/SM/NM) or savings
    (SB/SI/SL) class — there is no trust-charter or industrial-loan class left in
    the population, so BKCLASS alone drops nobody.
  * FDIC SPECGRP cleanly isolates CREDIT-CARD banks (SPECGRP==3) and nothing else
    we care about: the custody/trust banks (State Street, BNY Mellon, Bessemer,
    Deutsche Bank Trust, Sumitomo Mitsui Trust) all sit in the generic
    "All Other >$1B" bucket (SPECGRP==9) alongside SVB, First Republic and
    Goldman. SPECGRP cannot separate a custodian from SVB.
  * So the ONLY thing FDIC classifies cleanly enough to hard-drop is credit-card.
    Everything else that is "not really a deposit-funded commercial bank" is
    caught by transparent balance-sheet signals and FLAGGED, never silently
    dropped (belt-and-suspenders, per the task).

Signals (all reported, none hidden):
  A  loans_to_assets < LOAN_FLOOR and uninsured_share > UNINS_CEIL
       -> a bank that barely lends and is almost entirely uninsured wholesale
          deposits: custody / trust / servicer. Catches State Street, BNY Mellon,
          Bessemer, Sumitomo Mitsui Trust, Deutsche Bank Trust, Cenlar.
  B  fiduciary_ratio > FID_CEIL   (RC-T managed+non-managed+custody assets / assets)
       -> a bank whose off-balance-sheet fiduciary book dwarfs its balance sheet:
          catches Northern Trust and BNY Mellon, which DO lend enough to pass A.
  C  FDIC SPECGRP == 3            -> credit-card specialist.

is_trust_or_specialized = A or B or C.
is_creditcard           = C only  (the sole clean FDIC hard-drop for training).

CRITICAL: the three censored failures must never be flagged (they are real
commercial lenders — SVB loans/assets 0.67, uninsured 0.84; Signature 0.68;
First Republic 1.57). Verified in verify_flags().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _panel_narrow import get_numeric, read_schedule

RAW = Path("data/raw")
FDIC_CSV = RAW / "fdic" / "institutions_class.csv"

COMMERCIAL_CLASSES = {"N", "SM", "NM"}       # national + state commercial
SAVINGS_CLASSES = {"SB", "SI", "SL", "SA"}   # savings banks / associations
KEEP_CLASSES = COMMERCIAL_CLASSES | SAVINGS_CLASSES

# Signal thresholds (documented, deliberately conservative so ordinary banks and
# the failures are never caught).
LOAN_FLOOR = 0.40    # below this a "bank" is not primarily a lender
UNINS_CEIL = 0.55    # above this its deposits are near-entirely uninsured
FID_CEIL = 0.50      # fiduciary book > half the balance sheet -> trust/custody
CREDITCARD_SPECGRP = 3


def load_fdic() -> pd.DataFrame:
    if not FDIC_CSV.exists():
        raise FileNotFoundError(
            f"{FDIC_CSV} missing — run scripts/fetch_fdic_class.py first "
            f"(it STOPs rather than falling back to a name list).")
    f = pd.read_csv(FDIC_CSV)
    f["CERT"] = pd.to_numeric(f["CERT"], errors="coerce").astype("Int64")
    return f.rename(columns={"CERT": "fdic_cert"})


def _balance_sheet_signals(zip_path: Path, ids: set[int]) -> pd.DataFrame:
    """loans_to_assets and fiduciary_ratio for the given IDRSSDs, filer-aware."""
    rc = read_schedule(zip_path, "RC")
    rcci = read_schedule(zip_path, "RCCI")
    rct = read_schedule(zip_path, "RCT")

    def dom_or_cons(df, rcon, rcfd):
        return get_numeric(df, rcon).fillna(0) + get_numeric(df, rcfd).fillna(0)

    assets = pd.DataFrame({"bank_IDRSSD": rc["IDRSSD"],
                           "assets": dom_or_cons(rc, "RCON2170", "RCFD2170")})
    loans = pd.DataFrame({"bank_IDRSSD": rcci["IDRSSD"],
                          "loans": dom_or_cons(rcci, "RCON2122", "RCFD2122")})
    # RC-T fiduciary: managed (B868) + non-managed (B869) + custody/safekeeping (B871)
    fid = pd.DataFrame({"bank_IDRSSD": rct["IDRSSD"], "fiduciary": (
        dom_or_cons(rct, "RCONB868", "RCFDB868")
        + dom_or_cons(rct, "RCONB869", "RCFDB869")
        + dom_or_cons(rct, "RCONB871", "RCFDB871"))})

    m = assets.merge(loans, on="bank_IDRSSD", how="left") \
              .merge(fid, on="bank_IDRSSD", how="left")
    m = m[m.bank_IDRSSD.isin(ids)].copy()
    m["loans_to_assets"] = m["loans"] / m["assets"].replace(0, np.nan)
    m["fiduciary_ratio"] = m["fiduciary"].fillna(0) / m["assets"].replace(0, np.nan)
    return m[["bank_IDRSSD", "loans_to_assets", "fiduciary_ratio"]]


def classify(panel: pd.DataFrame, zip_path: Path,
             cert_col: str = "fdic_cert") -> pd.DataFrame:
    """
    Given a panel with columns [bank_IDRSSD, uninsured_share] and an FDIC cert
    column, return it with added columns:
        BKCLASS, SPECGRP, SPECGRPN, loans_to_assets, fiduciary_ratio,
        is_creditcard, is_trust_or_specialized, flag_reason.
    The panel's own rows are preserved (nothing dropped here — dropping is the
    caller's decision).
    """
    out = panel.copy()
    if cert_col not in out.columns:
        raise KeyError(f"classify() needs an FDIC cert column '{cert_col}'")

    fdic = load_fdic()[["fdic_cert", "BKCLASS", "SPECGRP", "SPECGRPN"]] \
        .rename(columns={"fdic_cert": "_fdic_cert"})
    out["_cert"] = pd.to_numeric(out[cert_col], errors="coerce").astype("Int64")
    out = out.merge(fdic, left_on="_cert", right_on="_fdic_cert", how="left")

    sig = _balance_sheet_signals(zip_path, set(out.bank_IDRSSD.astype(int)))
    out = out.merge(sig, on="bank_IDRSSD", how="left")

    a = (out["loans_to_assets"] < LOAN_FLOOR) & (out["uninsured_share"] > UNINS_CEIL)
    b = out["fiduciary_ratio"] > FID_CEIL
    c = out["SPECGRP"].eq(CREDITCARD_SPECGRP)

    out["is_creditcard"] = c.fillna(False)
    out["is_trust_or_specialized"] = (a | b | c).fillna(False)

    def reason(row_a, row_b, row_c):
        r = []
        if row_c:
            r.append("credit-card(SPECGRP3)")
        if row_a:
            r.append("low-loans+high-uninsured")
        if row_b:
            r.append("fiduciary>0.5")
        return "; ".join(r)

    out["flag_reason"] = [reason(x, y, z) for x, y, z in
                          zip(a.fillna(False), b.fillna(False), c.fillna(False))]
    return out.drop(columns=["_cert", "_fdic_cert"], errors="ignore")


def verify_flags(classified: pd.DataFrame, failure_ids: set[int]) -> list[str]:
    """Guard-rail checks. Returns a list of human-readable assertions/warnings."""
    msgs = []
    # 1. no censored failure may be flagged
    flagged_fail = classified[classified.bank_IDRSSD.isin(failure_ids)
                              & classified.is_trust_or_specialized]
    assert flagged_fail.empty, \
        f"FAILURE flagged as specialized: {list(flagged_fail.name)}"
    msgs.append(f"OK: none of the {len(failure_ids)} failures flagged")

    # 2. every kept bank is a commercial or savings class
    bad_class = classified[~classified.BKCLASS.isin(KEEP_CLASSES)
                           & classified.BKCLASS.notna()]
    if len(bad_class):
        msgs.append(f"WARN: {len(bad_class)} banks outside commercial/savings "
                    f"BKCLASS: {dict(bad_class.BKCLASS.value_counts())}")
    else:
        msgs.append("OK: all banks are commercial (N/SM/NM) or savings (SB/SI/SL)")
    return msgs
