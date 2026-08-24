"""
build_panel_wide.py — WIDE variant of Step 1: build the bank-characteristics
panel for the FULL population of 2022Q4 Call Report filers, not just the 278
listed banks that survive the RSSD->permno crosswalk.

Why: the narrow sample is conditioned on being publicly listed AND matchable to
CRSP. That is a selection on size and ownership structure, and it leaves N=277 —
too few observations for the tree models to have anything to learn. The wide
population drops the listing requirement and keeps every filer that is
economically comparable (a deposit-funded bank above a size floor), so the RQ1
models train on ~10x the data.

This file REUSES build_panel.py wholesale — the same MDRM codes, the same
filer-aware scope rule, the same variable formulas, the same dep_growth
definition. The ONLY thing that changes is which banks enter.

Reads ONLY from data/raw/ (incl. data/raw/fdic/fdic_failures.csv) and
data/frozen/sample_banks.csv (for the is_listed flag); writes ONLY
data/processed/panel_2022Q4_wide.csv. The narrow panel is never touched.

Population definition (constants at the top of this file):
    all 2022Q4 filers
      -> total_assets > $1bn          (ASSET_FLOOR, in $000 like the raw data)
      -> deposit_reliance >= 0.50     (DEP_RELIANCE_FLOOR)
      -> complete, finite features
      -> dep_growth observed (or censored)
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from _panel_narrow import (
    PROC,
    ZIP_2022Q4,
    assemble_raw_fields,
    build_variables,
    compute_dep_growth,
    load_filer_types,
)
from _charter_flags import classify, verify_flags

# --------------------------------------------------------------------------- #
#  Population constants
# --------------------------------------------------------------------------- #
# Call Report dollar amounts are in THOUSANDS, so $1bn = 1_000_000 here.
# Below this floor the balance sheet is a different animal (single-branch banks,
# no wholesale funding, no securities book worth speaking of) and the 2023 run
# never touched them.
ASSET_FLOOR = 1_000_000            # $1bn, expressed in $000

# Keep deposit-funded banks only. Below 0.5 the institution is funded by
# something other than deposits (trust-only charters, bankers' banks, credit-card
# funding vehicles) and a "deposit outflow" is not a meaningful stress for it.
DEP_RELIANCE_FLOOR = 0.50

# Size floors for the sensitivity table in run_rq1_wide.py.
SENSITIVITY_FLOORS = [1_000_000, 5_000_000, 10_000_000]   # $1bn / $5bn / $10bn

FAILURE_YEAR = 2023

FAILURES_CSV = Path("data/raw/fdic") / "fdic_failures.csv"
NARROW_SAMPLE = PROC.parent / "frozen" / "sample_banks.csv"
OUT_CSV = PROC / "panel_2022Q4_wide.csv"

# Same feature list the models use — completeness is enforced on exactly these.
FEATURE_COLS = [
    "uninsured_share", "unrealised_losses", "deposit_reliance", "liquidity",
    "capital", "size", "ROA", "NPL_ratio", "equity_ratio", "int_inc_ratio",
]


# --------------------------------------------------------------------------- #
#  Extra loaders (the narrow pipeline got name/failed from sample_banks.csv,
#  which does not exist for the full population)
# --------------------------------------------------------------------------- #
def load_names(zip_path: Path) -> pd.DataFrame:
    """(IDRSSD, name) from the POR directory inside a Call Report zip."""
    zf = zipfile.ZipFile(zip_path)
    por = sorted(n for n in zf.namelist() if "Bulk POR" in n)[0]
    df = pd.read_csv(zf.open(por), sep="\t", dtype=str, encoding="latin-1")
    df.columns = [c.strip().strip('"') for c in df.columns]
    idc = next(c for c in df.columns if c.upper() == "IDRSSD")
    nmc = next(c for c in df.columns if "Institution Name" in c)
    out = df[[idc, nmc]].rename(columns={idc: "bank_IDRSSD", nmc: "name"})
    out["bank_IDRSSD"] = out["bank_IDRSSD"].astype("int64")
    return out


def load_failures(zip_path: Path, year: int = FAILURE_YEAR) -> pd.DataFrame:
    """
    (bank_IDRSSD, failed) for banks that failed in `year`, bridged from the FDIC
    failed-bank list via FDIC certificate -> IDRSSD in the POR directory.

    The saved CSV has a non-breaking-space artefact in every header, hence the
    \\xa0 strip. Closing Date is 'DD-Mon-YY'.
    """
    f = pd.read_csv(FAILURES_CSV, encoding="latin-1")
    f.columns = [c.strip().replace("\xa0", "") for c in f.columns]
    f["Closing Date"] = pd.to_datetime(f["Closing Date"], format="%d-%b-%y",
                                       errors="coerce")
    f = f[f["Closing Date"].dt.year == year].copy()
    f["Cert"] = pd.to_numeric(f["Cert"], errors="coerce")

    por = load_filer_types(zip_path)
    por["fdic_cert"] = pd.to_numeric(por["fdic_cert"], errors="coerce")
    m = f.merge(por[["IDRSSD", "fdic_cert"]], left_on="Cert",
                right_on="fdic_cert", how="inner")
    out = pd.DataFrame({"bank_IDRSSD": m["IDRSSD"].astype("int64"), "failed": True})
    return out.drop_duplicates("bank_IDRSSD")


# --------------------------------------------------------------------------- #
#  Funnel bookkeeping
# --------------------------------------------------------------------------- #
class Funnel:
    """Records the count surviving each filter so the population is auditable."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, int, int, int]] = []
        self._prev: int | None = None

    def step(self, label: str, n: int, n_failed: int) -> None:
        dropped = 0 if self._prev is None else self._prev - n
        self.rows.append((label, n, dropped, n_failed))
        self._prev = n

    def render(self) -> str:
        w = max(len(r[0]) for r in self.rows)
        lines = [f"{'stage':<{w}}  {'N':>6}  {'dropped':>8}  {'failures':>8}",
                 "-" * (w + 28)]
        for label, n, dropped, nf in self.rows:
            d = "" if dropped == 0 and label.startswith("all ") else f"-{dropped}"
            lines.append(f"{label:<{w}}  {n:>6}  {d:>8}  {nf:>8}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    print("[1/7] enumerating all 2022Q4 filers ...")
    por = load_filer_types(ZIP_2022Q4)
    ids = set(por.IDRSSD.astype(int))
    print(f"      {len(ids)} filers  |  mix: {por.filing_type.value_counts().to_dict()}")

    print("[2/7] assembling filer-aware raw fields (full population) ...")
    wide_raw = assemble_raw_fields(ZIP_2022Q4, ids)

    print("[3/7] constructing Section 3.3 variables ...")
    panel = build_variables(wide_raw)
    panel = panel.merge(load_names(ZIP_2022Q4), on="bank_IDRSSD", how="left")

    print("[4/7] computing dep_growth (2022Q4 -> 2023Q1) ...")
    dep_q1 = compute_dep_growth(ids)
    panel = panel.merge(dep_q1, on="bank_IDRSSD", how="left")
    panel["dep_growth"] = ((panel["dep_2023Q1"] - panel["total_deposits"])
                           / panel["total_deposits"])

    print("[5/7] flagging + censoring 2023 failures ...")
    fails = load_failures(ZIP_2022Q4)
    panel = panel.merge(fails, on="bank_IDRSSD", how="left")
    panel["failed"] = panel["failed"].fillna(False).astype(bool)
    # Same rule as the narrow pipeline: a failed bank has no post-run deposits to
    # measure, so the outcome is right-censored at -1.0 and flagged. The flag is
    # what downstream code must read, never the bare number.
    #
    # WARNING — coupling with ASSET_FLOOR: load_failures() returns ALL five 2023
    # failures, but only the three March-May ones (SVB, Signature, First Republic)
    # exceed the $1bn floor and survive into the modelled panel. The other two —
    # Heartland Tri-State ($139m, failed Jul 2023) and Citizens Bank of Sac City
    # ($66m, Nov 2023) — filed a normal 2023Q1 report with REAL deposits and only
    # failed months later. Censoring them at -1.0 would be wrong for a
    # 2022Q4->2023Q1 outcome. Today the $1bn floor removes them before it matters.
    # If ASSET_FLOOR is ever lowered below their size, restrict censoring to the
    # three March-May failures explicitly — do NOT censor Heartland / Citizens.
    panel["censored"] = panel["failed"]
    panel.loc[panel["censored"], "dep_growth"] = -1.0

    # listed flag — lets run_rq1_wide.py hold the narrow 278 out for RQ3
    listed = pd.read_csv(NARROW_SAMPLE)[["bank_IDRSSD", "permno"]]
    panel = panel.merge(listed, on="bank_IDRSSD", how="left")
    panel["is_listed"] = panel["permno"].notna()

    # FDIC certificate — the join key for charter classification (added below)
    filers = load_filer_types(ZIP_2022Q4)
    filers["fdic_cert"] = pd.to_numeric(filers["fdic_cert"], errors="coerce")
    panel = panel.merge(filers[["IDRSSD", "fdic_cert"]],
                        left_on="bank_IDRSSD", right_on="IDRSSD", how="left") \
                 .drop(columns=["IDRSSD"], errors="ignore")

    print("[6/7] applying population filters ...")
    # non-finite values (division by a zero balance-sheet item) are treated as
    # missing so the completeness filter catches them
    panel[FEATURE_COLS] = panel[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)

    fn = Funnel()
    fn.step("all 2022Q4 filers", len(panel), int(panel.failed.sum()))

    panel = panel[panel["total_assets"] > ASSET_FLOOR]
    fn.step(f"total_assets > ${ASSET_FLOOR/1e6:.0f}bn", len(panel),
            int(panel.failed.sum()))

    panel = panel[panel["deposit_reliance"] >= DEP_RELIANCE_FLOOR]
    fn.step(f"deposit_reliance >= {DEP_RELIANCE_FLOOR}", len(panel),
            int(panel.failed.sum()))

    panel = panel[panel[FEATURE_COLS].notna().all(axis=1)]
    fn.step("complete + finite features", len(panel), int(panel.failed.sum()))

    # merger exits leave no 2023Q1 filing -> null outcome. Censored failures were
    # set to -1.0 above and survive this filter by construction.
    panel = panel[panel["dep_growth"].notna()]
    fn.step("dep_growth observed or censored", len(panel), int(panel.failed.sum()))

    panel = panel.reset_index(drop=True)

    # ----------------------------------------------------------------------- #
    #  Charter-type filter (training population only).
    #
    #  FDIC classification cleanly isolates exactly ONE non-commercial deposit
    #  type in this population — credit-card banks (SPECGRP==3) — which are
    #  HARD-DROPPED here. Everything else that is "not really a deposit-funded
    #  commercial bank" (trust/custody houses, mortgage servicers) shares FDIC's
    #  generic "All Other >$1B" bucket with SVB and First Republic and cannot be
    #  separated by any single FDIC field, so it is FLAGGED (is_trust_or_
    #  specialized) and KEPT — never a silent drop. See charter_flags.py.
    #
    #  The three censored failures are verified NOT flagged before any drop.
    # ----------------------------------------------------------------------- #
    print("[6b/7] charter classification (FDIC BKCLASS/SPECGRP + balance sheet) ...")
    panel = classify(panel, ZIP_2022Q4)
    failure_ids = set(panel.loc[panel.failed, "bank_IDRSSD"].astype(int))
    for msg in verify_flags(panel, failure_ids):
        print(f"      {msg}")

    dropped = panel[panel.is_creditcard].copy()
    print(f"\n--- credit-card banks HARD-DROPPED from the training population "
          f"({len(dropped)}) ---")
    for _, r in dropped.sort_values("total_assets", ascending=False).iterrows():
        print(f"    {r['name'][:42]:42s}  {r.BKCLASS:>3} SPECGRP={int(r.SPECGRP)}  "
              f"assets=${r.total_assets/1e6:,.1f}bn   reason: {r.flag_reason}")

    flagged = panel[panel.is_trust_or_specialized & ~panel.is_creditcard].copy()
    print(f"\n--- trust / custody / servicer FLAGGED but KEPT "
          f"({len(flagged)}; is_trust_or_specialized=True) ---")
    for _, r in flagged.sort_values("total_assets", ascending=False).iterrows():
        keep_note = " [LISTED->RQ3]" if r.is_listed else ""
        print(f"    {r['name'][:42]:42s}  {r.BKCLASS:>3} SPECGRP={int(r.SPECGRP)}  "
              f"L/A={r.loans_to_assets:.2f} unins={r.uninsured_share:.2f} "
              f"fid={r.fiduciary_ratio:.2f}   {r.flag_reason}{keep_note}")

    panel = panel[~panel.is_creditcard].reset_index(drop=True)
    fn.step("charter filter (credit-card dropped)", len(panel),
            int(panel.failed.sum()))

    print("\n--- WIDE population funnel ---")
    print(fn.render())

    print(f"\nlisted banks retained (overlap with the narrow 278): "
          f"{int(panel.is_listed.sum())}")
    print("failures retained:")
    for _, r in panel[panel.failed].iterrows():
        print(f"    {r.bank_IDRSSD}  {r['name'][:44]:44s} "
              f"assets=${r.total_assets/1e6:,.1f}bn")

    print("\n--- size-floor sensitivity (population sizes only; model metrics "
          "come from run_rq1_wide.py) ---")
    for floor in SENSITIVITY_FLOORS:
        sub = panel[panel["total_assets"] > floor]
        print(f"    > ${floor/1e6:>2.0f}bn : N={len(sub):>5}  "
              f"listed={int(sub.is_listed.sum()):>4}  "
              f"failures={int(sub.failed.sum())}")

    dg = panel.loc[~panel.censored, "dep_growth"]
    print("\n--- dep_growth (excluding censored failures) ---")
    print(f"    min={dg.min():.4f}  p25={dg.quantile(.25):.4f}  "
          f"median={dg.median():.4f}  p75={dg.quantile(.75):.4f}  max={dg.max():.4f}")
    print(f"    sd={dg.std():.4f}   banks below -0.10: {(dg < -0.10).sum()}")

    print(f"\ntrust/custody/servicer flagged & kept in the saved panel: "
          f"{int(panel.is_trust_or_specialized.sum())}")

    print("\n[7/7] saving ...")
    cols = ["bank_IDRSSD", "name", "permno", "is_listed", "filing_type",
            "fdic_cert", "BKCLASS", "SPECGRP",
            "total_assets", "total_deposits", "total_liabilities", "total_equity",
            *FEATURE_COLS, "loans_to_assets", "fiduciary_ratio",
            "is_creditcard", "is_trust_or_specialized", "flag_reason",
            "dep_2023Q1", "dep_growth", "censored", "failed"]
    PROC.mkdir(parents=True, exist_ok=True)
    panel[cols].to_csv(OUT_CSV, index=False)
    print(f"      wrote {OUT_CSV} ({len(panel)} rows, {len(cols)} cols)")


if __name__ == "__main__":
    main()
