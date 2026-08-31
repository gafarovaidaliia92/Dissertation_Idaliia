"""
build_panel.py — Step 1 of the empirical analysis: build the bank-characteristics
panel for the 278-bank sample (extraction + variable construction + cleaning).

Reads ONLY from data/raw/ and data/frozen/sample_banks.csv; writes ONLY to
data/processed/panel_2022Q4.csv. Re-runnable with no manual edits between runs.

Methodology anchor: the panel operationalises the Section 3.3 variables of the
dissertation. Predictors are measured at 2022Q4 (the quarter before the run);
the outcome dep_growth is 2022Q4 -> 2023Q1.

FILER-AWARE SCOPE (README trap #2: "scope is chosen per filer, never per column").
The FFIEC bulk files merge three report forms:
    FFIEC 031  -> consolidated, uses RCFD / RCFA / RCFN   (34 banks, the largest)
    FFIEC 041  -> domestic,     uses RCON / RCOA          (199 banks)
    FFIEC 051  -> domestic,     uses RCON / RCOA          (45 banks)
For a 031 filer the domestic RCON balance-sheet cells are EMPTY, so every
balance-sheet concept is taken as RCFD/RCFA for 031 and RCON/RCOA for 041/051,
consistently for that bank. Exceptions, per spec:
  * RC-O Memorandum item 1 (F047/F048/F051/F052) is domestic-office-only on ALL
    forms -> always RCON.
  * Income statement (RIAD*) is single-scope -> always RIAD.
  * Total deposits (consolidated) = RCON2200 (domestic) + RCFN2200 (foreign);
    RCFN2200 is empty for 041/051 so it contributes 0 there.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
RAW = Path("data/raw")
PROC = Path("data/processed")
SAMPLE_CSV = RAW.parent / "frozen" / "sample_banks.csv"
MDRM_CSV = RAW / "mdrm" / "MDRM_CSV.csv"
ZIP_2022Q4 = RAW / "call_12312022.zip"
ZIP_2023Q1 = RAW / "call_03312023.zip"
OUT_CSV = PROC / "panel_2022Q4.csv"

# --------------------------------------------------------------------------- #
#  MDRM code dictionary — single labelled place to find / change every code.
#  Each concept: schedule + {rcon: domestic code (041/051), cons: consolidated
#  code for 031}. Where a concept is single-scope, rcon == cons.
# --------------------------------------------------------------------------- #
CODES: dict[str, dict] = {
    # ---- Schedule RC (balance sheet) ----
    "total_assets":      {"sched": "RC",   "rcon": "RCON2170", "cons": "RCFD2170"},
    "total_liabilities": {"sched": "RC",   "rcon": "RCON2948", "cons": "RCFD2948"},
    "total_equity":      {"sched": "RC",   "rcon": "RCON3210", "cons": "RCFD3210"},
    "dep_domestic":      {"sched": "RC",   "rcon": "RCON2200", "cons": "RCON2200"},  # RCON on both forms
    "dep_foreign":       {"sched": "RC",   "rcon": None,       "cons": "RCFN2200"},  # 031 only
    "cash_noninterest":  {"sched": "RC",   "rcon": "RCON0081", "cons": "RCFD0081"},
    "cash_interest":     {"sched": "RC",   "rcon": "RCON0071", "cons": "RCFD0071"},
    # ---- Schedule RC-B (securities) ----
    "htm_amortized":     {"sched": "RCB",  "rcon": "RCON1754", "cons": "RCFD1754"},
    "htm_fairvalue":     {"sched": "RCB",  "rcon": "RCON1771", "cons": "RCFD1771"},
    "afs_amortized":     {"sched": "RCB",  "rcon": "RCON1772", "cons": "RCFD1772"},
    "afs_fairvalue":     {"sched": "RCB",  "rcon": "RCON1773", "cons": "RCFD1773"},
    "pledged_secs":      {"sched": "RCB",  "rcon": "RCON0416", "cons": "RCFD0416"},
    # ---- Schedule RC-O Memorandum item 1 (domestic-office only, always RCON) ----
    "unins_amt_gt":      {"sched": "RCO",  "rcon": "RCONF051", "cons": "RCONF051"},  # amount > $250k, ordinary
    "unins_num_gt":      {"sched": "RCO",  "rcon": "RCONF052", "cons": "RCONF052"},  # number > $250k, ordinary
    "unins_ret_amt_gt":  {"sched": "RCO",  "rcon": "RCONF047", "cons": "RCONF047"},  # amount > $250k, retirement
    "unins_ret_num_gt":  {"sched": "RCO",  "rcon": "RCONF048", "cons": "RCONF048"},  # number > $250k, retirement
    # ---- Schedule RC-R Part I (regulatory capital ratio, percent) ----
    "tier1_leverage":    {"sched": "RCRI", "rcon": "RCOA7204", "cons": "RCFA7204"},
    # ---- Schedule RI (income statement, single-scope RIAD) ----
    "net_income":        {"sched": "RI",   "rcon": "RIAD4340", "cons": "RIAD4340"},
    "interest_income":   {"sched": "RI",   "rcon": "RIAD4107", "cons": "RIAD4107"},
    # ---- Schedule RC-N (past due / nonaccrual) ----
    "nonaccrual":        {"sched": "RCN",  "rcon": "RCON1403", "cons": "RCFD1403"},
    "pastdue90":         {"sched": "RCN",  "rcon": "RCON1407", "cons": "RCFD1407"},
    # ---- Schedule RC-C Part I (total loans, NPL denominator) ----
    "total_loans":       {"sched": "RCCI", "rcon": "RCON2122", "cons": "RCFD2122"},
}

# uninsured-amount threshold per account (RC-O M.1 methodology, eq. 3.2).
# Call Report DOLLAR amounts (F051, F047) are reported in THOUSANDS of dollars,
# while F052, F048 are account COUNTS. So the $250,000 per-account limit must be
# expressed in $thousands (= 250) to match F051/F047 units. Using 250_000 here
# would subtract 250,000 * count and drive uninsured_share to ~ -100 (unit bug).
INSURANCE_LIMIT = 250  # $250,000 expressed in $thousands

# --------------------------------------------------------------------------- #
#  0. Parsing helpers
# --------------------------------------------------------------------------- #
def read_schedule(zip_path: Path, schedule: str) -> pd.DataFrame:
    """
    Open a call_MMDDYYYY.zip and read a named FFIEC schedule (tab-delimited .txt),
    handling the DOUBLE-HEADER: row 1 = MDRM codes (kept as the header), row 2 =
    plain-English descriptions (dropped). Multi-part schedules (columns split
    across "(1 of N)" files) are joined on IDRSSD. Returns a str dataframe keyed
    on a clean-integer IDRSSD.
    """
    zf = zipfile.ZipFile(zip_path)
    prefix = f"FFIEC CDR Call Schedule {schedule} "
    members = sorted(n for n in zf.namelist() if n.startswith(prefix))
    if not members:
        raise FileNotFoundError(f"schedule {schedule} not found in {zip_path.name}")
    merged: pd.DataFrame | None = None
    for name in members:
        with zf.open(name) as fh:
            rows = list(io.TextIOWrapper(fh, encoding="latin-1"))
        codes = [c.strip().strip('"') for c in rows[0].rstrip("\n").split("\t")]
        data = [r.rstrip("\n").split("\t") for r in rows[2:]]  # drop row 2 (descriptions)
        part = pd.DataFrame(data, columns=codes)
        part = part.loc[:, ~part.columns.duplicated()]
        if merged is None:
            merged = part
        else:  # parts split columns -> join on IDRSSD
            new_cols = [c for c in part.columns if c not in merged.columns or c == "IDRSSD"]
            merged = merged.merge(part[new_cols], on="IDRSSD", how="outer")
    # verify IDRSSD parses as clean integers
    idr = pd.to_numeric(merged["IDRSSD"], errors="coerce")
    assert idr.notna().all(), f"{schedule}: non-integer IDRSSD found"
    merged["IDRSSD"] = idr.astype("int64")
    return merged


def load_filer_types(zip_path: Path) -> pd.DataFrame:
    """Read the POR directory -> (IDRSSD, filing_type, fdic_cert). Filing type is
    '031' / '041' / '051' and drives the filer-aware scope choice."""
    zf = zipfile.ZipFile(zip_path)
    por = sorted(n for n in zf.namelist() if "Bulk POR" in n)[0]
    df = pd.read_csv(zf.open(por), sep="\t", dtype=str, encoding="latin-1")
    df.columns = [c.strip().strip('"') for c in df.columns]
    idc = next(c for c in df.columns if c.upper() == "IDRSSD")
    ftc = next(c for c in df.columns if "Filing Type" in c)
    ctc = next(c for c in df.columns if "Certificate" in c)
    out = df[[idc, ftc, ctc]].rename(columns={idc: "IDRSSD", ftc: "filing_type", ctc: "fdic_cert"})
    out["IDRSSD"] = out["IDRSSD"].astype("int64")
    return out


def get_numeric(df: pd.DataFrame, code: str | None) -> pd.Series:
    """Numeric column for an MDRM code, or all-NaN if the code/column is absent.
    Strips '%' (RC-R ratios are filed as e.g. '8.2978%') and thousands commas."""
    if code is None or code not in df.columns:
        return pd.Series(np.nan, index=df.index)
    cleaned = (df[code].astype(str)
               .str.replace("%", "", regex=False)
               .str.replace(",", "", regex=False)
               .str.strip())
    return pd.to_numeric(cleaned, errors="coerce")


# --------------------------------------------------------------------------- #
#  1. Filer-aware field assembly
# --------------------------------------------------------------------------- #
def assemble_raw_fields(zip_path: Path, ids: set[int]) -> pd.DataFrame:
    """
    For the sample banks, pull every raw MDRM code named in CODES (both the RCON
    and the consolidated variant), keyed on IDRSSD, plus filing_type. One wide row
    per bank of raw values — filer-aware selection happens in pick_filer_field().
    """
    filers = load_filer_types(zip_path)
    wide = filers[filers.IDRSSD.isin(ids)].copy()
    # cache schedules so each is read once
    needed_scheds = {c["sched"] for c in CODES.values()}
    sched_dfs = {s: read_schedule(zip_path, s) for s in needed_scheds}
    for concept, spec in CODES.items():
        sdf = sched_dfs[spec["sched"]]
        keep = ["IDRSSD"] + [c for c in {spec["rcon"], spec["cons"]} if c]
        sub = sdf[[c for c in keep if c in sdf.columns]].drop_duplicates("IDRSSD")
        wide = wide.merge(sub, on="IDRSSD", how="left")
    wide = wide.loc[:, ~wide.columns.duplicated()]
    return wide


def pick_filer_field(df: pd.DataFrame, concept: str) -> pd.Series:
    """
    Return the populated value for a concept per bank (README trap #2):
      031 filer -> consolidated code (RCFD/RCFA/RCFN)
      041/051   -> domestic code (RCON/RCOA)
    """
    spec = CODES[concept]
    dom = get_numeric(df, spec["rcon"])
    con = get_numeric(df, spec["cons"])
    is_031 = df["filing_type"].eq("031")
    return np.where(is_031, con, dom)


def total_deposits(df: pd.DataFrame) -> pd.Series:
    """Consolidated total deposits = domestic RCON2200 + foreign RCFN2200.
    RCFN2200 is empty for 041/051 (contributes 0)."""
    dom = get_numeric(df, CODES["dep_domestic"]["cons"])          # RCON2200 on all forms
    frn = get_numeric(df, CODES["dep_foreign"]["cons"]).fillna(0)  # RCFN2200, 031 only
    return dom + frn


# --------------------------------------------------------------------------- #
#  2/3. Variable construction (Section 3.3)
# --------------------------------------------------------------------------- #
def build_variables(wide: pd.DataFrame) -> pd.DataFrame:
    """Construct the Section 3.3 predictors + controls, one column each, from the
    filer-aware base fields. Formulas are exactly as specified in the brief."""
    v = pd.DataFrame({"bank_IDRSSD": wide["IDRSSD"].values,
                      "filing_type": wide["filing_type"].values})

    # filer-aware base magnitudes
    assets = pick_filer_field(wide, "total_assets")
    liabilities = pick_filer_field(wide, "total_liabilities")
    equity = pick_filer_field(wide, "total_equity")
    deposits = total_deposits(wide)
    cash = pick_filer_field(wide, "cash_noninterest") + pick_filer_field(wide, "cash_interest")
    htm_cost = pick_filer_field(wide, "htm_amortized")
    htm_fv = pick_filer_field(wide, "htm_fairvalue")
    afs_cost = pick_filer_field(wide, "afs_amortized")
    afs_fv = pick_filer_field(wide, "afs_fairvalue")
    pledged = pick_filer_field(wide, "pledged_secs")
    # RC-O M.1 (domestic, RCON on all forms)
    f051 = pick_filer_field(wide, "unins_amt_gt")
    f052 = pick_filer_field(wide, "unins_num_gt")
    f047 = pick_filer_field(wide, "unins_ret_amt_gt")
    f048 = pick_filer_field(wide, "unins_ret_num_gt")
    tier1 = pick_filer_field(wide, "tier1_leverage")
    net_income = pick_filer_field(wide, "net_income")
    int_income = pick_filer_field(wide, "interest_income")
    nonaccrual = pick_filer_field(wide, "nonaccrual")
    pastdue90 = pick_filer_field(wide, "pastdue90")
    loans = pick_filer_field(wide, "total_loans")

    # keep base magnitudes for auditing / dep_growth base
    v["total_assets"] = assets
    v["total_deposits"] = deposits
    v["total_liabilities"] = liabilities
    v["total_equity"] = equity

    # --- Section 3.3 predictors ---
    # uninsured_share, eq. 3.2: amount above $250k limit on large accounts
    # (ordinary + retirement branches of RC-O Memo item 1) over total deposits
    uninsured_amount = (f051 - INSURANCE_LIMIT * f052) + (f047 - INSURANCE_LIMIT * f048)
    v["uninsured_share"] = uninsured_amount / deposits
    # unrealised_losses: (HTM fair value − cost) + (AFS fair value − cost), over equity
    v["unrealised_losses"] = ((htm_fv - htm_cost) + (afs_fv - afs_cost)) / equity
    # deposit_reliance: deposits / total liabilities
    v["deposit_reliance"] = deposits / liabilities
    # liquidity: cash + unpledged securities (carrying basis) over assets, Section 3.3
    unpledged_secs = (htm_cost + afs_fv) - pledged      # carrying value basis
    v["liquidity"] = (cash + unpledged_secs) / assets
    # capital: tier-1 leverage ratio (percent, as filed)
    v["capital"] = tier1
    # size: natural log of total assets ($000)
    v["size"] = np.log(assets)

    # --- Controls ---
    v["ROA"] = net_income / assets
    v["NPL_ratio"] = (nonaccrual + pastdue90) / loans
    v["equity_ratio"] = equity / assets
    v["int_inc_ratio"] = int_income / assets
    return v


# --------------------------------------------------------------------------- #
#  4. Dependent variable
# --------------------------------------------------------------------------- #
def compute_dep_growth(ids: set[int]) -> pd.DataFrame:
    """2023Q1 deposits per bank, the numerator side of

        dep_growth = (deposits_2023Q1 − deposits_2022Q4) / deposits_2022Q4.

    The 2022Q4 base is held by the caller in panel['total_deposits'] and the
    ratio is formed there; this reads only 2023Q1, using the same filer-aware
    deposit rule (RCON2200 + RCFN2200)."""
    wide_q1 = load_filer_types(ZIP_2023Q1).merge(
        read_schedule(ZIP_2023Q1, "RC"), on="IDRSSD", how="right")
    wide_q1 = wide_q1[wide_q1.IDRSSD.isin(ids)]
    dep_q1 = pd.DataFrame({"bank_IDRSSD": wide_q1["IDRSSD"].values,
                           "dep_2023Q1": total_deposits(wide_q1).values})
    return dep_q1


# --------------------------------------------------------------------------- #
#  5. Censor failures
# --------------------------------------------------------------------------- #
def censor_failures(panel: pd.DataFrame) -> pd.DataFrame:
    """Right-censor failed banks (SVB, Signature, First Republic): override
    dep_growth = −1.0 and set censored = True (README: '−1.0 is an assumption,
    not a measurement')."""
    panel["censored"] = panel["failed"].astype(bool)
    panel.loc[panel["censored"], "dep_growth"] = -1.0
    return panel


# --------------------------------------------------------------------------- #
#  6. Diagnostics
# --------------------------------------------------------------------------- #
def run_diagnostics(panel: pd.DataFrame, both_null: dict) -> None:
    """Print (do not drop): missingness, dep_growth distribution, sanity checks."""
    varcols = ["uninsured_share", "unrealised_losses", "deposit_reliance", "liquidity",
               "capital", "size", "ROA", "NPL_ratio", "equity_ratio", "int_inc_ratio",
               "dep_growth"]
    print("\n--- missingness (nulls out of {}): ---".format(len(panel)))
    for c in varcols:
        print(f"    {c:18s}: {panel[c].isna().sum():3d}")

    print("\n--- fields empty under BOTH RCON and consolidated prefixes (031-safe check): ---")
    for concept, n in both_null.items():
        if n:
            print(f"    {concept:18s}: {n}")
    if not any(both_null.values()):
        print("    (none — every concept populated under at least one prefix for all banks)")

    dg = panel["dep_growth"].dropna()
    print("\n--- dep_growth distribution: ---")
    print(f"    min={dg.min():.4f}  p25={dg.quantile(.25):.4f}  median={dg.median():.4f}"
          f"  p75={dg.quantile(.75):.4f}  max={dg.max():.4f}")
    print(f"    censored (= −1.0): {(panel['dep_growth'] == -1.0).sum()}")
    print(f"    banks below −0.10: {(dg < -0.10).sum()}")

    print("\n--- sanity checks (listed = violations to inspect, not dropped): ---")
    bad_u = panel[(panel.uninsured_share < 0) | (panel.uninsured_share > 1.2)]
    bad_c = panel[panel.capital <= 0]
    bad_dr = panel[(panel.deposit_reliance < 0) | (panel.deposit_reliance > 1.2)]
    bad_l = panel[(panel.liquidity < 0) | (panel.liquidity > 1.0)]
    for label, bad, col in [("uninsured_share outside [0,1.2]", bad_u, "uninsured_share"),
                            ("capital <= 0", bad_c, "capital"),
                            ("deposit_reliance outside [0,1.2]", bad_dr, "deposit_reliance"),
                            ("liquidity outside [0,1.0]", bad_l, "liquidity")]:
        print(f"  {label}: {len(bad)}")
        for _, r in bad.iterrows():
            print(f"      {r.bank_IDRSSD}  {r['name'][:38]:38s} {col}={r[col]:.4f}  ({r.filing_type})")

    print(f"\n--- final panel shape: {panel.shape} ---")
    print(panel.head().to_string(index=False))


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    print("[1/7] loading sample ...")
    sample = pd.read_csv(SAMPLE_CSV)
    ids = set(sample.bank_IDRSSD.astype(int))
    print(f"      {len(ids)} sample banks (keyed on bank_IDRSSD)")

    print("[2/7] assembling filer-aware raw fields from 2022Q4 ...")
    wide = assemble_raw_fields(ZIP_2022Q4, ids)
    ftab = wide.filing_type.value_counts().to_dict()
    print(f"      filer mix: {ftab}")

    # step-1 coverage: fields empty under BOTH prefixes (truly missing)
    both_null = {}
    for concept, spec in CODES.items():
        dom = get_numeric(wide, spec["rcon"])
        con = get_numeric(wide, spec["cons"])
        both_null[concept] = int((dom.isna() & con.isna()).sum())

    print("[3/7] constructing Section 3.3 variables ...")
    v = build_variables(wide)
    # keep only identity/flag columns from the sample; magnitudes are recomputed
    sample_keep = sample[["bank_IDRSSD", "name", "permno", "holder_RSSD", "failed"]]
    panel = sample_keep.merge(v, on="bank_IDRSSD", how="left")

    print("[4/7] computing dep_growth (2022Q4 -> 2023Q1) ...")
    dep_q1 = compute_dep_growth(ids)
    panel = panel.merge(dep_q1, on="bank_IDRSSD", how="left")
    panel["dep_growth"] = (panel["dep_2023Q1"] - panel["total_deposits"]) / panel["total_deposits"]

    print("[5/7] censoring failed banks (dep_growth = -1.0) ...")
    panel = censor_failures(panel)

    print("[6/7] diagnostics ...")
    run_diagnostics(panel, both_null)

    print("[7/7] saving ...")
    cols = ["bank_IDRSSD", "name", "permno", "holder_RSSD", "filing_type",
            "total_assets", "total_deposits", "total_liabilities", "total_equity",
            "uninsured_share", "unrealised_losses", "deposit_reliance", "liquidity",
            "capital", "size", "ROA", "NPL_ratio", "equity_ratio", "int_inc_ratio",
            "dep_2023Q1", "dep_growth", "censored", "failed"]
    cols = [c for c in cols if c in panel.columns]
    PROC.mkdir(parents=True, exist_ok=True)
    panel[cols].to_csv(OUT_CSV, index=False)
    print(f"      wrote {OUT_CSV} ({len(panel)} rows, {len(cols)} cols)")


if __name__ == "__main__":
    main()
