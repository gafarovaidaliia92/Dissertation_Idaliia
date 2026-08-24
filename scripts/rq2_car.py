"""
rq2_car.py — RQ2 [B]: market-model event study around the 11 Fed CBDC speeches.

Reads CRSP daily files + the safeguard scores + sample_banks, estimates a market
model per bank x event, and writes the bank-event CAR panel with time-varying
Call Report characteristics attached.

Reads:  data/raw/wrds/{dsf,dsi,dsedelist}.parquet
        data/frozen/sample_banks.csv
        data/processed/rq2_safeguard_scores.csv
        data/raw/call_*.zip  (via scripts/build_panel.py extraction functions)
Writes: data/processed/rq2_car.csv

Design (Section 3.4):
  t0            = first CRSP trading day on or after the speech date
  estimation    = trading days [t0-230, t0-31]  (200 days, ending 30 days out)
                  requires >= 100 valid daily returns, else the bank-event drops
  market model  = OLS of ret on vwretd over the estimation window
  AR_it         = ret_it - (alpha + beta * vwretd_t)
  event window  = [t0-1, t0+5];  CAR = sum of AR over that window
  delisting     = dlret spliced into the return on dlstdt, series truncated after

Characteristics are taken from the most recent Call Report quarter whose
quarter-end falls STRICTLY BEFORE t0 (no look-ahead into the event). Coverage
starts at 2019Q3, which precedes the earliest event (2019-10-16), so every
bank-event now draws on a genuinely prior quarter and char_lookahead is False
throughout. The fallback below is kept as a guard in case an earlier event is
ever added without the matching zip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _panel_narrow as bp  # noqa: E402  — reuse MDRM / filer-aware extraction

RAW = Path("data/raw")
WRDS = RAW / "wrds"
PROC = Path("data/processed")
OUT_CSV = PROC / "rq2_car.csv"
DROP_LOG = PROC / "rq2_car_droplog.csv"

# --- event-study windows (in trading days relative to t0) ---
EST_START, EST_END = -230, -31     # 200 trading days
EVT_START, EVT_END = -1, 5         # 7 trading days
MIN_EST_OBS = 100

# --- quarter-end -> zip file (2019Q3 .. 2023Q2) ---
QUARTER_ZIPS = {
    "2019-09-30": "call_09302019.zip",
    "2019-12-31": "call_12312019.zip",
    "2020-03-31": "call_03312020.zip",
    "2020-06-30": "call_06302020.zip",
    "2020-09-30": "call_09302020.zip",
    "2020-12-31": "call_12312020.zip",
    "2021-03-31": "call_03312021.zip",
    "2021-06-30": "call_06302021.zip",
    "2021-09-30": "call_09302021.zip",
    "2021-12-31": "call_12312021.zip",
    "2022-03-31": "call_03312022.zip",
    "2022-06-30": "call_06302022.zip",
    "2022-09-30": "call_09302022.zip",
    "2022-12-31": "call_12312022.zip",
    "2023-03-31": "call_03312023.zip",
    "2023-06-30": "call_06302023.zip",
}
QUARTER_ENDS = sorted(QUARTER_ZIPS)

CHAR_COLS = ["uninsured_share", "size", "ROA", "NPL_ratio",
             "equity_ratio", "deposit_reliance"]


# --------------------------------------------------------------------------- #
#  1. CRSP returns, delisting-adjusted
# --------------------------------------------------------------------------- #
def load_returns(permnos: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Daily bank returns merged with the value-weighted market return, with CRSP
    delisting returns spliced in.

    CRSP's dsf already stops at the delisting date, so splicing means compounding
    the partial-day return with dlret on dlstdt:  (1+ret)(1+dlret) - 1. Where only
    one of the two exists that one is used; where neither exists the day is NaN
    and simply fails the estimation-window count. Rows after dlstdt are dropped
    (there are none in practice, but the truncation is kept explicit).
    """
    dsf = pd.read_parquet(WRDS / "dsf.parquet", columns=["permno", "date", "ret"])
    dsi = pd.read_parquet(WRDS / "dsi.parquet", columns=["date", "vwretd"])
    dl = pd.read_parquet(WRDS / "dsedelist.parquet")

    dsf = dsf[dsf.permno.isin(permnos)].copy()
    dsf["permno"] = dsf.permno.astype("int64")
    dsf["date"] = dsf.date.astype(str)
    dsf["ret"] = dsf.ret.astype(float)

    # dlstcd 100 = "issue still trading" (CRSP stamps the file end date); not a delisting
    dl = dl[(dl.dlstcd != 100) & dl.permno.isin(permnos)].copy()
    dl["permno"] = dl.permno.astype("int64")
    dl["dlstdt"] = dl.dlstdt.astype(str)
    dl["dlret"] = dl.dlret.astype(float)
    dl = dl.sort_values("dlstdt").drop_duplicates("permno", keep="first")

    dsf = dsf.merge(dl[["permno", "dlstdt", "dlret"]], on="permno", how="left")
    dsf = dsf[dsf.dlstdt.isna() | (dsf.date <= dsf.dlstdt)]          # truncate after delisting

    on_dlst = dsf.date.eq(dsf.dlstdt)
    r, d = dsf.ret, dsf.dlret
    spliced = np.where(r.notna() & d.notna(), (1 + r) * (1 + d) - 1,
                       np.where(d.notna(), d, r))
    dsf["ret"] = np.where(on_dlst, spliced, dsf.ret)

    dsi = dsi.copy()
    dsi["date"] = dsi.date.astype(str)
    dsi["vwretd"] = dsi.vwretd.astype(float)

    panel = dsf.merge(dsi, on="date", how="inner").sort_values(["permno", "date"])
    return panel, dl


# --------------------------------------------------------------------------- #
#  2. Event-day mapping
# --------------------------------------------------------------------------- #
def resolve_event_days(events: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """t0 = first CRSP trading day on or after the speech date (weekends/holidays
    shift forward); also record its integer index on the trading calendar."""
    cal = np.array(calendar)
    out = events.copy()
    pos = np.searchsorted(cal, out.date.astype(str).values, side="left")
    if (pos >= len(cal)).any():
        raise SystemExit("event date falls beyond the end of the CRSP calendar")
    out["t0"] = cal[pos]
    out["t0_idx"] = pos
    return out


# --------------------------------------------------------------------------- #
#  3. Market model per bank x event
# --------------------------------------------------------------------------- #
def compute_car(panel: pd.DataFrame, ev: pd.Series, calendar: list[str],
                delist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate alpha/beta on the estimation window and cumulate ARs over the
    event window, for every bank with enough estimation-window observations."""
    i = int(ev.t0_idx)
    est_lo, est_hi = max(i + EST_START, 0), i + EST_END
    evt_lo, evt_hi = i + EVT_START, i + EVT_END
    if evt_hi >= len(calendar):
        raise SystemExit(f"event {ev.doc_id}: event window runs past the CRSP calendar")

    est_dates = set(calendar[est_lo:est_hi + 1])
    evt_dates = calendar[evt_lo:evt_hi + 1]

    est = panel[panel.date.isin(est_dates) & panel.ret.notna()]
    evt = panel[panel.date.isin(evt_dates) & panel.ret.notna()]

    dl_map = dict(zip(delist.permno, delist.dlstdt))
    win_lo, win_hi = calendar[est_lo], evt_dates[-1]

    rows, dropped = [], []
    for permno, g in est.groupby("permno", sort=True):
        n_est = len(g)
        if n_est < MIN_EST_OBS:
            dropped.append({"permno": permno, "doc_id": ev.doc_id, "t0": ev.t0,
                            "n_est": n_est, "reason": "estimation window < 100 valid returns"})
            continue
        x, y = g.vwretd.values, g.ret.values
        beta, alpha = np.polyfit(x, y, 1)

        e = evt[evt.permno == permno]
        dlstdt = dl_map.get(permno)
        # a bank that delists inside the event window keeps its truncated CAR
        # (dlret is already spliced in); any other short window is a data gap
        delists_in_evt = dlstdt is not None and evt_dates[0] <= dlstdt <= evt_dates[-1]
        if len(e) < len(evt_dates) and not delists_in_evt:
            dropped.append({"permno": permno, "doc_id": ev.doc_id, "t0": ev.t0,
                            "n_est": n_est,
                            "reason": f"incomplete event window ({len(e)}/{len(evt_dates)} days)"})
            continue
        car = float(np.sum(e.ret.values - (alpha + beta * e.vwretd.values)))

        rows.append({
            "permno": permno, "doc_id": ev.doc_id, "event_date": ev.date, "t0": ev.t0,
            "CAR": car, "alpha": float(alpha), "beta": float(beta), "n_est": n_est,
            "delisted_flag": bool(dlstdt is not None and win_lo <= dlstdt <= win_hi),
        })

    # banks with no estimation-window rows at all never reach the loop above
    seen = {r["permno"] for r in rows} | {d["permno"] for d in dropped}
    for permno in sorted(set(panel.permno.unique()) - seen):
        dropped.append({"permno": permno, "doc_id": ev.doc_id, "t0": ev.t0, "n_est": 0,
                        "reason": "no CRSP returns in estimation window (not listed / already delisted)"})
    return pd.DataFrame(rows), pd.DataFrame(dropped)


# --------------------------------------------------------------------------- #
#  4. Time-varying characteristics
# --------------------------------------------------------------------------- #
def map_event_quarters(events: pd.DataFrame) -> pd.DataFrame:
    """Most recent quarter-end STRICTLY BEFORE t0; fall back to the earliest
    available quarter (flagged) for events that pre-date the zip coverage."""
    out = events.copy()
    qends, flags = [], []
    for t0 in out.t0:
        prior = [q for q in QUARTER_ENDS if q < t0]
        if prior:
            qends.append(prior[-1]); flags.append(False)
        else:
            qends.append(QUARTER_ENDS[0]); flags.append(True)
    out["char_quarter"] = qends
    out["char_lookahead"] = flags
    return out


def build_characteristics(quarters: list[str], ids: set[int]) -> pd.DataFrame:
    """Per (quarter, bank) characteristics via build_panel's filer-aware extraction.

    ROA is the only construction that differs from build_panel: RIAD4340 is a
    year-to-date figure, so a Q1 reading covers one quarter and a Q4 reading four.
    Since events map to different fiscal quarters, net income is annualised
    (x 4/quarter) to keep ROA comparable across events. Every other formula is
    build_panel's untouched.
    """
    frames = []
    for q in quarters:
        wide = bp.assemble_raw_fields(RAW / QUARTER_ZIPS[q], ids)
        v = bp.build_variables(wide)
        qnum = int(q[5:7]) // 3                       # 03->1, 06->2, 09->3, 12->4
        v["ROA"] = v["ROA"] * (4.0 / qnum)
        v["char_quarter"] = q
        missing = {c: int(v[c].isna().sum()) for c in CHAR_COLS}
        print(f"      {q}: {len(v):3d} banks   missing={ {k: n for k, n in missing.items() if n} or 'none' }")
        frames.append(v[["bank_IDRSSD", "char_quarter", "filing_type"] + CHAR_COLS])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    print("[1/5] loading sample + events ...")
    sample = pd.read_csv(PROC.parent / "frozen" / "sample_banks.csv")
    sample["permno"] = sample.permno.astype("int64")
    permnos = set(sample.permno)
    ids = set(sample.bank_IDRSSD.astype(int))
    events = pd.read_csv(PROC / "rq2_safeguard_scores.csv")
    print(f"      {len(sample)} banks, {len(events)} events")

    print("[2/5] loading CRSP returns (delisting-adjusted) ...")
    panel, delist = load_returns(permnos)
    calendar = sorted(pd.read_parquet(WRDS / "dsi.parquet", columns=["date"])
                      .date.astype(str).unique())
    events = resolve_event_days(events, calendar)
    n_dl = int(((delist.dlstdt >= "2019-01-01") & (delist.dlstdt <= "2023-12-31")).sum())
    print(f"      {panel.permno.nunique()} permnos, {len(calendar)} trading days, "
          f"{n_dl} in-sample delistings spliced")

    print("[3/5] estimating market models ...")
    cars, drops = [], []
    for _, ev in events.iterrows():
        c, d = compute_car(panel, ev, calendar, delist)
        cars.append(c); drops.append(d)
        print(f"      doc_id={ev.doc_id:>2}  {ev.date} -> t0={ev.t0}  "
              f"n_banks={len(c):3d}  dropped={len(d):3d}")
    car = pd.concat(cars, ignore_index=True)
    drop = pd.concat(drops, ignore_index=True)

    print("[4/5] attaching Call Report characteristics ...")
    events = map_event_quarters(events)
    for _, ev in events.iterrows():
        tag = "  (LOOKAHEAD: event pre-dates earliest zip)" if ev.char_lookahead else ""
        print(f"      doc_id={ev.doc_id:>2}  t0={ev.t0} -> {ev.char_quarter}{tag}")
    chars = build_characteristics(sorted(events.char_quarter.unique()), ids)

    car = car.merge(events[["doc_id", "S", "S_norm", "n_design",
                            "char_quarter", "char_lookahead"]], on="doc_id", how="left")
    car = car.merge(sample[["permno", "bank_IDRSSD", "name", "failed"]], on="permno", how="left")
    before = len(car)
    car = car.merge(chars, on=["bank_IDRSSD", "char_quarter"], how="left")
    assert len(car) == before, "characteristics merge changed row count (grain break)"

    n_nochar = int(car[CHAR_COLS].isna().any(axis=1).sum())
    if n_nochar:
        miss = car[car[CHAR_COLS].isna().any(axis=1)]
        print(f"      WARNING: {n_nochar} bank-events without characteristics "
              f"(banks: {sorted(miss.name.unique())[:5]} ...)")

    print("[5/5] saving ...")
    cols = ["permno", "bank_IDRSSD", "name", "doc_id", "event_date", "t0", "CAR",
            "alpha", "beta", "n_est", "delisted_flag", "S", "S_norm", "n_design",
            "char_quarter", "char_lookahead", "filing_type"] + CHAR_COLS + ["failed"]
    car = car.sort_values(["doc_id", "permno"])[cols]
    car.to_csv(OUT_CSV, index=False)
    drop.sort_values(["doc_id", "permno"]).to_csv(DROP_LOG, index=False)
    print(f"      wrote {OUT_CSV} ({len(car)} bank-events)")
    print(f"      wrote {DROP_LOG} ({len(drop)} dropped bank-events)")
    print("\n--- drop reasons ---")
    print(drop.reason.value_counts().to_string())


if __name__ == "__main__":
    main()
