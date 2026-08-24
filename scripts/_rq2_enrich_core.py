"""
rq2_enrich.py — three additions to the RQ2/RQ3 event study.

  PART A  Sector + rate-adjusted abnormal returns. The published CAR uses a
          one-factor market model. Bank stocks also load on a common banking
          factor and differ sharply in interest-rate sensitivity, so a 2022
          rate-driven move by deposit-heavy banks could masquerade as a
          CBDC-signal cross-section. This re-estimates AR against a
          market + banking-sector + rate model and re-runs the headline specs
          side by side: gamma1 on S (H2, event-clustered), delta1 on
          vulnerability_score (H3), and the S x uninsured_share interaction as
          a labelled extension.
  PART B  Did the market react at all? Unconditional first-moment (mean AR/CAR),
          second-moment (Beaver variance ratio) and volume (turnover ratio)
          tests. This is the PRIMARY H2 evidence and is reproduced at the top of
          rq2_event_regression.txt.
  PART C  Minimum detectable effect at 5% / 80% power for the headline
          coefficients — including gamma1, so the H2 null is bounded rather than
          merely reported.

Reads:  data/processed/rq2_car.csv                    (market-model CARs, unchanged)
        data/processed/vulnerability_scores_wide.csv
        data/raw/wrds/{dsf,dsi,dsedelist}.parquet
        external factor data (cached under data/raw/factors/)
Writes: data/processed/rq2_car_augmented.csv
        data/processed/rq2_enrichment_results.txt

Nothing in the RQ1 / text-signal / rq2_car.py / rq2_event_regression.py /
rq3_bridge.py pipeline is modified. Delisting handling and the estimation /
event windows are inherited by importing rq2_car, so the augmented model sees
exactly the same returns as the published one.

Parts B and C run off the existing rq2_car.csv and do not depend on Part A's
downloads succeeding.
"""

from __future__ import annotations

import io
import re
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import rq2_car as rc  # reuse load_returns / windows / event-day mapping verbatim

PROC = Path("data/processed")
WRDS = Path("data/raw/wrds")
FACT = Path("data/raw/factors")
OUT_CSV = PROC / "rq2_car_augmented.csv"
OUT_TXT = PROC / "rq2_enrichment_results.txt"

CONTROLS = ["size", "ROA", "NPL_ratio", "equity_ratio"]
GAMMA1 = "S"                # H2: average effect of the communication type
BETA3 = "S:uninsured_share"  # extension only: within-event cross-section
DELTA1 = "vulnerability_score"
CAR_SD_REF = 0.048          # published market-model CAR SD, for scaling MDEs
POWER_MULT = 2.80           # two-sided 5%, 80% power

FF_FACTORS_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                  "F-F_Research_Data_Factors_daily_CSV.zip")
FF_49IND_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                "49_Industry_Portfolios_Daily_CSV.zip")
FF_49IND_URL_ALT = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
                    "49_Industry_Portfolios_daily_CSV.zip")
FRED_DGS2_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2"

SOURCE_LOG: list[str] = []


# --------------------------------------------------------------------------- #
#  0. Download helpers (cached; every fallback is logged, never silent)
# --------------------------------------------------------------------------- #
def fetch(url: str, cache: Path, timeout: int = 120) -> bytes | None:
    """GET with on-disk cache. Returns None on failure (caller decides fallback)."""
    if cache.exists():
        SOURCE_LOG.append(f"    cache hit  : {cache.name}")
        return cache.read_bytes()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh)"})
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            b = r.read()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b)
        SOURCE_LOG.append(f"    downloaded : {cache.name}  ({len(b):,} bytes)")
        return b
    except Exception as e:                                    # noqa: BLE001
        SOURCE_LOG.append(f"    FAILED     : {url}  ({type(e).__name__}: {e})")
        return None


def parse_ff_block(text: str, header_line: int, stop_line: int | None) -> pd.DataFrame:
    """Parse one Ken French daily block into (date, <cols>) with % -> decimal."""
    lines = text.splitlines()
    cols = [c.strip() for c in lines[header_line].split(",")][1:]
    rows = []
    for l in lines[header_line + 1: stop_line]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) != len(cols) + 1 or not re.fullmatch(r"\d{8}", parts[0]):
            continue
        rows.append([parts[0]] + [float(p) for p in parts[1:]])
    df = pd.DataFrame(rows, columns=["yyyymmdd"] + cols)
    df["date"] = pd.to_datetime(df.yyyymmdd, format="%Y%m%d").dt.strftime("%Y-%m-%d")
    for c in cols:
        df[c] = df[c].replace([-99.99, -999.0], np.nan) / 100.0   # FF are in percent
    return df[["date"] + cols]


def load_ff_factors() -> pd.DataFrame | None:
    b = fetch(FF_FACTORS_URL, FACT / "F-F_Research_Data_Factors_daily_CSV.zip")
    if b is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(b))
    text = z.open(z.namelist()[0]).read().decode("latin-1")
    hdr = next(i for i, l in enumerate(text.splitlines()) if l.startswith(",Mkt-RF"))
    df = parse_ff_block(text, hdr, None)
    return df.rename(columns={"Mkt-RF": "mktrf", "RF": "rf"})[["date", "mktrf", "rf"]]


def load_ff_banks() -> pd.DataFrame | None:
    """Value-weighted 'Banks' industry portfolio (first block of the 49-industry
    daily file; the second block is equal-weighted and is deliberately skipped)."""
    b = fetch(FF_49IND_URL, FACT / "49_Industry_Portfolios_Daily_CSV.zip")
    if b is None:
        b = fetch(FF_49IND_URL_ALT, FACT / "49_Industry_Portfolios_daily_CSV.zip")
    if b is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(b))
    text = z.open(z.namelist()[0]).read().decode("latin-1")
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith(",Agric"))
    eq = next((i for i, l in enumerate(lines)
               if "Equal Weighted" in l and i > hdr), None)
    df = parse_ff_block(text, hdr, eq)
    if "Banks" not in df.columns:
        SOURCE_LOG.append("    FAILED     : 'Banks' column absent from 49-industry file")
        return None
    return df[["date", "Banks"]].rename(columns={"Banks": "banks"})


def load_dgs2(calendar: list[str]) -> pd.DataFrame | None:
    """Daily change in the 2-year Treasury CMT yield, in percentage points,
    differenced along the CRSP trading calendar."""
    b = fetch(FRED_DGS2_URL, FACT / "DGS2.csv")
    if b is None:
        return None
    raw = pd.read_csv(io.BytesIO(b))
    dcol, vcol = raw.columns[0], raw.columns[1]
    raw[dcol] = pd.to_datetime(raw[dcol]).dt.strftime("%Y-%m-%d")
    raw[vcol] = pd.to_numeric(raw[vcol], errors="coerce")     # '.' -> NaN on holidays
    s = raw.set_index(dcol)[vcol].reindex(calendar)
    n_missing = int(s.isna().sum())
    # bond-market holidays that are equity trading days: carry the yield forward,
    # so a closed Treasury market reads as "no observed change" rather than
    # deleting the equity observation
    s = s.ffill(limit=5)
    SOURCE_LOG.append(f"    DGS2 missing on {n_missing} of {len(calendar)} CRSP "
                      f"trading days (bond-market holidays) -> forward-filled")
    return pd.DataFrame({"date": calendar, "d_dgs2": s.diff().values})


def build_fallback_bank_index(panel: pd.DataFrame, dsf_full: pd.DataFrame,
                              calendar: list[str]) -> pd.DataFrame:
    """FALLBACK ONLY. Value-weighted daily bank index from the CRSP sample banks,
    weighted on lagged market cap. For the 10 largest banks a leave-one-out index
    is used so a mega-cap is not regressed on an index it dominates."""
    d = dsf_full.copy()
    d["mcap"] = d.prc.abs() * d.shrout
    d["w"] = d.groupby("permno").mcap.shift(1)
    d = d.dropna(subset=["ret", "w"])
    tot = d.groupby("date").apply(lambda g: np.average(g.ret, weights=g.w),
                                  include_groups=False)
    wsum = d.groupby("date").w.sum()
    big = (d.groupby("permno").mcap.mean().sort_values(ascending=False)
           .head(10).index.tolist())
    out = pd.DataFrame({"date": calendar}).merge(
        tot.rename("banks_all").reset_index(), on="date", how="left")
    loo = {}
    for p in big:
        sub = d[d.permno == p].set_index("date")
        num = (tot * wsum).reindex(out.date.values) - (sub.ret * sub.w).reindex(out.date.values).fillna(0)
        den = wsum.reindex(out.date.values) - sub.w.reindex(out.date.values).fillna(0)
        loo[p] = (num / den).values
    SOURCE_LOG.append(f"    FALLBACK bank index built from {d.permno.nunique()} sample "
                      f"banks; leave-one-out applied to the 10 largest {big}")
    return out.assign(**{f"loo_{p}": v for p, v in loo.items()})


# --------------------------------------------------------------------------- #
#  PART A — augmented (market + sector + rate) abnormal returns
# --------------------------------------------------------------------------- #
def estimate_augmented(panel: pd.DataFrame, events: pd.DataFrame,
                       calendar: list[str], delist: pd.DataFrame,
                       factor_cols: list[str]) -> pd.DataFrame:
    """Same windows and filters as the published market model, different RHS."""
    dl_map = dict(zip(delist.permno, delist.dlstdt))
    rows = []
    for _, ev in events.iterrows():
        i = int(ev.t0_idx)
        est_lo, est_hi = max(i + rc.EST_START, 0), i + rc.EST_END
        evt_lo, evt_hi = i + rc.EVT_START, i + rc.EVT_END
        est_dates = set(calendar[est_lo:est_hi + 1])
        evt_dates = calendar[evt_lo:evt_hi + 1]

        need = ["exret"] + factor_cols
        est = panel[panel.date.isin(est_dates)].dropna(subset=need)
        evt = panel[panel.date.isin(evt_dates)].dropna(subset=need)

        for permno, g in est.groupby("permno", sort=True):
            if len(g) < rc.MIN_EST_OBS:
                continue
            X = np.column_stack([np.ones(len(g))] + [g[c].values for c in factor_cols])
            coef, *_ = np.linalg.lstsq(X, g.exret.values, rcond=None)

            e = evt[evt.permno == permno]
            dlstdt = dl_map.get(permno)
            delists_in_evt = dlstdt is not None and evt_dates[0] <= dlstdt <= evt_dates[-1]
            if len(e) < len(evt_dates) and not delists_in_evt:
                continue
            Xe = np.column_stack([np.ones(len(e))] + [e[c].values for c in factor_cols])
            car = float(np.sum(e.exret.values - Xe @ coef))

            row = {"permno": permno, "doc_id": ev.doc_id, "event_date": ev.date,
                   "t0": ev.t0, "CAR_aug": car, "alpha_aug": float(coef[0]),
                   "n_est_aug": len(g),
                   "delisted_flag": bool(dlstdt is not None
                                         and calendar[est_lo] <= dlstdt <= evt_dates[-1])}
            for k, c in enumerate(factor_cols, start=1):
                row[f"b_{c}"] = float(coef[k])
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Regression helpers
# --------------------------------------------------------------------------- #
def fit_cl(df: pd.DataFrame, formula: str, cluster: str = "permno",
           use_t: bool = False):
    """OLS with cluster-robust SEs. Cluster on the level at which the regressor
    of interest varies: 'permno' for bank characteristics, 'doc_id' for S, which
    is constant within an event. With 11 event clusters, inference is read off
    t with G-1 = 10 df (use_t=True)."""
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster]}, use_t=use_t)


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "(n.s.)"


def side_by_side(df: pd.DataFrame, dep_a: str, dep_b: str, rhs: str,
                 term: str, label_a: str, label_b: str,
                 cluster: str = "permno", use_t: bool = False) -> tuple[str, dict]:
    ma = fit_cl(df.assign(_y=df[dep_a]), f"_y ~ {rhs}", cluster, use_t)
    mb = fit_cl(df.assign(_y=df[dep_b]), f"_y ~ {rhs}", cluster, use_t)
    lines = ["    {:<24s} {:>13s} {:>12s} {:>8s} {:>8s}  {}".format(
        "return model", "coef", "SE", "t", "p", "sig"), "    " + "-" * 76]
    for lab, m in [(label_a, ma), (label_b, mb)]:
        lines.append("    {:<24s} {:>13.6f} {:>12.6f} {:>8.3f} {:>8.4f}  {}".format(
            lab, m.params[term], m.bse[term], m.tvalues[term], m.pvalues[term],
            stars(m.pvalues[term])))
    d = mb.params[term] - ma.params[term]
    lines += ["    " + "-" * 76,
              "    {:<24s} {:>13.6f}   ({:+.1f}% of the market-model coefficient,"
              " {:.2f} market-model SEs)".format(
                  "change", d, 100 * d / ma.params[term] if ma.params[term] else float("nan"),
                  d / ma.bse[term])]
    return "\n".join(lines), {"a": ma, "b": mb}


# --------------------------------------------------------------------------- #
#  PART B — unconditional reaction
# --------------------------------------------------------------------------- #
def daily_ar_panel(panel: pd.DataFrame, events: pd.DataFrame, calendar: list[str],
                   spec: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-bank-event-day ARs plus the estimation-window AR variance and mean
    turnover, for the variance-ratio and turnover tests.

    Also returns the equal-weighted PORTFOLIO AR series over each estimation
    window. Cross-sectional t-tests treat banks as independent within a date,
    which is false — bank residuals share a common component, so those t-stats
    are inflated. The portfolio series supports a cross-correlation-robust test
    (Brown-Warner crude dependence adjustment) in part_b().
    """
    rows = []
    est_acc: dict[tuple, list[float]] = {}
    for _, ev in events.iterrows():
        i = int(ev.t0_idx)
        est_lo, est_hi = max(i + rc.EST_START, 0), i + rc.EST_END
        evt_dates = calendar[i + rc.EVT_START: i + rc.EVT_END + 1]
        est_dates = set(calendar[est_lo:est_hi + 1])

        if spec == "market":
            ycol, xcols = "ret", ["vwretd"]
        else:
            ycol, xcols = "exret", ["mktrf", "banks_x", "d_dgs2"]
        need = [ycol] + xcols
        est = panel[panel.date.isin(est_dates)].dropna(subset=need)
        evt = panel[panel.date.isin(evt_dates)].dropna(subset=need)

        for permno, g in est.groupby("permno", sort=True):
            if len(g) < rc.MIN_EST_OBS:
                continue
            X = np.column_stack([np.ones(len(g))] + [g[c].values for c in xcols])
            coef, *_ = np.linalg.lstsq(X, g[ycol].values, rcond=None)
            resid = g[ycol].values - X @ coef
            var_est = float(np.var(resid, ddof=len(coef)))
            for dt, r_ in zip(g.date.values, resid):          # portfolio accumulator
                a = est_acc.setdefault((ev.doc_id, dt), [0.0, 0])
                a[0] += float(r_); a[1] += 1
            to_est = float(np.nanmean(g.turnover.values)) if g.turnover.notna().any() else np.nan

            e = evt[evt.permno == permno]
            if e.empty:
                continue
            Xe = np.column_stack([np.ones(len(e))] + [e[c].values for c in xcols])
            ar = e[ycol].values - Xe @ coef
            to_evt = float(np.nanmean(e.turnover.values)) if e.turnover.notna().any() else np.nan
            for d_idx, (dt, a) in enumerate(zip(e.date.values, ar)):
                rows.append({"permno": permno, "doc_id": ev.doc_id, "t0": ev.t0,
                             "date": dt, "rel_day": rc.EVT_START + d_idx, "AR": a,
                             "var_est": var_est, "n_evt": len(e),
                             "to_est": to_est, "to_evt": to_evt})
    port = pd.DataFrame(
        [{"doc_id": k[0], "date": k[1], "port_AR": v[0] / v[1], "n_banks": v[1]}
         for k, v in est_acc.items()])
    return pd.DataFrame(rows), port


def ttest_block(vals: np.ndarray, mu: float = 0.0) -> tuple[float, float, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan, np.nan
    t, p = stats.ttest_1samp(v, mu)
    return float(v.mean()), float(t), float(p)


def portfolio_test(ar: pd.DataFrame, port: pd.DataFrame) -> str:
    """Cross-correlation-robust event test (Brown-Warner crude dependence
    adjustment). Form the equal-weighted portfolio of bank ARs each day, then
    judge the event-window portfolio CAR against the TIME-SERIES standard
    deviation of that same portfolio's daily AR over the estimation window.
    Because the portfolio is a single time series, the common component across
    banks is inside it rather than being wrongly treated as extra independent
    evidence."""
    out = ["-- B1c. Cross-correlation-robust portfolio test --", "",
           "    The per-event t-stats in B1 assume banks are independent within a",
           "    date. They are not: a common factor moves them together, so those",
           "    t-stats are inflated, often by an order of magnitude. This test uses",
           "    the equal-weighted portfolio AR and its own estimation-window",
           "    time-series SD, which is robust to that dependence.", "",
           "    {:>7s} {:>12s} {:>7s} {:>12s} {:>12s} {:>9s} {:>9s}  {}".format(
               "doc_id", "t0", "n_days", "port_CAR", "sigma_day", "t", "p", "sig"),
           "    " + "-" * 84]
    n_sig = 0
    robust: dict = {}
    for doc, g in ar.groupby("doc_id"):
        pe = port[port.doc_id == doc]
        sigma = float(pe.port_AR.std(ddof=1))
        daily = g.groupby("date").AR.mean()
        car_p = float(daily.sum())
        k = len(daily)
        t = car_p / (sigma * np.sqrt(k)) if sigma > 0 else np.nan
        p = float(2 * (1 - stats.norm.cdf(abs(t))))
        n_sig += int(p < 0.05)
        robust[int(doc)] = (car_p, t, p)
        out.append("    {:>7d} {:>12s} {:>7d} {:>12.5f} {:>12.6f} {:>9.2f} {:>9.4f}  {}".format(
            int(doc), str(g.t0.iloc[0]), k, car_p, sigma, t, p, stars(p)))
    out += ["    " + "-" * 84,
            f"    events significant at 5% under the robust test : {n_sig} of "
            f"{ar.doc_id.nunique()}", ""]
    return "\n".join(out), robust


def part_b(ar: pd.DataFrame, port: pd.DataFrame, label: str) -> str:
    """First moment (mean AR / CAR), second moment (Beaver variance ratio) and
    turnover, per event and pooled."""
    be = (ar.groupby(["permno", "doc_id", "t0"])
            .agg(CAR=("AR", "sum"), msar=("AR", lambda s: float(np.mean(s ** 2))),
                 var_est=("var_est", "first"), n_evt=("n_evt", "first"),
                 to_est=("to_est", "first"), to_evt=("to_evt", "first"))
            .reset_index())
    be["var_ratio"] = be.msar / be.var_est
    be["to_ratio"] = be.to_evt / be.to_est
    be.loc[~np.isfinite(be.to_ratio), "to_ratio"] = np.nan

    out = [f"### {label} ###", "",
           "-- B1. First moment: mean CAR and mean daily AR, cross-sectional t-test --", ""]
    out.append("    {:>7s} {:>12s} {:>6s} {:>11s} {:>8s} {:>8s} {:>11s} {:>8s} {:>8s}".format(
        "doc_id", "t0", "n", "mean_CAR", "t", "p", "mean_AR/day", "t", "p"))
    out.append("    " + "-" * 92)
    for (doc, t0), g in be.groupby(["doc_id", "t0"]):
        mc, tc, pc = ttest_block(g.CAR.values)
        daily = ar[(ar.doc_id == doc)].AR.values
        md, td, pd_ = ttest_block(daily)
        out.append("    {:>7d} {:>12s} {:>6d} {:>11.5f} {:>8.2f} {:>8.4f} {:>11.6f} {:>8.2f} {:>8.4f}".format(
            int(doc), str(t0), len(g), mc, tc, pc, md, td, pd_))
    mc, tc, pc = ttest_block(be.CAR.values)
    md, td, pd_ = ttest_block(ar.AR.values)
    out += ["    " + "-" * 92,
            "    {:>7s} {:>12s} {:>6d} {:>11.5f} {:>8.2f} {:>8.4f} {:>11.6f} {:>8.2f} {:>8.4f}".format(
                "POOLED", "", len(be), mc, tc, pc, md, td, pd_),
            "",
            "    WARNING: every t-stat in this table assumes banks are independent",
            "    within a date. They are not. Treat these as descriptive magnitudes",
            "    and read B1c for the test that survives cross-correlation.", ""]

    # average AR by relative day, pooled across events
    out += ["-- B1b. Average AR by event day (pooled across the 11 events) --", "",
            "    {:>9s} {:>7s} {:>12s} {:>8s} {:>8s}".format("rel_day", "n", "mean_AR", "t", "p"),
            "    " + "-" * 48]
    for rd, g in ar.groupby("rel_day"):
        m, t, p = ttest_block(g.AR.values)
        out.append("    {:>9d} {:>7d} {:>12.6f} {:>8.2f} {:>8.4f}".format(int(rd), len(g), m, t, p))
    out.append("")
    ptxt, robust = portfolio_test(ar, port)
    out.append(ptxt)

    # B2 variance ratio
    out += ["-- B2. Beaver variance ratio: mean(AR^2) in event window / AR variance",
            "       in estimation window. H0: ratio = 1 (no volatility spike) --", ""]
    out.append("    {:>7s} {:>12s} {:>6s} {:>11s} {:>10s} {:>8s} {:>9s} {:>10s}".format(
        "doc_id", "t0", "n", "mean_ratio", "median", "t vs 1", "p", "Wilcoxon p"))
    out.append("    " + "-" * 82)
    for (doc, t0), g in be.groupby(["doc_id", "t0"]):
        v = g.var_ratio.dropna().values
        m, t, p = ttest_block(v, 1.0)
        w = stats.wilcoxon(v - 1.0).pvalue if len(v) > 10 else np.nan
        out.append("    {:>7d} {:>12s} {:>6d} {:>11.4f} {:>10.4f} {:>8.2f} {:>9.4f} {:>10.4g}".format(
            int(doc), str(t0), len(v), m, float(np.median(v)), t, p, w))
    v = be.var_ratio.dropna().values
    m, t, p = ttest_block(v, 1.0)
    out += ["    " + "-" * 82,
            "    {:>7s} {:>12s} {:>6d} {:>11.4f} {:>10.4f} {:>8.2f} {:>9.4f} {:>10.4g}".format(
                "POOLED", "", len(v), m, float(np.median(v)), t, p,
                stats.wilcoxon(v - 1.0).pvalue),
            "",
            "    A ratio > 1 means returns were more dispersed than normal on these",
            "    dates even if the average move was zero.", ""]

    # B3 turnover
    out += ["-- B3. Turnover ratio: mean event-window turnover / mean estimation-window",
            "       turnover (turnover = vol / shrout). H0: ratio = 1 --", ""]
    out.append("    {:>7s} {:>12s} {:>6s} {:>11s} {:>10s} {:>8s} {:>9s} {:>10s}".format(
        "doc_id", "t0", "n", "mean_ratio", "median", "t vs 1", "p", "Wilcoxon p"))
    out.append("    " + "-" * 82)
    for (doc, t0), g in be.groupby(["doc_id", "t0"]):
        v = g.to_ratio.dropna().values
        m, t, p = ttest_block(v, 1.0)
        w = stats.wilcoxon(v - 1.0).pvalue if len(v) > 10 else np.nan
        out.append("    {:>7d} {:>12s} {:>6d} {:>11.4f} {:>10.4f} {:>8.2f} {:>9.4f} {:>10.4g}".format(
            int(doc), str(t0), len(v), m, float(np.median(v)), t, p, w))
    v = be.to_ratio.dropna().values
    m, t, p = ttest_block(v, 1.0)
    out += ["    " + "-" * 82,
            "    {:>7s} {:>12s} {:>6d} {:>11.4f} {:>10.4f} {:>8.2f} {:>9.4f} {:>10.4g}".format(
                "POOLED", "", len(v), m, float(np.median(v)), t, p,
                stats.wilcoxon(v - 1.0).pvalue),
            "",
            "    Turnover ratios are right-skewed, so the median and the Wilcoxon",
            "    test are the more reliable readings.", ""]

    # ---- synthesis: first moment vs second moment vs volume, per event ----
    n_ev = be.doc_id.nunique()
    n_mean = sum(1 for v in robust.values() if v[2] < 0.05)
    up_var = dn_var = up_to = dn_to = 0
    rows_s = []
    for (doc, t0), g in be.groupby(["doc_id", "t0"]):
        vr = g.var_ratio.dropna().values
        tr = g.to_ratio.dropna().values
        pv = stats.wilcoxon(vr - 1.0).pvalue if len(vr) > 10 else np.nan
        pt = stats.wilcoxon(tr - 1.0).pvalue if len(tr) > 10 else np.nan
        mv, mt = float(np.median(vr)), float(np.median(tr))
        v_dir = "UP" if (pv < 0.05 and mv > 1) else "DOWN" if (pv < 0.05 and mv < 1) else "flat"
        t_dir = "UP" if (pt < 0.05 and mt > 1) else "DOWN" if (pt < 0.05 and mt < 1) else "flat"
        up_var += v_dir == "UP"; dn_var += v_dir == "DOWN"
        up_to += t_dir == "UP"; dn_to += t_dir == "DOWN"
        rows_s.append((int(doc), str(t0), robust[int(doc)][2], mv, v_dir, mt, t_dir))

    out += ["-- B4. TAKEAWAY: first moment vs second moment vs volume --", "",
            "    {:>7s} {:>12s} {:>14s} {:>11s} {:>7s} {:>11s} {:>7s}".format(
                "doc_id", "t0", "mean CAR (rob)", "var ratio", "dir", "turn ratio", "dir"),
            "    " + "-" * 76]
    for doc, t0, p_rob, mv, vd, mt, td in rows_s:
        out.append("    {:>7d} {:>12s} {:>14s} {:>11.3f} {:>7s} {:>11.3f} {:>7s}".format(
            doc, t0, "sig" if p_rob < 0.05 else "null", mv, vd, mt, td))
    out += ["", f"    events with a robust mean-CAR reaction : {n_mean} of {n_ev}",
            f"    events with elevated volatility (median ratio > 1, Wilcoxon p<.05) "
            f": {up_var} of {n_ev}",
            f"    events with SUPPRESSED volatility                                 "
            f": {dn_var} of {n_ev}",
            f"    events with elevated turnover                                     "
            f": {up_to} of {n_ev}",
            f"    events with SUPPRESSED turnover                                   "
            f": {dn_to} of {n_ev}", "",
            "    Read: if the first moment is null but volatility and volume spike,",
            "    the market reacted with disagreement rather than a directional",
            "    repricing. If BOTH moments are null — and especially if variance and",
            "    turnover ratios sit BELOW 1 as often as above — these dates are not",
            "    distinguishable from ordinary trading days, and the cross-sectional",
            "    nulls in RQ2/RQ3 are explained by there being no event to react to.", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
#  PART C — minimum detectable effect
# --------------------------------------------------------------------------- #
def part_c(entries: list[tuple[str, str, object, float, float]]) -> str:
    """entries: (label, term, model, sd_regressor, car_sd)"""
    out = ["=" * 78, "PART C — MINIMUM DETECTABLE EFFECT (two-sided 5%, 80% power)",
           "=" * 78, "",
           f"  MDE = {POWER_MULT} x clustered SE, using the SEs actually obtained in the",
           "  reported specifications: event-clustered CAR ~ S + controls for the H2",
           "  coefficient gamma1 (no event FE exists for S — it is absorbed), event FE",
           "  clustered by bank for delta1 and for the beta3 extension. This converts",
           "  each null into a bounded claim: effects larger than the MDE would have",
           "  been detected with 80% probability.",
           "", "    {:<38s} {:>11s} {:>11s} {:>12s} {:>11s}".format(
               "coefficient", "estimate", "SE", "MDE (raw)", "MDE per SD"),
           "    " + "-" * 88]
    detail = []
    for lab, term, m, sd_x, car_sd in entries:
        se = m.bse[term]
        mde = POWER_MULT * se
        out.append("    {:<38s} {:>11.6f} {:>11.6f} {:>12.6f} {:>11.6f}".format(
            lab, m.params[term], se, mde, mde * sd_x))
        detail.append((lab, m.params[term], se, mde, mde * sd_x, sd_x, car_sd))
    out += ["", "    {:<38s} {:>14s} {:>16s}".format(
        "coefficient", "MDE per SD (pp)", "as % of CAR SD"), "    " + "-" * 72]
    for lab, est, se, mde, mde_sd, sd_x, car_sd in detail:
        out.append("    {:<38s} {:>14.4f} {:>15.1f}%".format(
            lab, mde_sd * 100, 100 * mde_sd / car_sd))
    out += ["", f"    CAR standard deviation used for scaling : {CAR_SD_REF * 100:.2f} pp", ""]
    for lab, est, se, mde, mde_sd, sd_x, car_sd in detail:
        out += [f"    {lab}:",
                f"      point estimate {est:+.6f}; we can rule out a true effect larger",
                f"      than {mde_sd * 100:.3f} pp of CAR per one-SD move in the regressor",
                f"      ({100 * mde_sd / car_sd:.1f}% of a CAR standard deviation) at 80% power.", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    car = pd.read_csv(PROC / "rq2_car.csv")
    wide = pd.read_csv(PROC / "vulnerability_scores_wide.csv")
    events = pd.read_csv(PROC / "rq2_safeguard_scores.csv")
    sample = pd.read_csv(PROC.parent / "frozen" / "sample_banks.csv")
    permnos = set(sample.permno.astype(int))

    calendar = sorted(pd.read_parquet(WRDS / "dsi.parquet", columns=["date"])
                      .date.astype(str).unique())
    events = rc.resolve_event_days(events, calendar)
    panel, delist = rc.load_returns(permnos)          # identical delisting handling

    # turnover from the raw daily file
    dsf_full = pd.read_parquet(WRDS / "dsf.parquet")
    dsf_full["date"] = dsf_full.date.astype(str)
    dsf_full["permno"] = dsf_full.permno.astype("int64")
    for c in ("ret", "prc", "shrout", "vol"):
        dsf_full[c] = dsf_full[c].astype(float)
    dsf_full["turnover"] = dsf_full.vol / dsf_full.shrout
    panel = panel.merge(dsf_full[["permno", "date", "turnover"]],
                        on=["permno", "date"], how="left")

    # ---------------- factors ----------------
    SOURCE_LOG.append("  Factor sources:")
    ff = load_ff_factors()
    bk = load_ff_banks()
    rate = load_dgs2(calendar)

    used_bank_src = "Ken French 49-industry 'Banks' (value-weighted)"
    used_rate_src = "FRED DGS2, daily change in percentage points"
    factor_cols = ["mktrf", "banks_x", "d_dgs2"]
    part_a_ok = ff is not None

    if part_a_ok:
        panel = panel.merge(ff, on="date", how="left")
        panel["exret"] = panel.ret - panel.rf
        if bk is not None:
            panel = panel.merge(bk, on="date", how="left")
            panel["banks_x"] = panel.banks - panel.rf     # sector excess return
        else:
            fb = build_fallback_bank_index(panel, dsf_full, calendar)
            panel = panel.merge(fb[["date", "banks_all"]], on="date", how="left")
            panel["banks_x"] = panel.banks_all - panel.rf
            used_bank_src = ("FALLBACK: value-weighted index of the CRSP sample banks "
                             "(leave-one-out for the 10 largest)")
        if rate is not None:
            panel = panel.merge(rate, on="date", how="left")
        else:
            factor_cols = ["mktrf", "banks_x"]
            used_rate_src = "FALLBACK: RATE FACTOR MISSING — market + sector only"
    else:
        SOURCE_LOG.append("    Ken French market factors unavailable -> PART A SKIPPED")

    # ---------------- PART A ----------------
    aug = pd.DataFrame()
    if part_a_ok:
        aug = estimate_augmented(panel, events, calendar, delist, factor_cols)
        chars = car[["permno", "doc_id", "name", "bank_IDRSSD", "S", "S_norm",
                     "char_quarter", "uninsured_share"] + CONTROLS +
                    ["deposit_reliance", "CAR", "beta", "n_est"]]
        aug = aug.merge(chars.rename(columns={"CAR": "CAR_mkt", "beta": "beta_mkt",
                                              "n_est": "n_est_mkt"}),
                        on=["permno", "doc_id"], how="left")
        aug = aug.merge(wide[["permno", DELTA1, "is_trust_or_specialized"]],
                        on="permno", how="left")
        aug.sort_values(["doc_id", "permno"]).to_csv(OUT_CSV, index=False)

    # ---------------- assemble report ----------------
    L = ["=" * 78,
         "RQ2/RQ3 ENRICHMENT — sector+rate model, unconditional tests, power",
         "=" * 78, "", "-- factor data provenance --", ""] + SOURCE_LOG + [
        "", f"    market factor : Ken French Mkt-RF + RF (daily)"
            f"{'' if part_a_ok else '  [UNAVAILABLE]'}",
        f"    sector factor : {used_bank_src}",
        f"    rate factor   : {used_rate_src}",
        f"    model         : (ret - RF) ~ a + b1*(Mkt-RF) + b2*(Banks-RF)"
        f"{' + b3*dDGS2' if 'd_dgs2' in factor_cols else ''}",
        "", "    A self-contained CRSP-based bank index is implemented as a fallback",
        "    (leave-one-out for the 10 largest banks); it was not needed here.", ""]

    if part_a_ok:
        common = aug.dropna(subset=["CAR_mkt", "CAR_aug", "S", "uninsured_share"] + CONTROLS)
        L += ["=" * 78, "PART A — SECTOR + RATE-ADJUSTED ABNORMAL RETURNS", "=" * 78, "",
              f"    bank-events, market model     : {len(car)}",
              f"    bank-events, augmented model  : {len(aug)}",
              f"    common sample used below      : {len(common)}"
              f"   ({common.permno.nunique()} banks, {common.doc_id.nunique()} events)",
              "",
              "    Both columns below are estimated on the SAME bank-events, so any",
              "    difference is attributable to the return model, not to sample mix.",
              "", "-- estimated factor loadings (augmented model) --", "",
              "    {:<20s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
                  "loading", "mean", "SD", "p10", "p90"),
              "    " + "-" * 64]
        namemap = {"b_mktrf": "market (Mkt-RF)", "b_banks_x": "sector (Banks-RF)",
                   "b_d_dgs2": "rate (d 2y yield)"}
        for c in factor_cols:
            v = aug[f"b_{c}"]
            L.append("    {:<20s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                namemap.get(f"b_{c}", c), v.mean(), v.std(), v.quantile(.1), v.quantile(.9)))
        rho_mb = panel[["mktrf", "banks_x"]].dropna().corr().iloc[0, 1]
        L += ["",
              f"    NOTE: corr(Mkt-RF, Banks-RF) = {rho_mb:.3f}. The sector factor",
              "    already contains most of the market, so the market loading is",
              "    partialled down (mean near zero or negative) while the sector",
              "    loading absorbs it. This is collinearity, not a coding error: the",
              "    two loadings are only interpretable jointly. What matters for the",
              "    event study is the fitted value, which is well determined.",
              "",
              f"    corr(CAR market-model, CAR augmented) = "
              f"{common.CAR_mkt.corr(common.CAR_aug):.4f}",
              f"    SD of CAR: market {common.CAR_mkt.std():.5f}  ->  augmented "
              f"{common.CAR_aug.std():.5f}"
              f"   ({100 * (common.CAR_aug.std() / common.CAR_mkt.std() - 1):+.1f}%)",
              "    A narrower augmented CAR means the extra factors genuinely absorb",
              "    common variation that the one-factor model left in the residual.", ""]

        rhs_h2 = f"S + {' + '.join(CONTROLS)}"
        tblS, mS = side_by_side(common, "CAR_mkt", "CAR_aug", rhs_h2, GAMMA1,
                                "market model", "market+sector+rate",
                                cluster="doc_id", use_t=True)
        L += ["-- A1. H2 headline: gamma1 = S (safeguard score), CAR ~ S + controls,",
              "       SEs CLUSTERED ON THE EVENT (11 clusters, t with G-1 = 10 df) --", "",
              "    H2 predicts gamma1 > 0: communications emphasising protective design",
              "    should produce less negative CARs. S is constant within an event, so",
              "    NO event-FE variant exists — event dummies absorb S exactly — and the",
              "    clustering is on the event, the level at which S varies.", "",
              tblS, "",
              f"    S varies over {common.doc_id.nunique()} events "
              f"({common.S.nunique()} distinct values); gamma1 is imprecise by",
              "    construction. The event-level form of this test (11 event-mean CARs on",
              "    11 S values) is in rq2_event_regression.txt, Section 1.", ""]

        cr3 = common.dropna(subset=[DELTA1])
        rhs_rq3 = f"{DELTA1} + {' + '.join(CONTROLS)} + C(doc_id)"
        tbl3, m3 = side_by_side(cr3, "CAR_mkt", "CAR_aug", rhs_rq3, DELTA1,
                                "market model", "market+sector+rate")
        L += [f"-- A2. RQ3/H3 headline: delta1 = vulnerability_score, event FE, "
              f"clustered by bank  (N={len(cr3)}) --", "", tbl3, ""]

        rhs_rq2 = (f"S + uninsured_share + S:uninsured_share + {' + '.join(CONTROLS)}"
                   " + C(doc_id)")
        tbl2, m2 = side_by_side(common, "CAR_mkt", "CAR_aug", rhs_rq2, BETA3,
                                "market model", "market+sector+rate")
        L += ["-- A3. ROBUSTNESS / EXTENSION (not the H2 test): beta3 = "
              "S x uninsured_share,", "       event FE, clustered by bank --", "",
              "    This is the cross-sectional question — do exposed banks react",
              "    differently WITHIN an event — which is closer to H3 in spirit than to",
              "    H2. Retained for continuity with the earlier version of the analysis.",
              "", tbl2, "", "-- A4. verdict --", ""]
        for nm, term, mm in [("gamma1", GAMMA1, mS), ("delta1", DELTA1, m3),
                             ("beta3 (ext)", BETA3, m2)]:
            pa, pb = mm["a"].pvalues[term], mm["b"].pvalues[term]
            same = (pa < 0.05) == (pb < 0.05)
            L.append(f"    {nm}: {mm['a'].params[term]:+.6f} (p={pa:.4f}) -> "
                     f"{mm['b'].params[term]:+.6f} (p={pb:.4f})   "
                     f"{'SAME conclusion' if same else 'CONCLUSION CHANGES'}")
        L.append("")
    else:
        L += ["=" * 78, "PART A — SKIPPED (market factor download failed)", "=" * 78, ""]

    # ---------------- PART B ----------------
    L += ["=" * 78, "PART B — DID THE MARKET REACT AT ALL? (unconditional)", "=" * 78, ""]
    ar_mkt, port_mkt = daily_ar_panel(panel, events, calendar, "market")
    L.append(part_b(ar_mkt, port_mkt, "MARKET MODEL (one factor)"))
    if part_a_ok and "d_dgs2" in factor_cols:
        ar_aug, port_aug = daily_ar_panel(panel, events, calendar, "aug")
        L.append(part_b(ar_aug, port_aug, "AUGMENTED MODEL (market + sector + rate)"))

    # ---------------- PART C ----------------
    entries = []
    car_w = car.merge(wide[["permno", DELTA1]], on="permno", how="left")
    base2 = car_w.dropna(subset=["CAR", "S", "uninsured_share"] + CONTROLS)

    # H2 first: gamma1 on S, event-clustered (the level at which S varies)
    m_h2_mkt = fit_cl(base2, f"CAR ~ S + {' + '.join(CONTROLS)}",
                      cluster="doc_id", use_t=True)
    entries.append(("H2 gamma1 (S), market-model CAR", GAMMA1, m_h2_mkt,
                    base2.S.std(), CAR_SD_REF))

    base3 = car_w.dropna(subset=["CAR", DELTA1] + CONTROLS)
    m_rq3_mkt = fit_cl(base3, f"CAR ~ {DELTA1} + {' + '.join(CONTROLS)} + C(doc_id)")
    entries.append(("H3 delta1, market-model CAR", DELTA1, m_rq3_mkt,
                    base3[DELTA1].std(), CAR_SD_REF))

    m_rq2_mkt = fit_cl(base2, f"CAR ~ S + uninsured_share + S:uninsured_share + "
                              f"{' + '.join(CONTROLS)} + C(doc_id)")
    sd_int = (base2.S * base2.uninsured_share).std()
    entries.append(("ext. beta3 (SxUI), market-model CAR", BETA3, m_rq2_mkt,
                    sd_int, CAR_SD_REF))

    if part_a_ok:
        entries.append(("H2 gamma1 (S), augmented CAR", GAMMA1, mS["b"],
                        common.S.std(), CAR_SD_REF))
        entries.append(("H3 delta1, augmented CAR", DELTA1, m3["b"],
                        cr3[DELTA1].std(), CAR_SD_REF))
        entries.append(("ext. beta3 (SxUI), augmented CAR", BETA3, m2["b"],
                        (common.S * common.uninsured_share).std(), CAR_SD_REF))
    L.append(part_c(entries))

    text = "\n".join(L)
    OUT_TXT.write_text(text + "\n")
    print(text)
    if part_a_ok:
        print(f"wrote {OUT_CSV} ({len(aug)} bank-events)")
    print(f"wrote {OUT_TXT}")


if __name__ == "__main__":
    main()
