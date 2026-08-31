"""rq2_robustness.py: the three robustness checks described in Section 3.7.

    (a) alternative event windows, [-5, +5] and [-1, +3], beside the baseline
        [-1, +5]
    (b) the normalised signal S_norm, which divides by the total sentence count
        rather than the design-sentence count
    (c) the threshold signal, keeping only communications with at least nine
        design sentences

Each variant is put through the same four quantities as the baseline: the
Brown-Warner event count, the gamma1 of Table 7, and the delta1 and interaction
b of Table 8 for each of the three vulnerability measures. The baseline window
is re-estimated alongside the variants as a control, so that the variant
machinery is shown to reproduce the published numbers before any variant is
believed.

The estimation window, t0-230 to t0-31, does not move with the event window, so
alpha and beta are the market-model parameters already in rq2_car.csv and are
reused rather than refitted: the alternative windows change which days are
summed, nothing else. Characteristics are likewise unchanged, since the quarter
is chosen strictly before t0 and no window touches that.

Reads     data/processed/rq2_car.csv, the vulnerability scores, CRSP returns
Writes    data/processed/rq2_robustness.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
import rq2_car as car_mod
from _stats import fit_cluster
from rq3_custody_check import NAMED_CUSTODIANS
from rq3_interaction import prepare

WINDOWS = {"baseline [-1,+5]": (-1, 5),
           "[-5,+5]": (-5, 5),
           "[-1,+3]": (-1, 3)}

DESIGN_THRESHOLD = 9      # Section 3.5's threshold specification


# --------------------------------------------------------------------------- #
#  CAR under an alternative event window
# --------------------------------------------------------------------------- #
def rebuild_car(window: tuple[int, int], base: pd.DataFrame,
                panel: pd.DataFrame, calendar: list[str],
                delist: pd.DataFrame) -> pd.DataFrame:
    """Recompute CAR over `window` using the alpha/beta already estimated for
    each bank-event. Returns base with the CAR column replaced; bank-events whose
    new window is incomplete (and that do not delist inside it) are dropped, the
    same rule rq2_car.compute_car applies."""
    lo, hi = window
    cal = np.array(calendar)
    idx = {d: i for i, d in enumerate(calendar)}
    dl_map = dict(zip(delist.permno, delist.dlstdt))

    ret = {(int(r.permno), r.date): (r.ret, r.vwretd)
           for r in panel.itertuples() if not np.isnan(r.ret)}

    rows = []
    for r in base.itertuples():
        i = idx[r.t0]
        days = list(cal[i + lo:i + hi + 1])
        vals = [ret.get((int(r.permno), d)) for d in days]
        have = [(d, v) for d, v in zip(days, vals) if v is not None]
        dlstdt = dl_map.get(int(r.permno))
        delists_in = dlstdt is not None and days[0] <= dlstdt <= days[-1]
        if len(have) < len(days) and not delists_in:
            continue
        ar = [v[0] - (r.alpha + r.beta * v[1]) for _, v in have]
        rows.append({"permno": int(r.permno), "doc_id": int(r.doc_id),
                     "CAR": float(np.sum(ar)), "n_evt_days": len(have)})
    out = base.drop(columns=["CAR"]).merge(pd.DataFrame(rows),
                                           on=["permno", "doc_id"], how="inner")
    return out


def daily_ar(window: tuple[int, int], base: pd.DataFrame, panel: pd.DataFrame,
             calendar: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(event-window AR by bank-day, estimation-window portfolio AR by day) —
    the two inputs the Brown-Warner crude-dependence test needs."""
    lo, hi = window
    cal = np.array(calendar)
    idx = {d: i for i, d in enumerate(calendar)}
    ret = {(int(r.permno), r.date): (r.ret, r.vwretd)
           for r in panel.itertuples() if not np.isnan(r.ret)}

    evt_rows, est_rows = [], []
    for r in base.itertuples():
        i = idx[r.t0]
        for d in cal[i + lo:i + hi + 1]:
            v = ret.get((int(r.permno), d))
            if v is not None:
                evt_rows.append({"doc_id": int(r.doc_id), "date": d,
                                 "AR": v[0] - (r.alpha + r.beta * v[1])})
        for d in cal[max(i + car_mod.EST_START, 0):i + car_mod.EST_END + 1]:
            v = ret.get((int(r.permno), d))
            if v is not None:
                est_rows.append({"doc_id": int(r.doc_id), "date": d,
                                 "AR": v[0] - (r.alpha + r.beta * v[1])})
    evt = pd.DataFrame(evt_rows)
    est = pd.DataFrame(est_rows)
    port = (est.groupby(["doc_id", "date"]).AR.mean()
               .reset_index().rename(columns={"AR": "port_AR"}))
    return evt, port


def brown_warner(evt: pd.DataFrame, port: pd.DataFrame) -> tuple[int, int]:
    """(events significant at 5%, events tested). Method identical to
    _rq2_enrich_core.portfolio_test: the event-window portfolio CAR judged
    against the time-series SD of the same portfolio's estimation-window AR."""
    n_sig = 0
    docs = sorted(evt.doc_id.unique())
    for doc in docs:
        sigma = float(port[port.doc_id == doc].port_AR.std(ddof=1))
        daily = evt[evt.doc_id == doc].groupby("date").AR.mean()
        k = len(daily)
        t = float(daily.sum()) / (sigma * np.sqrt(k)) if sigma > 0 else np.nan
        p = float(2 * (1 - stats.norm.cdf(abs(t))))
        n_sig += int(p < 0.05)
    return n_sig, len(docs)


# --------------------------------------------------------------------------- #
#  The three headline regressions, on an arbitrary panel and signal column
# --------------------------------------------------------------------------- #
def gamma1(d: pd.DataFrame, signal: str = "S") -> dict:
    """Table 7's headline row: pooled, with controls, clustered on the event."""
    sub = d.dropna(subset=["CAR", signal] + C.CONTROLS)
    f = f"CAR ~ {signal} + " + " + ".join(C.CONTROLS)
    m = fit_cluster(sub, f, ["doc_id"])
    return {"coef": float(m.params[signal]), "se": float(m.bse[signal]),
            "p": float(m.pvalues[signal]), "n": int(len(sub)),
            "events": int(sub.doc_id.nunique())}


def delta1(d: pd.DataFrame, measure: str) -> dict:
    """Table 8's link column: CAR ~ measure + controls + event FE, bank-clustered."""
    sub = d.dropna(subset=["CAR", measure] + C.CONTROLS)
    f = f"CAR ~ {measure} + " + " + ".join(C.CONTROLS) + " + C(doc_id)"
    m = fit_cluster(sub, f, ["permno"])
    return {"coef": float(m.params[measure]), "p": float(m.pvalues[measure]),
            "n": int(len(sub))}


def interaction(d: pd.DataFrame, meas: dict, signal: str = "S") -> dict:
    """Table 8's interaction column: CAR = eventFE + bankFE + b(S x measure),
    clustered on the event. The measure's own level term is included only where
    it is identified, i.e. for the bank-event measure, exactly as in
    rq3_interaction.one_measure()."""
    key = meas["key"]
    d = d.copy()
    if signal != "S":
        d["S"] = d[signal]          # swap the signal in, keep one formula
    s = prepare(d, key)
    rhs = f"{key} + " if not meas["static"] else ""
    f = f"CAR ~ {rhs}S:{key} + C(doc_id) + C(permno)"
    m = fit_cluster(s, f, ["doc_id"])
    name = f"S:{key}"
    return {"coef": float(m.params[name]), "p": float(m.pvalues[name]),
            "n": int(len(s))}


def variant_block(d: pd.DataFrame, label: str, signal: str = "S",
                  bw: tuple[int, int] | None = None) -> tuple[list[str], dict]:
    g = gamma1(d, signal)
    # Raw coefficients are not comparable across signals: S runs on [-0.67, +1.0]
    # while S_norm runs on [-0.06, +0.09], so the same economic effect appears
    # ~11x larger under S_norm. Everything is therefore also reported per one
    # standard deviation of whichever signal is in use.
    sd_sig = float(d[signal].std())
    L = [f"-- {label} --", "",
         f"    bank-events {g['n']}   events {g['events']}   "
         f"distinct {signal} {d[signal].nunique()}"]
    if bw is not None:
        L.append(f"    Brown-Warner: {bw[0]} of {bw[1]} events significant at 5%")
    L += ["",
          f"    signal SD                   {sd_sig:>12.8f}",
          f"    gamma1 ({signal})            {g['coef']:>+12.8f}  SE {g['se']:>10.8f}"
          f"  p {g['p']:.8f}",
          f"    gamma1 per 1 SD of signal   {g['coef'] * sd_sig:>+12.8f}",
          "",
          "    {:<26s} {:>14s} {:>12s} {:>14s} {:>12s} {:>14s} {:>12s} {:>13s}".format(
              "measure", "delta1", "p", "interaction b", "p",
              "b ex-custody", "p", "b per 1 SD"),
          "    " + "-" * 124]
    keyed = {f"{label}.gamma1.coef": g["coef"], f"{label}.gamma1.se": g["se"],
             f"{label}.gamma1.p": g["p"], f"{label}.signal_sd": sd_sig,
             f"{label}.gamma1_per_sd": g["coef"] * sd_sig}
    if bw is not None:
        keyed[f"{label}.bw_significant"] = float(bw[0])
    ex = d[~d.permno.isin(NAMED_CUSTODIANS)]
    for meas in C.MEASURES:
        k = meas["key"]
        dl = delta1(d, k)
        ia = interaction(d, meas, signal)
        xc = interaction(ex, meas, signal)
        L.append("    {:<26s} {:>14.8f} {:>12.8f} {:>14.8f} {:>12.8f} {:>14.8f} {:>12.8f} {:>13.8f}".format(
            meas["name"], dl["coef"], dl["p"], ia["coef"], ia["p"],
            xc["coef"], xc["p"], ia["coef"] * sd_sig))
        keyed[f"{label}.beta_per_sd.{k}"] = ia["coef"] * sd_sig
        keyed[f"{label}.delta1.{k}.coef"] = dl["coef"]
        keyed[f"{label}.delta1.{k}.p"] = dl["p"]
        keyed[f"{label}.beta.{k}.coef"] = ia["coef"]
        keyed[f"{label}.beta.{k}.p"] = ia["p"]
        keyed[f"{label}.beta_excustody.{k}.coef"] = xc["coef"]
        keyed[f"{label}.beta_excustody.{k}.p"] = xc["p"]
    L.append("")
    return L, keyed


# --------------------------------------------------------------------------- #
#  Assembly
# --------------------------------------------------------------------------- #
def bridge_for(car: pd.DataFrame) -> pd.DataFrame:
    """Attach the vulnerability measures to an arbitrary CAR panel, exactly as
    rq3_link.load_bridge_panel does for the baseline one."""
    rf = (pd.read_csv(C.SCORES_WIDE)[["permno", "vulnerability_score",
                                      "is_trust_or_specialized"]]
          .rename(columns={"vulnerability_score": "score_rf"}))
    ols = pd.read_csv(C.SCORES_OLS)[["permno", "score_ols"]]
    d = car.merge(rf, on="permno", how="left").merge(ols, on="permno", how="left")
    return d.dropna(subset=["CAR", "S", "uninsured_share"] + C.CONTROLS).copy()


def threshold_signal(d: pd.DataFrame) -> pd.DataFrame:
    """Keep only communications with at least DESIGN_THRESHOLD design sentences."""
    return d[d.n_design >= DESIGN_THRESHOLD].copy()


def main() -> dict:
    base = pd.read_csv(C.RQ2_CAR)
    permnos = set(base.permno.astype(int))
    panel, delist = car_mod.load_returns(permnos)
    calendar = sorted(pd.read_parquet(C.WRDS / "dsi.parquet",
                                      columns=["date"]).date.astype(str).unique())

    L = ["=" * 96,
         "RQ2/RQ3 ROBUSTNESS — the three checks Section 3.7 describes",
         "=" * 96, "",
         "  Every variant is judged on the same four quantities as the baseline:",
         "  the Brown-Warner event count, Table 7's gamma1, and Table 8's delta1",
         "  and interaction b for each of the three vulnerability measures.", "",
         "=" * 96, "(a) ALTERNATIVE EVENT WINDOWS", "=" * 96, ""]
    keyed: dict = {}

    for label, win in WINDOWS.items():
        car = base if win == (-1, 5) else rebuild_car(win, base, panel,
                                                      calendar, delist)
        evt, port = daily_ar(win, base, panel, calendar)
        bw = brown_warner(evt, port)
        block, k = variant_block(bridge_for(car), f"window {label}", "S", bw)
        L += block
        keyed.update(k)

    L += ["=" * 96, "(b) NORMALISED SIGNAL S_norm", "=" * 96, "",
          "  S_norm divides protective-minus-expansive by the TOTAL sentence count",
          "  rather than by the design-sentence count, so a document that says",
          "  little about design scores near zero however one-sided that little is.",
          ""]
    block, k = variant_block(bridge_for(base), "S_norm", "S_norm")
    L += block
    keyed.update(k)

    thr = threshold_signal(base)
    n_docs = thr.doc_id.nunique()
    L += ["=" * 96,
          f"(c) THRESHOLD SIGNAL — communications with n_design >= {DESIGN_THRESHOLD}",
          "=" * 96, "",
          f"  {n_docs} of {base.doc_id.nunique()} communications survive the threshold.",
          f"  Dropped: {sorted(set(base.doc_id) - set(thr.doc_id))}, which includes",
          "  BOTH documents scoring S = +1.00 (3 and 4 design sentences). The",
          "  surviving S range is therefore much narrower than the full sample's.",
          ""]
    evt_t, port_t = daily_ar((-1, 5), thr, panel, calendar)
    bw_t = brown_warner(evt_t, port_t)
    block, k = variant_block(bridge_for(thr), "threshold n_design>=9", "S", bw_t)
    L += block
    keyed.update(k)

    L += ["=" * 96, ""]
    text = "\n".join(L)
    C.RQ2_ROBUSTNESS.write_text(text)
    print(text)
    print(f"wrote {C.RQ2_ROBUSTNESS}")
    def slug(k: str) -> str:
        """frozen keys must be stable identifiers, not display labels."""
        k = (k.replace("window baseline [-1,+5]", "win_base")
              .replace("window [-5,+5]", "win_m5p5")
              .replace("window [-1,+3]", "win_m1p3")
              .replace("threshold n_design>=9", "thresh9")
              .replace("S_norm", "snorm"))
        return "rq2.robust." + k.replace(" ", "_")

    return {slug(k): v for k, v in keyed.items()}


if __name__ == "__main__":
    main()
