"""
rq3_link.py — RQ3, part 1 of 3: the LEVEL link between vulnerability and CAR.

HYPOTHESIS   H3 — banks the RQ1 model calls more vulnerable should show more
             negative abnormal returns around CBDC communications.
COEFFICIENT  delta1 in   CAR = delta0 + delta1 * vulnerability + controls
             H3 predicts delta1 < 0.

Run for all THREE vulnerability measures, every one oriented HIGHER = MORE
VULNERABLE (see config.MEASURES for the exact construction of each):
    uninsured_share   raw balance-sheet ratio, bank-EVENT level
    score_rf          leakage-free RandomForest score, STATIC per bank
    score_ols         leakage-free OLS score, same split, STATIC per bank

This file answers only the LEVEL question: do vulnerable banks have lower CARs in
these windows? Whether that is a REACTION to the communication is a different
question, and it is answered by the interaction in rq3_interaction.py. The two
are put side by side in rq3_measures.py, which is where the level-vs-reaction
diagnostic lives.

Inputs   data/processed/rq2_car.csv                    (rq2_car.py, unchanged)
         data/processed/vulnerability_scores_wide.csv  (rq1_scores.py)
         data/processed/vulnerability_scores_ols.csv   (rq1_scores.py)
Output   data/processed/rq3_link.txt
         data/processed/rq3_bridge.csv   (the merged bank-event panel RQ3 uses)

Standard errors are clustered on the BANK here, the level at which the
vulnerability measures vary. The interaction script clusters differently and
says why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from _stats import coef_header, coef_row, fit_cluster, mde, stars


# --------------------------------------------------------------------------- #
#  The RQ3 panel: CAR + all three vulnerability measures on one frame
# --------------------------------------------------------------------------- #
def load_bridge_panel(save: bool = False) -> pd.DataFrame:
    """CAR bank-events joined to every vulnerability measure. Shared by all three
    RQ3 scripts so they cannot disagree about the sample."""
    car = pd.read_csv(C.RQ2_CAR)
    rf = (pd.read_csv(C.SCORES_WIDE)[["permno", "vulnerability_score",
                                      "is_trust_or_specialized"]]
          .rename(columns={"vulnerability_score": "score_rf"}))
    ols = pd.read_csv(C.SCORES_OLS)[["permno", "score_ols"]]

    d = car.merge(rf, on="permno", how="left").merge(ols, on="permno", how="left")
    d = d.dropna(subset=["CAR", "S", "uninsured_share"] + C.CONTROLS).copy()
    if save:
        d.to_csv(C.RQ3_BRIDGE_CSV, index=False)
    return d


def bank_level(d: pd.DataFrame) -> pd.DataFrame:
    """One row per bank, carrying all three measures (for correlations)."""
    return (d.dropna(subset=C.MEASURE_KEYS)
              .groupby("permno")[C.MEASURE_KEYS].first().reset_index())


# --------------------------------------------------------------------------- #
#  delta1 for one measure
# --------------------------------------------------------------------------- #
def link_models(d: pd.DataFrame) -> dict:
    """CAR ~ measure + controls, pooled and with event FE, per measure."""
    out = {}
    for m in C.MEASURE_KEYS:
        sub = d.dropna(subset=["CAR", m] + C.CONTROLS)
        base = f"CAR ~ {m} + " + " + ".join(C.CONTROLS)
        out[(m, "pooled")] = (fit_cluster(sub, base, ["permno"]), sub)
        out[(m, "event FE")] = (fit_cluster(sub, base + " + C(doc_id)",
                                            ["permno"]), sub)
    return out


def report(d: pd.DataFrame, models: dict) -> str:
    car_sd = float(d.CAR.std())
    nice = {m["key"]: m["name"] for m in C.MEASURES}

    L = ["=" * 78,
         "RQ3 (1/3) — THE LEVEL LINK: CAR ~ vulnerability + controls",
         "=" * 78, "",
         "  H3 predicts delta1 < 0: a bank the RQ1 model calls more vulnerable",
         "  should earn a more negative abnormal return around CBDC communications.",
         "",
         f"  sample: {len(d)} bank-events, {d.permno.nunique()} banks, "
         f"{d.doc_id.nunique()} events",
         f"  CAR standard deviation: {car_sd:.8f}  ({car_sd * 100:.6f} pp)",
         "  SEs clustered on the bank, t with G-1 df.",
         "",
         "  All three measures are oriented HIGHER = MORE VULNERABLE. Because they",
         "  live on very different scales, the comparable column is 'per 1 SD' —",
         "  delta1 x SD(measure), in CAR percentage points — and the MDE beside it.",
         "",
         "    {:<26s} {:<10s} {:>14s} {:>13s} {:>10s} {:>12s} {:>12s}  {}".format(
             "measure", "spec", "delta1", "SE", "p", "per 1 SD pp", "MDE pp", "sig"),
         "    " + "-" * 108]
    for m in C.MEASURE_KEYS:
        for spec in ("pooled", "event FE"):
            mod, sub = models[(m, spec)]
            sd_m = float(sub[m].std())
            b, se, p = mod.params[m], mod.bse[m], mod.pvalues[m]
            e = mde(se, sd_m, car_sd)
            L.append("    {:<26s} {:<10s} {:>14.8f} {:>13.8f} {:>10.6f} "
                     "{:>12.6f} {:>12.6f}  {}".format(
                         nice[m], spec, b, se, p, b * sd_m * 100,
                         e["per_sd_pp"], stars(p)))
        L.append("")

    # --- what the table says, stated rather than left to the reader ---
    ui = models[("uninsured_share", "event FE")][0]
    rf = models[("score_rf", "event FE")][0]
    ols = models[("score_ols", "event FE")][0]
    L += ["-- reading --", "",
          f"    uninsured_share  delta1 = {ui.params['uninsured_share']:+.8f}"
          f"   p = {ui.pvalues['uninsured_share']:.8f}  "
          f"{stars(ui.pvalues['uninsured_share'])}",
          f"    RF score         delta1 = {rf.params['score_rf']:+.8f}"
          f"   p = {rf.pvalues['score_rf']:.8f}  {stars(rf.pvalues['score_rf'])}",
          f"    OLS score        delta1 = {ols.params['score_ols']:+.8f}"
          f"   p = {ols.pvalues['score_ols']:.8f}  {stars(ols.pvalues['score_ols'])}",
          "",
          "    The two model-based scores are null and carry the WRONG sign for H3",
          "    (positive = more vulnerable, higher CAR). uninsured_share is",
          "    significant with the H3-predicted NEGATIVE sign.",
          "",
          "    Do not over-read the significant one. A negative level effect of",
          "    uninsured_share on CAR in these windows says vulnerable banks sat at a",
          "    lower average abnormal return; it does NOT say they were reacting to",
          "    the CBDC communication. Separating those two readings is exactly what",
          "    the interaction in rq3_interaction.py does, and the verdict is in",
          "    rq3_measures.py. Read them before quoting this coefficient.",
          "",
          "    Note also the published RQ3 null is a statement about the ML SCORE.",
          "    It does not extend to uninsured_share, and saying so would be wrong.",
          ""]
    return "\n".join(L)


def main() -> dict:
    d = load_bridge_panel(save=True)
    models = link_models(d)
    text = report(d, models)
    C.RQ3_LINK.write_text(text + "\n")
    print(text)
    print(f"wrote {C.RQ3_LINK}")
    print(f"wrote {C.RQ3_BRIDGE_CSV}")

    return {
        "rq3.delta1.uninsured.coef":
            models[("uninsured_share", "event FE")][0].params["uninsured_share"],
        "rq3.delta1.uninsured.p":
            models[("uninsured_share", "event FE")][0].pvalues["uninsured_share"],
        "rq3.delta1.score_rf.coef":
            models[("score_rf", "event FE")][0].params["score_rf"],
        "rq3.delta1.score_rf.p":
            models[("score_rf", "event FE")][0].pvalues["score_rf"],
        "rq3.delta1.score_ols.coef":
            models[("score_ols", "event FE")][0].params["score_ols"],
        "rq3.delta1.score_ols.p":
            models[("score_ols", "event FE")][0].pvalues["score_ols"],
    }


if __name__ == "__main__":
    main()
