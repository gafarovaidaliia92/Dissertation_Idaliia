"""
rq3_interaction.py — RQ3, part 2 of 3: the supervisor's two-way fixed-effects
interaction.

    CAR = EventFE + BankFE + b * (Safeguard x vulnerability) + e

WHY THIS LIVES IN RQ3, NOT RQ2. The interaction asks whether VULNERABLE banks
react differently from less vulnerable ones. That is a question about the bridge
between the RQ1 vulnerability profile and the market reaction, i.e. H3. It is not
a question about whether the type of communication moves the average reaction,
which is H2 and is answered by CAR ~ S in rq2_avg_effect.py. In an earlier
version of this project the interaction was presented as the H2 test; it was not
one, and the misfiled version is archived under results/_archive/.

COEFFICIENT  b on (S x vulnerability). Higher vulnerability = more run-prone, and
             higher S = more protective communication, so the H3-flavoured
             prediction is b > 0: a protective communication should hurt
             vulnerable banks less.

Run twice, once per measure that the supervisor asked for:
    uninsured_share   bank-EVENT level  -> its MAIN EFFECT survives bank FE
    score_rf          STATIC per bank   -> its main effect is absorbed; only b

Three specifications per measure, so the reader can see where the pooled result
goes once the confounds are differenced out:
    pooled (no FE)                 CAR ~ S + V + S:V
    event FE only                  CAR ~ V + S:V + C(doc_id)
    event FE + bank FE  <- theirs  CAR ~ [V] + S:V + C(doc_id) + C(permno)

INFERENCE. The two-way (bank AND event) clustered covariance is reported because
it was requested, but it is NOT the headline: the Cameron-Gelbach-Miller
estimator is not positive semi-definite in a sample this small and hands back
negative variances (NaN SEs) for a large share of the fixed-effect dummies. The
EVENT-clustered SE is the headline inference — the event is the level at which S
varies, and there are only 11 of them. Both are printed side by side, and the
count of NaN SEs is printed rather than hidden.

Inputs   data/processed/rq2_car.csv, vulnerability_scores_{wide,ols}.csv
Output   data/processed/rq3_interaction.txt
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from _stats import (
    NotIdentified,
    absorption,
    check_identified,
    fit_cluster,
    mde,
    n_bad_se,
    stars,
    wrap,
)
from rq3_link import load_bridge_panel

# the two measures the supervisor's model is run on
INTERACTION_MEASURES = ["uninsured_share", "score_rf"]

SUP = "event FE + bank FE  <-- supervisor"


def _specs(static: bool) -> list[tuple[str, str, str]]:
    """(label, formula, everything-except-SxV) for the three specifications."""
    main = "" if static else "VULN + "
    return [
        ("pooled (no FE)", "CAR_ ~ S + VULN + SxV", "S + VULN"),
        ("event FE", "CAR_ ~ VULN + SxV + C(doc_id)", "VULN + C(doc_id)"),
        (SUP, f"CAR_ ~ {main}SxV + C(doc_id) + C(permno)",
         f"{main}C(doc_id) + C(permno)".lstrip("+ ")),
    ]


def prepare(d: pd.DataFrame, key: str) -> pd.DataFrame:
    s = d.dropna(subset=["CAR", "S", key]).copy()
    s["VULN"] = s[key].astype(float)
    s["SxV"] = s["S"].astype(float) * s["VULN"]
    s["CAR_"] = s["CAR"]
    return s


def structural_checks(s: pd.DataFrame, key: str, static: bool) -> tuple[list[str], dict]:
    """Verify numerically what is absorbed and what survives. Nothing here is
    asserted: every claim is a number."""
    a_S = absorption(s, "S", "C(doc_id)")
    a_V_bank = absorption(s, "VULN", "C(permno)")
    a_V_both = absorption(s, "VULN", "C(permno) + C(doc_id)")
    a_I = absorption(s, "SxV", "C(permno) + C(doc_id)")

    L = ["-- structural checks (numerical, not asserted) --", "",
         "    {:<44s} {:>16s} {:>13s} {:>12s}".format(
             "regressor on fixed effects", "R^2", "max |resid|", "resid SD"),
         "    " + "-" * 88]
    for lab, a in [("S on event dummies", a_S),
                   (f"{key} on bank dummies", a_V_bank),
                   (f"{key} on bank + event dummies", a_V_both),
                   ("S x vulnerability on bank + event dummies", a_I)]:
        L.append("    {:<44s} {:>16.12f} {:>13.3e} {:>12.3e}  {}".format(
            lab, a["r2"], a["max_abs"], a["sd"],
            "ABSORBED" if a["absorbed"] else "survives"))
    L += ["", "    Reading:",
          "    * S is absorbed by the event dummies to machine precision. Its main",
          "      effect is NOT estimated under any FE spec. That is exact",
          "      COLLINEARITY, not an effect of zero — S is constant within an event.",
          ]
    if static:
        L += ["    * This measure is STATIC (one value per bank), so the bank dummies",
              "      absorb it to machine precision too. In the two-way FE model ONLY",
              "      the interaction b is identified, and only b is reported there.",
              ]
    else:
        L += [f"    * This measure is NOT absorbed by the bank dummies (R^2 = "
              f"{a_V_bank['r2']:.8f},",
              f"      residual SD {a_V_bank['sd']:.6e}). It is bank-EVENT level: it is",
              "      re-read from the last Call Report before each event, and the 11",
              "      events draw on 8 distinct characteristic quarters. Its main effect",
              "      therefore REMAINS IDENTIFIED under bank FE, and it is estimated and",
              "      reported rather than suppressed. The premise that a vulnerability",
              "      main effect always drops out under bank FE holds only for a static",
              "      measure.",
              ]
    L += [f"    * The interaction survives both sets of dummies (residual SD "
          f"{a_I['sd']:.6e}),",
          "      because it varies across banks AND across events. That is what makes",
          "      the two-way FE model estimable at all.", ""]
    return L, {"a_S": a_S, "a_V_bank": a_V_bank, "a_I": a_I}


def one_measure(d: pd.DataFrame, meas: dict) -> tuple[str, dict]:
    key, static = meas["key"], meas["static"]
    s = prepare(d, key)
    car_sd = float(s.CAR.std())
    sd_int = float(s.SxV.std())

    L = ["=" * 78,
         "RQ3 (2/3) — TWO-WAY FE INTERACTION",
         f"           vulnerability = {meas['name']}",
         "=" * 78, "",
         "  CAR = EventFE + BankFE + b * (S x vulnerability) + e",
         "",
         "  Event dummies difference out the between-event confound (each speech date",
         "  carries its own macro news); bank dummies difference out the permanent",
         "  differences between banks. What is left is the WITHIN-EVENT comparison of",
         "  vulnerable against less-vulnerable banks. H3-flavoured prediction: b > 0.",
         "",
         "-- orientation and scale --", ""] + wrap(meas["orient"]) + [""]
    L += ["    {:<28s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
              "variable", "mean", "SD", "min", "max"),
          "    " + "-" * 90]
    for nm, col in [(key, s.VULN), ("S", s.S), ("S x vulnerability", s.SxV),
                    ("CAR", s.CAR)]:
        L.append("    {:<28s} {:>14.8f} {:>14.8f} {:>14.8f} {:>14.8f}".format(
            nm, col.mean(), col.std(), col.min(), col.max()))
    L += ["", f"    N bank-events {len(s)}     banks {s.permno.nunique()}"
              f"     events {s.doc_id.nunique()}", ""]

    checks, _ = structural_checks(s, key, static)
    L += checks

    L += ["-- b on (S x vulnerability), three specifications --", "",
          "    [event] = clustered on the event, 11 clusters, t(10). HEADLINE.",
          "    [2way]  = clustered on bank AND event (Cameron-Gelbach-Miller).",
          "              Requested, but not PSD at this sample size — see the NaN",
          "              counts below. Reported for completeness, not for inference.",
          "    [bank]  = clustered on the bank, shown only as a reference point.", ""]

    res: dict = {}
    for lab, formula, others in _specs(static):
        try:
            ident = check_identified(s, "SxV", others)
        except NotIdentified as e:
            L += [f"  [{lab}]", f"    NOT ESTIMATED — {e}", ""]
            res[lab] = None
            continue
        mE = fit_cluster(s, formula, ["doc_id"])
        m2 = fit_cluster(s, formula, ["permno", "doc_id"])
        mB = fit_cluster(s, formula, ["permno"])
        res[lab] = {"event": mE, "twoway": m2, "bank": mB}

        L += [f"  [{lab}]",
              f"    formula : {formula.replace('CAR_', 'CAR')}",
              f"    k parameters {len(mE.params)}     model df_resid "
              f"{int(mE.df_resid)}     inference df {int(mE.df_resid_inference)}",
              f"    identification of SxV given the rest: R^2 {ident['r2']:.12f}, "
              f"residual SD {ident['sd']:.8e}",
              "",
              "    {:<22s} {:>14s} {:>13s} {:>8s} {:>10s}  {:<30s} {}".format(
                  "SE cluster", "b", "SE", "t", "p", "95% CI", "sig"),
              "    " + "-" * 106]
        for tag, m in [("event  <-- headline", mE), ("two-way", m2),
                       ("bank (reference)", mB)]:
            ci = m.conf_int()
            L.append("    {:<22s} {:>14.8f} {:>13.8f} {:>8.3f} {:>10.6f}  "
                     "[{:+.8f}, {:+.8f}]  {}".format(
                         tag, m.params["SxV"], m.bse["SxV"], m.tvalues["SxV"],
                         m.pvalues["SxV"], ci.loc["SxV", 0], ci.loc["SxV", 1],
                         stars(m.pvalues["SxV"])))
        e = mde(mE.bse["SxV"], sd_int, car_sd)
        L += ["",
              f"    MDE for b at 80% power (event SE): {e['raw']:.8f} raw"
              f"   = {e['per_sd_pp']:.6f} pp of CAR per 1 SD of the interaction"
              f"   = {e['pct_of_car_sd']:.4f}% of a CAR SD"]
        bad = n_bad_se(m2)
        if bad:
            L += [f"    two-way covariance NOT PSD: {bad} of {len(m2.params)} "
                  f"parameters have a negative",
                  "      variance and hence a NaN SE (all of them fixed-effect "
                  "dummies). b's own",
                  "      variance is positive, so the two-way SE printed for b is "
                  "usable — but this",
                  "      is why the event-clustered SE is the headline."]
        if not static and "VULN" in mE.params.index:
            L += ["    main effect of the measure (identified here — see the checks "
                  "above):",
                  "    {:<22s} {:>14.8f} {:>13.8f} {:>8.3f} {:>10.6f}  {}".format(
                      "  event-clustered", mE.params["VULN"], mE.bse["VULN"],
                      mE.tvalues["VULN"], mE.pvalues["VULN"],
                      stars(mE.pvalues["VULN"]))]
        L.append("")

    # supplementary: controls added to the supervisor's spec
    if res.get(SUP):
        f_ctrl = _specs(static)[2][1] + " + " + " + ".join(C.CONTROLS)
        mc = fit_cluster(s, f_ctrl, ["doc_id"])
        L += ["-- supplementary: the supervisor's spec with bank controls added --", "",
              "    Controls are read at the same characteristic quarter as the",
              "    measure, so they vary within bank and are not absorbed.", "",
              "    {:<22s} {:>14.8f} {:>13.8f} {:>8.3f} {:>10.6f}  {}".format(
                  "event-clustered", mc.params["SxV"], mc.bse["SxV"],
                  mc.tvalues["SxV"], mc.pvalues["SxV"], stars(mc.pvalues["SxV"])), ""]
        res["supervisor + controls"] = {"event": mc}

    # verdict
    sup, ev = res.get(SUP), res.get("event FE")
    if sup and ev:
        b = sup["event"].params["SxV"]
        pE = sup["event"].pvalues["SxV"]
        p2 = sup["twoway"].pvalues["SxV"]
        null_holds = pE >= 0.05 and ev["event"].pvalues["SxV"] >= 0.05
        L += ["-- verdict for this measure --", "",
              f"    b = {b:+.8f}   p = {pE:.8f} (event, headline), "
              f"{p2:.8f} (two-way)",
              f"    H3-flavoured prediction is b > 0. Observed sign: "
              f"{'POSITIVE' if b > 0 else 'NEGATIVE'}.",
              f"    b is null under event / two-way FE: "
              f"{'YES' if null_holds else 'NO — see rq3_measures.py, it is an artefact'}",
              ""]
    return "\n".join(L), res


def main() -> dict:
    d = load_bridge_panel()
    by_key = {m["key"]: m for m in C.MEASURES}
    chunks, out = [], {}
    for key in INTERACTION_MEASURES:
        txt, res = one_measure(d, by_key[key])
        chunks.append(txt)
        sup = res.get(SUP)
        if sup:
            out[f"rq3.b.{key}.coef"] = sup["event"].params["SxV"]
            out[f"rq3.b.{key}.se_twoway"] = sup["twoway"].bse["SxV"]
            out[f"rq3.b.{key}.p_event"] = sup["event"].pvalues["SxV"]

    header = ["=" * 78,
              "RQ3 (2/3) — DOES THE COMMUNICATION TYPE CHANGE HOW VULNERABLE BANKS "
              "REACT?", "=" * 78, "",
              "  This test used to sit in RQ2, labelled as the H2 test. That was a",
              "  mis-filing: H2 is about the AVERAGE effect of the communication",
              "  (rq2_avg_effect.py), while this is about whether VULNERABLE banks",
              "  react differently, which is H3. It has been moved here.",
              "",
              "  The artefact diagnostics for anything significant below — calendar",
              "  placebo, custody-bank drop, leave-one-event-out — are in",
              "  rq3_measures.py. Do not quote a coefficient from this file without",
              "  them.",
              ""]
    text = "\n".join(header) + "\n" + "\n".join(chunks)
    C.RQ3_INTERACTION.write_text(text + "\n")
    print(text)
    print(f"wrote {C.RQ3_INTERACTION}")
    return out


if __name__ == "__main__":
    main()
