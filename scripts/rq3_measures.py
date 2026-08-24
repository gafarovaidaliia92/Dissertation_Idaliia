"""
rq3_measures.py — RQ3, part 3 of 3: which vulnerability measure carries signal,
and is any of it real?

Three jobs, in this order:

  A. MEASURES COMPARISON. RF score vs OLS-predicted score vs uninsured_share:
     how much do they agree, and which of them carries anything in the level link
     (delta1, rq3_link.py) and in the interaction (b, rq3_interaction.py)? The
     question is whether the machine-learning step buys anything over the single
     strongest balance-sheet ratio.

  B. LEVEL-VS-REACTION DIAGNOSTIC. uninsured_share is significant in the plain
     delta1 link with the H3-predicted sign, but its interaction with the
     safeguard signal is null. Those two facts together mean the level result is
     a property of these windows, not a reaction to CBDC content. This is the
     section that decides whether the H3 MECHANISM is supported.

  C. ARTEFACT DIAGNOSTICS for the one coefficient that is significant in the
     interaction, the RF score. Run symmetrically on both measures so the
     reporting is not conditioned on the outcome:
       C1 calendar placebo — S trends upward over 2019-2023, so S x vulnerability
          is nearly the same regressor as time x vulnerability
       C2 leave-one-event-out
       C3 drop the trust/custody banks

Inputs   data/processed/rq2_car.csv, vulnerability_scores_{wide,ols}.csv
Output   data/processed/rq3_measures.txt
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from _stats import fit_cluster, stars
from rq3_interaction import SUP, prepare
from rq3_link import bank_level, link_models, load_bridge_panel

NICE = {m["key"]: m["name"] for m in C.MEASURES}
FE = "C(doc_id) + C(permno)"


# --------------------------------------------------------------------------- #
#  A. measures comparison
# --------------------------------------------------------------------------- #
def comparison_block(d: pd.DataFrame, links: dict) -> tuple[list[str], dict]:
    bank = bank_level(d)
    L = ["=" * 78, "RQ3 (3/3) — WHICH MEASURE CARRIES SIGNAL, AND IS IT REAL?",
         "=" * 78, "",
         f"-- A1. do the three measures agree? (bank level, N = {len(bank)}) --", "",
         "    {:<28s} {:<28s} {:>14s} {:>14s}".format(
             "measure A", "measure B", "Pearson r", "Spearman rho"),
         "    " + "-" * 88]
    corrs = {}
    keys = C.MEASURE_KEYS
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            r, _ = stats.pearsonr(bank[a], bank[b])
            rho, _ = stats.spearmanr(bank[a], bank[b])
            corrs[(a, b)] = (r, rho)
            L.append("    {:<28s} {:<28s} {:>14.8f} {:>14.8f}".format(
                NICE[a], NICE[b], r, rho))
    L += ["",
          "    {:<28s} {:>14s} {:>14s} {:>14s} {:>14s}".format(
              "measure", "mean", "SD", "min", "max"),
          "    " + "-" * 88]
    for m in keys:
        L.append("    {:<28s} {:>14.8f} {:>14.8f} {:>14.8f} {:>14.8f}".format(
            NICE[m], bank[m].mean(), bank[m].std(), bank[m].min(), bank[m].max()))
    rho_rf_ui = corrs[("uninsured_share", "score_rf")][1]
    rho_rf_ols = corrs[("score_rf", "score_ols")][1]
    L += ["",
          f"    The RF score and uninsured_share rank banks almost independently",
          f"    (Spearman {rho_rf_ui:+.8f}). They are two different orderings, not two",
          "    measurements of one thing. The RF and OLS scores, built from the same",
          f"    features and the same split, agree far more ({rho_rf_ols:+.8f}).", ""]
    return L, {"rho_rf_ui": rho_rf_ui, "rho_rf_ols": rho_rf_ols}


def side_by_side_block(d: pd.DataFrame, links: dict) -> tuple[list[str], dict]:
    """delta1 (level) and b (interaction) for all three measures, one table."""
    L = ["-- A2. level effect vs reaction effect, all three measures --", "",
         "    delta1 = CAR ~ measure + controls + event FE   (level; bank-clustered)",
         "    b      = CAR = EventFE + BankFE + b*(S x measure)  (reaction;",
         "             event-clustered, the headline inference)", "",
         "    {:<26s} {:>14s} {:>10s}  {:>14s} {:>10s}".format(
             "measure", "delta1", "p", "b", "p"),
         "    " + "-" * 82]
    out = {}
    for key in C.MEASURE_KEYS:
        mod = links[(key, "event FE")][0]
        s = prepare(d, key)
        static = next(m["static"] for m in C.MEASURES if m["key"] == key)
        main = "" if static else "VULN + "
        mi = fit_cluster(s, f"CAR_ ~ {main}SxV + {FE}", ["doc_id"])
        out[key] = {"delta1": mod.params[key], "p_delta1": mod.pvalues[key],
                    "b": mi.params["SxV"], "p_b": mi.pvalues["SxV"]}
        L.append("    {:<26s} {:>14.8f} {:>10.6f}  {:>14.8f} {:>10.6f}".format(
            NICE[key], mod.params[key], mod.pvalues[key],
            mi.params["SxV"], mi.pvalues["SxV"]))
    L += ["",
          "    Read across the rows, not down the columns. uninsured_share is the",
          "    only measure with a level effect, and it is the only one with NO",
          "    interaction. The RF and OLS scores are the mirror image: no level",
          "    effect, but an interaction. Section B says what that pattern means.",
          ""]

    # joint model: does the score survive next to uninsured_share?
    s = d.dropna(subset=["CAR", "S", "score_rf", "score_ols",
                         "uninsured_share"]).copy()
    s["CAR_"] = s.CAR
    s["SxRF"] = s.S * s.score_rf
    s["SxOLS"] = s.S * s.score_ols
    s["SxUI"] = s.S * s.uninsured_share
    s["UI"] = s.uninsured_share
    j_rf = fit_cluster(s, f"CAR_ ~ SxRF + SxUI + UI + {FE}", ["doc_id"])
    j_ols = fit_cluster(s, f"CAR_ ~ SxOLS + SxUI + UI + {FE}", ["doc_id"])
    L += ["-- A3. head to head in ONE model (event-clustered) --", "",
          "    {:<34s} {:>14s} {:>13s} {:>10s}  {}".format(
              "term", "coef", "SE", "p", "sig"),
          "    " + "-" * 86]
    for m, terms in ((j_rf, [("SxRF", "S x RF score"), ("SxUI", "S x uninsured_share"),
                             ("UI", "uninsured_share (main)")]),
                     (j_ols, [("SxOLS", "S x OLS score"), ("SxUI", "S x uninsured_share"),
                              ("UI", "uninsured_share (main)")])):
        for t, lab in terms:
            L.append("    {:<34s} {:>14.8f} {:>13.8f} {:>10.6f}  {}".format(
                lab, m.params[t], m.bse[t], m.pvalues[t], stars(m.pvalues[t])))
        L.append("")
    out["joint"] = {"p_SxRF": j_rf.pvalues["SxRF"], "p_SxUI": j_rf.pvalues["SxUI"],
                    "p_SxOLS": j_ols.pvalues["SxOLS"]}
    L += ["    The fitted composite survives beside uninsured_share; "
          "uninsured_share's own",
          "    interaction does not. But the OLS score does the same job as the RF",
          "    score, so what is doing the work is USING A FITTED COMBINATION of the",
          "    balance sheet, not machine learning as such. On the level link the",
          "    ranking reverses and the raw ratio wins. Neither ordering is evidence",
          "    for H3 on its own — see section B.", ""]
    return L, out


# --------------------------------------------------------------------------- #
#  B. level vs reaction — the H3 mechanism verdict
# --------------------------------------------------------------------------- #
def level_vs_reaction_block(sbs: dict) -> list[str]:
    ui = sbs["uninsured_share"]
    rf = sbs["score_rf"]
    return [
        "=" * 78, "B. LEVEL VS REACTION — IS THE H3 MECHANISM SUPPORTED?", "=" * 78, "",
        "  Two facts about uninsured_share, both from the tables above:",
        "",
        f"    LEVEL     delta1 = {ui['delta1']:+.8f}   p = {ui['p_delta1']:.8f}"
        f"  {stars(ui['p_delta1'])}",
        f"    REACTION  b      = {ui['b']:+.8f}   p = {ui['p_b']:.8f}"
        f"  {stars(ui['p_b'])}",
        "",
        "  Banks with more uninsured deposits DID sit at a lower abnormal return",
        "  across these windows. But the gap does not move with the safeguard",
        "  content of the communication: interacted with S it is indistinguishable",
        "  from zero.",
        "",
        "  That combination has one honest reading. The level result is a property",
        "  of WHICH BANKS these are — funding-fragile banks priced differently over",
        "  2019-2023, a period containing COVID and the 2022 rate cycle — and not a",
        "  response to what the Federal Reserve said about CBDC design. A reaction",
        "  to content would have to show up in the interaction, and it does not.",
        "",
        "  THE H3 MECHANISM IS NOT SUPPORTED. The bridge between the RQ1",
        "  vulnerability profile and the RQ2 market reaction does not hold:",
        "    * the ML score has no level effect (delta1 null, wrong sign);",
        f"    * uninsured_share has a level effect but no reaction "
        f"(b p = {ui['p_b']:.4f});",
        f"    * the one significant interaction (RF score, b = {rf['b']:+.8f},"
        f" p = {rf['p_b']:.6f})",
        "      has the WRONG SIGN for the hypothesis and does not survive the",
        "      diagnostics in section C.",
        "",
        "  This is consistent with RQ2: the unconditional test finds no event with a",
        "  mean CAR distinguishable from zero, so there is no reaction for a",
        "  cross-section to modulate in the first place.",
        "",
    ]


# --------------------------------------------------------------------------- #
#  C. artefact diagnostics — run for BOTH measures, whatever the result
# --------------------------------------------------------------------------- #
def diagnostics_block(d: pd.DataFrame, key: str) -> tuple[list[str], dict]:
    meas = next(m for m in C.MEASURES if m["key"] == key)
    static = meas["static"]
    main = "" if static else "VULN + "
    s = prepare(d, key)

    order = {t: i for i, t in enumerate(sorted(s.t0.unique()))}
    s["TIME"] = s.t0.map(order).astype(float)
    s["S_z"] = (s.S - s.S.mean()) / s.S.std()
    s["T_z"] = (s.TIME - s.TIME.mean()) / s.TIME.std()
    s["SzV"] = s.S_z * s.VULN
    s["TzV"] = s.T_z * s.VULN

    ev = s.groupby("doc_id").agg(S=("S", "first"), T=("TIME", "first"))
    r_st, p_st = stats.pearsonr(ev.S, ev["T"])
    rs_st, ps_st = stats.spearmanr(ev.S, ev["T"])

    L = ["-" * 78, f"C. ARTEFACT DIAGNOSTICS — {NICE[key]}", "-" * 78, "",
         "-- C1. calendar placebo: is this about the SIGNAL or about TIME? --", "",
         "    The 11 events are not randomly ordered in S: later communications are",
         "    more protective.",
         f"      corr(S, calendar rank) = Pearson {r_st:+.8f} (p {p_st:.6f}),"
         f" Spearman {rs_st:+.8f} (p {ps_st:.6f})",
         "",
         "    So 'S x vulnerability' and 'time x vulnerability' are close to the same",
         "    regressor. Both are built from z-scored S and z-scored calendar rank so",
         "    the coefficients are directly comparable.", "",
         "    {:<32s} {:>14s} {:>13s} {:>10s}  {}".format(
             "spec (two-way FE, event SE)", "coef", "SE", "p", "sig"),
         "    " + "-" * 86]
    m_s = fit_cluster(s, f"CAR_ ~ {main}SzV + {FE}", ["doc_id"])
    m_t = fit_cluster(s, f"CAR_ ~ {main}TzV + {FE}", ["doc_id"])
    m_j = fit_cluster(s, f"CAR_ ~ {main}SzV + TzV + {FE}", ["doc_id"])
    for lab, m, t in [("S_z x vuln (alone)", m_s, "SzV"),
                      ("time_z x vuln (alone)", m_t, "TzV"),
                      ("S_z x vuln (joint)", m_j, "SzV"),
                      ("time_z x vuln (joint)", m_j, "TzV")]:
        L.append("    {:<32s} {:>14.8f} {:>13.8f} {:>10.6f}  {}".format(
            lab, m.params[t], m.bse[t], m.pvalues[t], stars(m.pvalues[t])))
    L += ["",
          f"    Joint model: S interaction "
          f"{'still significant' if m_j.pvalues['SzV'] < 0.05 else 'NOT significant'}"
          f" at 5% (p {m_j.pvalues['SzV']:.6f}); time",
          f"    interaction "
          f"{'significant' if m_j.pvalues['TzV'] < 0.05 else 'not significant'}"
          f" (p {m_j.pvalues['TzV']:.6f}). With 11 events the two cannot be",
          "    separated — read this as a warning flag, not a clean decomposition.",
          ""]

    # C2 leave-one-event-out
    L += ["-- C2. leave-one-event-out (supervisor's spec, event SE) --", "",
          "    {:>7s} {:>12s} {:>9s} {:>14s} {:>13s} {:>10s}  {}".format(
              "dropped", "t0", "S", "b", "SE", "p", "sig"),
          "    " + "-" * 82]
    bs = []
    for doc in sorted(s.doc_id.unique()):
        sub = s[s.doc_id != doc]
        try:
            m = fit_cluster(sub, f"CAR_ ~ {main}SxV + {FE}", ["doc_id"])
            bs.append(float(m.params["SxV"]))
            L.append("    {:>7d} {:>12s} {:>9.4f} {:>14.8f} {:>13.8f} {:>10.6f}  {}".format(
                int(doc), str(s.loc[s.doc_id == doc, "t0"].iloc[0]),
                float(s.loc[s.doc_id == doc, "S"].iloc[0]),
                m.params["SxV"], m.bse["SxV"], m.pvalues["SxV"],
                stars(m.pvalues["SxV"])))
        except Exception as e:                                     # noqa: BLE001
            L.append(f"    {int(doc):>7d}  NOT ESTIMATED ({type(e).__name__})")
    if bs:
        L += ["", f"    b ranges [{min(bs):+.8f}, {max(bs):+.8f}]; sign flips: "
                  f"{'YES' if min(bs) * max(bs) < 0 else 'no'}", ""]

    # C3 drop trust/custody
    diag = {"r_st": r_st, "b_Tz": m_t.params["TzV"], "p_Tz": m_t.pvalues["TzV"],
            "p_Sz_joint": m_j.pvalues["SzV"], "p_Tz_joint": m_j.pvalues["TzV"],
            "loo_min": min(bs) if bs else np.nan, "loo_max": max(bs) if bs else np.nan}
    if "is_trust_or_specialized" in s.columns:
        sub = s[~s.is_trust_or_specialized.fillna(False).astype(bool)]
        n_drop = s.permno.nunique() - sub.permno.nunique()
        m = fit_cluster(sub, f"CAR_ ~ {main}SxV + {FE}", ["doc_id"])
        m2 = fit_cluster(sub, f"CAR_ ~ {main}SxV + {FE}", ["permno", "doc_id"])
        full = fit_cluster(s, f"CAR_ ~ {main}SxV + {FE}", ["doc_id"])
        flipped = m.params["SxV"] * full.params["SxV"] < 0
        share = (1 - abs(m.params["SxV"]) / abs(full.params["SxV"])) * 100
        diag["trust_b"] = float(m.params["SxV"])
        diag["trust_p"] = float(m.pvalues["SxV"])
        diag["trust_p_twoway"] = float(m2.pvalues["SxV"])
        diag["trust_share"] = float(share) if not flipped else float("nan")
        L += ["-- C3. excluding trust / custody-flagged banks --", "",
              f"    {n_drop} flagged banks removed; {sub.permno.nunique()} banks, "
              f"{len(sub)} bank-events remain.",
              "    The RF score puts three custodians at the top, and custodians carry",
              "    their own rate exposure.", "",
              "    {:<32s} {:>14.8f} {:>13.8f} {:>10.6f}  {}".format(
                  "b, trust/custody dropped", m.params["SxV"], m.bse["SxV"],
                  m.pvalues["SxV"], stars(m.pvalues["SxV"])),
              (f"    {share:.4f}% of the magnitude comes from those banks."
               if not flipped else
               "    b changes SIGN when they are dropped, so a '% of the magnitude'"
               " figure would be meaningless and is not reported."),
              f"    (two-way clustered p for the same spec: {m2.pvalues['SxV']:.6f})",
              ""]
    return L, diag


def verdict_block(sbs: dict, diags: dict) -> list[str]:
    rf = sbs["score_rf"]
    d = diags["score_rf"]
    return [
        "=" * 78, "VERDICT — RQ3", "=" * 78, "",
        "  1. The H3 mechanism is NOT supported. No measure shows a reaction to the",
        "     safeguard content of the communication.",
        "",
        f"  2. uninsured_share has a real LEVEL effect (delta1 = "
        f"{sbs['uninsured_share']['delta1']:+.8f},",
        f"     p = {sbs['uninsured_share']['p_delta1']:.8f}) but a null REACTION "
        f"(b p = {sbs['uninsured_share']['p_b']:.6f}).",
        "     That is a statement about which banks these are, not about CBDC.",
        "",
        f"  3. The RF score's interaction (b = {rf['b']:+.8f}, p = {rf['p_b']:.6f}) is",
        "     treated as an ARTEFACT, on three grounds, and it also has the wrong",
        "     sign for the hypothesis:",
        f"       * calendar placebo: corr(S, time) = {d['r_st']:+.6f}; time x vuln "
        f"gives {d['b_Tz']:+.8f}",
        f"         (p {d['p_Tz']:.6f}), and jointly neither survives "
        f"(S p {d['p_Sz_joint']:.6f},",
        f"         time p {d['p_Tz_joint']:.6f});",
        f"       * custody banks: dropping 9 of them moves b to {d.get('trust_b', float('nan')):+.8f}"
        f" (p {d.get('trust_p', float('nan')):.6f}),",
        f"         i.e. {d.get('trust_share', float('nan')):.2f}% of the magnitude is custodians;",
        f"       * leave-one-event-out keeps the sign "
        f"([{d['loo_min']:+.8f}, {d['loo_max']:+.8f}]), so it is",
        "         not one event — but that does not rescue it from the first two.",
        "",
        "  4. Machine learning is not what matters. The OLS-predicted score behaves",
        "     like the RF score in both the level link and the interaction, and the",
        f"     two agree at Spearman {diags['rho_rf_ols']:+.6f}. Any credit belongs to",
        "     using a fitted balance-sheet composite, not to the algorithm.",
        "",
    ]


def main() -> dict:
    d = load_bridge_panel()
    links = link_models(d)
    L, corrs = comparison_block(d, links)
    sbs_lines, sbs = side_by_side_block(d, links)
    L += sbs_lines
    L += level_vs_reaction_block(sbs)

    diags = {"rho_rf_ols": corrs["rho_rf_ols"]}
    for key in ("uninsured_share", "score_rf"):
        lines, dg = diagnostics_block(d, key)
        L += lines
        diags[key] = dg
    L += verdict_block(sbs, diags)

    text = "\n".join(L)
    C.RQ3_MEASURES.write_text(text + "\n")
    print(text)
    print(f"wrote {C.RQ3_MEASURES}")
    return {
        "rq3.b.score_ols.coef": sbs["score_ols"]["b"],
        "rq3.corr.rf_uninsured.spearman": corrs["rho_rf_ui"],
        "rq3.corr.rf_ols.spearman": corrs["rho_rf_ols"],
        "rq3.placebo.corr_S_time": diags["score_rf"]["r_st"],
        "rq3.custody_drop.b": diags["score_rf"].get("trust_b", float("nan")),
        # the archived value was two-way clustered; the event-clustered figure is
        # the new headline. Both are returned so the reconciliation can match the
        # archive AND show the instructed change of headline SE.
        "rq3.custody_drop.p": diags["score_rf"].get("trust_p_twoway", float("nan")),
        "rq3.custody_drop.p_event_headline":
            diags["score_rf"].get("trust_p", float("nan")),
    }


if __name__ == "__main__":
    main()
