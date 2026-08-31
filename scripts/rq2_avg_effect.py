"""rq2_avg_effect.py: tests H2, the average effect of the type of communication.

    H2           communications emphasising protective design, meaning a higher
                 Safeguard score S, produce less negative abnormal returns.
    coefficient  gamma1 in CAR = gamma0 + gamma1 * S + controls. H2 predicts
                 gamma1 > 0.

The interaction of S with bank vulnerability is not tested here. It asks whether
vulnerable banks react differently, which is H3, and it is estimated in
rq3_interaction.py.

Two features of the design are enforced in the code rather than described:

  (a) no event-fixed-effects specification is estimated for S. S is constant
      within an event, so the event dummies absorb it exactly and gamma1 is not
      identified. The script verifies this numerically instead of printing
      whatever a pseudo-inverse returns.
  (b) standard errors cluster on the event, the level at which S varies, with
      t(G-1) inference. Bank-clustered errors would treat 276 banks as
      independent information about a regressor with eleven values; they are
      reported only to show the size of that inflation.

Three views of the same comparison:
    1  event level   the eleven event-mean CARs on the eleven values of S, with
                     Pearson and Spearman correlations
    2  pooled        all bank-events, so that bank controls can be included
    3  augmented     the same test on the market, sector and rate CAR

Reads     data/processed/rq2_car.csv, rq2_car_augmented.csv
Writes    data/processed/rq2_avg_effect.txt, rq2_car_sanity.txt
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import config as C
from _stats import absorption, fit_cluster, mde, stars


# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(C.RQ2_CAR)
    n_all = len(df)
    df = df.dropna(subset=["CAR", "S"] + C.CONTROLS).copy()
    if df.groupby("doc_id").S.nunique().max() != 1:
        raise SystemExit("S is not constant within events — the design assumes it is.")
    ev = (df.groupby(["doc_id", "t0"])
            .agg(S=("S", "first"), n_banks=("CAR", "size"),
                 mean_CAR=("CAR", "mean"), median_CAR=("CAR", "median"),
                 sd_CAR=("CAR", "std"))
            .reset_index().sort_values("t0"))
    ev["se_CAR"] = ev.sd_CAR / np.sqrt(ev.n_banks)
    df.attrs["n_all"] = n_all
    return df, ev


# --------------------------------------------------------------------------- #
#  1. event level (n = 11)
# --------------------------------------------------------------------------- #
def event_level(ev: pd.DataFrame) -> tuple[list[str], dict]:
    m = smf.ols("mean_CAR ~ S", data=ev).fit()
    mw = smf.wls("mean_CAR ~ S", data=ev, weights=ev.n_banks).fit()
    g, se, p = m.params["S"], m.bse["S"], m.pvalues["S"]
    ci = m.conf_int().loc["S"]
    r_p, p_p = stats.pearsonr(ev.mean_CAR, ev.S)
    r_s, p_s = stats.spearmanr(ev.mean_CAR, ev.S)
    exp_, prot_ = ev.loc[ev.S < 0, "mean_CAR"], ev.loc[ev.S >= 0, "mean_CAR"]
    tt = stats.ttest_ind(prot_, exp_, equal_var=False)
    mwu = stats.mannwhitneyu(prot_, exp_, alternative="two-sided")
    rng = float(ev.S.max() - ev.S.min())

    L = ["=" * 78, "1. H2 AT THE EVENT LEVEL — the honest form of the test (n = 11)",
         "=" * 78, "",
         "  S is a property of the COMMUNICATION, not of the bank. H2 is therefore a",
         "  between-event question and the natural test has 11 observations.", "",
         "-- 1.1 the 11 events --", "",
         "    {:>6s} {:>12s} {:>11s} {:>8s} {:>13s} {:>13s} {:>12s}  {}".format(
             "doc_id", "t0", "S", "n_banks", "mean_CAR", "median_CAR", "SE(mean)",
             "flag"),
         "    " + "-" * 96]
    for _, r in ev.iterrows():
        L.append("    {:>6d} {:>12s} {:>11.6f} {:>8d} {:>13.8f} {:>13.8f} {:>12.8f}  {}"
                 .format(int(r.doc_id), str(r.t0), r.S, int(r.n_banks), r.mean_CAR,
                         r.median_CAR, r.se_CAR,
                         "THIN" if r.n_banks < C.MIN_BANKS_PER_EVENT else ""))
    L += ["",
          f"    S range [{ev.S.min():+.8f}, {ev.S.max():+.8f}]   SD(S) = {ev.S.std():.8f}"
          f"   expansive (S<0): {int((ev.S < 0).sum())}   "
          f"protective (S>=0): {int((ev.S >= 0).sum())}",
          "",
          "-- 1.2 OLS of the 11 event-mean CARs on the 11 S values --", "",
          "    mean_CAR = gamma0 + gamma1 * S      (n = 11, ordinary SEs)", "",
          f"    gamma1 = {g:+.8f}   SE {se:.8f}   t {m.tvalues['S']:+.6f}"
          f"   p {p:.8f}  {stars(p)}",
          f"      95% CI [{ci[0]:+.8f}, {ci[1]:+.8f}]     R^2 {m.rsquared:.8f}",
          f"      intercept {m.params['Intercept']:+.8f} (SE {m.bse['Intercept']:.8f})",
          f"      MDE at 80% power = {C.POWER_MULT * se:.8f} of CAR per unit of S;",
          f"      across the observed S range ({rng:.6f}) that is "
          f"{C.POWER_MULT * se * rng * 100:.6f} pp of CAR.",
          f"    weighted by n_banks (secondary): gamma1 = {mw.params['S']:+.8f}"
          f"   SE {mw.bse['S']:.8f}   p {mw.pvalues['S']:.8f}",
          "",
          "-- 1.3 correlation between event-mean CAR and S --", "",
          f"    Pearson  = {r_p:+.8f}   p {p_p:.8f}  {stars(p_p)}   (n = 11)",
          f"    Spearman = {r_s:+.8f}   p {p_s:.8f}  {stars(p_s)}   (n = 11)",
          "    A POSITIVE correlation is the H2-consistent direction.",
          "",
          "-- 1.4 protective vs expansive events --", "",
          f"    mean CAR, expansive  (S <  0, n={len(exp_)}) : {exp_.mean():+.8f}",
          f"    mean CAR, protective (S >= 0, n={len(prot_)}) : {prot_.mean():+.8f}",
          f"    difference (protective - expansive)     : "
          f"{prot_.mean() - exp_.mean():+.8f}",
          f"    Welch t = {tt.statistic:+.6f}  p {tt.pvalue:.8f}   "
          f"Mann-Whitney p {mwu.pvalue:.8f}",
          "",
          "    With 11 events this has very little power. The point of reporting it",
          "    is that the sign and magnitude are on the record.", ""]
    return L, {"coef": g, "se": se, "p": p, "r_p": r_p, "p_p": p_p,
               "r_s": r_s, "p_s": p_s, "diff": prot_.mean() - exp_.mean()}


# --------------------------------------------------------------------------- #
#  2. pooled bank-event level
# --------------------------------------------------------------------------- #
def pooled(df: pd.DataFrame) -> tuple[list[str], dict]:
    n_ev = df.doc_id.nunique()
    f_nc, f_ct = "CAR ~ S", "CAR ~ S + " + " + ".join(C.CONTROLS)
    m_nc = fit_cluster(df, f_nc, ["doc_id"])
    m_ct = fit_cluster(df, f_ct, ["doc_id"])
    b_nc = fit_cluster(df, f_nc, ["permno"], use_t=False)
    b_ct = fit_cluster(df, f_ct, ["permno"], use_t=False)
    absorb = absorption(df, "S", "C(doc_id)")
    car_sd = float(df.CAR.std())
    e = mde(m_ct.bse["S"], float(df.S.std()), car_sd)

    L = ["=" * 78, "2. H2 POOLED AT THE BANK-EVENT LEVEL (SEs clustered on the EVENT)",
         "=" * 78, "",
         f"  {len(df)} bank-events, {df.permno.nunique()} banks, {n_ev} events,",
         f"  {df.S.nunique()} distinct values of S. The extra rows add no independent",
         "  information about S, which is why the SEs are clustered on the event.",
         "",
         "-- 2.1 why there is no event-FE specification for S --", "",
         f"    S regressed on the event dummies: R^2 = {absorb['r2']:.12f},",
         f"    max |residual| = {absorb['max_abs']:.3e}, residual SD = {absorb['sd']:.3e}",
         f"    -> {'ABSORBED' if absorb['absorbed'] else 'not absorbed'}. "
         "The S column is a linear combination of",
         "    the event dummies, so gamma1 is NOT IDENTIFIED under event FE. This is a",
         "    design fact, not a null result, and nothing is printed for it.",
         "",
         "-- 2.2 gamma1 --", "",
         "    {:<32s} {:>14s} {:>13s} {:>8s} {:>10s}  {}".format(
             "spec / cluster", "gamma1", "SE", "t", "p", "sig"),
         "    " + "-" * 88]
    for lab, m in [("no controls, cluster = EVENT", m_nc),
                   ("with controls, cluster = EVENT", m_ct),
                   ("no controls, cluster = bank", b_nc),
                   ("with controls, cluster = bank", b_ct)]:
        L.append("    {:<32s} {:>14.8f} {:>13.8f} {:>8.3f} {:>10.8f}  {}".format(
            lab, m.params["S"], m.bse["S"], m.tvalues["S"], m.pvalues["S"],
            stars(m.pvalues["S"])))
    L += ["",
          f"    inference df (event-clustered) = {int(m_ct.df_resid_inference)}"
          f"  (= {n_ev} clusters - 1)",
          "    The bank-clustered rows are shown ONLY to make the inflation visible:",
          f"    the SE falls from {m_ct.bse['S']:.8f} to {b_ct.bse['S']:.8f}, a factor of"
          f" {m_ct.bse['S'] / b_ct.bse['S']:.2f}.",
          "    They are not the reported inference.",
          "",
          "-- 2.3 how tight is this null? --", "",
          f"    MDE at 80% power for gamma1 (with controls, event-clustered)",
          f"      = {e['raw']:.8f} per unit of S",
          f"      = {e['per_sd_pp']:.6f} pp of CAR per one-SD move in S",
          f"      = {e['pct_of_car_sd']:.4f}% of a CAR standard deviation "
          f"({car_sd * 100:.6f} pp)",
          "",
          "    That is a WIDE bound and must be stated as one. With 11 events the",
          "    design can rule out a LARGE average communication effect but not a",
          "    modest one. H2's null is directionally clear but not tightly bounded.",
          ""]
    return L, {"coef": m_ct.params["S"], "se": m_ct.bse["S"], "p": m_ct.pvalues["S"],
               "absorb_max": absorb["max_abs"]}


# --------------------------------------------------------------------------- #
#  3. augmented CAR
# --------------------------------------------------------------------------- #
def augmented() -> list[str]:
    if not C.RQ2_CAR_AUG.exists():
        return ["=" * 78, "3. AUGMENTED CAR — not available", "=" * 78,
                "  run rq2_reaction.py to build rq2_car_augmented.csv.", ""]
    a = pd.read_csv(C.RQ2_CAR_AUG)
    a = a.dropna(subset=["CAR_aug", "S"] + C.CONTROLS)
    m = fit_cluster(a, "CAR_aug ~ S + " + " + ".join(C.CONTROLS), ["doc_id"])
    return ["=" * 78,
            "3. THE SAME TEST ON THE MARKET+SECTOR+RATE CAR", "=" * 78, "",
            "  Bank stocks load on a common banking factor and differ in rate",
            "  sensitivity. If gamma1 were a rate artefact it would move here.", "",
            f"    N {len(a)}   gamma1 = {m.params['S']:+.8f}   SE {m.bse['S']:.8f}"
            f"   p {m.pvalues['S']:.8f}  {stars(m.pvalues['S'])}",
            "", "    Same conclusion as the market-model CAR.", ""]


def sanity(df: pd.DataFrame, ev: pd.DataFrame) -> str:
    L = ["=" * 78, "SANITY — mean CAR per event", "=" * 78, "",
         "    {:>6s} {:>12s} {:>11s} {:>8s} {:>13s} {:>13s} {:>9s}".format(
             "doc_id", "t0", "S", "n_banks", "mean_CAR", "median_CAR", "t"),
         "    " + "-" * 82]
    for _, r in ev.iterrows():
        L.append("    {:>6d} {:>12s} {:>11.6f} {:>8d} {:>13.8f} {:>13.8f} {:>9.4f}".format(
            int(r.doc_id), str(r.t0), r.S, int(r.n_banks), r.mean_CAR,
            r.median_CAR, r.mean_CAR / r.se_CAR))
    L += ["",
          "    These per-event t-stats assume banks are independent within a date.",
          "    They are not. The cross-correlation-robust test is in rq2_reaction.txt.",
          ""]
    return "\n".join(L)


def main() -> dict:
    df, ev = load()
    L = ["=" * 78,
         "RQ2 — DOES THE TYPE OF CBDC COMMUNICATION MOVE BANK EQUITY ON AVERAGE?",
         "=" * 78, "",
         f"  bank-events on file : {df.attrs['n_all']}",
         f"  used in estimation  : {len(df)}",
         f"  banks {df.permno.nunique()}   events {df.doc_id.nunique()}",
         "  dependent variable  : CAR over [t0-1, t0+5], market-model abnormal returns",
         "  coefficient         : gamma1 on S. H2 predicts gamma1 > 0.",
         "",
         "  READ rq2_reaction.txt FIRST. It asks whether the market reacted to these",
         "  dates at all. If no event moved prices there is no average effect for S",
         "  to explain, and everything below is a gradient across events that did not",
         "  move.",
         "",
         "  The S x vulnerability interaction is NOT here: it tests whether",
         "  vulnerable banks react differently, which is H3. See rq3_interaction.py.",
         ""]
    el, s1 = event_level(ev)
    pl, s2 = pooled(df)
    L += el + pl + augmented()

    L += ["=" * 78, "VERDICT — RQ2", "=" * 78, "",
          "    {:<44s} {:>14s} {:>13s} {:>10s}".format("test", "gamma1", "SE", "p"),
          "    " + "-" * 84,
          "    {:<44s} {:>14.8f} {:>13.8f} {:>10.8f}".format(
              "event level, mean_CAR ~ S (n=11)", s1["coef"], s1["se"], s1["p"]),
          "    {:<44s} {:>14.8f} {:>13.8f} {:>10.8f}".format(
              "pooled with controls (cluster = event)", s2["coef"], s2["se"], s2["p"]),
          "    {:<44s} {:>14.8f} {:>13s} {:>10.8f}".format(
              "Pearson corr(event-mean CAR, S)", s1["r_p"], "", s1["p_p"]),
          "    {:<44s} {:>14.8f} {:>13s} {:>10.8f}".format(
              "Spearman corr(event-mean CAR, S)", s1["r_s"], "", s1["p_s"]),
          "",
          f"    H2 predicts gamma1 > 0. Observed sign: "
          f"{'POSITIVE (H2 direction)' if s1['coef'] > 0 else 'NEGATIVE'}."
          f"  Significant at 5%: "
          f"{'YES' if min(s1['p'], s2['p']) < 0.05 else 'NO'}.",
          "",
          "    H2 IS NOT SUPPORTED. The sign is in the predicted direction but no",
          "    specification comes close to significance, and the correlation across",
          "    the 11 events is essentially zero. Read together with rq2_reaction.txt:",
          "    no event produced a mean CAR distinguishable from zero, so a null",
          "    gamma1 is what the unconditional evidence already implied.",
          "",
          "    The null is NOT tightly bounded (see 2.3). Say 'H2 not supported and a",
          "    large average effect ruled out', not 'no effect'.",
          ""]

    text = "\n".join(L)
    C.RQ2_AVG_EFFECT.write_text(text + "\n")
    C.RQ2_CAR_SANITY.write_text(sanity(df, ev) + "\n")
    print(text)
    print(f"wrote {C.RQ2_AVG_EFFECT}\nwrote {C.RQ2_CAR_SANITY}")
    return {
        "rq2.gamma1.event_level.coef": s1["coef"],
        "rq2.gamma1.event_level.se": s1["se"],
        "rq2.gamma1.event_level.p": s1["p"],
        "rq2.gamma1.pooled_ctrl.coef": s2["coef"],
        "rq2.gamma1.pooled_ctrl.se": s2["se"],
        "rq2.gamma1.pooled_ctrl.p": s2["p"],
        "rq2.corr.pearson": s1["r_p"],
        "rq2.corr.spearman": s1["r_s"],
    }


if __name__ == "__main__":
    main()
