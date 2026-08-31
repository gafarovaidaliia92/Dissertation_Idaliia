"""rq3_custody_check.py: the custody-bank robustness check behind Appendix C.

Re-estimates the interaction with custody and trust banks excluded, and reports
how the Safeguard signal correlates with calendar time. The specification is
exactly the one in rq3_interaction.py, with nothing changed but the rows in the
sample:

    static measure (score_rf, score_ols)
        CAR ~ S:vulnerability + C(doc_id) + C(permno)
    bank-event measure (uninsured_share)
        CAR ~ vulnerability + S:vulnerability + C(doc_id) + C(permno)

The vulnerability main effect is included only where it is identified: the two
model scores are static per bank and absorbed by the bank dummies, whereas
uninsured_share is re-read from eight characteristic quarters and survives them.
Standard errors cluster on the event, eleven clusters with t(10), which is the
headline inference in rq3_interaction.py; the two-way clustered p-value is
printed beside it.

Three exclusion definitions are reported, because they do not coincide:
    A  the three named custodians: BNY Mellon, State Street, Northern Trust
    B  trust and fiduciary only, which is A plus the other fiduciary-flagged
       banks, excluding credit-card banks
    C  the is_trust_or_specialized flag as it stands, nine banks, four of them
       credit-card rather than custody

Definition C is the one the diagnostic in rq3_measures.py uses, so it reproduces
that figure; definition A is the one Appendix C reports.

Reads     data/processed/rq3_bridge.csv
Writes    data/processed/rq3_custody_check.txt
"""

from __future__ import annotations

import contextlib
import io
import sys

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from _stats import fit_cluster, stars
from rq3_interaction import prepare
from rq3_link import load_bridge_panel

MEASURES = ["score_rf", "score_ols", "uninsured_share"]
NICE = {"score_rf": "RF score", "score_ols": "OLS score",
        "uninsured_share": "uninsured_share"}
FE = "C(doc_id) + C(permno)"

# the three named custodians, by permno so a name string cannot drift
NAMED_CUSTODIANS = {
    49656: "BANK OF NEW YORK MELLON, THE",
    72726: "STATE STREET BANK AND TRUST COMPANY",
    58246: "NORTHERN TRUST COMPANY, THE",
}


def identify(d: pd.DataFrame) -> dict[str, dict]:
    """Build the three exclusion sets and say exactly how each was derived."""
    is_flag = d.is_trust_or_specialized.astype("boolean").fillna(False).astype(bool)
    flag = d[is_flag].groupby("permno").first().reset_index()
    scores = pd.read_csv(C.SCORES_WIDE)
    reason = dict(zip(scores.permno, scores.flag_reason.fillna("")))
    names = dict(zip(scores.permno, scores.name))

    all_flagged = sorted(flag.permno.unique())
    credit_card = [p for p in all_flagged if "credit-card" in reason.get(p, "")]
    trust_only = [p for p in all_flagged if p not in credit_card]
    named = sorted(NAMED_CUSTODIANS)

    return {
        "A": {"permnos": named,
              "how": "named custodians, matched on permno "
                     "(BNY Mellon 49656, State Street 72726, Northern Trust 58246)"},
        "B": {"permnos": trust_only,
              "how": "is_trust_or_specialized = True AND flag_reason is not "
                     "credit-card(SPECGRP3) — i.e. fiduciary / low-loans+high-uninsured"},
        "C": {"permnos": all_flagged,
              "how": "is_trust_or_specialized = True as it stands in "
                     "vulnerability_scores_wide.csv (the flag rq3_measures.py uses)"},
    }, names, reason


def fit_beta(d: pd.DataFrame, key: str) -> dict:
    """The rq3_interaction.py two-way FE specification, unchanged."""
    s = prepare(d, key)
    static = next(m["static"] for m in C.MEASURES if m["key"] == key)
    main = "" if static else "VULN + "
    f = f"CAR_ ~ {main}SxV + {FE}"
    mE = fit_cluster(s, f, ["doc_id"])
    m2 = fit_cluster(s, f, ["permno", "doc_id"])
    return {"beta": float(mE.params["SxV"]), "se": float(mE.bse["SxV"]),
            "p": float(mE.pvalues["SxV"]), "p_twoway": float(m2.pvalues["SxV"]),
            "n": int(mE.nobs), "banks": int(s.permno.nunique()),
            "ci": [float(x) for x in mE.conf_int().loc["SxV"]]}


def _report() -> None:
    d = load_bridge_panel()
    sets, names, reason = identify(d)

    print("=" * 100)
    print("RQ3 INTERACTION — ROBUSTNESS TO EXCLUDING CUSTODY / TRUST BANKS")
    print("=" * 100)
    print()
    print("Specification (unchanged from rq3_interaction.py):")
    print("    static measure : CAR ~ S:vulnerability + C(doc_id) + C(permno)")
    print("    uninsured_share: CAR ~ vulnerability + S:vulnerability + "
          "C(doc_id) + C(permno)")
    print("    SEs clustered on the EVENT (11 clusters, t with 10 df).")
    print(f"    Full sample: {len(d)} bank-events, {d.permno.nunique()} banks, "
          f"{d.doc_id.nunique()} events.")
    print()

    # ---------- 1. which banks, and how identified ----------
    print("-" * 100)
    print("1. WHICH BANKS ARE EXCLUDED, AND HOW THEY WERE IDENTIFIED")
    print("-" * 100)
    print()
    print("A flag already exists in the data: `is_trust_or_specialized`, built by")
    print("_charter_flags.py and carried in vulnerability_scores_wide.csv and")
    print("rq3_bridge.csv. It marks 9 banks, but 4 of them are CREDIT-CARD banks,")
    print("not custodians, so the flag is broader than 'custody/trust'. All three")
    print("definitions are reported below.")
    print()
    for tag in ("A", "B", "C"):
        ps = sets[tag]["permnos"]
        print(f"  Definition {tag} — {len(ps)} banks")
        print(f"    how: {sets[tag]['how']}")
        for p in ps:
            print(f"      permno {p:<7d} {names.get(p, '?'):<40s} "
                  f"[{reason.get(p, '') or 'not flagged'}]")
        print()

    # ---------- 2. beta, full vs ex-custody ----------
    print("-" * 100)
    print("2. INTERACTION COEFFICIENT, FULL SAMPLE vs EX-CUSTODY")
    print("-" * 100)
    print()
    full = {k: fit_beta(d, k) for k in MEASURES}
    results: dict[str, dict] = {}
    for tag in ("A", "B", "C"):
        drop = set(sets[tag]["permnos"])
        sub = d[~d.permno.isin(drop)]
        results[tag] = {k: fit_beta(sub, k) for k in MEASURES}

    hdr = ("  {:<17s} {:>15s} {:>11s} {:>7s} {:>15s} {:>11s} {:>7s}"
           .format("measure", "beta full", "p full", "N full",
                   "beta ex-cust", "p ex-cust", "N ex"))
    for tag in ("A", "B", "C"):
        n_drop = len(sets[tag]["permnos"])
        print(f"  Definition {tag} — dropping {n_drop} banks")
        print(hdr)
        print("  " + "-" * 96)
        for k in MEASURES:
            f_, e_ = full[k], results[tag][k]
            print("  {:<17s} {:>15.8f} {:>11.8f} {:>7d} {:>15.8f} {:>11.8f} {:>7d}  {}"
                  .format(NICE[k], f_["beta"], f_["p"], f_["n"],
                          e_["beta"], e_["p"], e_["n"], stars(e_["p"])))
        print()

    print("  Full detail (event-clustered SE and 95% CI, plus the two-way p):")
    print()
    print("  {:<17s} {:<14s} {:>15s} {:>13s} {:>11s} {:>11s}  {:<30s}"
          .format("measure", "sample", "beta", "SE", "p (event)", "p (2way)", "95% CI"))
    print("  " + "-" * 120)
    for k in MEASURES:
        f_ = full[k]
        print("  {:<17s} {:<14s} {:>15.8f} {:>13.8f} {:>11.8f} {:>11.8f}  "
              "[{:+.8f}, {:+.8f}]".format(
                  NICE[k], "full", f_["beta"], f_["se"], f_["p"], f_["p_twoway"],
                  f_["ci"][0], f_["ci"][1]))
        for tag in ("A", "B", "C"):
            e_ = results[tag][k]
            print("  {:<17s} {:<14s} {:>15.8f} {:>13.8f} {:>11.8f} {:>11.8f}  "
                  "[{:+.8f}, {:+.8f}]".format(
                      "", f"ex-custody {tag}", e_["beta"], e_["se"], e_["p"],
                      e_["p_twoway"], e_["ci"][0], e_["ci"][1]))
        print()

    # ---------- 3. custody contribution ----------
    print("-" * 100)
    print("3. HOW MUCH OF THE INTERACTION IS THE CUSTODY BANKS?")
    print("-" * 100)
    print()
    print("  share = (beta_full - beta_ex) / beta_full, in per cent. A share of 100%")
    print("  means the coefficient collapses to zero once they are dropped; above")
    print("  100% means it crosses zero and changes sign.")
    print()
    print("  {:<17s} {:<8s} {:>15s} {:>15s} {:>15s} {:>12s}"
          .format("measure", "defn", "beta full", "beta ex-cust", "change", "share %"))
    print("  " + "-" * 92)
    for k in MEASURES:
        for tag in ("A", "B", "C"):
            bf, be = full[k]["beta"], results[tag][k]["beta"]
            change = be - bf
            share = (bf - be) / bf * 100 if bf != 0 else float("nan")
            flip = " (sign flips)" if bf * be < 0 else ""
            print("  {:<17s} {:<8s} {:>15.8f} {:>15.8f} {:>15.8f} {:>12.4f}{}"
                  .format(NICE[k] if tag == "A" else "", tag, bf, be, change,
                          share, flip))
        print()

    # ---------- 4. calendar confound ----------
    print("-" * 100)
    print("4. IS THE SAFEGUARD SIGNAL CONFOUNDED WITH CALENDAR TIME?")
    print("-" * 100)
    print()
    ev = (d.groupby("doc_id").agg(S=("S", "first"), t0=("t0", "first"))
            .reset_index().sort_values("t0"))
    ev["date"] = pd.to_datetime(ev.t0)
    ev["rank"] = np.arange(1, len(ev) + 1)
    ev["days"] = (ev.date - ev.date.min()).dt.days

    print(f"  The {len(ev)} events, in calendar order:")
    print()
    print("  {:>7s} {:>13s} {:>7s} {:>9s} {:>14s}"
          .format("doc_id", "date", "rank", "days", "S"))
    print("  " + "-" * 56)
    for _, r in ev.iterrows():
        print("  {:>7d} {:>13s} {:>7d} {:>9d} {:>14.8f}".format(
            int(r.doc_id), str(r.t0), int(r["rank"]), int(r.days), r.S))
    print()
    for lab, col in (("calendar rank (1..11)", "rank"), ("days since first event", "days")):
        rp, pp = stats.pearsonr(ev.S, ev[col])
        rs, ps = stats.spearmanr(ev.S, ev[col])
        print(f"  corr(S, {lab}):")
        print(f"      Pearson  r   = {rp:+.8f}   p = {pp:.8f}")
        print(f"      Spearman rho = {rs:+.8f}   p = {ps:.8f}")
    print()
    print(f"  n = {len(ev)} events.")
    print()

    # ---------- 5. read-out ----------
    print("=" * 100)
    print("5. WHAT THIS SHOWS")
    print("=" * 100)
    print()
    a = results["A"]
    for k in MEASURES:
        surv = "SURVIVES at 5%" if a[k]["p"] < 0.05 else "does NOT survive at 5%"
        print(f"  {NICE[k]:<17s} full beta {full[k]['beta']:+.8f} "
              f"(p {full[k]['p']:.8f})  ->  ex-custody A {a[k]['beta']:+.8f} "
              f"(p {a[k]['p']:.8f})  {surv}")
    print()
    rp, _ = stats.pearsonr(ev.S, ev["rank"])
    print(f"  corr(S, calendar rank) = {rp:+.8f} across {len(ev)} events.")
    print()


class _Tee:
    """Write to the console and to a buffer at once, so the saved file is exactly
    what the reader sees. Capturing instead of restructuring keeps every print()
    and therefore every number identical to the terminal-only version."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def main() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        _report()
    C.RQ3_CUSTODY.write_text(buf.getvalue())
    print(f"wrote {C.RQ3_CUSTODY}")


if __name__ == "__main__":
    main()
