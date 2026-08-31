"""rq2_reaction.py: the unconditional test of whether the communications moved bank
equity at all.

This is the primary RQ2 evidence and should be read before rq2_avg_effect.py. If
no event moved prices, there is no average reaction for the type of
communication to modulate, and the null on gamma1 follows.

Reports:
    1. the Brown-Warner cross-correlation-robust mean-CAR test, per event. Naive
       per-event cross-sectional t-statistics treat banks as independent within
       a date; they are not, and those statistics are inflated by roughly an
       order of magnitude. The robust test forms the equal-weighted portfolio of
       bank abnormal returns and judges the event-window portfolio CAR against
       that same portfolio's estimation-window time-series standard deviation.
    2. abnormal volatility, as a Beaver variance ratio, and abnormal turnover,
       so that a reaction with a zero mean but heavy trading could not be missed.
    3. the minimum detectable effect for gamma1, which turns the null into a
       bounded claim rather than an absence of evidence.

The augmented market, sector and rate model, the daily abnormal-return panel and
the portfolio test are computed in _rq2_enrich_core.py; this script drives that
computation and writes the RQ2 report.

Reads     data/processed/rq2_car.csv, the CRSP daily files, cached factor data
Writes    data/processed/rq2_reaction.txt
          data/processed/rq2_car_augmented.csv
          data/processed/rq2_enrichment_results.txt
"""

from __future__ import annotations

import sys

import pandas as pd

import config as C
from _stats import fit_cluster, mde, stars

ENRICH_TXT = C.PROC / "rq2_enrichment_results.txt"


def run_core(force: bool = False) -> None:
    """Execute the enrichment pipeline that produces the augmented CAR and the
    Part B unconditional tests. Skipped when its outputs already exist, so the
    pipeline stays idempotent and cheap to re-run."""
    if ENRICH_TXT.exists() and C.RQ2_CAR_AUG.exists() and not force:
        print(f"[rq2_reaction] reusing existing {ENRICH_TXT.name} "
              f"(pass --force to recompute)")
        return
    import _rq2_enrich_core as core
    core.main()


def _slice(text: str, start: str, end: str | None) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start)) if end else -1
    return text[i: j if j > 0 else len(text)].rstrip()


def part_b_blocks() -> tuple[list[str], dict]:
    """Pull the market-model Part B blocks verbatim, plus the augmented count."""
    txt = ENRICH_TXT.read_text()
    mkt = _slice(txt, "### MARKET MODEL", "### AUGMENTED MODEL") or txt
    aug = _slice(txt, "### AUGMENTED MODEL", None)

    b1c = _slice(mkt, "-- B1c.", "-- B2.")
    b2 = _slice(mkt, "-- B2.", "-- B3.")
    b3 = _slice(mkt, "-- B3.", "-- B4.")
    b4 = _slice(mkt, "-- B4.", "    Read:")

    def count(block: str, needle: str) -> str:
        for line in block.splitlines():
            if needle in line:
                return line.strip()
        return ""

    n_mkt = count(mkt, "events significant at 5% under the robust test")
    n_aug = count(aug, "events significant at 5% under the robust test")
    n_sig = int(n_mkt.split(":")[1].split("of")[0]) if n_mkt else -1
    return [b1c, "", b2, "", b3, "", b4, ""], {
        "n_significant": n_sig, "line_mkt": n_mkt, "line_aug": n_aug}


def gamma1_mde() -> tuple[list[str], dict]:
    """The bound on the RQ2 average effect, computed from the same specification
    that rq2_avg_effect.py reports."""
    df = pd.read_csv(C.RQ2_CAR).dropna(subset=["CAR", "S"] + C.CONTROLS)
    m = fit_cluster(df, "CAR ~ S + " + " + ".join(C.CONTROLS), ["doc_id"])
    car_sd = float(df.CAR.std())
    e = mde(m.bse["S"], float(df.S.std()), car_sd)
    L = ["=" * 78, "3. HOW BIG AN AVERAGE EFFECT COULD WE HAVE MISSED?", "=" * 78, "",
         "  MDE = 2.80 x SE, i.e. the effect that would have been detected with 80%",
         "  probability at a two-sided 5% level. This turns the RQ2 null into a",
         "  bounded claim.", "",
         f"    gamma1 estimate            {m.params['S']:+.8f}",
         f"    SE (event-clustered)       {m.bse['S']:.8f}",
         f"    MDE, raw                   {e['raw']:.8f} per unit of S",
         f"    MDE, per one-SD move in S  {e['per_sd_pp']:.6f} pp of CAR",
         f"    as a share of a CAR SD     {e['pct_of_car_sd']:.4f}%"
         f"   (CAR SD = {car_sd * 100:.6f} pp)",
         "",
         "  This bound is WIDE. gamma1 is identified off 11 events, so the design can",
         "  rule out a large average communication effect but not a modest one. The",
         "  correct sentence is 'H2 not supported, and a large effect ruled out',",
         "  not 'there is no effect'.",
         "",
         "  For contrast, the RQ3 coefficients are bounded far more tightly (~5% and",
         "  ~9.5% of a CAR SD) — see rq3_link.py. Do not describe all the nulls in",
         "  this project as equally informative; they are not.",
         ""]
    return L, {"mde_pct_car_sd": e["pct_of_car_sd"]}


def main() -> dict:
    run_core(force="--force" in sys.argv)
    blocks, info = part_b_blocks()
    mde_lines, mde_info = gamma1_mde()

    L = ["=" * 78, "RQ2 — DID THE EQUITY MARKET REACT TO CBDC COMMUNICATIONS AT ALL?",
         "=" * 78, "",
         "  The primary RQ2 result. Read this before rq2_avg_effect.txt.",
         "",
         "  Design: 11 Federal Reserve CBDC communications, 2019-2023. Market-model",
         "  abnormal returns, estimation window [t0-230, t0-31], event window",
         "  [t0-1, t0+5]. 2956 bank-events, 276 banks. CAR construction is in",
         "  rq2_car.py and is unchanged.",
         "",
         "=" * 78, "1. FIRST MOMENT — cross-correlation-robust mean-CAR test", "=" * 78,
         "",
         "  Banks are not independent within a date: a common factor moves them",
         "  together. The naive cross-sectional t-stats therefore overstate",
         "  significance badly. The Brown-Warner portfolio test below is robust to",
         "  that dependence and is the one to quote.",
         ""] + blocks

    L += ["=" * 78, "2. VERDICT ON THE UNCONDITIONAL REACTION", "=" * 78, "",
          f"    {info['line_mkt']}   (market model — the published CAR)",
          f"    {info['line_aug']}   (market+sector+rate model)",
          "",
          "  The two events that reach 5% under the augmented model are doc_id 3 and",
          "  6, and they carry OPPOSITE signs (+0.028 and -0.027), so they do not add",
          "  up to a directional reaction.",
          "",
          "  Volatility and turnover are SUPPRESSED on these dates at least as often",
          "  as they are elevated (7 of 11 against 3 of 11 on both measures). This is",
          "  not 'a null mean with violent trading underneath'. On most of these",
          "  dates bank stocks were quieter than on an ordinary day.",
          "",
          "  RQ2 CONCLUSION: the market did not react to CBDC communications on",
          "  average. Federal Reserve CBDC speeches were not events for bank equity.",
          "  Everything conditional — the average effect of communication type in",
          "  rq2_avg_effect.py, and the whole RQ3 bridge — inherits this. There is no",
          "  reaction for a cross-section to modulate.",
          ""] + mde_lines

    text = "\n".join(L)
    C.RQ2_REACTION.write_text(text + "\n")
    print(text)
    print(f"wrote {C.RQ2_REACTION}")
    return {"rq2.unconditional.n_significant": info["n_significant"],
            **mde_info}


if __name__ == "__main__":
    main()
